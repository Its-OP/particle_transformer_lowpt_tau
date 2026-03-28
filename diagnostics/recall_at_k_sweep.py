"""R@K diagnostic sweep for TrackPreFilter checkpoints.

Loads a pre-filter checkpoint, runs inference on validation data, and
reports R@K, P@K, d-prime, and GT rank statistics for a range of K values.

This is Phase 2.1 of the improvement blueprint: confirm cascade viability
by verifying R@K₁ ≥ 0.90 at the chosen cascade cut K₁ ≈ 600.

Usage:
    python diagnostics/recall_at_k_sweep.py \\
        --checkpoint models/prefilter_best.pt \\
        --data-config data/low-pt/lowpt_tau_trackfinder.yaml \\
        --data-dir data/low-pt/val/ \\
        --device mps
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weaver.nn.model.TrackPreFilter import TrackPreFilter
from weaver.utils.dataset import SimpleIterDataset

from utils.training_utils import (
    MetricsAccumulator,
    extract_label_from_inputs,
    trim_to_max_valid_tracks,
)

logger = logging.getLogger('recall_at_k_sweep')


def load_prefilter_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> TrackPreFilter:
    """Load a TrackPreFilter model from a training checkpoint.

    Reads the model constructor arguments from the checkpoint's saved args
    and reconstructs the model with the same configuration.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        device: Target device.

    Returns:
        Loaded model in eval mode.
    """
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False,
    )

    # Reconstruct model config from saved training args
    saved_args = checkpoint.get('args', {})
    network_path = saved_args.get('network', 'networks/lowpt_tau_TrackPreFilter.py')

    # Load network wrapper to get model config
    # The wrapper's get_model() needs a data_config object.
    # Instead, we extract config from the state dict shape.
    state_dict = checkpoint['model_state_dict']

    # Infer input_dim from the first Conv1d layer weight shape.
    # TrackPreFilter uses track_mlp.0.weight: (hidden_dim, input_dim, 1)
    first_layer_key = 'track_mlp.0.weight'
    if first_layer_key not in state_dict:
        raise ValueError(
            f'Cannot infer model dimensions from checkpoint. '
            f'Expected key "{first_layer_key}" not found. '
            f'Keys: {list(state_dict.keys())[:10]}'
        )
    hidden_dim = state_dict[first_layer_key].shape[0]
    input_dim = state_dict[first_layer_key].shape[1]

    # Infer num_message_rounds from state dict keys
    message_round_keys = [
        key for key in state_dict
        if key.startswith('message_mlps.')
    ]
    if message_round_keys:
        round_indices = {
            int(key.split('.')[1]) for key in message_round_keys
        }
        num_message_rounds = max(round_indices) + 1
    else:
        num_message_rounds = 2

    model = TrackPreFilter(
        mode='mlp',
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_neighbors=16,
        num_message_rounds=num_message_rounds,
        ranking_num_samples=50,
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', '?')
    best_recall = checkpoint.get('best_val_recall_at_200', 0.0)
    logger.info(
        f'Loaded checkpoint: {checkpoint_path} '
        f'(epoch {epoch}, R@200={best_recall:.4f})'
    )
    return model


def run_recall_sweep(
    checkpoint_path: str,
    data_config_path: str,
    data_dir: str,
    k_values: tuple[int, ...] = (
        10, 20, 30, 50, 100, 200, 300, 400, 500, 600, 800,
    ),
    device: torch.device | None = None,
    batch_size: int = 32,
    max_steps: int | None = None,
    num_workers: int = 0,
) -> dict[str, float]:
    """Run R@K sweep on validation data.

    Args:
        checkpoint_path: Path to pre-filter checkpoint.
        data_config_path: Path to data config YAML.
        data_dir: Directory with validation parquet files.
        k_values: K values for recall@K computation.
        device: Compute device (defaults to cpu).
        batch_size: Batch size for inference.
        max_steps: Optional limit on number of batches (for smoke tests).
        num_workers: DataLoader worker count.

    Returns:
        Dict with recall_at_K, perfect_at_K, d_prime, median_gt_rank, etc.
    """
    if device is None:
        device = torch.device('cpu')

    # Load model
    model = load_prefilter_from_checkpoint(checkpoint_path, device)

    # Load data
    parquet_files = sorted(glob.glob(os.path.join(data_dir, '*.parquet')))
    if not parquet_files:
        raise FileNotFoundError(f'No parquet files in {data_dir}')

    logger.info(f'Found {len(parquet_files)} parquet files in {data_dir}')

    dataset = SimpleIterDataset(
        {'data': parquet_files},
        data_config_file=data_config_path,
        for_training=False,
        load_range_and_fraction=((0.0, 1.0), 1.0),
        fetch_by_files=True,
        fetch_step=len(parquet_files),
        in_memory=True,
    )
    data_config = dataset.config

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        pin_memory=False,
        num_workers=num_workers,
    )

    input_names = list(data_config.input_names)
    mask_input_index = input_names.index('pf_mask')
    label_input_index = input_names.index('pf_label')

    # Run inference and accumulate metrics
    accumulator = MetricsAccumulator(k_values=k_values)

    # Use train() mode for BatchNorm: the training validate() function runs
    # compute_loss() in train() mode (batch statistics), so running statistics
    # may not reflect the true distribution. Using train() mode here matches
    # the conditions under which R@K was measured during training.
    model.train()
    with torch.no_grad():
        for batch_index, (X, y, _) in enumerate(data_loader):
            if max_steps is not None and batch_index >= max_steps:
                break

            inputs = [X[key].to(device) for key in data_config.input_names]
            inputs = trim_to_max_valid_tracks(inputs, mask_input_index)
            model_inputs, track_labels = extract_label_from_inputs(
                inputs, label_input_index,
            )
            points, features, lorentz_vectors, mask = model_inputs

            # Forward pass — returns (B, P) per-track scores
            per_track_scores = model(points, features, lorentz_vectors, mask)

            accumulator.update(per_track_scores, track_labels, mask)

            if (batch_index + 1) % 10 == 0:
                logger.info(
                    f'Batch {batch_index + 1}: '
                    f'{accumulator.total_events_with_gt} events processed'
                )

            del inputs, model_inputs, track_labels, per_track_scores

    metrics = accumulator.compute()

    logger.info(
        f'Sweep complete: {accumulator.total_events_with_gt} events, '
        f'{accumulator.total_gt_tracks} GT tracks'
    )

    return metrics


def print_sweep_results(
    metrics: dict[str, float],
    k_values: tuple[int, ...],
) -> None:
    """Print a formatted R@K sweep table."""
    print('\n' + '=' * 65)
    print('  R@K DIAGNOSTIC SWEEP')
    print('=' * 65)
    print(f'  Events: {metrics.get("total_events_with_gt", 0):,}')
    print(f'  GT tracks: {metrics.get("total_gt_tracks", 0):,}')
    print(f'  d-prime: {metrics.get("d_prime", 0):.3f}')
    print(f'  Median GT rank: {metrics.get("median_gt_rank", 0):.0f}')
    print(
        f'  GT rank p90: {metrics.get("gt_rank_p90", 0):.0f}  '
        f'p95: {metrics.get("gt_rank_p95", 0):.0f}'
    )
    print('-' * 65)
    print(f'  {"K":>6}  {"R@K":>8}  {"P@K":>8}  {"Cascade viable?":>18}')
    print('-' * 65)

    for k in k_values:
        recall = metrics.get(f'recall_at_{k}', 0.0)
        perfect = metrics.get(f'perfect_at_{k}', 0.0)
        viable = 'YES' if recall >= 0.90 else ('close' if recall >= 0.85 else '')
        print(f'  {k:>6}  {recall:>8.4f}  {perfect:>8.4f}  {viable:>18}')

    print('=' * 65)

    # Identify the smallest K where R@K >= 0.90
    for k in k_values:
        recall = metrics.get(f'recall_at_{k}', 0.0)
        if recall >= 0.90:
            print(f'\n  Cascade cut recommendation: K₁ = {k} (R@{k} = {recall:.4f})')
            break
    else:
        print('\n  WARNING: No K achieves R@K >= 0.90')

    print()


def main():
    parser = argparse.ArgumentParser(
        description='R@K diagnostic sweep for TrackPreFilter',
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to pre-filter checkpoint (.pt)',
    )
    parser.add_argument('--data-config', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Optional path to save metrics JSON',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    device = torch.device(args.device)
    k_values = (10, 20, 30, 50, 100, 200, 300, 400, 500, 600, 800)

    metrics = run_recall_sweep(
        checkpoint_path=args.checkpoint,
        data_config_path=args.data_config,
        data_dir=args.data_dir,
        k_values=k_values,
        device=device,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        num_workers=args.num_workers,
    )

    print_sweep_results(metrics, k_values)

    if args.output:
        with open(args.output, 'w') as output_file:
            json.dump(metrics, output_file, indent=2)
        logger.info(f'Saved metrics to {args.output}')


if __name__ == '__main__':
    main()
