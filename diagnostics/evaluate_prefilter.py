"""Evaluate and compare TrackPreFilter checkpoints on validation data.

Produces per-model metrics stratified by:
  - Overall aggregate (R@K, P@K, d', median GT rank)
  - Number of tracks per event (low/medium/high)
  - Number of GT tracks per event (1, 2, 3)
  - Score distribution analysis (GT vs background)
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from weaver.utils.dataset import SimpleIterDataset

from utils.training_utils import (
    extract_label_from_inputs,
    load_network_module,
    trim_to_max_valid_tracks,
)

logger = logging.getLogger('evaluate_prefilter')


def compute_per_event_metrics(
    per_track_scores: torch.Tensor,
    track_labels: torch.Tensor,
    mask: torch.Tensor,
    k_values: tuple[int, ...] = (10, 20, 30, 100, 200),
) -> list[dict]:
    """Compute metrics for each event individually.

    Returns a list of dicts, one per event in the batch.
    Each dict contains:
        - num_tracks: number of valid tracks
        - num_gt_tracks: number of GT tracks
        - recall_at_K: fraction of GT found in top-K
        - perfect_at_K: 1 if all GT found in top-K, else 0
        - gt_scores: list of GT track scores
        - background_scores: list of background track scores (sampled)
        - gt_ranks: list of ranks for each GT track
    """
    batch_size = per_track_scores.shape[0]
    labels_flat = track_labels.squeeze(1) * mask.squeeze(1).float()
    valid_mask = mask.squeeze(1).bool()

    masked_scores = per_track_scores.clone()
    masked_scores[~valid_mask] = float('-inf')

    # Compute ranks: argsort(argsort(descending)) gives rank of each element
    rank_lookup = torch.argsort(
        torch.argsort(masked_scores, dim=1, descending=True), dim=1,
    )

    sorted_indices = masked_scores.argsort(dim=1, descending=True)

    results = []
    for batch_index in range(batch_size):
        event_valid = valid_mask[batch_index]
        event_labels = labels_flat[batch_index]
        event_scores = per_track_scores[batch_index]

        num_tracks = event_valid.sum().item()
        gt_mask = (event_labels == 1.0) & event_valid
        background_mask = (event_labels == 0.0) & event_valid
        num_gt = gt_mask.sum().item()

        event_result = {
            'num_tracks': num_tracks,
            'num_gt_tracks': int(num_gt),
        }

        # Score distributions
        if gt_mask.any():
            event_result['gt_scores'] = event_scores[gt_mask].cpu().numpy()
        else:
            event_result['gt_scores'] = np.array([])

        if background_mask.any():
            # Sample up to 200 background scores to keep memory manageable
            background_scores = event_scores[background_mask].cpu().numpy()
            if len(background_scores) > 200:
                rng = np.random.default_rng(batch_index)
                indices = rng.choice(len(background_scores), 200, replace=False)
                background_scores = background_scores[indices]
            event_result['background_scores'] = background_scores
        else:
            event_result['background_scores'] = np.array([])

        if num_gt == 0:
            for k in k_values:
                event_result[f'recall_at_{k}'] = float('nan')
                event_result[f'perfect_at_{k}'] = float('nan')
            event_result['gt_ranks'] = np.array([])
            event_result['median_gt_rank'] = float('nan')
        else:
            gt_positions = gt_mask.nonzero(as_tuple=True)[0]
            gt_ranks = rank_lookup[batch_index, gt_positions].cpu().numpy()
            event_result['gt_ranks'] = gt_ranks
            event_result['median_gt_rank'] = float(np.median(gt_ranks))

            for k in k_values:
                top_k_indices = sorted_indices[batch_index, :k]
                found = torch.isin(gt_positions, top_k_indices).sum().item()
                event_result[f'recall_at_{k}'] = found / num_gt
                event_result[f'perfect_at_{k}'] = 1.0 if found == num_gt else 0.0

        results.append(event_result)

    return results


def aggregate_metrics(
    event_metrics: list[dict],
    k_values: tuple[int, ...] = (10, 20, 30, 100, 200),
) -> dict:
    """Aggregate per-event metrics into summary statistics."""
    # Filter to events with GT tracks
    events_with_gt = [
        event for event in event_metrics if event['num_gt_tracks'] > 0
    ]

    if not events_with_gt:
        return {'num_events': 0}

    result = {
        'num_events': len(events_with_gt),
        'num_events_total': len(event_metrics),
    }

    # Recall and perfect metrics
    for k in k_values:
        recalls = [event[f'recall_at_{k}'] for event in events_with_gt]
        perfects = [event[f'perfect_at_{k}'] for event in events_with_gt]
        result[f'recall_at_{k}'] = np.mean(recalls)
        result[f'perfect_at_{k}'] = np.mean(perfects)

    # Median GT rank
    median_ranks = [event['median_gt_rank'] for event in events_with_gt]
    result['median_gt_rank'] = np.mean(median_ranks)

    # d-prime from pooled score distributions
    all_gt_scores = np.concatenate(
        [event['gt_scores'] for event in events_with_gt if len(event['gt_scores']) > 0]
    )
    all_background_scores = np.concatenate(
        [event['background_scores'] for event in events_with_gt
         if len(event['background_scores']) > 0]
    )
    if len(all_gt_scores) > 1 and len(all_background_scores) > 1:
        gt_mean = np.mean(all_gt_scores)
        background_mean = np.mean(all_background_scores)
        gt_std = np.std(all_gt_scores, ddof=1)
        background_std = np.std(all_background_scores, ddof=1)
        # d' = (mu_gt - mu_bg) / sqrt(0.5 * (sigma_gt^2 + sigma_bg^2))
        pooled_std = np.sqrt(0.5 * (gt_std**2 + background_std**2))
        result['d_prime'] = (gt_mean - background_mean) / max(pooled_std, 1e-8)
        result['gt_score_mean'] = float(gt_mean)
        result['gt_score_std'] = float(gt_std)
        result['background_score_mean'] = float(background_mean)
        result['background_score_std'] = float(background_std)
    else:
        result['d_prime'] = float('nan')

    return result


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    data_config,
    mask_input_index: int,
    label_input_index: int,
    max_steps: int | None = None,
) -> list[dict]:
    """Run model on data and return per-event metrics."""
    model.eval()
    all_event_metrics = []

    for batch_index, (features, labels, _) in enumerate(data_loader):
        if max_steps is not None and batch_index >= max_steps:
            break

        inputs = [features[k].to(device) for k in data_config.input_names]
        inputs = trim_to_max_valid_tracks(inputs, mask_input_index)
        model_inputs, track_labels = extract_label_from_inputs(
            inputs, label_input_index,
        )
        points, feature_tensor, lorentz_vectors, mask = model_inputs

        # Forward pass to get scores
        per_track_scores = model(
            points, feature_tensor, lorentz_vectors, mask,
        )

        batch_metrics = compute_per_event_metrics(
            per_track_scores, track_labels, mask,
        )
        all_event_metrics.extend(batch_metrics)

        if (batch_index + 1) % 20 == 0:
            logger.info(
                f'  Processed {batch_index + 1} batches '
                f'({len(all_event_metrics)} events)'
            )

        del inputs, model_inputs, track_labels, per_track_scores

    return all_event_metrics


def stratify_by_track_count(
    event_metrics: list[dict],
) -> dict[str, list[dict]]:
    """Split events into track count buckets.

    Buckets:
      - low: <500 tracks
      - medium: 500-1500 tracks
      - high: >1500 tracks
    """
    buckets = {
        'low (<500)': [],
        'medium (500-1500)': [],
        'high (>1500)': [],
    }
    for event in event_metrics:
        num_tracks = event['num_tracks']
        if num_tracks < 500:
            buckets['low (<500)'].append(event)
        elif num_tracks <= 1500:
            buckets['medium (500-1500)'].append(event)
        else:
            buckets['high (>1500)'].append(event)
    return buckets


def stratify_by_gt_count(
    event_metrics: list[dict],
) -> dict[str, list[dict]]:
    """Split events by number of GT tracks."""
    buckets = defaultdict(list)
    for event in event_metrics:
        num_gt = event['num_gt_tracks']
        buckets[f'{num_gt} GT tracks'].append(event)
    return dict(sorted(buckets.items()))


def print_comparison_table(
    name: str,
    metrics_widened: dict,
    metrics_phase_a: dict,
    k_values: tuple[int, ...] = (10, 20, 30, 100, 200),
):
    """Print a formatted comparison table."""
    print(f'\n{"=" * 70}')
    print(f'  {name}')
    print(f'{"=" * 70}')

    n_widened = metrics_widened.get('num_events', 0)
    n_phase_a = metrics_phase_a.get('num_events', 0)
    print(f'  Events with GT:  widened={n_widened}  phase-a={n_phase_a}')
    print(f'{"-" * 70}')

    header = f'  {"Metric":<20} {"Widened":>10} {"Phase-A":>10} {"Delta":>10}'
    print(header)
    print(f'  {"-" * 50}')

    rows = []
    for k in k_values:
        key = f'recall_at_{k}'
        value_widened = metrics_widened.get(key, float('nan'))
        value_phase_a = metrics_phase_a.get(key, float('nan'))
        delta = value_phase_a - value_widened
        rows.append((f'R@{k}', value_widened, value_phase_a, delta))

    for k in k_values:
        key = f'perfect_at_{k}'
        value_widened = metrics_widened.get(key, float('nan'))
        value_phase_a = metrics_phase_a.get(key, float('nan'))
        delta = value_phase_a - value_widened
        rows.append((f'P@{k}', value_widened, value_phase_a, delta))

    for key, label in [('d_prime', "d'"), ('median_gt_rank', 'Med. rank')]:
        value_widened = metrics_widened.get(key, float('nan'))
        value_phase_a = metrics_phase_a.get(key, float('nan'))
        delta = value_phase_a - value_widened
        rows.append((label, value_widened, value_phase_a, delta))

    for label, value_widened, value_phase_a, delta in rows:
        sign = '+' if delta >= 0 else ''
        print(
            f'  {label:<20} {value_widened:>10.4f} '
            f'{value_phase_a:>10.4f} {sign}{delta:>9.4f}'
        )


def load_model_from_checkpoint(
    checkpoint_path: str,
    model_config: dict,
    device: torch.device,
) -> torch.nn.Module:
    """Load a TrackPreFilter model from a checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        model_config: Dict of model constructor kwargs.
        device: Target device.

    Returns:
        Loaded model in eval mode.
    """
    from weaver.nn.model.TrackPreFilter import TrackPreFilter

    model = TrackPreFilter(**model_config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    logger.info(
        f'Loaded checkpoint from {checkpoint_path} '
        f'(epoch {checkpoint.get("epoch", "?")})'
    )
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate and compare TrackPreFilter checkpoints',
    )
    parser.add_argument(
        '--widened-checkpoint', type=str, required=True,
        help='Path to widened (hybrid) model checkpoint',
    )
    parser.add_argument(
        '--phase-a-checkpoint', type=str, required=True,
        help='Path to Phase-A (MLP) model checkpoint',
    )
    parser.add_argument('--data-config', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Optional JSON output path for detailed results',
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    device = torch.device(args.device)
    logger.info(f'Device: {device}')

    # ---- Data loading ----
    import glob
    parquet_files = sorted(glob.glob(f'{args.data_dir}/*.parquet'))
    logger.info(f'Found {len(parquet_files)} parquet files in {args.data_dir}')
    file_dict = {'data': parquet_files}

    dataset = SimpleIterDataset(
        file_dict,
        data_config_file=args.data_config,
        for_training=False,
        load_range_and_fraction=((0.0, 1.0), 1.0),
        fetch_by_files=True,
        fetch_step=len(parquet_files),
        in_memory=True,
    )
    data_config = dataset.config

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        drop_last=False,
        pin_memory=False,
        num_workers=args.num_workers,
    )

    input_names = list(data_config.input_names)
    mask_input_index = input_names.index('pf_mask')
    label_input_index = input_names.index('pf_label')

    # ---- Model configs ----
    widened_config = dict(
        mode='hybrid',
        input_dim=13,
        hidden_dim=192,
        latent_dim=48,
        num_neighbors=16,
        num_message_rounds=2,
        ranking_num_samples=50,
    )
    phase_a_config = dict(
        mode='mlp',
        input_dim=13,
        hidden_dim=192,
        num_neighbors=16,
        num_message_rounds=2,
        ranking_num_samples=50,
        ranking_temperature_start=2.0,
        ranking_temperature_end=0.5,
        denoising_sigma_start=1.0,
        denoising_sigma_end=0.1,
        drw_warmup_fraction=0.3,
        drw_positive_weight=2.0,
    )

    # ---- Evaluate widened model ----
    logger.info('=' * 50)
    logger.info('Evaluating WIDENED (hybrid) model...')
    widened_model = load_model_from_checkpoint(
        args.widened_checkpoint, widened_config, device,
    )
    widened_events = evaluate_model(
        widened_model, data_loader, device, data_config,
        mask_input_index, label_input_index, args.max_steps,
    )
    del widened_model
    torch.mps.empty_cache() if device.type == 'mps' else None
    logger.info(f'Widened: {len(widened_events)} events evaluated')

    # ---- Re-create data loader for phase-a ----
    # (SimpleIterDataset is consumed after one pass)
    dataset_phase_a = SimpleIterDataset(
        file_dict,
        data_config_file=args.data_config,
        for_training=False,
        load_range_and_fraction=((0.0, 1.0), 1.0),
        fetch_by_files=True,
        fetch_step=len(parquet_files),
        in_memory=True,
    )
    data_loader_phase_a = DataLoader(
        dataset_phase_a,
        batch_size=args.batch_size,
        drop_last=False,
        pin_memory=False,
        num_workers=args.num_workers,
    )

    # ---- Evaluate phase-a model ----
    logger.info('=' * 50)
    logger.info('Evaluating PHASE-A (MLP) model...')
    phase_a_model = load_model_from_checkpoint(
        args.phase_a_checkpoint, phase_a_config, device,
    )
    phase_a_events = evaluate_model(
        phase_a_model, data_loader_phase_a, device, data_config,
        mask_input_index, label_input_index, args.max_steps,
    )
    del phase_a_model
    logger.info(f'Phase-A: {len(phase_a_events)} events evaluated')

    # ---- Overall comparison ----
    widened_overall = aggregate_metrics(widened_events)
    phase_a_overall = aggregate_metrics(phase_a_events)
    print_comparison_table('OVERALL', widened_overall, phase_a_overall)

    # ---- Stratify by track count ----
    widened_by_tracks = stratify_by_track_count(widened_events)
    phase_a_by_tracks = stratify_by_track_count(phase_a_events)

    for bucket_name in widened_by_tracks:
        widened_bucket = aggregate_metrics(widened_by_tracks[bucket_name])
        phase_a_bucket = aggregate_metrics(phase_a_by_tracks[bucket_name])
        if widened_bucket['num_events'] > 0 or phase_a_bucket['num_events'] > 0:
            print_comparison_table(
                f'Track count: {bucket_name}',
                widened_bucket, phase_a_bucket,
            )

    # ---- Stratify by GT count ----
    widened_by_gt = stratify_by_gt_count(widened_events)
    phase_a_by_gt = stratify_by_gt_count(phase_a_events)

    all_gt_keys = sorted(set(widened_by_gt.keys()) | set(phase_a_by_gt.keys()))
    for gt_key in all_gt_keys:
        widened_bucket = aggregate_metrics(widened_by_gt.get(gt_key, []))
        phase_a_bucket = aggregate_metrics(phase_a_by_gt.get(gt_key, []))
        if widened_bucket.get('num_events', 0) > 0 or phase_a_bucket.get('num_events', 0) > 0:
            print_comparison_table(gt_key, widened_bucket, phase_a_bucket)

    # ---- Score distribution summary ----
    print(f'\n{"=" * 70}')
    print('  SCORE DISTRIBUTION SUMMARY')
    print(f'{"=" * 70}')
    for model_name, overall in [('Widened', widened_overall), ('Phase-A', phase_a_overall)]:
        print(f'  {model_name}:')
        print(f'    GT scores:    mean={overall.get("gt_score_mean", 0):.4f}  '
              f'std={overall.get("gt_score_std", 0):.4f}')
        print(f'    Bkg scores:   mean={overall.get("background_score_mean", 0):.4f}  '
              f'std={overall.get("background_score_std", 0):.4f}')
        print(f"    d':           {overall.get('d_prime', 0):.4f}")
        print()

    # ---- Rank distribution for GT tracks ----
    print(f'{"=" * 70}')
    print('  GT RANK DISTRIBUTION')
    print(f'{"=" * 70}')

    for model_name, events in [('Widened', widened_events), ('Phase-A', phase_a_events)]:
        all_ranks = np.concatenate([
            event['gt_ranks'] for event in events
            if event['num_gt_tracks'] > 0 and len(event['gt_ranks']) > 0
        ])
        percentiles = np.percentile(all_ranks, [5, 10, 25, 50, 75, 90, 95])
        print(f'  {model_name} GT rank percentiles:')
        print(f'    p5={percentiles[0]:.0f}  p10={percentiles[1]:.0f}  '
              f'p25={percentiles[2]:.0f}  p50={percentiles[3]:.0f}  '
              f'p75={percentiles[4]:.0f}  p90={percentiles[5]:.0f}  '
              f'p95={percentiles[6]:.0f}')

        # Fraction of GT tracks in various rank buckets
        in_top10 = np.mean(all_ranks < 10) * 100
        in_top50 = np.mean(all_ranks < 50) * 100
        in_top100 = np.mean(all_ranks < 100) * 100
        in_top200 = np.mean(all_ranks < 200) * 100
        in_top500 = np.mean(all_ranks < 500) * 100
        print(f'    GT in top-10: {in_top10:.1f}%  top-50: {in_top50:.1f}%  '
              f'top-100: {in_top100:.1f}%  top-200: {in_top200:.1f}%  '
              f'top-500: {in_top500:.1f}%')
        print()

    # ---- Save detailed results ----
    if args.output:
        output_data = {
            'overall': {
                'widened': {k: v for k, v in widened_overall.items()
                           if not isinstance(v, np.ndarray)},
                'phase_a': {k: v for k, v in phase_a_overall.items()
                           if not isinstance(v, np.ndarray)},
            },
        }
        with open(args.output, 'w') as output_file:
            json.dump(output_data, output_file, indent=2, default=float)
        logger.info(f'Detailed results saved to {args.output}')


if __name__ == '__main__':
    main()
