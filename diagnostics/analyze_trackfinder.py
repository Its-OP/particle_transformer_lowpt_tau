"""Diagnostic analysis of a trained track finder model.

Loads a trained checkpoint, runs inference on the full dataset, and prints
per-event recall statistics, score distributions, and rank distributions.
Helps identify WHERE the model fails before redesigning the architecture.

Usage:
    cd part/
    python analyze_trackfinder.py \
        --data-config data/low-pt/lowpt_tau_trackfinder.yaml \
        --data-dir data/low-pt/ \
        --network networks/lowpt_tau_TrackFinderV2.py \
        --checkpoint models/debug_checkpoints/trackfinder_simple_improved/checkpoints/best_model.pt \
        --pretrained-backbone models/backbone_best.pt \
        --device mps
"""
from __future__ import annotations

import argparse
import os
import sys

# Add parent directory (part/) to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from weaver.utils.dataset import SimpleIterDataset

from utils.training_utils import (
    extract_label_from_inputs,
    extract_per_track_scores,
    load_network_module,
    trim_to_max_valid_tracks,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Analyze track finder model')
    parser.add_argument(
        '--data-config', required=True, help='Path to YAML data config',
    )
    parser.add_argument(
        '--data-dir', required=True, help='Directory with parquet files',
    )
    parser.add_argument(
        '--network', required=True, help='Path to network wrapper .py file',
    )
    parser.add_argument(
        '--checkpoint', required=True, help='Path to model checkpoint .pt',
    )
    parser.add_argument(
        '--pretrained-backbone', default=None,
        help='Path to pretrained backbone .pt (needed for model construction)',
    )
    parser.add_argument(
        '--device', default='mps', help='Device (mps, cuda:0, cpu)',
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Batch size for inference',
    )
    parser.add_argument(
        '--max-batches', type=int, default=None,
        help='Max batches to process (None = all)',
    )
    parser.add_argument(
        '--backbone-mode', default=None,
        help='Backbone mode override (frozen, parallel). '
             'Passed to network wrapper as backbone_mode kwarg.',
    )
    return parser.parse_args()


def build_model_and_load_checkpoint(
    network_path: str,
    data_config,
    checkpoint_path: str,
    pretrained_backbone_path: str | None,
    device: torch.device,
    backbone_mode: str | None = None,
) -> torch.nn.Module:
    """Construct model from network wrapper and load checkpoint weights."""
    network_module = load_network_module(network_path)

    model_kwargs = {}
    if pretrained_backbone_path is not None:
        model_kwargs['pretrained_backbone_path'] = pretrained_backbone_path
    if backbone_mode is not None:
        model_kwargs['backbone_mode'] = backbone_mode

    model, _ = network_module.get_model(data_config, **model_kwargs)
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f'Model loaded: {total_params:,} params ({trainable_params:,} trainable)')

    return model


@torch.no_grad()
def collect_per_event_statistics(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    label_input_index: int,
    max_batches: int | None = None,
) -> dict[str, list]:
    """Run inference on all batches, collect per-event statistics.

    Returns dict with lists of per-event values:
        - recall_at_K: fraction of GT found in top K
        - gt_pion_ranks: ranks of GT pions in the sorted score list
        - gt_pion_scores: model scores for GT pions
        - background_scores: model scores for background tracks (sampled)
        - event_track_count: number of valid tracks per event
        - gt_pion_pt: transverse momentum of GT pions
        - gt_pion_dxy_sig: dxy_significance of GT pions
    """
    results = {
        'recall_at_10': [],
        'recall_at_20': [],
        'recall_at_30': [],
        'recall_at_50': [],
        'recall_at_100': [],
        'gt_pion_ranks': [],
        'gt_pion_scores': [],
        'background_scores': [],
        'event_track_count': [],
        'gt_pion_pt': [],
        'gt_pion_dxy_significance': [],
        'num_gt_per_event': [],
    }

    num_batches_processed = 0

    for inputs in data_loader:
        # SimpleIterDataset returns (input_dict, label_dict, observer_dict)
        # Unpack input tensors in the order defined by data_config.input_names
        input_dict = inputs[0]
        input_names = list(input_dict.keys())

        # Build flat tensor list matching the training script's expected order
        flat_inputs = [input_dict[name].to(device) for name in input_names]
        model_inputs, track_labels = extract_label_from_inputs(
            flat_inputs, label_input_index,
        )

        # Trim padding (same as training)
        # mask is at index 3 in model_inputs (after label removal at index 4)
        mask_index = input_names.index('pf_mask')
        if mask_index > label_input_index:
            mask_index -= 1  # Adjust for removed label
        model_inputs = list(trim_to_max_valid_tracks(model_inputs, mask_index))

        # Run inference
        output_dict = model(*model_inputs)
        per_track_scores = extract_per_track_scores(output_dict)

        # Get mask and labels
        mask = model_inputs[3]  # (B, 1, P)
        mask_flat = mask.squeeze(1).bool()  # (B, P)
        labels_flat = (
            track_labels.squeeze(1)[:, :mask_flat.shape[1]]
            * mask_flat.float()
        )  # (B, P)

        # Raw features for physics analysis (after trim)
        features = model_inputs[1]  # (B, 7, P) — standardized features
        lorentz_vectors = model_inputs[2]  # (B, 4, P) — raw (px, py, pz, E)

        batch_size = per_track_scores.shape[0]

        for event_index in range(batch_size):
            event_mask = mask_flat[event_index]  # (P,)
            event_labels = labels_flat[event_index]  # (P,)
            event_scores = per_track_scores[event_index]  # (P,)

            # Valid track count
            num_valid = event_mask.sum().item()
            results['event_track_count'].append(num_valid)

            # GT positions
            gt_positions = event_labels.nonzero(as_tuple=True)[0]
            num_gt = len(gt_positions)
            results['num_gt_per_event'].append(num_gt)

            if num_gt == 0:
                continue

            # Track count aligned with recall arrays (only events with GT)
            results.setdefault('event_track_count_with_gt', []).append(num_valid)

            # Sort scores (descending), masking padded tracks
            masked_event_scores = event_scores.clone()
            masked_event_scores[~event_mask] = float('-inf')
            sorted_indices = masked_event_scores.argsort(descending=True)

            # Compute rank of each GT pion
            rank_lookup = torch.zeros_like(sorted_indices)
            rank_lookup[sorted_indices] = torch.arange(
                len(sorted_indices), device=sorted_indices.device,
            )

            gt_set = set(gt_positions.tolist())
            sorted_list = sorted_indices.tolist()

            # Recall@K
            for k_value in [10, 20, 30, 50, 100]:
                actual_k = min(k_value, num_valid)
                top_k_set = set(sorted_list[:actual_k])
                found = len(top_k_set & gt_set)
                recall = found / num_gt
                results[f'recall_at_{k_value}'].append(recall)

            # Per-GT-pion statistics
            for gt_position in gt_positions:
                gt_idx = gt_position.item()
                rank = rank_lookup[gt_idx].item()
                score = event_scores[gt_idx].item()
                results['gt_pion_ranks'].append(rank)
                results['gt_pion_scores'].append(score)

                # Compute pT from Lorentz vectors: pT = sqrt(px^2 + py^2)
                px_value = lorentz_vectors[event_index, 0, gt_idx].item()
                py_value = lorentz_vectors[event_index, 1, gt_idx].item()
                pt_value = np.sqrt(px_value**2 + py_value**2)
                results['gt_pion_pt'].append(pt_value)

                # dxy_significance is feature index 6 (standardized)
                dxy_significance_value = features[event_index, 6, gt_idx].item()
                results['gt_pion_dxy_significance'].append(dxy_significance_value)

            # Sample background scores (up to 50 per event for memory)
            background_positions = (
                (event_labels == 0) & event_mask
            ).nonzero(as_tuple=True)[0]
            if len(background_positions) > 50:
                sample_indices = torch.randperm(len(background_positions))[:50]
                background_positions = background_positions[sample_indices]
            for background_position in background_positions:
                results['background_scores'].append(
                    event_scores[background_position.item()].item(),
                )

        num_batches_processed += 1
        if num_batches_processed % 10 == 0:
            print(
                f'  Processed {num_batches_processed} batches '
                f'({num_batches_processed * batch_size} events)...',
            )

        if max_batches is not None and num_batches_processed >= max_batches:
            break

    print(f'Total: {num_batches_processed} batches processed')
    return results


def print_recall_statistics(results: dict[str, list]) -> None:
    """Print recall@K statistics."""
    print('\n' + '=' * 70)
    print('RECALL@K STATISTICS')
    print('=' * 70)

    for k_value in [10, 20, 30, 50, 100]:
        key = f'recall_at_{k_value}'
        values = results[key]
        if not values:
            continue
        values_array = np.array(values)
        print(
            f'  R@{k_value:3d}: '
            f'mean={values_array.mean():.4f}  '
            f'median={np.median(values_array):.4f}  '
            f'std={values_array.std():.4f}  '
            f'min={values_array.min():.4f}  '
            f'max={values_array.max():.4f}',
        )


def print_recall_distribution(results: dict[str, list]) -> None:
    """Print distribution of per-event recall (what fraction of events get X% recall)."""
    print('\n' + '=' * 70)
    print('PER-EVENT RECALL DISTRIBUTION (R@30)')
    print('=' * 70)

    values = np.array(results['recall_at_30'])
    if len(values) == 0:
        print('  No events with GT tracks')
        return

    # For 3-prong tau: recall can be 0/3, 1/3, 2/3, 3/3
    bins = [(-0.01, 0.01), (0.01, 0.34), (0.34, 0.67), (0.67, 1.01)]
    labels = ['0% (0/3)', '33% (1/3)', '67% (2/3)', '100% (3/3)']

    for (low, high), label in zip(bins, labels):
        count = np.sum((values > low) & (values <= high))
        fraction = count / len(values)
        bar = '#' * int(fraction * 40)
        print(f'  {label:>12s}: {count:5d} ({fraction:5.1%}) {bar}')


def print_rank_statistics(results: dict[str, list]) -> None:
    """Print rank distribution of GT pions."""
    print('\n' + '=' * 70)
    print('GT PION RANK DISTRIBUTION (lower = better)')
    print('=' * 70)

    ranks = np.array(results['gt_pion_ranks'])
    if len(ranks) == 0:
        print('  No GT pions found')
        return

    print(f'  Total GT pions: {len(ranks)}')
    print(
        f'  Rank:  mean={ranks.mean():.1f}  '
        f'median={np.median(ranks):.1f}  '
        f'p75={np.percentile(ranks, 75):.1f}  '
        f'p90={np.percentile(ranks, 90):.1f}  '
        f'p99={np.percentile(ranks, 99):.1f}  '
        f'max={ranks.max()}',
    )

    # Rank buckets
    buckets = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, 500), (500, 3000)]
    for low, high in buckets:
        count = np.sum((ranks >= low) & (ranks < high))
        fraction = count / len(ranks)
        bar = '#' * int(fraction * 40)
        print(f'  [{low:4d}, {high:4d}): {count:5d} ({fraction:5.1%}) {bar}')


def print_score_statistics(results: dict[str, list]) -> None:
    """Print score distributions for GT pions vs background."""
    print('\n' + '=' * 70)
    print('SCORE DISTRIBUTIONS')
    print('=' * 70)

    gt_scores = np.array(results['gt_pion_scores'])
    background_scores = np.array(results['background_scores'])

    if len(gt_scores) == 0:
        print('  No GT pions found')
        return

    print(f'  GT pion scores (n={len(gt_scores)}):')
    print(
        f'    mean={gt_scores.mean():.6f}  '
        f'median={np.median(gt_scores):.6f}  '
        f'std={gt_scores.std():.6f}  '
        f'min={gt_scores.min():.6f}  '
        f'max={gt_scores.max():.6f}',
    )

    print(f'  Background scores (n={len(background_scores)}):')
    print(
        f'    mean={background_scores.mean():.6f}  '
        f'median={np.median(background_scores):.6f}  '
        f'std={background_scores.std():.6f}  '
        f'min={background_scores.min():.6f}  '
        f'max={background_scores.max():.6f}',
    )

    # Score separation
    if gt_scores.mean() > background_scores.mean():
        separation = (gt_scores.mean() - background_scores.mean()) / (
            np.sqrt(0.5 * (gt_scores.std()**2 + background_scores.std()**2)) + 1e-10
        )
        print(f'  Score separation (d-prime): {separation:.3f}')


def print_recall_vs_track_count(results: dict[str, list]) -> None:
    """Print recall@30 binned by event track count."""
    print('\n' + '=' * 70)
    print('RECALL@30 vs EVENT TRACK COUNT')
    print('=' * 70)

    recalls = np.array(results['recall_at_30'])
    track_counts = np.array(results['event_track_count_with_gt'][:len(recalls)])

    if len(recalls) == 0:
        print('  No events with GT tracks')
        return

    bins = [(0, 500), (500, 800), (800, 1100), (1100, 1400), (1400, 1700), (1700, 3000)]
    for low, high in bins:
        selected_mask = (track_counts >= low) & (track_counts < high)
        if selected_mask.sum() == 0:
            continue
        bin_recalls = recalls[selected_mask]
        print(
            f'  [{low:4d}, {high:4d}) tracks: '
            f'n={len(bin_recalls):4d}  '
            f'R@30={bin_recalls.mean():.4f}  '
            f'median={np.median(bin_recalls):.4f}',
        )


def print_recall_vs_gt_pt(results: dict[str, list]) -> None:
    """Print per-pion recall (found in top-30?) binned by pion pT."""
    print('\n' + '=' * 70)
    print('GT PION FOUND IN TOP-30 vs PION pT')
    print('=' * 70)

    ranks = np.array(results['gt_pion_ranks'])
    pt_values = np.array(results['gt_pion_pt'])

    if len(ranks) == 0:
        print('  No GT pions found')
        return

    found_in_top30 = (ranks < 30).astype(float)

    bins = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 10.0), (10.0, 50.0)]
    for low, high in bins:
        selected_mask = (pt_values >= low) & (pt_values < high)
        if selected_mask.sum() == 0:
            continue
        bin_found = found_in_top30[selected_mask]
        print(
            f'  pT [{low:5.1f}, {high:5.1f}) GeV: '
            f'n={len(bin_found):4d}  '
            f'found_rate={bin_found.mean():.4f}',
        )


def print_recall_vs_dxy_significance(results: dict[str, list]) -> None:
    """Print per-pion recall binned by dxy_significance (standardized)."""
    print('\n' + '=' * 70)
    print('GT PION FOUND IN TOP-30 vs dxy_significance (standardized)')
    print('=' * 70)

    ranks = np.array(results['gt_pion_ranks'])
    dxy_values = np.array(results['gt_pion_dxy_significance'])

    if len(ranks) == 0:
        print('  No GT pions found')
        return

    found_in_top30 = (ranks < 30).astype(float)

    bins = [(-3, -1), (-1, 0), (0, 1), (1, 2), (2, 3), (3, 5), (5, 20)]
    for low, high in bins:
        selected_mask = (dxy_values >= low) & (dxy_values < high)
        if selected_mask.sum() == 0:
            continue
        bin_found = found_in_top30[selected_mask]
        print(
            f'  dxy_sig [{low:5.1f}, {high:5.1f}): '
            f'n={len(bin_found):4d}  '
            f'found_rate={bin_found.mean():.4f}',
        )


def main() -> None:
    args = parse_arguments()
    device = torch.device(args.device)

    print(f'Device: {device}')
    print(f'Checkpoint: {args.checkpoint}')

    # ---- Data loading ----
    import glob
    parquet_files = sorted(glob.glob(f'{args.data_dir}/*.parquet'))
    file_dict = {'data': parquet_files}
    print(f'Found {len(parquet_files)} parquet files')

    # Use full dataset (not split into train/val), streaming to avoid OOM
    dataset = SimpleIterDataset(
        file_dict,
        data_config_file=args.data_config,
        for_training=False,
        load_range_and_fraction=((0.0, 1.0), 1.0),
        fetch_by_files=True,
        fetch_step=len(parquet_files),
        in_memory=False,
    )
    data_config = dataset.config

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        drop_last=False,
        pin_memory=False,
        num_workers=0,
    )

    # Find label input index
    input_names = list(data_config.input_names)
    label_input_index = input_names.index('pf_label')
    print(f'Input names: {input_names}')
    print(f'Label input index: {label_input_index}')

    # ---- Model ----
    model = build_model_and_load_checkpoint(
        args.network, data_config, args.checkpoint,
        args.pretrained_backbone, device,
        backbone_mode=args.backbone_mode,
    )

    # ---- Inference ----
    print('\nRunning inference...')
    results = collect_per_event_statistics(
        model, data_loader, device, label_input_index,
        max_batches=args.max_batches,
    )

    # ---- Print statistics ----
    num_events_with_gt = len(results['recall_at_30'])
    num_events_total = len(results['event_track_count'])
    print(f'\nEvents total: {num_events_total}')
    print(f'Events with GT tracks: {num_events_with_gt}')

    gt_per_event = np.array(results['num_gt_per_event'])
    gt_nonzero = gt_per_event[gt_per_event > 0]
    if len(gt_nonzero) > 0:
        print(
            f'GT tracks per event (non-zero): '
            f'mean={gt_nonzero.mean():.2f}  '
            f'median={np.median(gt_nonzero):.1f}  '
            f'min={gt_nonzero.min()}  '
            f'max={gt_nonzero.max()}',
        )

    print_recall_statistics(results)
    print_recall_distribution(results)
    print_rank_statistics(results)
    print_score_statistics(results)
    print_recall_vs_track_count(results)
    print_recall_vs_gt_pt(results)
    print_recall_vs_dxy_significance(results)


if __name__ == '__main__':
    main()
