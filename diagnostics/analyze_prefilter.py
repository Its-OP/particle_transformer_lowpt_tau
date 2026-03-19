"""Diagnostic analysis of a trained TrackPreFilter model.

Loads a checkpoint, runs inference on the subset, and prints per-event
and per-pion statistics binned by track count, pT, and dxy_significance.

Usage:
    cd part/
    python analyze_prefilter.py \
        --data-config data/low-pt/lowpt_tau_trackfinder.yaml \
        --data-dir data/low-pt/subset/ \
        --network networks/lowpt_tau_TrackPreFilter.py \
        --checkpoint models/debug_checkpoints/prefilter/checkpoints/best_model.pt \
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
    load_network_module,
    trim_to_max_valid_tracks,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Analyze TrackPreFilter')
    parser.add_argument('--data-config', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--network', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-batches', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--latent-dim', type=int, default=None,
                        help='Override latent_dim (for loading old checkpoints)')
    parser.add_argument('--hidden-dim', type=int, default=None,
                        help='Override hidden_dim')
    parser.add_argument('--ranking-num-samples', type=int, default=None,
                        help='Override ranking_num_samples')
    return parser.parse_args()


@torch.no_grad()
def collect_statistics(
    model, data_loader, device, input_names, label_input_index,
    mask_input_index, max_batches,
) -> dict[str, list]:
    """Run inference and collect per-event / per-pion statistics."""
    mask_idx_adj = mask_input_index
    if mask_input_index > label_input_index:
        mask_idx_adj -= 1

    results = {
        'recall_at_30': [], 'recall_at_100': [], 'recall_at_200': [],
        'gt_pion_ranks': [], 'gt_pion_scores': [], 'background_scores': [],
        'event_track_count': [], 'num_gt_per_event': [],
        'gt_pion_pt': [], 'gt_pion_dxy_significance': [],
        'gt_pion_found_200': [],
        # Failure cases: GT pion features for events where R@200=0
        'failure_gt_pt': [], 'failure_gt_dxy': [],
    }

    num_batches = 0
    for batch_data in data_loader:
        inputs_dict = batch_data[0]
        flat_inputs = [inputs_dict[name].to(device) for name in input_names]
        model_inputs, track_labels = extract_label_from_inputs(
            flat_inputs, label_input_index,
        )
        model_inputs = list(trim_to_max_valid_tracks(model_inputs, mask_idx_adj))
        points, features, lorentz_vectors, mask = model_inputs
        track_labels = track_labels[:, :, :mask.shape[2]]

        scores = model(points, features, lorentz_vectors, mask)

        mask_flat = mask.squeeze(1).bool()
        labels_flat = track_labels.squeeze(1) * mask_flat.float()
        batch_size = scores.shape[0]

        for event_index in range(batch_size):
            event_mask = mask_flat[event_index]
            event_labels = labels_flat[event_index]
            event_scores = scores[event_index]

            num_valid = event_mask.sum().item()
            results['event_track_count'].append(num_valid)

            gt_positions = event_labels.nonzero(as_tuple=True)[0]
            num_gt = len(gt_positions)
            results['num_gt_per_event'].append(num_gt)

            if num_gt == 0:
                continue

            # Track count aligned with recall arrays (only events with GT)
            results.setdefault('event_track_count_with_gt', []).append(num_valid)

            # Sort scores descending
            masked_scores = event_scores.clone()
            masked_scores[~event_mask] = float('-inf')
            sorted_indices = masked_scores.argsort(descending=True)

            # Rank lookup
            rank_lookup = torch.zeros_like(sorted_indices)
            rank_lookup[sorted_indices] = torch.arange(
                len(sorted_indices), device=sorted_indices.device,
            )

            gt_set = set(gt_positions.tolist())
            sorted_list = sorted_indices.tolist()

            # Recall@K
            for k_value in [30, 100, 200]:
                actual_k = min(k_value, num_valid)
                top_k_set = set(sorted_list[:actual_k])
                found = len(top_k_set & gt_set)
                results[f'recall_at_{k_value}'].append(found / num_gt)

            # Check if this is a failure case (R@200 = 0)
            actual_200 = min(200, num_valid)
            top_200_set = set(sorted_list[:actual_200])
            found_200 = len(top_200_set & gt_set)
            is_failure = (found_200 == 0)

            # Per-GT-pion statistics
            for gt_pos in gt_positions:
                gt_idx = gt_pos.item()
                rank = rank_lookup[gt_idx].item()
                score = event_scores[gt_idx].item()
                found_in_200 = 1.0 if rank < 200 else 0.0

                results['gt_pion_ranks'].append(rank)
                results['gt_pion_scores'].append(score)
                results['gt_pion_found_200'].append(found_in_200)

                # pT from Lorentz vectors
                px = lorentz_vectors[event_index, 0, gt_idx].item()
                py = lorentz_vectors[event_index, 1, gt_idx].item()
                pt = np.sqrt(px**2 + py**2)
                results['gt_pion_pt'].append(pt)

                # dxy_significance (feature index 6, standardized)
                dxy = features[event_index, 6, gt_idx].item()
                results['gt_pion_dxy_significance'].append(dxy)

                if is_failure:
                    results['failure_gt_pt'].append(pt)
                    results['failure_gt_dxy'].append(dxy)

            # Sample background scores
            bg_positions = (
                (event_labels == 0) & event_mask
            ).nonzero(as_tuple=True)[0]
            if len(bg_positions) > 30:
                sample_idx = torch.randperm(len(bg_positions))[:30]
                bg_positions = bg_positions[sample_idx]
            for bg_pos in bg_positions:
                results['background_scores'].append(
                    event_scores[bg_pos.item()].item(),
                )

        num_batches += 1
        if num_batches % 50 == 0:
            print(f'  Processed {num_batches} batches...')
        if max_batches and num_batches >= max_batches:
            break

    print(f'Total: {num_batches} batches')
    return results


def print_section(title):
    print(f'\n{"=" * 70}')
    print(title)
    print('=' * 70)


def main():
    args = parse_arguments()
    device = torch.device(args.device)

    # Data
    import glob
    parquet_files = sorted(glob.glob(f'{args.data_dir}/*.parquet'))
    dataset = SimpleIterDataset(
        {'data': parquet_files},
        data_config_file=args.data_config,
        for_training=False,
        load_range_and_fraction=((0.0, 1.0), 1.0),
        fetch_by_files=True, fetch_step=len(parquet_files),
        in_memory=False,
    )
    data_config = dataset.config
    loader = DataLoader(
        dataset, batch_size=args.batch_size, drop_last=False,
        num_workers=args.num_workers,
    )

    input_names = list(data_config.input_names)
    label_idx = input_names.index('pf_label')
    mask_idx = input_names.index('pf_mask')

    # Model
    network_module = load_network_module(args.network)
    model_kwargs = {}
    if args.latent_dim is not None:
        model_kwargs['latent_dim'] = args.latent_dim
    if args.hidden_dim is not None:
        model_kwargs['hidden_dim'] = args.hidden_dim
    if args.ranking_num_samples is not None:
        model_kwargs['ranking_num_samples'] = args.ranking_num_samples
    model, _ = network_module.get_model(data_config, **model_kwargs)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {total_params:,} params')

    # Collect
    print('Running inference...')
    results = collect_statistics(
        model, loader, device, input_names, label_idx, mask_idx,
        args.max_batches,
    )

    # ---- 1. Per-event R@200 distribution ----
    print_section('1. PER-EVENT R@200 DISTRIBUTION')
    r200 = np.array(results['recall_at_200'])
    n_events = len(r200)
    print(f'Events with GT: {n_events}')
    bins = [(-0.01, 0.01), (0.01, 0.34), (0.34, 0.67), (0.67, 1.01)]
    labels_str = ['0% (0/3)', '33% (1/3)', '67% (2/3)', '100% (3/3)']
    for (lo, hi), label in zip(bins, labels_str):
        count = np.sum((r200 > lo) & (r200 <= hi))
        frac = count / n_events
        bar = '#' * int(frac * 40)
        print(f'  {label:>12s}: {count:5d} ({frac:5.1%}) {bar}')

    # ---- 2. R@200 vs event track count ----
    print_section('2. R@200 vs EVENT TRACK COUNT')
    track_counts = np.array(results['event_track_count_with_gt'][:n_events])
    for lo, hi in [(0, 500), (500, 800), (800, 1100), (1100, 1400), (1400, 1700), (1700, 3000)]:
        sel = (track_counts >= lo) & (track_counts < hi)
        if sel.sum() == 0:
            continue
        print(f'  [{lo:4d}, {hi:4d}): n={sel.sum():5d}  R@200={r200[sel].mean():.4f}  perfect={np.mean(r200[sel] > 0.99):.4f}')

    # ---- 3. Found-in-top-200 vs pT ----
    print_section('3. GT PION FOUND IN TOP-200 vs pT')
    found = np.array(results['gt_pion_found_200'])
    pt = np.array(results['gt_pion_pt'])
    for lo, hi in [(0, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 50.0)]:
        sel = (pt >= lo) & (pt < hi)
        if sel.sum() == 0:
            continue
        print(f'  pT [{lo:5.1f}, {hi:5.1f}) GeV: n={sel.sum():5d}  found_rate={found[sel].mean():.4f}')

    # ---- 4. Found-in-top-200 vs dxy_significance ----
    print_section('4. GT PION FOUND IN TOP-200 vs |dxy_significance| (standardized)')
    dxy = np.abs(np.array(results['gt_pion_dxy_significance']))
    for lo, hi in [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 20.0)]:
        sel = (dxy >= lo) & (dxy < hi)
        if sel.sum() == 0:
            continue
        print(f'  |dxy_sig| [{lo:5.1f}, {hi:5.1f}): n={sel.sum():5d}  found_rate={found[sel].mean():.4f}')

    # ---- 5. Score distributions ----
    print_section('5. SCORE DISTRIBUTIONS')
    gt_scores = np.array(results['gt_pion_scores'])
    bg_scores = np.array(results['background_scores'])
    print(f'  GT pion scores (n={len(gt_scores)}):')
    print(f'    mean={gt_scores.mean():.4f}  median={np.median(gt_scores):.4f}  std={gt_scores.std():.4f}')
    print(f'  Background scores (n={len(bg_scores)}):')
    print(f'    mean={bg_scores.mean():.4f}  median={np.median(bg_scores):.4f}  std={bg_scores.std():.4f}')
    pooled = np.sqrt(0.5 * (gt_scores.std()**2 + bg_scores.std()**2))
    dprime = (gt_scores.mean() - bg_scores.mean()) / max(pooled, 1e-10)
    print(f'  d-prime: {dprime:.3f}')

    # ---- 6. GT pion rank distribution ----
    print_section('6. GT PION RANK DISTRIBUTION')
    ranks = np.array(results['gt_pion_ranks'])
    print(f'  Total GT pions: {len(ranks)}')
    print(f'  Rank: mean={ranks.mean():.1f}  median={np.median(ranks):.1f}  p75={np.percentile(ranks,75):.1f}  p90={np.percentile(ranks,90):.1f}  p99={np.percentile(ranks,99):.1f}')
    for lo, hi in [(0, 30), (30, 100), (100, 200), (200, 500), (500, 3000)]:
        count = np.sum((ranks >= lo) & (ranks < hi))
        print(f'  [{lo:4d}, {hi:4d}): {count:5d} ({count/len(ranks):5.1%})')

    # ---- 7. Failure case analysis ----
    print_section('7. FAILURE CASES (events where R@200 = 0%)')
    n_failures = np.sum(r200 < 0.01)
    print(f'  Events with 0/3 in top-200: {n_failures} ({n_failures/n_events:.1%})')
    if len(results['failure_gt_pt']) > 0:
        fpt = np.array(results['failure_gt_pt'])
        fdxy = np.abs(np.array(results['failure_gt_dxy']))
        print(f'  Failed GT pions (n={len(fpt)}):')
        print(f'    pT:  mean={fpt.mean():.3f}  median={np.median(fpt):.3f}')
        print(f'    |dxy|: mean={fdxy.mean():.3f}  median={np.median(fdxy):.3f}')
        print(f'    pT < 0.5 GeV: {np.mean(fpt < 0.5)*100:.1f}%')
        print(f'    |dxy| < 1.0:  {np.mean(fdxy < 1.0)*100:.1f}%')
    else:
        print('  No failure cases found!')


if __name__ == '__main__':
    main()
