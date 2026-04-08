"""Phase 1 of `reports/triplet_reranking/triplet_research_plan_20260408.md`.

Couple-count and recall measurement on the cascade_soap (cutoff dataset)
output. For each event:

1. Run the cascade Stage 1 → Stage 2 to obtain the ParT top-50 (sorted by
   the trained Stage 2 score within the Stage 1 top-256).
2. Enumerate C(50, 2) = 1225 candidate couples.
3. Apply the LOOSE physics filter:
       m(i, j) <= m_tau = 1.77693 GeV
   No charge cut. No rho-mass window. The omitted cuts are exactly the
   biases the new framing must avoid (`reports/triplet_reranking/triplet_combinatorics.md`).
4. Identify GT couples (both members are GT pions).
5. Aggregate per-event statistics across val.

Output: a markdown report with the surviving-couple count distribution,
the GT-couple count distribution, the GT-survival-of-mass-cut rate (a
sanity check on physics conservation), and a random-rank baseline that
defines what the future couple reranker must beat.

Usage (MPS, ~5k events):
    /opt/miniconda3/envs/part/bin/python diagnostics/couple_count_diagnostic.py \\
        --checkpoint models/cascade_best.pt \\
        --stage1-checkpoint models/prefilter_best.pt \\
        --data-config data/low-pt/lowpt_tau_trackfinder.yaml \\
        --data-dir data/low-pt/val/ \\
        --output reports/triplet_reranking/couple_count_20260408.md \\
        --device mps --batch-size 16 --max-events 5000
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger('couple_count_diagnostic')

# PDG 2024
M_TAU_GEV = 1.77693


# ---------------------------------------------------------------------------
# Model + data loading (mirrors `train_cascade.py` patterns)
# ---------------------------------------------------------------------------

def load_cascade_with_trained_weights(
    checkpoint_path: str,
    stage1_checkpoint_path: str,
    data_config,
    device: torch.device,
) -> torch.nn.Module:
    """Build the cascade via the network wrapper, then load the trained state.

    The wrapper handles auto-detection of Stage 1 hidden_dim from the
    checkpoint shape (so this works for the dim256 cutoff prefilter as well
    as the older dim192 ones).
    """
    from utils.training_utils import load_network_module

    network_module = load_network_module(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'networks/lowpt_tau_CascadeReranker.py',
        ),
    )

    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False,
    )
    saved_args = checkpoint.get('args', {})

    pair_embed_dims_raw = saved_args.get('stage2_pair_embed_dims', '64,64,64')
    if isinstance(pair_embed_dims_raw, str):
        pair_embed_dims = [int(x) for x in pair_embed_dims_raw.split(',')]
    else:
        pair_embed_dims = pair_embed_dims_raw

    model, _ = network_module.get_model(
        data_config,
        stage1_checkpoint=stage1_checkpoint_path,
        top_k1=saved_args.get('top_k1', 256),
        stage2_embed_dim=saved_args.get('stage2_embed_dim', 512),
        stage2_num_heads=saved_args.get('stage2_num_heads', 8),
        stage2_num_layers=saved_args.get('stage2_num_layers', 2),
        stage2_pair_embed_dims=pair_embed_dims,
        stage2_pair_extra_dim=saved_args.get('stage2_pair_extra_dim', 6),
        stage2_pair_embed_mode=saved_args.get('stage2_pair_embed_mode', 'concat'),
        stage2_ffn_ratio=saved_args.get('stage2_ffn_ratio', 4),
        stage2_dropout=saved_args.get('stage2_dropout', 0.1),
        stage2_loss_mode=saved_args.get('stage2_loss_mode', 'pairwise'),
        stage2_rs_at_k_target=saved_args.get('stage2_rs_at_k_target', 200),
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.train()  # BatchNorm batch stats — same as training-time validation
    logger.info(
        f'Loaded cascade from {checkpoint_path} '
        f'(epoch={checkpoint.get("epoch", "?")}, '
        f'top_k1={saved_args.get("top_k1", "?")})',
    )
    return model


# ---------------------------------------------------------------------------
# Couple statistics
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Filter definitions
# ---------------------------------------------------------------------------
#
# The plan's primary filter is "loose" — only the m_tau kinematic bound. Two
# additional filter levels are reported for context, NOT as recommendations:
# they characterize the cost in surviving-couple count of progressively
# tighter physics assumptions, so the Phase 2 physics baseline can pick its
# own filter knowing the trade-off. The diagnostic does NOT bake any of the
# filters into the architectural choice — that's the entire point of the new
# framing (`reports/triplet_reranking/triplet_combinatorics.md` failed because of hardcoded
# physics cuts).

FILTER_LEVELS = [
    'A: m(ij) <= m_tau                      (loose, no bias)',
    'B: A + opposite-sign                   (kills SS pairs, +charge bias)',
    'C: A + 0.4 GeV <= m(ij) <= 1.2 GeV     (wide a1 mass window, +mass bias)',
    'D: A + B + C                           (all three together, ~prior physics filter)',
]


def couple_stats_for_event(
    top50_lorentz_vectors: np.ndarray,
    top50_is_ground_truth: np.ndarray,
    top50_charges_raw: np.ndarray,
    n_ground_truth_in_top50: int,
) -> dict:
    """Compute couple-level statistics for a single event.

    Args:
        top50_lorentz_vectors: (4, 50) array of (px, py, pz, E) for the
            top-50 tracks selected by the trained Stage 2.
        top50_is_ground_truth: (50,) bool — True for GT pion positions.
        top50_charges_raw: (50,) signed charges in {-1, +1}, recovered from
            the standardized feature channel by reversing the same scale
            the cascade applies.
        n_ground_truth_in_top50: how many GT pions are in top-50 (0-3).

    Returns:
        dict with the per-event metrics that get aggregated downstream.
    """
    upper_i, upper_j = np.triu_indices(top50_lorentz_vectors.shape[1], k=1)

    # Sum 4-vectors per couple
    sum_energy = top50_lorentz_vectors[3, upper_i] + top50_lorentz_vectors[3, upper_j]
    sum_px = top50_lorentz_vectors[0, upper_i] + top50_lorentz_vectors[0, upper_j]
    sum_py = top50_lorentz_vectors[1, upper_i] + top50_lorentz_vectors[1, upper_j]
    sum_pz = top50_lorentz_vectors[2, upper_i] + top50_lorentz_vectors[2, upper_j]

    # Invariant mass m(i, j) = sqrt(E^2 - |p|^2)
    invariant_mass_squared = (
        sum_energy ** 2 - sum_px ** 2 - sum_py ** 2 - sum_pz ** 2
    )
    invariant_mass = np.sqrt(np.maximum(invariant_mass_squared, 0.0))

    # Charge product per pair: q_i * q_j (-1 = OS, +1 = SS)
    charge_product = top50_charges_raw[upper_i] * top50_charges_raw[upper_j]

    # Filter masks
    mask_a = invariant_mass <= M_TAU_GEV
    mask_b = mask_a & (charge_product < 0)
    mask_c = mask_a & (invariant_mass >= 0.4) & (invariant_mass <= 1.2)
    mask_d = mask_b & (invariant_mass >= 0.4) & (invariant_mass <= 1.2)

    # GT couple = both members are GT pions
    is_ground_truth_couple = (
        top50_is_ground_truth[upper_i] & top50_is_ground_truth[upper_j]
    )

    return {
        'n_couples_total': len(upper_i),
        'n_couples_surviving_a': int(mask_a.sum()),
        'n_couples_surviving_b': int(mask_b.sum()),
        'n_couples_surviving_c': int(mask_c.sum()),
        'n_couples_surviving_d': int(mask_d.sum()),
        'n_ground_truth_couples': int(is_ground_truth_couple.sum()),
        'n_gt_couples_surviving_a': int((is_ground_truth_couple & mask_a).sum()),
        'n_gt_couples_surviving_b': int((is_ground_truth_couple & mask_b).sum()),
        'n_gt_couples_surviving_c': int((is_ground_truth_couple & mask_c).sum()),
        'n_gt_couples_surviving_d': int((is_ground_truth_couple & mask_d).sum()),
        'n_ground_truth_in_top50': n_ground_truth_in_top50,
    }


# ---------------------------------------------------------------------------
# Aggregation + report
# ---------------------------------------------------------------------------

def percentiles(values: np.ndarray) -> dict:
    return {
        'mean': float(values.mean()),
        'median': float(np.median(values)),
        'p25': float(np.percentile(values, 25)),
        'p75': float(np.percentile(values, 75)),
        'p90': float(np.percentile(values, 90)),
        'p95': float(np.percentile(values, 95)),
        'min': float(values.min()),
        'max': float(values.max()),
    }


def random_rank_recall(
    surviving_counts: np.ndarray,
    surviving_gt_counts: np.ndarray,
    k_targets: tuple[int, ...] = (50, 100, 200),
) -> dict:
    """Expected recall@K under a uniformly random couple ordering.

    For an event with N surviving couples and G surviving ground-truth
    couples, the probability that AT LEAST ONE GT couple lands in top-K
    under a uniform random permutation is:

        P(>=1 GT in top-K) = 1 - C(N - G, K) / C(N, K)

    Computed in log-space to handle large N. Events with G == 0 are
    excluded from the average since the metric is undefined for them.
    """
    from math import lgamma

    def log_combinations(n: int, k: int) -> float:
        if k < 0 or k > n:
            return float('-inf')
        return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)

    results: dict[str, float] = {}
    eligible = surviving_gt_counts > 0
    if not eligible.any():
        return {f'random_recall_at_{k}': 0.0 for k in k_targets}

    for k in k_targets:
        per_event_prob: list[float] = []
        for surviving, n_gt in zip(
            surviving_counts[eligible], surviving_gt_counts[eligible],
        ):
            n = int(surviving)
            g = int(n_gt)
            if g == 0:
                continue
            if k >= n:
                per_event_prob.append(1.0)
                continue
            log_no_gt = log_combinations(n - g, k) - log_combinations(n, k)
            per_event_prob.append(float(1.0 - np.exp(log_no_gt)))
        results[f'random_recall_at_{k}'] = (
            float(np.mean(per_event_prob)) if per_event_prob else 0.0
        )
    return results


def write_report(
    output_path: str,
    n_events: int,
    per_filter_surviving: dict[str, np.ndarray],
    per_filter_gt_surviving: dict[str, np.ndarray],
    ground_truth_couple_counts: np.ndarray,
    n_ground_truth_in_top50_counts: np.ndarray,
    random_recall_per_filter: dict[str, dict],
) -> None:
    n_gt_in_top50_hist = {
        i: int((n_ground_truth_in_top50_counts == i).sum())
        for i in range(4)
    }
    n_gt_couple_hist = {
        i: int((ground_truth_couple_counts == i).sum())
        for i in range(4)
    }
    expected_gt_couples = (
        n_ground_truth_in_top50_counts
        * (n_ground_truth_in_top50_counts - 1)
        // 2
    )
    sanity_pass = int(
        (ground_truth_couple_counts == expected_gt_couples).sum()
    )

    def gt_survival_rate(filter_letter: str) -> float:
        eligible = ground_truth_couple_counts > 0
        if not eligible.any():
            return 0.0
        survived = per_filter_gt_surviving[filter_letter][eligible]
        total = ground_truth_couple_counts[eligible]
        return float((survived / total).mean())

    lines: list[str] = []
    add = lines.append

    add('# Phase 1 — Couple-Count Diagnostic')
    add('')
    add(f'**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    add(f'**Events:** {n_events}')
    add('**Source:** `cascade_soap_Cascade_20260406_202001/checkpoints/best_model.pt`')
    add('**Top-50 selection:** trained Stage 2 score within Stage 1 top-256.')
    add('')
    add('Filters reported (Filter A is the plan-mandated loose baseline; B/C/D')
    add('are for context, NOT recommended — they bake in physics biases the new')
    add('framing must avoid):')
    add('')
    for label in FILTER_LEVELS:
        add(f'- {label}')
    add('')
    add('---')
    add('')
    add('## 1. Surviving couples per event (per filter)')
    add('')
    add('| filter | mean | median | p25 | p75 | p90 | p95 | min | max |')
    add('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for letter in ('a', 'b', 'c', 'd'):
        stats = percentiles(per_filter_surviving[letter])
        add(
            f'| {letter.upper()} '
            f'| {stats["mean"]:.0f} '
            f'| {stats["median"]:.0f} '
            f'| {stats["p25"]:.0f} '
            f'| {stats["p75"]:.0f} '
            f'| {stats["p90"]:.0f} '
            f'| {stats["p95"]:.0f} '
            f'| {stats["min"]:.0f} '
            f'| {stats["max"]:.0f} |'
        )
    add('')
    add('Theoretical max per event: 1225 = C(50, 2).')
    add('')
    add('---')
    add('')
    add('## 2. GT pions in top-50 (histogram)')
    add('')
    add('| n_gt_in_top50 | events | fraction |')
    add('|---:|---:|---:|')
    for i in range(4):
        n = n_gt_in_top50_hist[i]
        add(f'| {i} | {n} | {n / max(1, n_events):.4f} |')
    add('')
    add(
        f'**Duplet rate (>=2/3 GT in top-50): '
        f'{(n_gt_in_top50_hist[2] + n_gt_in_top50_hist[3]) / max(1, n_events):.4f}**'
    )
    add(
        f'**Triplet rate (3/3 GT in top-50): '
        f'{n_gt_in_top50_hist[3] / max(1, n_events):.4f}**'
    )
    add('')
    add('---')
    add('')
    add('## 3. GT couples per event (histogram)')
    add('')
    add('| n_gt_couples | events | fraction |')
    add('|---:|---:|---:|')
    for i in range(4):
        n = n_gt_couple_hist[i]
        add(f'| {i} | {n} | {n / max(1, n_events):.4f} |')
    add('')
    add(
        f'Sanity (n_gt_couples == C(n_gt_in_top50, 2)): '
        f'**{sanity_pass}/{n_events}** events match.'
    )
    add('')
    add('### GT couple survival rate per filter')
    add('')
    add('Fraction of GT couples that pass each filter (averaged over events with')
    add('at least one GT couple). Filter A should be ~1.0 — kinematic conservation')
    add('guarantees `m(GT pair) <= m_tau`. Filters B/C/D will drop below 1.0 by')
    add('the amount the imposed bias kills GT couples.')
    add('')
    add('| filter | GT survival |')
    add('|---|---:|')
    for letter in ('a', 'b', 'c', 'd'):
        add(f'| {letter.upper()} | {gt_survival_rate(letter):.4f} |')
    add('')
    add('---')
    add('')
    add('## 4. Random-rank baseline (per filter)')
    add('')
    add('Expected recall@K under a uniformly random ranking of the surviving')
    add('couples, averaged over events with at least one surviving GT couple.')
    add('**This is the floor the future couple reranker must beat.**')
    add('')
    add('| filter | recall@50 | recall@100 | recall@200 |')
    add('|---|---:|---:|---:|')
    for letter in ('a', 'b', 'c', 'd'):
        rr = random_recall_per_filter[letter]
        add(
            f'| {letter.upper()} '
            f'| {rr["random_recall_at_50"]:.4f} '
            f'| {rr["random_recall_at_100"]:.4f} '
            f'| {rr["random_recall_at_200"]:.4f} |'
        )
    add('')
    add('---')
    add('')
    add('## 5. Notes for Phase 2 / Phase 3')
    add('')
    add(
        '- **Filter A is the framing-correct baseline** (no charge or mass '
        'biases), and it leaves a much larger candidate pool than the user '
        'intuition of "a couple hundred". The neural couple reranker should '
        'be trained on Filter A inputs.'
    )
    add(
        '- **Filters B/C/D** show the cost (in surviving-couple count and GT '
        'survival rate) of progressively tighter physics. Phase 2 (physics-only '
        'baseline) can pick whichever filter minimizes the surviving count '
        'without dropping GT survival meaningfully. Filter D is the closest to '
        'the prior (failed) physics-cascade approach in `triplet_combinatorics.md`.'
    )
    add(
        '- **Random-rank baseline** under Filter A is the conservative '
        'floor. The neural reranker is interesting only if it beats Filter A '
        'random recall by a meaningful margin.'
    )
    add('')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    logger.info(f'Wrote report: {output_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Phase 1 couple-count diagnostic for the post-ParT '
                    'reranking research plan.',
    )
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--stage1-checkpoint', type=str, required=True)
    parser.add_argument('--data-config', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='reports/triplet_reranking/couple_count.md')
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-events', type=int, default=5000)
    parser.add_argument('--top-k2', type=int, default=50,
                        help='Number of tracks per event to enumerate couples '
                             'over (default 50, matching the ParT top-K used '
                             'as the couple anchor pool).')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    device = torch.device(args.device)

    # ---- Data ----
    from torch.utils.data import DataLoader

    from utils.training_utils import (
        extract_label_from_inputs,
        trim_to_max_valid_tracks,
    )
    from weaver.utils.dataset import SimpleIterDataset

    parquet_files = sorted(glob.glob(os.path.join(args.data_dir, '*.parquet')))
    if not parquet_files:
        raise FileNotFoundError(f'No parquet files in {args.data_dir}')
    logger.info(f'Loading {len(parquet_files)} parquet files from {args.data_dir}')

    dataset = SimpleIterDataset(
        {'data': parquet_files},
        data_config_file=args.data_config,
        for_training=False,
        load_range_and_fraction=((0.0, 1.0), 1.0),
        fetch_by_files=True,
        fetch_step=len(parquet_files),
        in_memory=True,
    )
    data_config = dataset.config
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size,
        drop_last=False, pin_memory=False, num_workers=0,
    )
    input_names = list(data_config.input_names)
    mask_input_index = input_names.index('pf_mask')
    label_input_index = input_names.index('pf_label')

    # ---- Model ----
    cascade = load_cascade_with_trained_weights(
        args.checkpoint, args.stage1_checkpoint, data_config, device,
    )

    # ---- Iterate ----
    per_filter_surviving: dict[str, list[int]] = {k: [] for k in 'abcd'}
    per_filter_gt_surviving: dict[str, list[int]] = {k: [] for k in 'abcd'}
    n_ground_truth_couples_per_event: list[int] = []
    n_ground_truth_in_top50_counts: list[int] = []
    events_processed = 0
    max_steps = (args.max_events // args.batch_size) + 1

    logger.info(
        f'Processing up to {args.max_events} events in batches of '
        f'{args.batch_size}',
    )
    with torch.no_grad():
        for batch_index, (X, _, _) in enumerate(data_loader):
            if events_processed >= args.max_events or batch_index >= max_steps:
                break

            inputs = [X[key].to(device) for key in input_names]
            inputs = trim_to_max_valid_tracks(inputs, mask_input_index)
            model_inputs, track_labels = extract_label_from_inputs(
                inputs, label_input_index,
            )
            points, features, lorentz_vectors, mask = model_inputs

            filtered = cascade._run_stage1(
                points, features, lorentz_vectors, mask, track_labels,
            )
            stage2_scores = cascade.stage2(
                filtered['points'], filtered['features'],
                filtered['lorentz_vectors'], filtered['mask'],
                filtered['stage1_scores'],
            )

            selected_indices_cpu = filtered['selected_indices'].cpu().numpy()
            stage2_scores_cpu = stage2_scores.detach().cpu().numpy()
            lorentz_vectors_cpu = lorentz_vectors.cpu().numpy()
            features_cpu = features.cpu().numpy()
            track_labels_cpu = track_labels.squeeze(1).cpu().numpy()
            mask_cpu = mask.squeeze(1).cpu().numpy().astype(bool)

            batch_size_actual = points.shape[0]
            for event_index in range(batch_size_actual):
                if events_processed >= args.max_events:
                    break

                event_stage2_scores = stage2_scores_cpu[event_index]
                event_selected = selected_indices_cpu[event_index]

                top_k2_in_k1 = np.argsort(-event_stage2_scores)[: args.top_k2]
                top_k2_original = event_selected[top_k2_in_k1]

                if not mask_cpu[event_index, top_k2_original].all():
                    continue

                top_k2_lorentz = lorentz_vectors_cpu[event_index][:, top_k2_original]
                # Recover raw {-1, +1} charges from the standardized feature
                # channel: the cascade applies (raw - 1) * 0.5 → standardized,
                # so raw = standardized / 0.5 + 1.
                charge_standardized = features_cpu[event_index, 5, top_k2_original]
                top_k2_charges_raw = charge_standardized / 0.5 + 1.0
                top_k2_is_gt = track_labels_cpu[event_index][top_k2_original] == 1.0
                n_gt_in_top50 = int(top_k2_is_gt.sum())

                event_stats = couple_stats_for_event(
                    top_k2_lorentz, top_k2_is_gt, top_k2_charges_raw, n_gt_in_top50,
                )

                for letter in 'abcd':
                    per_filter_surviving[letter].append(
                        event_stats[f'n_couples_surviving_{letter}'],
                    )
                    per_filter_gt_surviving[letter].append(
                        event_stats[f'n_gt_couples_surviving_{letter}'],
                    )
                n_ground_truth_couples_per_event.append(
                    event_stats['n_ground_truth_couples'],
                )
                n_ground_truth_in_top50_counts.append(
                    event_stats['n_ground_truth_in_top50'],
                )
                events_processed += 1

            if events_processed % (args.batch_size * 10) == 0:
                logger.info(f'Processed {events_processed} events')

    logger.info(f'Done. Processed {events_processed} events.')

    # ---- Aggregate ----
    per_filter_surviving_arr = {
        k: np.array(v) for k, v in per_filter_surviving.items()
    }
    per_filter_gt_surviving_arr = {
        k: np.array(v) for k, v in per_filter_gt_surviving.items()
    }
    n_gt_couples_arr = np.array(n_ground_truth_couples_per_event)
    n_gt_in_top50_arr = np.array(n_ground_truth_in_top50_counts)

    random_recall_per_filter = {
        letter: random_rank_recall(
            per_filter_surviving_arr[letter],
            per_filter_gt_surviving_arr[letter],
            k_targets=(50, 100, 200),
        )
        for letter in 'abcd'
    }

    write_report(
        output_path=args.output,
        n_events=events_processed,
        per_filter_surviving=per_filter_surviving_arr,
        per_filter_gt_surviving=per_filter_gt_surviving_arr,
        ground_truth_couple_counts=n_gt_couples_arr,
        n_ground_truth_in_top50_counts=n_gt_in_top50_arr,
        random_recall_per_filter=random_recall_per_filter,
    )


if __name__ == '__main__':
    main()
