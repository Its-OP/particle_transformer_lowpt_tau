"""Combinatorial triplet feasibility experiment.

Enumerates candidate triplets from the top-K tracks of the Physics ParT
cascade, applies progressive physics + Dalitz filters, and measures how
many survive at each stage. Reports GT triplet survival rate and purity.

Usage:
    python diagnostics/triplet_combinatorics.py \\
        --checkpoint models/debug_checkpoints/partfull_physics_concat_Cascade_20260329_213437/checkpoints/best_model.pt \\
        --stage1-checkpoint models/prefilter_best.pt \\
        --data-config data/low-pt/lowpt_tau_trackfinder.yaml \\
        --data-dir data/low-pt/subset/val/ \\
        --output reports/triplet_reranking/triplet_combinatorics.md \\
        --device mps --max-events 150
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from datetime import datetime
from itertools import combinations

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger('triplet_combinatorics')

# ---------------------------------------------------------------------------
# Physics constants (PDG 2024)
# ---------------------------------------------------------------------------
M_TAU = 1.77693    # GeV
M_PI = 0.13957     # GeV
M_RHO = 0.77526    # GeV
GAMMA_RHO = 0.1474 # GeV
M_A1 = 1.233       # GeV
GAMMA_A1 = 0.431   # GeV
# Bachelor pion energy in a1 rest frame:
# E_b = (m_a1^2 + m_pi^2 - m_rho^2) / (2 * m_a1) ~ 0.380 GeV
E_BACHELOR_A1_FRAME = (M_A1**2 + M_PI**2 - M_RHO**2) / (2 * M_A1)

FILTER_NAMES = [
    'F1: Charge (|Q|=1)',
    'F2: Tau mass (m3pi < 1.777)',
    'F3: a1 window (0.6 < m3pi < 1.5)',
    'F4: Rho resonance (OS pair near 770)',
    'D1: Dalitz boundary',
    'D2: Rho band structure (s_high > s_low)',
    'D3: Bachelor energy (250-500 MeV in a1 frame)',
    'D4: Rho helicity (|cos theta| > 0.3)',
]


# ---------------------------------------------------------------------------
# Triplet physics computations
# ---------------------------------------------------------------------------

def invariant_mass(four_vectors: np.ndarray) -> np.ndarray:
    """Compute invariant mass from summed 4-vectors.

    Args:
        four_vectors: (..., 4) array with (px, py, pz, E).

    Returns:
        (...,) array of invariant masses in GeV.
    """
    e = four_vectors[..., 3]
    px = four_vectors[..., 0]
    py = four_vectors[..., 1]
    pz = four_vectors[..., 2]
    m_squared = e**2 - px**2 - py**2 - pz**2
    return np.sqrt(np.maximum(m_squared, 0.0))


def boost_to_rest_frame(
    particle_4v: np.ndarray,
    frame_4v: np.ndarray,
) -> np.ndarray:
    """Boost particle 4-vectors to the rest frame of frame_4v.

    Args:
        particle_4v: (..., 4) array (px, py, pz, E).
        frame_4v: (..., 4) array — the system whose rest frame we boost to.

    Returns:
        (..., 4) boosted 4-vectors.
    """
    # beta = p / E
    e_frame = frame_4v[..., 3:4]
    p_frame = frame_4v[..., :3]
    beta = p_frame / np.maximum(e_frame, 1e-10)
    beta_sq = np.sum(beta**2, axis=-1, keepdims=True)
    gamma = 1.0 / np.sqrt(np.maximum(1.0 - beta_sq, 1e-10))

    # Boost
    p_particle = particle_4v[..., :3]
    e_particle = particle_4v[..., 3:4]

    beta_dot_p = np.sum(beta * p_particle, axis=-1, keepdims=True)

    p_boosted = (
        p_particle
        + beta * ((gamma - 1.0) * beta_dot_p / np.maximum(beta_sq, 1e-10) - gamma * e_particle)
    )
    e_boosted = gamma * (e_particle - beta_dot_p)

    return np.concatenate([p_boosted, e_boosted], axis=-1)


def analyze_triplets_for_event(
    four_vectors: np.ndarray,
    charges: np.ndarray,
    gt_indices: set[int],
    k_value: int,
) -> dict:
    """Run the full filter cascade on one event's top-K tracks.

    Args:
        four_vectors: (K, 4) array of (px, py, pz, E) for top-K tracks.
        charges: (K,) array of charges {-1, +1}.
        gt_indices: Set of indices (within top-K) that are GT pions.
        k_value: K value used (for reporting).

    Returns:
        Dict with counts at each filter stage and GT survival info.
    """
    n_tracks = len(charges)
    positive_indices = np.where(charges > 0)[0]
    negative_indices = np.where(charges < 0)[0]
    n_pos = len(positive_indices)
    n_neg = len(negative_indices)

    # Build GT triplet (if all 3 are present in top-K)
    gt_in_topk = sorted(gt_indices)
    gt_triplet_present = len(gt_in_topk) == 3

    results = {
        'k': k_value,
        'n_tracks': n_tracks,
        'n_positive': n_pos,
        'n_negative': n_neg,
        'gt_triplet_in_topk': gt_triplet_present,
        'n_gt_in_topk': len(gt_in_topk),
    }

    # --- F1: Charge filter ---
    # |Q| = 1 triplets: (2 same-sign + 1 opposite-sign)
    # tau- → pi-pi-pi+: 2 negative + 1 positive
    # tau+ → pi+pi+pi-: 2 positive + 1 negative
    triplets_tau_minus = []  # (neg_i, neg_j, pos_k)
    for i, j in combinations(range(n_neg), 2):
        for k in range(n_pos):
            triplets_tau_minus.append((negative_indices[i], negative_indices[j], positive_indices[k]))

    triplets_tau_plus = []
    for i, j in combinations(range(n_pos), 2):
        for k in range(n_neg):
            triplets_tau_plus.append((positive_indices[i], positive_indices[j], negative_indices[k]))

    all_charge_valid = triplets_tau_minus + triplets_tau_plus
    results['F1_count'] = len(all_charge_valid)

    if len(all_charge_valid) == 0:
        for stage in FILTER_NAMES[1:]:
            key = stage.split(':')[0] + '_count'
            results[key] = 0
        results['gt_killed_at'] = 'F1' if gt_triplet_present else 'not_in_topk'
        return results

    # Convert to numpy for vectorized mass computation
    triplet_indices = np.array(all_charge_valid)  # (N, 3)
    fv = four_vectors  # (K, 4)

    # 4-vectors of each track in each triplet
    fv_a = fv[triplet_indices[:, 0]]  # (N, 4)
    fv_b = fv[triplet_indices[:, 1]]  # (N, 4)
    fv_c = fv[triplet_indices[:, 2]]  # (N, 4)

    # Triplet 4-vector sum
    fv_triplet = fv_a + fv_b + fv_c  # (N, 4)
    m_3pi = invariant_mass(fv_triplet)  # (N,)

    # Charges for identifying OS pairs
    ch_a = charges[triplet_indices[:, 0]]
    ch_b = charges[triplet_indices[:, 1]]
    ch_c = charges[triplet_indices[:, 2]]

    # In tau- triplets: a,b are same-sign (negative), c is opposite (positive)
    # In tau+ triplets: a,b are same-sign (positive), c is opposite (negative)
    # The OS pairs are always (a,c) and (b,c)
    fv_os1 = fv_a + fv_c  # OS pair 1
    fv_os2 = fv_b + fv_c  # OS pair 2
    m_os1 = invariant_mass(fv_os1)  # (N,)
    m_os2 = invariant_mass(fv_os2)  # (N,)

    # Check GT triplet survival at each stage
    gt_killed_at = None
    if gt_triplet_present:
        gt_set = frozenset(gt_in_topk)
        gt_mask = np.array([
            frozenset(triplet_indices[i].tolist()) == gt_set
            for i in range(len(triplet_indices))
        ])
    else:
        gt_mask = np.zeros(len(triplet_indices), dtype=bool)

    def check_gt(surviving_mask, stage_name):
        nonlocal gt_killed_at
        if gt_killed_at is not None:
            return
        if gt_triplet_present and not np.any(gt_mask & surviving_mask):
            gt_killed_at = stage_name

    # Start with all charge-valid triplets surviving
    surviving = np.ones(len(triplet_indices), dtype=bool)
    check_gt(surviving, 'F1')

    # --- F2: Tau mass ---
    f2_pass = m_3pi < M_TAU
    surviving &= f2_pass
    results['F2_count'] = int(surviving.sum())
    check_gt(surviving, 'F2')

    # --- F3: a1 mass window ---
    f3_pass = (m_3pi > 0.6) & (m_3pi < 1.5)
    surviving &= f3_pass
    results['F3_count'] = int(surviving.sum())
    check_gt(surviving, 'F3')

    # --- F4: Rho resonance (at least one OS pair near 770 MeV) ---
    rho_window = 0.150  # GeV
    os1_near_rho = np.abs(m_os1 - M_RHO) < rho_window
    os2_near_rho = np.abs(m_os2 - M_RHO) < rho_window
    f4_pass = os1_near_rho | os2_near_rho
    surviving &= f4_pass
    results['F4_count'] = int(surviving.sum())
    check_gt(surviving, 'F4')

    # --- Dalitz features (applied on F4 survivors) ---

    # Symmetrized Dalitz variables
    s1 = m_os1**2  # m^2(a, c)
    s2 = m_os2**2  # m^2(b, c)
    s_high = np.maximum(s1, s2)
    s_low = np.minimum(s1, s2)

    # D1: Dalitz boundary — s1 + s2 <= m_tau^2 - 3*m_pi^2
    # (kinematic constraint from energy-momentum conservation)
    dalitz_bound = M_TAU**2 - 3 * M_PI**2
    d1_pass = (s1 + s2) <= dalitz_bound
    surviving &= d1_pass
    results['D1_count'] = int(surviving.sum())
    check_gt(surviving, 'D1')

    # D2: Rho band structure — s_high should dominate
    # In a1→rho+pi, one OS pair carries the rho mass.
    # Require s_high > s_low + margin (the rho pair is distinguishable)
    d2_pass = s_high > (s_low + 0.1)  # 0.1 GeV^2 margin
    surviving &= d2_pass
    results['D2_count'] = int(surviving.sum())
    check_gt(surviving, 'D2')

    # D3: Bachelor energy in a1 rest frame
    # The "bachelor" is the same-sign track NOT paired with the OS track
    # in the rho. Identify which OS pair is the rho candidate (closer to m_rho).
    rho_is_os1 = np.abs(m_os1 - M_RHO) < np.abs(m_os2 - M_RHO)  # (N,)

    # Bachelor is track b when rho=(a,c), track a when rho=(b,c)
    bachelor_fv = np.where(rho_is_os1[:, None], fv_b, fv_a)  # (N, 4)

    # Boost bachelor to a1 (triplet) rest frame
    bachelor_boosted = np.zeros_like(bachelor_fv)
    surv_idx = np.where(surviving)[0]
    if len(surv_idx) > 0:
        bachelor_boosted[surv_idx] = boost_to_rest_frame(
            bachelor_fv[surv_idx], fv_triplet[surv_idx],
        )
    bachelor_energy = bachelor_boosted[:, 3]

    d3_pass = (bachelor_energy > 0.250) & (bachelor_energy < 0.500)
    # Only apply to survivors
    surviving_d3 = surviving.copy()
    surviving_d3 &= d3_pass
    results['D3_count'] = int(surviving_d3.sum())
    surviving = surviving_d3
    check_gt(surviving, 'D3')

    # D4: Rho helicity angle
    # cos(theta) = angle of the OS pion (track c) in rho rest frame
    # w.r.t. rho flight direction in a1 rest frame
    rho_fv = np.where(rho_is_os1[:, None], fv_os1, fv_os2)  # (N, 4)

    cos_theta = np.zeros(len(triplet_indices))
    if len(surv_idx) > 0:
        # Boost track c to rho rest frame
        c_in_rho = boost_to_rest_frame(fv_c[surv_idx], rho_fv[surv_idx])
        # Boost rho to a1 rest frame to get rho flight direction
        rho_in_a1 = boost_to_rest_frame(rho_fv[surv_idx], fv_triplet[surv_idx])

        # cos(theta) = p_c_in_rho · p_rho_in_a1 / (|p_c| * |p_rho|)
        p_c = c_in_rho[:, :3]
        p_rho = rho_in_a1[:, :3]
        dot = np.sum(p_c * p_rho, axis=1)
        norm_c = np.sqrt(np.sum(p_c**2, axis=1) + 1e-10)
        norm_rho = np.sqrt(np.sum(p_rho**2, axis=1) + 1e-10)
        cos_theta[surv_idx] = dot / (norm_c * norm_rho)

    d4_pass = np.abs(cos_theta) > 0.3
    surviving &= d4_pass
    results['D4_count'] = int(surviving.sum())
    check_gt(surviving, 'D4')

    # Final
    if gt_killed_at is None and gt_triplet_present:
        gt_killed_at = 'survived'
    elif not gt_triplet_present:
        gt_killed_at = 'not_in_topk'

    results['gt_killed_at'] = gt_killed_at
    results['final_survivors'] = int(surviving.sum())

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Combinatorial triplet feasibility experiment',
    )
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--stage1-checkpoint', type=str, required=True)
    parser.add_argument('--data-config', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='reports/triplet_reranking/triplet_combinatorics.md')
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-events', type=int, default=150)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    device = torch.device(args.device)

    # Load model
    from diagnostics.cascade_model_analysis import (
        load_cascade_from_checkpoint,
    )
    cascade = load_cascade_from_checkpoint(
        args.checkpoint, args.stage1_checkpoint, device,
    )

    # Load data
    from torch.utils.data import DataLoader
    from weaver.utils.dataset import SimpleIterDataset
    from utils.training_utils import (
        extract_label_from_inputs,
        trim_to_max_valid_tracks,
    )

    parquet_files = sorted(glob.glob(os.path.join(args.data_dir, '*.parquet')))
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

    # pT bins for per-bin analysis
    pt_bins = [(0, 0.5, 'low (0-0.5)'), (0.5, 1.0, 'mid (0.5-1)'), (1.0, 100, 'high (1+)')]

    # Run inference and collect per-event data
    max_steps = (args.max_events * 3 // args.batch_size) + 1  # generous margin for skips
    all_results = {200: [], 300: []}
    events_per_bin = {label: 0 for _, _, label in pt_bins}
    target_per_bin = args.max_events
    events_processed = 0

    logger.info(f'Processing up to {args.max_events} events per pT bin (3 bins)...')

    with torch.no_grad():
        for batch_index, (X, y, _) in enumerate(data_loader):
            if all(v >= target_per_bin for v in events_per_bin.values()):
                break
            if batch_index >= max_steps:
                break

            inputs = [X[key].to(device) for key in data_config.input_names]
            inputs = trim_to_max_valid_tracks(inputs, mask_input_index)
            model_inputs, track_labels = extract_label_from_inputs(
                inputs, label_input_index,
            )
            points, features, lorentz_vectors, mask = model_inputs

            # Run cascade
            cascade.train()
            filtered = cascade._run_stage1(
                points, features, lorentz_vectors, mask, track_labels,
            )
            stage2_scores = cascade.stage2(
                filtered['points'], filtered['features'],
                filtered['lorentz_vectors'], filtered['mask'],
                filtered['stage1_scores'],
            )

            # Scatter back to full event
            batch_size_actual = points.shape[0]
            full_scores = torch.full(
                (batch_size_actual, mask.shape[2]), float('-inf'), device=device,
            )
            full_scores.scatter_(1, filtered['selected_indices'], stage2_scores)

            # Process each event
            labels_flat = track_labels.squeeze(1)
            valid_mask = mask.squeeze(1).bool()

            for event_index in range(batch_size_actual):
                # Check if all bins are full
                if all(v >= target_per_bin for v in events_per_bin.values()):
                    break

                event_labels = labels_flat[event_index]
                event_valid = valid_mask[event_index]
                gt_positions = (
                    (event_labels == 1.0) & event_valid
                ).nonzero(as_tuple=True)[0]

                # Skip events without exactly 3 GT pions
                if len(gt_positions) != 3:
                    continue

                # Compute mean GT pT to assign bin
                px_gt = lorentz_vectors[event_index, 0, gt_positions]
                py_gt = lorentz_vectors[event_index, 1, gt_positions]
                pt_gt = torch.sqrt(px_gt**2 + py_gt**2)
                mean_gt_pt = pt_gt.mean().item()

                # Find pT bin
                event_bin = None
                for bin_lo, bin_hi, bin_label in pt_bins:
                    if bin_lo <= mean_gt_pt < bin_hi:
                        event_bin = bin_label
                        break
                if event_bin is None:
                    continue

                # Skip if this bin is already full
                if events_per_bin[event_bin] >= target_per_bin:
                    continue

                events_per_bin[event_bin] += 1
                events_processed += 1

                # Get event data on CPU
                event_scores = full_scores[event_index].cpu().numpy()
                event_lv = lorentz_vectors[event_index].cpu().numpy()  # (4, P)
                event_charges_std = features[event_index, 5, :].cpu().numpy()
                event_charges = event_charges_std / 0.5 + 1.0  # recover raw {-1, +1}
                event_gt = set(gt_positions.cpu().tolist())

                # For each K value
                for k_value in [200, 300]:
                    # Get top-K indices by score
                    scores_tensor = torch.from_numpy(event_scores)
                    scores_tensor[~event_valid[: len(scores_tensor)].cpu()] = float('-inf')
                    topk_indices = scores_tensor.topk(
                        min(k_value, int(event_valid.sum().item())),
                    ).indices.numpy()

                    # Extract 4-vectors and charges for top-K
                    topk_fv = event_lv[:, topk_indices].T  # (K, 4)
                    topk_charges = event_charges[topk_indices]  # (K,)

                    # Map GT indices to top-K positions
                    gt_in_topk = set()
                    for gt_pos in event_gt:
                        matches = np.where(topk_indices == gt_pos)[0]
                        if len(matches) > 0:
                            gt_in_topk.add(matches[0])

                    result = analyze_triplets_for_event(
                        topk_fv, topk_charges, gt_in_topk, k_value,
                    )
                    result['pt_bin'] = event_bin
                    result['mean_gt_pt'] = mean_gt_pt
                    all_results[k_value].append(result)

                if events_processed % 10 == 0:
                    filled = ', '.join(f'{k}:{v}' for k, v in events_per_bin.items())
                    logger.info(f'Processed {events_processed} events ({filled})')

    filled = ', '.join(f'{k}: {v}' for k, v in events_per_bin.items())
    logger.info(f'Total events processed: {events_processed} ({filled})')

    # Generate report
    report_lines = []

    def add(text=''):
        report_lines.append(text)

    add('# Combinatorial Triplet Feasibility Experiment')
    add()
    add(f'**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    add(f'**Events:** {events_processed} (3-GT only)')
    add(f'**Events per bin:** {filled}')
    add(f'**Model:** Physics ParT cascade (pair_extra_dim=5, concat mode)')
    add()
    add('---')

    bin_labels = ['all'] + [label for _, _, label in pt_bins]

    for k_value in [200, 300]:
        results = all_results[k_value]
        if not results:
            continue

        add()
        add(f'## K = {k_value}')

        for bin_label in bin_labels:
            if bin_label == 'all':
                bin_results = results
                section_title = 'All events'
            else:
                bin_results = [r for r in results if r.get('pt_bin') == bin_label]
                section_title = f'pT bin: {bin_label}'

            if not bin_results:
                continue

            n_events = len(bin_results)
            add()
            add(f'### {section_title} (n={n_events})')
            add()

            # GT in top-K
            gt_present = sum(1 for r in bin_results if r['gt_triplet_in_topk'])
            add(f'GT triplet fully in top-{k_value}: {gt_present}/{n_events} ({100*gt_present/max(1,n_events):.1f}%)')
            add()

            # Filter cascade table
            add('| Stage | Survivors (mean) | Survivors (median) | GT survival |')
            add('|-------|-----------------|-------------------|-------------|')

            filter_keys = ['F1', 'F2', 'F3', 'F4', 'D1', 'D2', 'D3', 'D4']
            for i, (stage_key, stage_name) in enumerate(zip(filter_keys, FILTER_NAMES)):
                count_key = f'{stage_key}_count'
                counts = [r.get(count_key, 0) for r in bin_results]
                counts_arr = np.array(counts)

                gt_survived = sum(
                    1 for r in bin_results
                    if r['gt_triplet_in_topk'] and r['gt_killed_at'] not in filter_keys[:i+1]
                )
                gt_rate = gt_survived / max(1, gt_present) if gt_present > 0 else 0

                short_name = stage_name.split('(')[0].strip()
                add(f'| {stage_key}: {short_name} | {counts_arr.mean():.0f} | {np.median(counts_arr):.0f} | {100*gt_rate:.1f}% |')

            # GT loss breakdown
            add()
            kill_counts = {}
            for r in bin_results:
                stage = r['gt_killed_at']
                kill_counts[stage] = kill_counts.get(stage, 0) + 1

            kill_summary = ', '.join(
                f'{stage}={count}'
                for stage in ['not_in_topk', 'F3', 'F4', 'D2', 'D3', 'D4', 'survived']
                if (count := kill_counts.get(stage, 0)) > 0
            )
            add(f'GT killed at: {kill_summary}')

            # Purity
            surviving_events = [r for r in bin_results if r['gt_killed_at'] == 'survived']
            if surviving_events:
                final_counts = np.array([r['final_survivors'] for r in surviving_events])
                add(f'Survivors when GT survives: mean={final_counts.mean():.0f}, median={np.median(final_counts):.0f}')

        add()

    # Write report
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        f.write('\n'.join(report_lines))

    logger.info(f'Report saved to {args.output}')


if __name__ == '__main__':
    main()
