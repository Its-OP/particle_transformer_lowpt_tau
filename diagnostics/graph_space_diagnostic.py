"""Graph space diagnostic: GT-neighbor connectivity across kNN feature spaces.

Evaluates how many ground-truth pion pairs appear as kNN neighbors in
different feature spaces. This determines which spaces to include in the
composite graph for TrackPreFilter.

Baseline: kNN(k=16) in (eta, phi) has only 28% of GT pions with >= 1 GT neighbor.
The goal is to find spaces where this fraction is significantly higher.

Usage:
    python diagnostics/graph_space_diagnostic.py \\
        --data-config data/low-pt/lowpt_tau_trackfinder.yaml \\
        --data-dir data/low-pt/val/ \\
        --device mps \\
        --batch-size 8 \\
        --max-steps 50
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger('graph_space_diagnostic')


# ---------------------------------------------------------------------------
# kNN utilities
# ---------------------------------------------------------------------------

def knn_l2(
    coordinates: torch.Tensor,
    num_neighbors: int,
    mask: torch.Tensor,
) -> torch.Tensor:
    """kNN in arbitrary feature space using L2 distance.

    Args:
        coordinates: (B, D, P) — D-dimensional coordinates per track.
        num_neighbors: K.
        mask: (B, 1, P) — validity mask (1=valid, 0=padded).

    Returns:
        indices: (B, P, K) — neighbor indices into the P dimension.
    """
    # (B, P, D) pairwise L2 distances → (B, P, P)
    coords_transposed = coordinates.permute(0, 2, 1)  # (B, P, D)
    distances = torch.cdist(coords_transposed, coords_transposed)  # (B, P, P)

    # Exclude self-matches
    distances.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))

    # Mask invalid reference tracks (columns)
    invalid_mask = ~mask.squeeze(1).bool()  # (B, P)
    distances.masked_fill_(invalid_mask.unsqueeze(1), float('inf'))

    # Clamp k to available valid tracks
    effective_k = min(num_neighbors, distances.shape[-1] - 1)

    return distances.topk(effective_k, dim=-1, largest=False).indices


def knn_eta_phi(
    points: torch.Tensor,
    num_neighbors: int,
    mask: torch.Tensor,
) -> torch.Tensor:
    """kNN in (eta, phi) space with phi wrapping.

    Uses delta_phi = (phi_a - phi_b + pi) mod 2pi - pi for periodic phi.

    Args:
        points: (B, 2, P) — (eta, phi) coordinates.
        num_neighbors: K.
        mask: (B, 1, P) — validity mask.

    Returns:
        indices: (B, P, K).
    """
    eta = points[:, 0:1, :]  # (B, 1, P)
    phi = points[:, 1:2, :]  # (B, 1, P)

    # ΔR² = (Δη)² + (Δφ_wrapped)²
    # Broadcast: (B, 1, P, 1) vs (B, 1, 1, P) → (B, 1, P, P)
    delta_eta = eta.unsqueeze(-1) - eta.unsqueeze(-2)
    delta_phi = phi.unsqueeze(-1) - phi.unsqueeze(-2)
    # Phi wrapping: (Δφ + π) mod 2π − π
    delta_phi = (delta_phi + torch.pi) % (2 * torch.pi) - torch.pi
    distances = (delta_eta.square() + delta_phi.square()).squeeze(1)  # (B, P, P)

    distances.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))
    invalid_mask = ~mask.squeeze(1).bool()
    distances.masked_fill_(invalid_mask.unsqueeze(1), float('inf'))

    effective_k = min(num_neighbors, distances.shape[-1] - 1)
    return distances.topk(effective_k, dim=-1, largest=False).indices


# ---------------------------------------------------------------------------
# Space-specific kNN builders
# ---------------------------------------------------------------------------

def build_knn_eta_phi(
    points: torch.Tensor,
    mask: torch.Tensor,
    num_neighbors: int = 16,
) -> torch.Tensor:
    """Baseline: kNN in (eta, phi) with phi wrapping."""
    return knn_eta_phi(points, num_neighbors, mask)


def build_knn_dz_sig(
    features: torch.Tensor,
    mask: torch.Tensor,
    num_neighbors: int = 12,
) -> torch.Tensor:
    """Space S1: kNN in dz_significance (1D).

    All 3 pions share the tau vertex z_v, so their dz ~ z_v - z_PV
    are similar. Background clusters at dz~0 (primary vertex).
    Feature index 7 = track_log_dz_significance.
    """
    dz_coords = features[:, 7:8, :]  # (B, 1, P)
    return knn_l2(dz_coords, num_neighbors, mask)


def build_knn_vertex_proxy(
    points: torch.Tensor,
    features: torch.Tensor,
    mask: torch.Tensor,
    num_neighbors: int = 12,
) -> torch.Tensor:
    """Space S2: kNN in transverse vertex proxy (v_x, v_y).

    d₀ ≈ -x_v sin(φ) + y_v cos(φ), so:
        v_x = -dxy_sig * sin(φ)
        v_y =  dxy_sig * cos(φ)
    maps same-vertex tracks to similar (v_x, v_y).

    Uses raw phi from points (unstandardized) and dxy_significance
    from features (index 6). Note: using dxy_sig instead of raw d₀
    introduces noise from per-track sigma, but the mapping is still
    useful since sigma is largely monotonic in pT.
    """
    dxy_sig = features[:, 6:7, :]  # (B, 1, P)
    phi_raw = points[:, 1:2, :]    # (B, 1, P) — unstandardized

    vertex_x = -dxy_sig * torch.sin(phi_raw)  # (B, 1, P)
    vertex_y = dxy_sig * torch.cos(phi_raw)   # (B, 1, P)
    vertex_coords = torch.cat([vertex_x, vertex_y], dim=1)  # (B, 2, P)

    return knn_l2(vertex_coords, num_neighbors, mask)


def build_knn_opposite_sign(
    points: torch.Tensor,
    features: torch.Tensor,
    mask: torch.Tensor,
    num_neighbors: int = 16,
) -> torch.Tensor:
    """Space S3: kNN in (eta, phi) restricted to opposite-sign pairs.

    The rho(770) → pi+pi- decay mandates that the rho daughter pair
    carries opposite charges. By filtering to OS pairs only, each track's
    neighborhood is drawn from ~550 candidates (half the event) instead
    of ~1100, effectively doubling the ΔR reach for the same k.

    Feature index 5 = charge (standardized: center=1.0, scale=0.5).
    Raw charge ∈ {-1, +1}.
    """
    eta = points[:, 0:1, :]  # (B, 1, P)
    phi = points[:, 1:2, :]  # (B, 1, P)

    # ΔR² distance matrix
    delta_eta = eta.unsqueeze(-1) - eta.unsqueeze(-2)
    delta_phi = phi.unsqueeze(-1) - phi.unsqueeze(-2)
    delta_phi = (delta_phi + torch.pi) % (2 * torch.pi) - torch.pi
    distances = (delta_eta.square() + delta_phi.square()).squeeze(1)  # (B, P, P)

    # Charge filtering: set same-sign pairs to inf
    # Recover raw charge sign from standardized features
    charge_raw = features[:, 5, :]  # (B, P)
    charge_sign = torch.sign(charge_raw)  # (B, P) — {-1, +1}
    # charge_product[i,j] > 0 means same sign → exclude
    charge_product = charge_sign.unsqueeze(-1) * charge_sign.unsqueeze(-2)  # (B, P, P)
    same_sign_mask = charge_product > 0  # (B, P, P)
    distances.masked_fill_(same_sign_mask, float('inf'))

    # Standard masking
    distances.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))
    invalid_mask = ~mask.squeeze(1).bool()
    distances.masked_fill_(invalid_mask.unsqueeze(1), float('inf'))

    effective_k = min(num_neighbors, distances.shape[-1] - 1)
    return distances.topk(effective_k, dim=-1, largest=False).indices


def build_knn_logpt_eta_phi(
    points: torch.Tensor,
    features: torch.Tensor,
    mask: torch.Tensor,
    num_neighbors: int = 12,
) -> torch.Tensor:
    """Space S4: kNN in (log_pT, eta, phi) with per-event z-score.

    The tau mass constraint m(3pi) <= 1.777 GeV correlates daughter
    momenta. Adding log_pT separates low-pT signal pions from high-pT
    background along the pT axis. Z-scoring ensures balanced weighting.

    Feature index 14 = track_log_pt.
    """
    log_pt = features[:, 14:15, :]  # (B, 1, P)
    coords_3d = torch.cat([log_pt, points], dim=1)  # (B, 3, P)

    # Per-event z-score normalization
    mask_float = mask.float()  # (B, 1, P)
    num_valid = mask_float.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, 1, 1)

    mean = (coords_3d * mask_float).sum(dim=-1, keepdim=True) / num_valid
    variance = (
        ((coords_3d - mean) * mask_float).square().sum(dim=-1, keepdim=True)
        / num_valid
    )
    std = variance.sqrt().clamp(min=1e-6)
    coords_normed = ((coords_3d - mean) / std) * mask_float

    return knn_l2(coords_normed, num_neighbors, mask)


def build_composite_graph(
    points: torch.Tensor,
    features: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Union of all kNN spaces.

    Returns concatenated neighbor indices from all spaces.
    Duplicates are fine — max-pool/PNA aggregation is permutation-invariant.
    """
    indices_eta_phi = build_knn_eta_phi(points, mask, num_neighbors=16)
    indices_dz = build_knn_dz_sig(features, mask, num_neighbors=12)
    indices_vertex = build_knn_vertex_proxy(points, features, mask, num_neighbors=12)
    indices_os = build_knn_opposite_sign(points, features, mask, num_neighbors=16)
    indices_logpt = build_knn_logpt_eta_phi(points, features, mask, num_neighbors=12)

    return torch.cat([
        indices_eta_phi, indices_dz, indices_vertex, indices_os, indices_logpt,
    ], dim=-1)


# ---------------------------------------------------------------------------
# GT-neighbor metrics
# ---------------------------------------------------------------------------

def compute_gt_neighbor_metrics(
    neighbor_indices: torch.Tensor,
    track_labels: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    """Compute GT-neighbor connectivity statistics.

    For each GT pion, counts how many of its kNN neighbors are also GT pions.

    Args:
        neighbor_indices: (B, P, K) kNN indices.
        track_labels: (B, 1, P) binary labels (1.0 = GT pion).
        mask: (B, 1, P) validity mask.

    Returns:
        Dict with:
        - gt_with_at_least_1_gt_neighbor: fraction of GT pions with >= 1 GT neighbor
        - gt_with_at_least_2_gt_neighbors: fraction with >= 2
        - mean_gt_neighbors_per_gt: average GT neighbors per GT pion
        - total_gt_pions: total GT pions evaluated
    """
    labels_flat = track_labels.squeeze(1)  # (B, P)
    valid_mask = mask.squeeze(1).bool()    # (B, P)

    total_gt_with_1_plus = 0
    total_gt_with_2_plus = 0
    total_gt_neighbor_count = 0
    total_gt_pions = 0

    batch_size = neighbor_indices.shape[0]
    for batch_index in range(batch_size):
        event_labels = labels_flat[batch_index]
        event_valid = valid_mask[batch_index]
        event_neighbors = neighbor_indices[batch_index]  # (P, K)

        # GT pion positions
        gt_positions = (
            (event_labels == 1.0) & event_valid
        ).nonzero(as_tuple=True)[0]

        if len(gt_positions) == 0:
            continue

        for gt_pos in gt_positions:
            # Get this GT pion's neighbors
            neighbors = event_neighbors[gt_pos]  # (K,)
            # Count how many neighbors are also GT pions
            neighbor_labels = event_labels[neighbors]
            gt_neighbor_count = (neighbor_labels == 1.0).sum().item()

            total_gt_pions += 1
            total_gt_neighbor_count += gt_neighbor_count
            if gt_neighbor_count >= 1:
                total_gt_with_1_plus += 1
            if gt_neighbor_count >= 2:
                total_gt_with_2_plus += 1

    if total_gt_pions == 0:
        return {
            'gt_with_at_least_1_gt_neighbor': 0.0,
            'gt_with_at_least_2_gt_neighbors': 0.0,
            'mean_gt_neighbors_per_gt': 0.0,
            'total_gt_pions': 0,
        }

    return {
        'gt_with_at_least_1_gt_neighbor': total_gt_with_1_plus / total_gt_pions,
        'gt_with_at_least_2_gt_neighbors': total_gt_with_2_plus / total_gt_pions,
        'mean_gt_neighbors_per_gt': total_gt_neighbor_count / total_gt_pions,
        'total_gt_pions': total_gt_pions,
    }


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

SPACE_BUILDERS = {
    'eta_phi': lambda pts, fts, msk, k: build_knn_eta_phi(pts, msk, k),
    'dz_sig': lambda pts, fts, msk, k: build_knn_dz_sig(fts, msk, k),
    'vertex_proxy': lambda pts, fts, msk, k: build_knn_vertex_proxy(pts, fts, msk, k),
    'opposite_sign': lambda pts, fts, msk, k: build_knn_opposite_sign(pts, fts, msk, k),
    'logpt_eta_phi': lambda pts, fts, msk, k: build_knn_logpt_eta_phi(pts, fts, msk, k),
}

K_VALUES = [4, 8, 12, 16, 24, 32]


def run_diagnostic(
    data_config_path: str,
    data_dir: str,
    device: torch.device | None = None,
    batch_size: int = 8,
    max_steps: int | None = None,
    num_workers: int = 0,
) -> dict[str, dict[int, dict[str, float]]]:
    """Run GT-neighbor diagnostic across all spaces and k values.

    Returns:
        Nested dict: results[space_name][k] = {metric: value}
    """
    from torch.utils.data import DataLoader

    from weaver.utils.dataset import SimpleIterDataset

    from utils.training_utils import (
        extract_label_from_inputs,
        trim_to_max_valid_tracks,
    )

    if device is None:
        device = torch.device('cpu')

    parquet_files = sorted(glob.glob(os.path.join(data_dir, '*.parquet')))
    if not parquet_files:
        raise FileNotFoundError(f'No parquet files in {data_dir}')

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

    # Accumulate across batches per (space, k)
    accumulators: dict[str, dict[int, dict[str, float]]] = {}
    for space_name in SPACE_BUILDERS:
        accumulators[space_name] = {}
        for k in K_VALUES:
            accumulators[space_name][k] = {
                'gt_1plus': 0, 'gt_2plus': 0,
                'gt_neighbors_total': 0, 'gt_total': 0,
            }

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

            for space_name, builder_fn in SPACE_BUILDERS.items():
                for k in K_VALUES:
                    indices = builder_fn(points, features, mask, k)
                    metrics = compute_gt_neighbor_metrics(
                        indices, track_labels, mask,
                    )

                    accumulator = accumulators[space_name][k]
                    n_gt = metrics['total_gt_pions']
                    accumulator['gt_1plus'] += int(
                        metrics['gt_with_at_least_1_gt_neighbor'] * n_gt
                    )
                    accumulator['gt_2plus'] += int(
                        metrics['gt_with_at_least_2_gt_neighbors'] * n_gt
                    )
                    accumulator['gt_neighbors_total'] += (
                        metrics['mean_gt_neighbors_per_gt'] * n_gt
                    )
                    accumulator['gt_total'] += n_gt

            if (batch_index + 1) % 10 == 0:
                logger.info(f'Batch {batch_index + 1} processed')

    # Compute final fractions
    results: dict[str, dict[int, dict[str, float]]] = {}
    for space_name in SPACE_BUILDERS:
        results[space_name] = {}
        for k in K_VALUES:
            accumulator = accumulators[space_name][k]
            n = max(1, accumulator['gt_total'])
            results[space_name][k] = {
                'gt_with_at_least_1_gt_neighbor': accumulator['gt_1plus'] / n,
                'gt_with_at_least_2_gt_neighbors': accumulator['gt_2plus'] / n,
                'mean_gt_neighbors_per_gt': accumulator['gt_neighbors_total'] / n,
                'total_gt_pions': accumulator['gt_total'],
            }

    return results


def print_diagnostic_results(
    results: dict[str, dict[int, dict[str, float]]],
) -> None:
    """Print formatted diagnostic table."""
    print('\n' + '=' * 90)
    print('  GT-NEIGHBOR CONNECTIVITY DIAGNOSTIC')
    print('  Fraction of GT pions with >= 1 GT neighbor in kNN graph')
    print('=' * 90)

    # Header row
    header = f'  {"Space":<20}'
    for k in K_VALUES:
        header += f'  k={k:<4}'
    print(header)
    print('-' * 90)

    for space_name in SPACE_BUILDERS:
        row = f'  {space_name:<20}'
        for k in K_VALUES:
            fraction = results[space_name][k]['gt_with_at_least_1_gt_neighbor']
            row += f'  {fraction:.3f}'
        print(row)

    print()
    print('  >= 2 GT neighbors:')
    print('-' * 90)
    for space_name in SPACE_BUILDERS:
        row = f'  {space_name:<20}'
        for k in K_VALUES:
            fraction = results[space_name][k]['gt_with_at_least_2_gt_neighbors']
            row += f'  {fraction:.3f}'
        print(row)

    n_gt = results['eta_phi'][K_VALUES[0]]['total_gt_pions']
    print(f'\n  Total GT pions evaluated: {n_gt}')
    print('=' * 90)


def main():
    parser = argparse.ArgumentParser(
        description='GT-neighbor connectivity diagnostic across kNN spaces',
    )
    parser.add_argument('--data-config', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='mps')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    device = torch.device(args.device)
    results = run_diagnostic(
        data_config_path=args.data_config,
        data_dir=args.data_dir,
        device=device,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        num_workers=args.num_workers,
    )
    print_diagnostic_results(results)


if __name__ == '__main__':
    main()
