"""Deep diagnostics for Phase-B TrackPreFilter model.

Analyzes where the model excels and fails by comparing success vs failure
events, characterizing missed GT tracks, examining score distributions,
and inspecting kNN neighborhood composition.

Usage:
    python diagnostics/model_failure_analysis.py \
        --checkpoint models/debug_checkpoints/phase-b/checkpoints/best_model.pt \
        --data-config data/low-pt/lowpt_tau_trackfinder.yaml \
        --val-data-dir data/low-pt/val/ \
        --val-files val_00.parquet val_01.parquet \
        --batch-size 8 \
        --max-steps 100 \
        --device mps
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys

import numpy as np
import torch

# Project root on PYTHONPATH for weaver and utils imports
sys.path.insert(0, '.')

from torch.utils.data import DataLoader

from weaver.nn.model.HierarchicalGraphBackbone import cross_set_knn, cross_set_gather
from weaver.nn.model.TrackPreFilter import TrackPreFilter
from weaver.utils.dataset import SimpleIterDataset

from utils.training_utils import (
    extract_label_from_inputs,
    load_network_module,
    trim_to_max_valid_tracks,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('model_failure_analysis')

# Feature names corresponding to pf_features channels (13-feature config)
FEATURE_NAMES = [
    'track_px', 'track_py', 'track_pz', 'track_eta', 'track_phi',
    'track_charge', 'track_dxy_significance', 'track_dz_significance',
    'track_pt_error', 'track_n_valid_pixel_hits', 'track_dca_significance',
    'track_covariance_phi_phi', 'track_covariance_lambda_lambda',
]

# Preprocessing params from the auto.yaml — used to un-standardize features
# for human-readable analysis. Formula: raw = standardized * scale + center
PREPROCESS_PARAMS = {
    'track_px': {'center': -0.006082190433517098, 'scale': 2.048686774561125},
    'track_py': {'center': 0.0012929992517456412, 'scale': 2.067582670032769},
    'track_pz': {'center': 0.018174415454268456, 'scale': 0.5446895030631395},
    'track_eta': {'center': 0.0391845703125, 'scale': 0.5039680098431252},
    'track_phi': {'center': 0.005466938018798828, 'scale': 0.4615527163060676},
    'track_charge': {'center': 1.0, 'scale': 0.5},
    'track_dxy_significance': {'center': 0.005108157638460398, 'scale': 0.8543357188021373},
    'track_dz_significance': {'center': 0.198388010263443, 'scale': 0.0034129274432297303},
    'track_pt_error': {'center': 0.009976133704185486, 'scale': 67.14158713102799},
    'track_n_valid_pixel_hits': {'center': 4.0, 'scale': 1.0},
    'track_dca_significance': {'center': 0.7176337540149689, 'scale': 0.6822560876403961},
    'track_covariance_phi_phi': {'center': 2.9802322387695312e-05, 'scale': 7530.168761220826},
    'track_covariance_lambda_lambda': {'center': 4.857778549194336e-06, 'scale': 86703.95865633075},
}


def unstandardize_feature(standardized_values: np.ndarray, feature_name: str) -> np.ndarray:
    """Convert standardized feature values back to raw (physical) units.

    Inverse of weaver's auto-preprocessing:
        raw = standardized * scale + center
    """
    params = PREPROCESS_PARAMS.get(feature_name)
    if params is None or params['center'] is None:
        return standardized_values
    return standardized_values * params['scale'] + params['center']


def compute_pt_from_features(features_px: np.ndarray, features_py: np.ndarray) -> np.ndarray:
    """Compute pT from raw px and py components.

    pT = sqrt(px^2 + py^2)
    """
    return np.sqrt(features_px ** 2 + features_py ** 2)


# ---------------------------------------------------------------------------
# Data collection: per-event and per-GT-track records
# ---------------------------------------------------------------------------

def collect_event_records(
    model: torch.nn.Module,
    data_loader: DataLoader,
    data_config,
    device: torch.device,
    mask_input_index: int,
    label_input_index: int,
    max_steps: int,
    num_neighbors: int = 16,
) -> list[dict]:
    """Run inference on val data and collect detailed per-event records.

    For each event, records track count, GT count, per-GT-track features and
    rank, top-5 false positive scores, and kNN neighbor composition.

    Uses the EXACT validate() pattern from train_prefilter.py:
    model.train() before compute_loss(), then model.eval().
    """
    all_events = []

    with torch.no_grad():
        for batch_index, (X, y, _) in enumerate(data_loader):
            if batch_index >= max_steps:
                break

            if batch_index % 20 == 0:
                logger.info(f'Processing batch {batch_index}/{max_steps}...')

            inputs = [X[k].to(device) for k in data_config.input_names]
            inputs = trim_to_max_valid_tracks(inputs, mask_input_index)
            model_inputs, track_labels = extract_label_from_inputs(
                inputs, label_input_index,
            )
            points, features, lorentz_vectors, mask = model_inputs

            # Exact validate() pattern: model.train() for compute_loss(), then eval()
            model.train()
            loss_dict = model.compute_loss(
                points, features, lorentz_vectors, mask, track_labels,
            )
            model.eval()

            per_track_scores = loss_dict.pop('_scores').detach()

            # Compute kNN indices for neighborhood analysis
            neighbor_indices = cross_set_knn(
                query_coordinates=points,
                reference_coordinates=points,
                num_neighbors=num_neighbors,
                reference_mask=mask,
                query_reference_indices=None,
            )  # (B, P, K)

            batch_size = per_track_scores.shape[0]
            labels_flat = track_labels.squeeze(1)[:, :per_track_scores.shape[1]]
            valid_mask = mask.squeeze(1).bool()

            # Compute ranks: argsort(argsort(descending)) gives rank of each element
            masked_scores = per_track_scores.clone()
            masked_scores[~valid_mask] = float('-inf')
            rank_lookup = torch.argsort(
                torch.argsort(masked_scores, dim=1, descending=True), dim=1,
            )

            sorted_indices = masked_scores.argsort(dim=1, descending=True)

            for event_index in range(batch_size):
                event_valid = valid_mask[event_index]
                event_labels = labels_flat[event_index] * event_valid.float()
                event_scores = per_track_scores[event_index]
                event_features = features[event_index]  # (C, P)
                event_neighbors = neighbor_indices[event_index]  # (P, K)

                num_tracks = event_valid.sum().item()
                gt_mask_event = (event_labels == 1.0) & event_valid
                background_mask_event = (event_labels == 0.0) & event_valid
                num_gt = gt_mask_event.sum().item()

                if num_gt == 0:
                    continue

                gt_positions = gt_mask_event.nonzero(as_tuple=True)[0]
                background_positions = background_mask_event.nonzero(as_tuple=True)[0]

                # Per-GT-track: rank, score, raw features, kNN neighbor GT count
                gt_ranks = rank_lookup[event_index, gt_positions].cpu().numpy()
                gt_scores = event_scores[gt_positions].cpu().numpy()
                gt_features_standardized = event_features[:, gt_positions].cpu().numpy()  # (C, num_gt)

                # kNN neighborhood: count GT neighbors for each GT track
                gt_neighbor_indices = event_neighbors[gt_positions]  # (num_gt, K)
                # For each GT track, check how many of its K neighbors are also GT
                gt_set = set(gt_positions.cpu().tolist())
                gt_neighbor_gt_counts = []
                for gt_track_index in range(len(gt_positions)):
                    neighbors_of_this_gt = gt_neighbor_indices[gt_track_index].cpu().tolist()
                    gt_count_in_neighbors = sum(
                        1 for neighbor in neighbors_of_this_gt if neighbor in gt_set
                    )
                    gt_neighbor_gt_counts.append(gt_count_in_neighbors)

                # Background kNN analysis: for a sample of background tracks,
                # count how many GT neighbors they have
                max_background_sample = min(200, len(background_positions))
                if max_background_sample > 0:
                    background_sample_indices = background_positions[
                        torch.randperm(len(background_positions))[:max_background_sample]
                    ]
                    background_neighbor_indices = event_neighbors[background_sample_indices]
                    background_neighbor_gt_counts = []
                    for background_track_index in range(len(background_sample_indices)):
                        neighbors_of_bg = background_neighbor_indices[background_track_index].cpu().tolist()
                        gt_count_in_neighbors = sum(
                            1 for neighbor in neighbors_of_bg if neighbor in gt_set
                        )
                        background_neighbor_gt_counts.append(gt_count_in_neighbors)
                else:
                    background_neighbor_gt_counts = []

                # Recall@200 and perfect@200
                top_200_indices = sorted_indices[event_index, :200]
                found_gt_count = torch.isin(gt_positions, top_200_indices).sum().item()
                recall_at_200 = found_gt_count / num_gt
                perfect_at_200 = 1.0 if found_gt_count == num_gt else 0.0

                # Top-5 false positives: highest-scoring noise tracks
                top_200_labels = event_labels[top_200_indices]
                top_200_scores = event_scores[top_200_indices]
                false_positive_mask = (top_200_labels == 0.0)
                if false_positive_mask.any():
                    fp_scores_all = top_200_scores[false_positive_mask].cpu().numpy()
                    fp_scores_sorted = np.sort(fp_scores_all)[::-1]
                    top_5_fp_scores = fp_scores_sorted[:5].tolist()
                else:
                    top_5_fp_scores = []

                # Background score sample for distribution analysis
                background_scores = event_scores[background_positions].cpu().numpy()

                # Un-standardize key features for GT tracks
                gt_raw_features = {}
                for feature_index, feature_name in enumerate(FEATURE_NAMES):
                    raw_values = unstandardize_feature(
                        gt_features_standardized[feature_index], feature_name,
                    )
                    gt_raw_features[feature_name] = raw_values

                # Compute pT for GT tracks
                gt_raw_features['track_pt'] = compute_pt_from_features(
                    gt_raw_features['track_px'], gt_raw_features['track_py'],
                )

                # Background features (sample for comparison)
                background_feature_sample_size = min(500, len(background_positions))
                if background_feature_sample_size > 0:
                    background_sample_positions = background_positions[
                        torch.randperm(len(background_positions))[:background_feature_sample_size]
                    ]
                    background_features_standardized = (
                        event_features[:, background_sample_positions].cpu().numpy()
                    )
                    background_raw_features = {}
                    for feature_index, feature_name in enumerate(FEATURE_NAMES):
                        background_raw_features[feature_name] = unstandardize_feature(
                            background_features_standardized[feature_index], feature_name,
                        )
                    background_raw_features['track_pt'] = compute_pt_from_features(
                        background_raw_features['track_px'],
                        background_raw_features['track_py'],
                    )
                else:
                    background_raw_features = {}

                all_events.append({
                    'num_tracks': int(num_tracks),
                    'num_gt': int(num_gt),
                    'recall_at_200': recall_at_200,
                    'perfect_at_200': perfect_at_200,
                    'gt_ranks': gt_ranks,
                    'gt_scores': gt_scores,
                    'gt_raw_features': gt_raw_features,
                    'gt_neighbor_gt_counts': gt_neighbor_gt_counts,
                    'background_scores': background_scores,
                    'background_raw_features': background_raw_features,
                    'background_neighbor_gt_counts': background_neighbor_gt_counts,
                    'top_5_fp_scores': top_5_fp_scores,
                })

            del inputs, model_inputs, track_labels, loss_dict
            del per_track_scores, neighbor_indices

    logger.info(f'Collected {len(all_events)} events with GT tracks.')
    return all_events


# ---------------------------------------------------------------------------
# Analysis A: Success vs Failure breakdown
# ---------------------------------------------------------------------------

def analyze_success_vs_failure(events: list[dict]) -> None:
    """Split events into success (R@200=1.0) and failure, compare distributions."""
    print('\n' + '=' * 80)
    print('SECTION A: SUCCESS vs FAILURE BREAKDOWN')
    print('=' * 80)

    success_events = [e for e in events if e['perfect_at_200'] == 1.0]
    failure_events = [e for e in events if e['perfect_at_200'] < 1.0]

    total = len(events)
    print(f'\nTotal events with GT: {total}')
    print(f'Success (all GT in top-200): {len(success_events)} ({100*len(success_events)/max(1,total):.1f}%)')
    print(f'Failure (some GT missed):    {len(failure_events)} ({100*len(failure_events)/max(1,total):.1f}%)')

    # R@200 distribution
    recall_values = [e['recall_at_200'] for e in events]
    print(f'\nR@200 distribution:')
    print(f'  Mean:   {np.mean(recall_values):.4f}')
    print(f'  Median: {np.median(recall_values):.4f}')
    print(f'  Std:    {np.std(recall_values):.4f}')
    print(f'  Min:    {np.min(recall_values):.4f}')

    # Compare distributions
    def print_comparison(label: str, success_values: list, failure_values: list) -> None:
        """Print side-by-side comparison of success vs failure distributions."""
        success_array = np.array(success_values) if success_values else np.array([0.0])
        failure_array = np.array(failure_values) if failure_values else np.array([0.0])
        print(f'\n  {label}:')
        print(f'    {"":20s} {"Success":>12s} {"Failure":>12s} {"Delta":>12s}')
        print(f'    {"Mean":20s} {np.mean(success_array):12.2f} {np.mean(failure_array):12.2f} {np.mean(failure_array)-np.mean(success_array):+12.2f}')
        print(f'    {"Median":20s} {np.median(success_array):12.2f} {np.median(failure_array):12.2f} {np.median(failure_array)-np.median(success_array):+12.2f}')
        print(f'    {"Std":20s} {np.std(success_array):12.2f} {np.std(failure_array):12.2f} {"":12s}')

    success_track_counts = [e['num_tracks'] for e in success_events]
    failure_track_counts = [e['num_tracks'] for e in failure_events]
    print_comparison('Track count per event', success_track_counts, failure_track_counts)

    success_gt_counts = [e['num_gt'] for e in success_events]
    failure_gt_counts = [e['num_gt'] for e in failure_events]
    print_comparison('GT count per event', success_gt_counts, failure_gt_counts)

    # GT pT spectrum
    success_gt_pts = []
    for e in success_events:
        success_gt_pts.extend(e['gt_raw_features']['track_pt'].tolist())
    failure_gt_pts = []
    for e in failure_events:
        failure_gt_pts.extend(e['gt_raw_features']['track_pt'].tolist())
    print_comparison('GT track pT (GeV)', success_gt_pts, failure_gt_pts)

    # GT eta
    success_gt_etas = []
    for e in success_events:
        success_gt_etas.extend(e['gt_raw_features']['track_eta'].tolist())
    failure_gt_etas = []
    for e in failure_events:
        failure_gt_etas.extend(e['gt_raw_features']['track_eta'].tolist())
    print_comparison('GT track |eta|', [abs(x) for x in success_gt_etas], [abs(x) for x in failure_gt_etas])

    # GT dxy_significance
    success_dxy = []
    for e in success_events:
        success_dxy.extend(e['gt_raw_features']['track_dxy_significance'].tolist())
    failure_dxy = []
    for e in failure_events:
        failure_dxy.extend(e['gt_raw_features']['track_dxy_significance'].tolist())
    print_comparison('GT track dxy_significance', success_dxy, failure_dxy)

    # GT dz_significance
    success_dz = []
    for e in success_events:
        success_dz.extend(e['gt_raw_features']['track_dz_significance'].tolist())
    failure_dz = []
    for e in failure_events:
        failure_dz.extend(e['gt_raw_features']['track_dz_significance'].tolist())
    print_comparison('GT track dz_significance', success_dz, failure_dz)

    # GT count histogram
    print('\n  GT count distribution:')
    all_gt_counts = [e['num_gt'] for e in events]
    for gt_count in sorted(set(all_gt_counts)):
        total_with_count = sum(1 for e in events if e['num_gt'] == gt_count)
        success_with_count = sum(1 for e in success_events if e['num_gt'] == gt_count)
        failure_with_count = sum(1 for e in failure_events if e['num_gt'] == gt_count)
        success_rate = success_with_count / max(1, total_with_count)
        print(
            f'    GT={gt_count}: {total_with_count:4d} events | '
            f'success={success_with_count:4d} ({100*success_rate:5.1f}%) | '
            f'failure={failure_with_count:4d}'
        )


# ---------------------------------------------------------------------------
# Analysis B: Per-GT-track failure analysis
# ---------------------------------------------------------------------------

def analyze_per_gt_track_failures(events: list[dict]) -> None:
    """Compare features of found GT vs missed GT vs noise tracks."""
    print('\n' + '=' * 80)
    print('SECTION B: PER-GT-TRACK FAILURE ANALYSIS')
    print('=' * 80)

    found_gt_features = {name: [] for name in FEATURE_NAMES + ['track_pt']}
    missed_gt_features = {name: [] for name in FEATURE_NAMES + ['track_pt']}
    noise_features = {name: [] for name in FEATURE_NAMES + ['track_pt']}

    found_gt_count = 0
    missed_gt_count = 0

    for event in events:
        gt_ranks = event['gt_ranks']
        gt_raw = event['gt_raw_features']
        background_raw = event['background_raw_features']

        for gt_track_index in range(len(gt_ranks)):
            is_found = gt_ranks[gt_track_index] < 200
            target_dict = found_gt_features if is_found else missed_gt_features
            if is_found:
                found_gt_count += 1
            else:
                missed_gt_count += 1
            for feature_name in FEATURE_NAMES + ['track_pt']:
                target_dict[feature_name].append(gt_raw[feature_name][gt_track_index])

        # Collect noise track features
        if background_raw:
            for feature_name in FEATURE_NAMES + ['track_pt']:
                if feature_name in background_raw:
                    noise_features[feature_name].extend(
                        background_raw[feature_name].tolist()
                    )

    total_gt = found_gt_count + missed_gt_count
    print(f'\nTotal GT tracks: {total_gt}')
    print(f'Found (rank < 200):  {found_gt_count} ({100*found_gt_count/max(1,total_gt):.1f}%)')
    print(f'Missed (rank >= 200): {missed_gt_count} ({100*missed_gt_count/max(1,total_gt):.1f}%)')

    if missed_gt_count == 0:
        print('\nNo missed GT tracks -- model achieves perfect R@200 on all events!')
        return

    # Feature comparison table
    key_features = ['track_pt', 'track_eta', 'track_dxy_significance',
                    'track_dz_significance', 'track_dca_significance',
                    'track_pt_error', 'track_n_valid_pixel_hits']

    print(f'\n  Feature comparison (mean +/- std):')
    print(f'    {"Feature":30s} {"Found GT":>20s} {"Missed GT":>20s} {"Noise":>20s}')
    print(f'    {"-"*30} {"-"*20} {"-"*20} {"-"*20}')

    for feature_name in key_features:
        found_array = np.array(found_gt_features[feature_name])
        missed_array = np.array(missed_gt_features[feature_name])
        noise_array = np.array(noise_features.get(feature_name, [0.0]))

        found_str = f'{np.mean(found_array):.4f} +/- {np.std(found_array):.4f}'
        missed_str = f'{np.mean(missed_array):.4f} +/- {np.std(missed_array):.4f}'
        noise_str = f'{np.mean(noise_array):.4f} +/- {np.std(noise_array):.4f}'
        print(f'    {feature_name:30s} {found_str:>20s} {missed_str:>20s} {noise_str:>20s}')

    # Overlap analysis: identify features where missed GT looks most like noise
    # Use Bhattacharyya coefficient as overlap measure
    print(f'\n  Feature overlap (missed-GT vs noise) — higher = more overlap:')
    print(f'    {"Feature":30s} {"Bhattacharyya coeff":>20s} {"Interpretation":>20s}')
    print(f'    {"-"*30} {"-"*20} {"-"*20}')

    overlap_results = []
    for feature_name in key_features:
        missed_array = np.array(missed_gt_features[feature_name])
        noise_array = np.array(noise_features.get(feature_name, [0.0]))

        if len(missed_array) < 2 or len(noise_array) < 2:
            continue

        # Bhattacharyya coefficient for Gaussian approximation:
        # BC = exp(-1/8 * (mu1-mu2)^2 / ((s1^2+s2^2)/2)) * sqrt(2*s1*s2 / (s1^2+s2^2))
        mu_missed = np.mean(missed_array)
        mu_noise = np.mean(noise_array)
        sigma_missed = max(np.std(missed_array), 1e-10)
        sigma_noise = max(np.std(noise_array), 1e-10)

        pooled_variance = (sigma_missed ** 2 + sigma_noise ** 2) / 2.0
        exponent_term = -0.125 * (mu_missed - mu_noise) ** 2 / pooled_variance
        scale_term = math.sqrt(sigma_missed * sigma_noise / pooled_variance)
        bhattacharyya_coefficient = math.exp(exponent_term) * scale_term

        interpretation = 'HIGH' if bhattacharyya_coefficient > 0.8 else (
            'MODERATE' if bhattacharyya_coefficient > 0.5 else 'LOW'
        )
        overlap_results.append((feature_name, bhattacharyya_coefficient, interpretation))

    overlap_results.sort(key=lambda x: x[1], reverse=True)
    for feature_name, coefficient, interpretation in overlap_results:
        print(f'    {feature_name:30s} {coefficient:20.4f} {interpretation:>20s}')

    # Missed GT rank distribution
    all_missed_ranks = []
    for event in events:
        for gt_track_index, rank in enumerate(event['gt_ranks']):
            if rank >= 200:
                all_missed_ranks.append(rank)

    if all_missed_ranks:
        missed_ranks_array = np.array(all_missed_ranks)
        print(f'\n  Missed GT rank distribution:')
        print(f'    Mean rank:   {np.mean(missed_ranks_array):.0f}')
        print(f'    Median rank: {np.median(missed_ranks_array):.0f}')
        print(f'    P90 rank:    {np.percentile(missed_ranks_array, 90):.0f}')
        print(f'    Max rank:    {np.max(missed_ranks_array):.0f}')

        # How close are the missed GT to the boundary?
        near_boundary = sum(1 for r in all_missed_ranks if r < 300)
        print(f'    Within rank 200-300 (near miss): {near_boundary} ({100*near_boundary/len(all_missed_ranks):.1f}%)')
        deep_miss = sum(1 for r in all_missed_ranks if r >= 500)
        print(f'    Rank >= 500 (deep miss):         {deep_miss} ({100*deep_miss/len(all_missed_ranks):.1f}%)')


# ---------------------------------------------------------------------------
# Analysis C: Score distribution analysis
# ---------------------------------------------------------------------------

def analyze_score_distributions(events: list[dict]) -> None:
    """Analyze score distributions for different track categories."""
    print('\n' + '=' * 80)
    print('SECTION C: SCORE DISTRIBUTION ANALYSIS')
    print('=' * 80)

    found_gt_scores = []
    missed_gt_scores = []
    top_200_noise_scores = []
    bottom_noise_scores = []

    for event in events:
        gt_ranks = event['gt_ranks']
        gt_event_scores = event['gt_scores']
        background_event_scores = event['background_scores']

        for gt_track_index in range(len(gt_ranks)):
            if gt_ranks[gt_track_index] < 200:
                found_gt_scores.append(gt_event_scores[gt_track_index])
            else:
                missed_gt_scores.append(gt_event_scores[gt_track_index])

        # Split background scores into top-200-range and bottom
        if len(background_event_scores) > 0:
            sorted_background = np.sort(background_event_scores)[::-1]
            top_200_count = min(200, len(sorted_background))
            top_200_noise_scores.extend(sorted_background[:top_200_count].tolist())
            if len(sorted_background) > 200:
                bottom_noise_scores.extend(sorted_background[200:].tolist())

    found_array = np.array(found_gt_scores) if found_gt_scores else np.array([0.0])
    missed_array = np.array(missed_gt_scores) if missed_gt_scores else np.array([0.0])
    top_noise_array = np.array(top_200_noise_scores) if top_200_noise_scores else np.array([0.0])
    bottom_noise_array = np.array(bottom_noise_scores) if bottom_noise_scores else np.array([0.0])

    print(f'\n  Score statistics by category:')
    print(f'    {"Category":25s} {"Count":>8s} {"Mean":>10s} {"Std":>10s} {"Median":>10s} {"P10":>10s} {"P90":>10s}')
    print(f'    {"-"*25} {"-"*8} {"-"*10} {"-"*10} {"-"*10} {"-"*10} {"-"*10}')

    for label, array in [
        ('Found GT', found_array),
        ('Missed GT', missed_array),
        ('Top-200 noise', top_noise_array),
        ('Bottom noise', bottom_noise_array),
    ]:
        if len(array) == 0:
            continue
        print(
            f'    {label:25s} {len(array):8d} '
            f'{np.mean(array):10.4f} {np.std(array):10.4f} '
            f'{np.median(array):10.4f} '
            f'{np.percentile(array, 10):10.4f} '
            f'{np.percentile(array, 90):10.4f}'
        )

    # Score threshold analysis
    if len(found_gt_scores) > 0 and len(missed_gt_scores) > 0:
        # What score threshold separates found from missed?
        found_min = np.min(found_array)
        missed_max = np.max(missed_array)
        print(f'\n  Score boundary analysis:')
        print(f'    Lowest found-GT score:  {found_min:.4f}')
        print(f'    Highest missed-GT score: {missed_max:.4f}')
        print(f'    Gap (found_min - missed_max): {found_min - missed_max:.4f}')

        overlap_count = sum(1 for s in missed_gt_scores if s > found_min)
        print(f'    Missed GT scoring above lowest found: {overlap_count}/{len(missed_gt_scores)}')

    # d' between found GT and missed GT
    # d' = (mu1 - mu2) / sqrt(0.5 * (s1^2 + s2^2))
    if len(found_gt_scores) > 1 and len(missed_gt_scores) > 1:
        mu_found = np.mean(found_array)
        mu_missed = np.mean(missed_array)
        sigma_found = np.std(found_array)
        sigma_missed = np.std(missed_array)
        pooled_std = math.sqrt(0.5 * (sigma_found ** 2 + sigma_missed ** 2))
        d_prime = (mu_found - mu_missed) / pooled_std if pooled_std > 1e-10 else 0.0
        print(f'\n  d\' (found GT vs missed GT): {d_prime:.4f}')
        print(f'    Interpretation: ', end='')
        if d_prime < 0.5:
            print('POOR — scores barely distinguish found from missed GT')
        elif d_prime < 1.0:
            print('WEAK — some separation but heavy overlap')
        elif d_prime < 2.0:
            print('MODERATE — reasonable separation')
        else:
            print('STRONG — good separation between found and missed')

    # d' between found GT and top-200 noise
    if len(found_gt_scores) > 1 and len(top_200_noise_scores) > 1:
        mu_found = np.mean(found_array)
        mu_noise = np.mean(top_noise_array)
        sigma_found = np.std(found_array)
        sigma_noise = np.std(top_noise_array)
        pooled_std = math.sqrt(0.5 * (sigma_found ** 2 + sigma_noise ** 2))
        d_prime_noise = (mu_found - mu_noise) / pooled_std if pooled_std > 1e-10 else 0.0
        print(f'  d\' (found GT vs top-200 noise): {d_prime_noise:.4f}')

    # Top-5 false positive score analysis
    all_top_fp_scores = []
    for event in events:
        all_top_fp_scores.extend(event['top_5_fp_scores'])
    if all_top_fp_scores:
        fp_array = np.array(all_top_fp_scores)
        print(f'\n  Top-5 false positive scores (across all events):')
        print(f'    Mean:   {np.mean(fp_array):.4f}')
        print(f'    Median: {np.median(fp_array):.4f}')
        print(f'    Max:    {np.max(fp_array):.4f}')
        if len(found_gt_scores) > 0:
            print(f'    vs Found-GT mean ({np.mean(found_array):.4f}): '
                  f'gap = {np.mean(found_array) - np.mean(fp_array):.4f}')


# ---------------------------------------------------------------------------
# Analysis D: kNN neighborhood analysis
# ---------------------------------------------------------------------------

def analyze_knn_neighborhoods(events: list[dict]) -> None:
    """Analyze kNN neighborhood composition for found vs missed GT tracks."""
    print('\n' + '=' * 80)
    print('SECTION D: kNN NEIGHBORHOOD ANALYSIS')
    print('=' * 80)

    found_gt_neighbor_counts = []
    missed_gt_neighbor_counts = []
    background_neighbor_counts = []

    for event in events:
        gt_ranks = event['gt_ranks']
        gt_neighbor_counts = event['gt_neighbor_gt_counts']

        for gt_track_index in range(len(gt_ranks)):
            gt_count = gt_neighbor_counts[gt_track_index]
            if gt_ranks[gt_track_index] < 200:
                found_gt_neighbor_counts.append(gt_count)
            else:
                missed_gt_neighbor_counts.append(gt_count)

        background_neighbor_counts.extend(event['background_neighbor_gt_counts'])

    found_array = np.array(found_gt_neighbor_counts) if found_gt_neighbor_counts else np.array([0])
    missed_array = np.array(missed_gt_neighbor_counts) if missed_gt_neighbor_counts else np.array([0])
    background_array = np.array(background_neighbor_counts) if background_neighbor_counts else np.array([0])

    print(f'\n  GT neighbors in k=16 neighborhood:')
    print(f'    {"Category":25s} {"Count":>8s} {"Mean":>10s} {"Median":>10s} {"Zero%":>10s} {">=2":>10s}')
    print(f'    {"-"*25} {"-"*8} {"-"*10} {"-"*10} {"-"*10} {"-"*10}')

    for label, array in [
        ('Found GT tracks', found_array),
        ('Missed GT tracks', missed_array),
        ('Background tracks', background_array),
    ]:
        if len(array) == 0:
            continue
        zero_pct = 100 * np.sum(array == 0) / len(array)
        ge2_pct = 100 * np.sum(array >= 2) / len(array)
        print(
            f'    {label:25s} {len(array):8d} '
            f'{np.mean(array):10.2f} {np.median(array):10.1f} '
            f'{zero_pct:9.1f}% {ge2_pct:9.1f}%'
        )

    # Is having a GT neighbor predictive of being found?
    if len(found_gt_neighbor_counts) > 0 and len(missed_gt_neighbor_counts) > 0:
        found_has_gt_neighbor = sum(1 for c in found_gt_neighbor_counts if c > 0)
        missed_has_gt_neighbor = sum(1 for c in missed_gt_neighbor_counts if c > 0)
        found_rate = found_has_gt_neighbor / len(found_gt_neighbor_counts)
        missed_rate = missed_has_gt_neighbor / len(missed_gt_neighbor_counts)

        print(f'\n  Predictive value of having >= 1 GT neighbor:')
        print(f'    Found GT with GT neighbor:  {found_has_gt_neighbor}/{len(found_gt_neighbor_counts)} ({100*found_rate:.1f}%)')
        print(f'    Missed GT with GT neighbor: {missed_has_gt_neighbor}/{len(missed_gt_neighbor_counts)} ({100*missed_rate:.1f}%)')
        if missed_rate > 0:
            odds_ratio = (found_rate / (1 - found_rate + 1e-10)) / (missed_rate / (1 - missed_rate + 1e-10))
            print(f'    Odds ratio (found/missed): {odds_ratio:.2f}x')

    # Distribution of GT neighbor counts
    if len(found_gt_neighbor_counts) > 0:
        print(f'\n  Histogram of GT neighbor counts (found GT):')
        max_count = int(max(found_array)) if len(found_array) > 0 else 0
        for count_value in range(min(max_count + 1, 8)):
            num_with_count = np.sum(found_array == count_value)
            bar = '#' * int(50 * num_with_count / max(1, len(found_array)))
            print(f'    {count_value:3d}: {num_with_count:6d} ({100*num_with_count/len(found_array):5.1f}%) {bar}')
        if max_count >= 8:
            num_with_count = np.sum(found_array >= 8)
            bar = '#' * int(50 * num_with_count / max(1, len(found_array)))
            print(f'    8+:  {num_with_count:6d} ({100*num_with_count/len(found_array):5.1f}%) {bar}')

    if len(missed_gt_neighbor_counts) > 0:
        print(f'\n  Histogram of GT neighbor counts (missed GT):')
        max_count = int(max(missed_array)) if len(missed_array) > 0 else 0
        for count_value in range(min(max_count + 1, 8)):
            num_with_count = np.sum(missed_array == count_value)
            bar = '#' * int(50 * num_with_count / max(1, len(missed_array)))
            print(f'    {count_value:3d}: {num_with_count:6d} ({100*num_with_count/len(missed_array):5.1f}%) {bar}')
        if max_count >= 8:
            num_with_count = np.sum(missed_array >= 8)
            bar = '#' * int(50 * num_with_count / max(1, len(missed_array)))
            print(f'    8+:  {num_with_count:6d} ({100*num_with_count/len(missed_array):5.1f}%) {bar}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Deep diagnostics for Phase-B TrackPreFilter model',
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to best_model.pt checkpoint',
    )
    parser.add_argument(
        '--data-config', type=str,
        default='data/low-pt/lowpt_tau_trackfinder.yaml',
        help='Path to YAML data config',
    )
    parser.add_argument(
        '--val-data-dir', type=str, default='data/low-pt/val/',
        help='Directory containing validation parquet files',
    )
    parser.add_argument(
        '--val-files', type=str, nargs='+',
        default=['val_00.parquet', 'val_01.parquet'],
        help='Specific val files to use (relative to val-data-dir)',
    )
    parser.add_argument(
        '--network', type=str,
        default='networks/lowpt_tau_TrackPreFilter.py',
        help='Path to network wrapper module',
    )
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--num-workers', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f'Device: {device}')

    # ---- Data loading ----
    val_parquet_files = [
        os.path.join(args.val_data_dir, filename) for filename in args.val_files
    ]
    for filepath in val_parquet_files:
        if not os.path.exists(filepath):
            logger.error(f'Val file not found: {filepath}')
            sys.exit(1)

    logger.info(f'Val files: {val_parquet_files}')
    val_file_dict = {'data': val_parquet_files}

    val_dataset = SimpleIterDataset(
        val_file_dict,
        data_config_file=args.data_config,
        for_training=False,
        load_range_and_fraction=((0.0, 1.0), 1.0),
        fetch_by_files=True,
        fetch_step=len(val_parquet_files),
        in_memory=True,
    )
    data_config = val_dataset.config

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        drop_last=False,
        pin_memory=False,
        num_workers=args.num_workers,
    )

    # ---- Model ----
    network_module = load_network_module(args.network)
    model, model_info = network_module.get_model(data_config)

    # Load checkpoint
    logger.info(f'Loading checkpoint: {args.checkpoint}')
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(
        f'Loaded checkpoint from epoch {checkpoint.get("epoch", "?")} '
        f'(best R@200: {checkpoint.get("best_val_recall_at_200", "?")})'
    )

    model = model.to(device)
    model.eval()

    # Set temperature/sigma to end-of-training values for consistent scoring
    model.set_temperature_progress(1.0)
    model.set_drw_active(True)

    input_names = list(data_config.input_names)
    mask_input_index = input_names.index('pf_mask')
    label_input_index = input_names.index('pf_label')
    logger.info(f'Mask index: {mask_input_index} | Label index: {label_input_index}')

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model parameters: {total_params:,}')

    # ---- Collect event records ----
    logger.info(f'Running inference (batch_size={args.batch_size}, max_steps={args.max_steps})...')
    events = collect_event_records(
        model=model,
        data_loader=val_loader,
        data_config=data_config,
        device=device,
        mask_input_index=mask_input_index,
        label_input_index=label_input_index,
        max_steps=args.max_steps,
        num_neighbors=16,
    )

    if not events:
        logger.error('No events with GT tracks found!')
        sys.exit(1)

    # ---- Run all analyses ----
    print('\n' + '#' * 80)
    print('# PHASE-B TRACKPREFILTER DEEP DIAGNOSTICS')
    print(f'# Checkpoint: {args.checkpoint}')
    print(f'# Events analyzed: {len(events)}')
    print(f'# Val files: {args.val_files}')
    print('#' * 80)

    analyze_success_vs_failure(events)
    analyze_per_gt_track_failures(events)
    analyze_score_distributions(events)
    analyze_knn_neighborhoods(events)

    print('\n' + '=' * 80)
    print('DIAGNOSTICS COMPLETE')
    print('=' * 80)


if __name__ == '__main__':
    main()
