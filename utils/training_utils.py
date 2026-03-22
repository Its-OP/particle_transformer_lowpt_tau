"""Shared training utilities used by training scripts, diagnostics, and tests.

Extracted from train_trackfinder.py and pretrain_backbone.py to avoid fragile
cross-script imports. Both original scripts re-export these for backwards
compatibility.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import math
import os

import torch

logger = logging.getLogger(__name__)


def trim_to_max_valid_tracks(
    inputs: list[torch.Tensor],
    mask_input_index: int,
) -> list[torch.Tensor]:
    """Trim padded tensors to the maximum number of valid tracks in the batch.

    Weaver pads all events to a fixed sequence length (e.g. 3500) defined in
    the YAML config. Most of this is padding zeros — median track count is
    ~1130. This wastes GPU compute and, critically, corrupts BatchNorm
    statistics (the input embedding's BN1d sees ~60-80% zeros).

    This function finds the maximum number of valid tracks across the batch
    using pf_mask, then slices all input tensors to that length. Since FPS,
    kNN, and EdgeConv operate on variable-length point clouds, no architecture
    changes are needed.

    Args:
        inputs: List of input tensors, each (B, C_i, P) where P is the padded
            sequence length. Order follows data_config.input_names.
        mask_input_index: Index of the pf_mask tensor in the inputs list.

    Returns:
        List of trimmed tensors, each (B, C_i, P_trimmed) where
        P_trimmed = max valid tracks in the batch.
    """
    mask = inputs[mask_input_index]  # (B, 1, P)

    # Sum over the sequence dimension to count valid tracks per event,
    # then take the batch maximum. This is the tightest trim that
    # preserves all real data in the batch.
    max_valid_tracks = int(mask.sum(dim=2).max().item())

    # Safety: ensure at least 1 track (handles empty-event edge case)
    max_valid_tracks = max(1, max_valid_tracks)

    # Round up to the nearest multiple of 128 to reduce the number of
    # distinct tensor shapes. torch.compile with dynamic=True recompiles
    # for each new shape; bucketing avoids this by limiting to ~22 possible
    # sizes (128, 256, ..., 2816) instead of thousands of unique values.
    bucket_size = 128
    max_valid_tracks = min(
        ((max_valid_tracks + bucket_size - 1) // bucket_size) * bucket_size,
        inputs[0].shape[2],  # don't exceed original padded length
    )

    return [tensor[:, :, :max_valid_tracks] for tensor in inputs]


def load_network_module(network_path: str):
    """Load get_model() from the network wrapper file.

    Args:
        network_path: Path to the network wrapper Python file.

    Returns:
        Module with get_model() function.
    """
    spec = importlib.util.spec_from_file_location('network', network_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_label_from_inputs(
    inputs: list[torch.Tensor],
    label_input_index: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Extract pf_label from inputs list and return remaining inputs + label.

    The data config loads track_label_from_tau as an input group (pf_label)
    to work around weaver's lack of native per-track label support. This
    function separates it from the model inputs before forward pass.

    Args:
        inputs: List of input tensors including pf_label.
        label_input_index: Index of pf_label in the inputs list.

    Returns:
        Tuple of (model_inputs, track_labels) where model_inputs has
        pf_label removed and track_labels is (B, 1, P).
    """
    track_labels = inputs[label_input_index]  # (B, 1, P)
    model_inputs = [
        tensor for index, tensor in enumerate(inputs)
        if index != label_input_index
    ]
    return model_inputs, track_labels


def extract_per_track_scores(
    output_dict: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Extract per-track ranking scores from any head's inference output.

    Supports all heads:
        - DETR hybrid: output_dict['per_track_logits'] -> (B, P) direct scores
        - OC: output_dict['beta_scores'] -> (B, P) direct scores
        - DETR query-only: output_dict['mask_logits'] -> max over queries -> (B, P)

    Args:
        output_dict: Model inference output dict.

    Returns:
        per_track_scores: (B, P) scores for ranking tracks (higher = more likely tau).
    """
    if 'per_track_logits' in output_dict:
        return output_dict['per_track_logits']
    elif 'beta_scores' in output_dict:
        return output_dict['beta_scores']
    elif 'mask_logits' in output_dict:
        return output_dict['mask_logits'].max(dim=1).values
    else:
        raise KeyError(
            f'Cannot extract per-track scores from output keys: '
            f'{list(output_dict.keys())}.',
        )


@torch.no_grad()
def compute_recall_at_k_metrics(
    per_track_scores: torch.Tensor,
    track_labels: torch.Tensor,
    mask: torch.Tensor,
    k_values: tuple[int, ...] = (10, 20, 30, 100),
) -> dict[str, float]:
    """Compute recall@K, d-prime, and median GT rank for track finding.

    Tracks are ranked by score (descending). For each K, recall@K is
    the fraction of GT pion tracks found in the top-K predictions.

    Additional metrics:
        - d_prime: separation between GT and background score distributions,
          d' = (mu_gt - mu_bg) / sqrt(0.5 * (sigma_gt^2 + sigma_bg^2)).
          Higher = better separation.
        - median_gt_rank: median rank of GT pions in the sorted score list.
          Lower = better (0 = top-ranked).

    Args:
        per_track_scores: (B, P) per-track ranking scores.
        track_labels: (B, 1, P) binary labels (1.0 = tau pion).
        mask: (B, 1, P) boolean mask (True = valid track).
        k_values: Tuple of K values for recall@K (default: 10, 20, 30, 100).

    Returns:
        Dict with recall_at_K for each K, d_prime, median_gt_rank,
        and total_gt_tracks.
    """
    batch_size = per_track_scores.shape[0]
    labels_flat = track_labels.squeeze(1) * mask.squeeze(1).float()
    valid_mask = mask.squeeze(1).bool()

    masked_scores = per_track_scores.clone()
    masked_scores[~valid_mask] = float('-inf')
    sorted_indices = masked_scores.argsort(dim=1, descending=True)

    # Build rank lookup vectorized: rank_of[i] = position of track i
    # argsort(argsort(x)) gives the rank of each element
    rank_lookup = torch.argsort(
        torch.argsort(masked_scores, dim=1, descending=True), dim=1,
    )

    recall_sums = {k: 0.0 for k in k_values}
    perfect_event_counts = {k: 0 for k in k_values}
    total_events_with_gt = 0
    total_gt_tracks = 0

    # Collect scores and ranks for d-prime and median rank
    all_gt_scores = []
    all_background_scores = []
    all_gt_ranks = []

    # Per-event breakdown at K=200: count events by (found, total_gt)
    breakdown_k = 200
    compute_breakdown = breakdown_k in k_values
    event_breakdown_counts: dict[str, int] = {}

    for batch_index in range(batch_size):
        gt_positions = labels_flat[batch_index].nonzero(as_tuple=True)[0]
        num_gt = len(gt_positions)

        # Collect scores for d-prime
        event_valid = valid_mask[batch_index]
        event_labels = labels_flat[batch_index]
        event_scores = per_track_scores[batch_index]

        gt_mask = (event_labels == 1.0) & event_valid
        background_mask = (event_labels == 0.0) & event_valid

        if gt_mask.any():
            all_gt_scores.append(event_scores[gt_mask])
        if background_mask.any():
            all_background_scores.append(event_scores[background_mask])

        if num_gt == 0:
            continue

        total_events_with_gt += 1
        total_gt_tracks += num_gt

        # Recall@K: use torch.isin instead of converting to Python sets
        found_at_breakdown_k = 0
        for k in k_values:
            top_k_indices = sorted_indices[batch_index, :k]
            found = torch.isin(gt_positions, top_k_indices).sum().item()
            recall_sums[k] += found / num_gt
            if found == num_gt:
                perfect_event_counts[k] += 1
            if k == breakdown_k:
                found_at_breakdown_k = found

        # Per-event breakdown at K=200
        if compute_breakdown:
            breakdown_key = f'found_{found_at_breakdown_k}_of_{num_gt}'
            event_breakdown_counts[breakdown_key] = (
                event_breakdown_counts.get(breakdown_key, 0) + 1
            )

        # GT pion ranks: batch gather, single CPU transfer
        event_gt_ranks = rank_lookup[batch_index, gt_positions]
        all_gt_ranks.extend(event_gt_ranks.cpu().tolist())

    metrics = {}
    for k in k_values:
        metrics[f'recall_at_{k}'] = recall_sums[k] / max(1, total_events_with_gt)
        metrics[f'perfect_at_{k}'] = perfect_event_counts[k] / max(1, total_events_with_gt)
    metrics['total_gt_tracks'] = total_gt_tracks
    metrics['total_events_with_gt'] = total_events_with_gt

    # d-prime: score separation between GT and background
    # d' = (mu_gt - mu_bg) / sqrt(0.5 * (sigma_gt^2 + sigma_bg^2))
    if all_gt_scores and all_background_scores:
        gt_scores_cat = torch.cat(all_gt_scores)
        background_scores_cat = torch.cat(all_background_scores)
        mu_gt = gt_scores_cat.mean().item()
        mu_background = background_scores_cat.mean().item()
        sigma_gt = gt_scores_cat.std().item()
        sigma_background = background_scores_cat.std().item()
        pooled_std = (0.5 * (sigma_gt ** 2 + sigma_background ** 2)) ** 0.5
        metrics['d_prime'] = (
            (mu_gt - mu_background) / pooled_std if pooled_std > 1e-10 else 0.0
        )
    else:
        metrics['d_prime'] = 0.0

    # GT rank statistics: median + percentiles
    if all_gt_ranks:
        sorted_ranks = sorted(all_gt_ranks)
        num_ranks = len(sorted_ranks)
        midpoint = num_ranks // 2
        if num_ranks % 2 == 0:
            metrics['median_gt_rank'] = (
                sorted_ranks[midpoint - 1] + sorted_ranks[midpoint]
            ) / 2.0
        else:
            metrics['median_gt_rank'] = float(sorted_ranks[midpoint])

        # Percentiles: p75, p90, p95
        # p-th percentile = value at index ceil(p/100 * n) - 1
        for percentile in (75, 90, 95):
            index = min(
                int(math.ceil(percentile / 100.0 * num_ranks)) - 1,
                num_ranks - 1,
            )
            metrics[f'gt_rank_p{percentile}'] = float(sorted_ranks[index])
    else:
        metrics['median_gt_rank'] = float('inf')
        for percentile in (75, 90, 95):
            metrics[f'gt_rank_p{percentile}'] = float('inf')

    # Per-event breakdown at K=200: normalize to fractions
    if compute_breakdown:
        for key, count in event_breakdown_counts.items():
            metrics[f'{key}_at_{breakdown_k}'] = (
                count / max(1, total_events_with_gt)
            )

    return metrics


class MetricsAccumulator:
    """Accumulates raw per-event data across batches for global metric computation.

    Percentiles and per-event breakdowns cannot be averaged across batches.
    This class collects raw GT ranks, per-event recall counts, and score
    statistics, then computes final metrics from the full distribution.

    Usage:
        accumulator = MetricsAccumulator(k_values=(10, 200, 500, 600))
        for batch in val_loader:
            scores, labels, mask = ...
            accumulator.update(scores, labels, mask)
        metrics = accumulator.compute()

    Args:
        k_values: Tuple of K values for recall@K computation.
    """

    def __init__(self, k_values: tuple[int, ...] = (10, 20, 30, 100, 200)):
        self.k_values = k_values

        # Accumulators for global computation
        self.all_gt_ranks: list[int] = []
        self.all_gt_scores: list[torch.Tensor] = []
        self.all_background_scores: list[torch.Tensor] = []

        # Per-K accumulators: recall sums and perfect event counts
        self.recall_sums: dict[int, float] = {k: 0.0 for k in k_values}
        self.perfect_event_counts: dict[int, int] = {k: 0 for k in k_values}

        # Per-event breakdown at K=200
        self.breakdown_k = 200
        self.compute_breakdown = self.breakdown_k in k_values
        self.event_breakdown_counts: dict[str, int] = {}

        # Counters
        self.total_events_with_gt = 0
        self.total_gt_tracks = 0

    @torch.no_grad()
    def update(
        self,
        per_track_scores: torch.Tensor,
        track_labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """Accumulate data from a single batch.

        Args:
            per_track_scores: (B, P) per-track ranking scores.
            track_labels: (B, 1, P) binary labels (1.0 = tau pion).
            mask: (B, 1, P) boolean mask (True = valid track).
        """
        batch_size = per_track_scores.shape[0]
        labels_flat = track_labels.squeeze(1) * mask.squeeze(1).float()
        valid_mask = mask.squeeze(1).bool()

        masked_scores = per_track_scores.clone()
        masked_scores[~valid_mask] = float('-inf')
        sorted_indices = masked_scores.argsort(dim=1, descending=True)

        # Build rank lookup: rank_of[i] = position of track i in sorted order
        # argsort(argsort(x, descending), descending=False) gives the rank
        rank_lookup = torch.argsort(
            torch.argsort(masked_scores, dim=1, descending=True), dim=1,
        )

        for batch_index in range(batch_size):
            gt_positions = labels_flat[batch_index].nonzero(as_tuple=True)[0]
            num_gt = len(gt_positions)

            # Collect scores for d-prime
            event_valid = valid_mask[batch_index]
            event_labels = labels_flat[batch_index]
            event_scores = per_track_scores[batch_index]

            gt_mask = (event_labels == 1.0) & event_valid
            background_mask = (event_labels == 0.0) & event_valid

            if gt_mask.any():
                self.all_gt_scores.append(event_scores[gt_mask].cpu())
            if background_mask.any():
                self.all_background_scores.append(
                    event_scores[background_mask].cpu(),
                )

            if num_gt == 0:
                continue

            self.total_events_with_gt += 1
            self.total_gt_tracks += num_gt

            # Recall@K
            found_at_breakdown_k = 0
            for k in self.k_values:
                top_k_indices = sorted_indices[batch_index, :k]
                found = torch.isin(gt_positions, top_k_indices).sum().item()
                self.recall_sums[k] += found / num_gt
                if found == num_gt:
                    self.perfect_event_counts[k] += 1
                if k == self.breakdown_k:
                    found_at_breakdown_k = found

            # Per-event breakdown at K=200
            if self.compute_breakdown:
                breakdown_key = f'found_{found_at_breakdown_k}_of_{num_gt}'
                self.event_breakdown_counts[breakdown_key] = (
                    self.event_breakdown_counts.get(breakdown_key, 0) + 1
                )

            # Collect raw GT ranks for global percentile computation
            event_gt_ranks = rank_lookup[batch_index, gt_positions]
            self.all_gt_ranks.extend(event_gt_ranks.cpu().tolist())

    def compute(self) -> dict[str, float]:
        """Compute final metrics from all accumulated data.

        Returns:
            Dict with recall_at_K, perfect_at_K, d_prime, median_gt_rank,
            gt_rank percentiles, per-event breakdown, and counts.
        """
        num_events = max(1, self.total_events_with_gt)
        metrics = {}

        # R@K and P@K
        for k in self.k_values:
            metrics[f'recall_at_{k}'] = self.recall_sums[k] / num_events
            metrics[f'perfect_at_{k}'] = (
                self.perfect_event_counts[k] / num_events
            )

        metrics['total_gt_tracks'] = self.total_gt_tracks
        metrics['total_events_with_gt'] = self.total_events_with_gt

        # d-prime: score separation between GT and background
        # d' = (mu_gt - mu_bg) / sqrt(0.5 * (sigma_gt^2 + sigma_bg^2))
        if self.all_gt_scores and self.all_background_scores:
            gt_scores_cat = torch.cat(self.all_gt_scores)
            background_scores_cat = torch.cat(self.all_background_scores)
            mu_gt = gt_scores_cat.mean().item()
            mu_background = background_scores_cat.mean().item()
            sigma_gt = gt_scores_cat.std().item()
            sigma_background = background_scores_cat.std().item()
            pooled_std = (
                0.5 * (sigma_gt ** 2 + sigma_background ** 2)
            ) ** 0.5
            metrics['d_prime'] = (
                (mu_gt - mu_background) / pooled_std
                if pooled_std > 1e-10
                else 0.0
            )
        else:
            metrics['d_prime'] = 0.0

        # GT rank statistics: median + percentiles (GLOBAL, not per-batch)
        if self.all_gt_ranks:
            sorted_ranks = sorted(self.all_gt_ranks)
            num_ranks = len(sorted_ranks)
            midpoint = num_ranks // 2
            if num_ranks % 2 == 0:
                metrics['median_gt_rank'] = (
                    sorted_ranks[midpoint - 1] + sorted_ranks[midpoint]
                ) / 2.0
            else:
                metrics['median_gt_rank'] = float(sorted_ranks[midpoint])

            # Percentiles: p75, p90, p95
            # p-th percentile = value at index ceil(p/100 * n) - 1
            for percentile in (75, 90, 95):
                index = min(
                    int(math.ceil(percentile / 100.0 * num_ranks)) - 1,
                    num_ranks - 1,
                )
                metrics[f'gt_rank_p{percentile}'] = float(
                    sorted_ranks[index],
                )
        else:
            metrics['median_gt_rank'] = float('inf')
            for percentile in (75, 90, 95):
                metrics[f'gt_rank_p{percentile}'] = float('inf')

        # Per-event breakdown at K=200: normalize to fractions
        if self.compute_breakdown:
            for key, count in self.event_breakdown_counts.items():
                metrics[f'{key}_at_{self.breakdown_k}'] = count / num_events

        return metrics


@torch.no_grad()
def compute_conditional_recall(
    per_track_scores: torch.Tensor,
    track_labels: torch.Tensor,
    mask: torch.Tensor,
    raw_features: torch.Tensor,
    feature_index_pt: int = 0,
    feature_index_dxy_significance: int = 6,
    top_k: int = 200,
) -> dict[str, float]:
    """Compute recall@K conditioned on pT and |dxy_significance| bins.

    For each GT pion track, checks whether it lands in the top-K.
    Reports found rate per pT bin, per |dxy_sig| bin, and a 2D grid.

    Args:
        per_track_scores: (B, P) per-track ranking scores.
        track_labels: (B, 1, P) binary labels (1.0 = tau pion).
        mask: (B, 1, P) boolean mask (True = valid track).
        raw_features: (B, C, P) raw features (before standardization).
        feature_index_pt: Index of pT in feature channels (default: 0).
        feature_index_dxy_significance: Index of dxy_significance (default: 6).
        top_k: K value for recall computation (default: 200).

    Returns:
        Dict with recall_pt_{bin}, recall_dxy_{bin}, recall_2d_pt{i}_dxy{j},
        and corresponding count_* entries (70 metrics total).
    """
    batch_size = per_track_scores.shape[0]
    labels_flat = track_labels.squeeze(1) * mask.squeeze(1).float()
    valid_mask = mask.squeeze(1).bool()

    masked_scores = per_track_scores.clone()
    masked_scores[~valid_mask] = float('-inf')
    sorted_indices = masked_scores.argsort(dim=1, descending=True)

    # Bin edges for pT (GeV) and |dxy_significance|
    pt_bin_edges = [0.0, 0.3, 0.5, 1.0, 2.0, float('inf')]
    dxy_bin_edges = [0.0, 0.5, 1.0, 2.0, 5.0, float('inf')]
    num_pt_bins = len(pt_bin_edges) - 1
    num_dxy_bins = len(dxy_bin_edges) - 1

    # Accumulators: [found, total] per bin
    pt_counts = [[0, 0] for _ in range(num_pt_bins)]
    dxy_counts = [[0, 0] for _ in range(num_dxy_bins)]
    grid_counts = [
        [[0, 0] for _ in range(num_dxy_bins)]
        for _ in range(num_pt_bins)
    ]

    for batch_index in range(batch_size):
        gt_positions = labels_flat[batch_index].nonzero(as_tuple=True)[0]
        if len(gt_positions) == 0:
            continue

        top_k_indices = sorted_indices[batch_index, :top_k]
        top_k_set = set(top_k_indices.cpu().tolist())
        features_event = raw_features[batch_index]  # (C, P)

        for gt_pos in gt_positions.cpu().tolist():
            pt_value = features_event[feature_index_pt, gt_pos].item()
            dxy_value = abs(
                features_event[feature_index_dxy_significance, gt_pos].item(),
            )
            found = 1 if gt_pos in top_k_set else 0

            # Find pT bin
            pt_bin = num_pt_bins - 1
            for bin_index in range(num_pt_bins):
                if pt_value < pt_bin_edges[bin_index + 1]:
                    pt_bin = bin_index
                    break

            # Find |dxy_sig| bin
            dxy_bin = num_dxy_bins - 1
            for bin_index in range(num_dxy_bins):
                if dxy_value < dxy_bin_edges[bin_index + 1]:
                    dxy_bin = bin_index
                    break

            pt_counts[pt_bin][0] += found
            pt_counts[pt_bin][1] += 1
            dxy_counts[dxy_bin][0] += found
            dxy_counts[dxy_bin][1] += 1
            grid_counts[pt_bin][dxy_bin][0] += found
            grid_counts[pt_bin][dxy_bin][1] += 1

    metrics = {}
    pt_labels = ['0_0.3', '0.3_0.5', '0.5_1', '1_2', '2+']
    for bin_index, label in enumerate(pt_labels):
        found, total = pt_counts[bin_index]
        metrics[f'recall_pt_{label}'] = found / max(1, total)
        metrics[f'count_pt_{label}'] = total

    dxy_labels = ['0_0.5', '0.5_1', '1_2', '2_5', '5+']
    for bin_index, label in enumerate(dxy_labels):
        found, total = dxy_counts[bin_index]
        metrics[f'recall_dxy_{label}'] = found / max(1, total)
        metrics[f'count_dxy_{label}'] = total

    for pt_index, pt_label in enumerate(pt_labels):
        for dxy_index, dxy_label in enumerate(dxy_labels):
            found, total = grid_counts[pt_index][dxy_index]
            metrics[f'recall_2d_pt{pt_label}_dxy{dxy_label}'] = (
                found / max(1, total)
            )
            metrics[f'count_2d_pt{pt_label}_dxy{dxy_label}'] = total

    return metrics


def save_epoch_metrics(
    metrics: dict[str, float | int],
    experiment_directory: str,
    epoch: int,
) -> str:
    """Save all metrics for an epoch as JSON for cross-experiment comparison.

    Args:
        metrics: Flat dict of metric name -> value.
        experiment_directory: Path to experiment directory.
        epoch: Current epoch number.

    Returns:
        Path to the saved JSON file.
    """
    metrics_directory = os.path.join(experiment_directory, 'metrics')
    os.makedirs(metrics_directory, exist_ok=True)
    filepath = os.path.join(metrics_directory, f'epoch_{epoch}.json')
    with open(filepath, 'w') as file_handle:
        json.dump(metrics, file_handle, indent=2, default=float)
    return filepath


class CheckpointManager:
    """Manages rolling top-K best checkpoints to limit disk usage.

    Tracks saved checkpoint files ranked by a task metric and deletes
    those that fall outside the top K. The special ``best_model.pt`` file
    is always maintained as a copy of the rank-1 checkpoint.

    Args:
        checkpoints_directory: Path to the checkpoints directory.
        keep_best_k: Maximum number of best checkpoints to retain.
            When a new checkpoint is saved and the count exceeds this limit,
            the checkpoint with the worst metric value is deleted.
            Set to 0 to disable cleanup (keep all checkpoints).
        criterion_mode: 'max' if higher metric is better (e.g. R@200),
            'min' if lower is better (e.g. val loss). Defaults to 'max'.
        criterion_name: Display name for the criterion in log messages.
    """

    def __init__(
        self,
        checkpoints_directory: str,
        keep_best_k: int = 5,
        criterion_mode: str = 'max',
        criterion_name: str = 'R@200',
    ):
        self.checkpoints_directory = checkpoints_directory
        self.keep_best_k = keep_best_k
        self.criterion_mode = criterion_mode
        self.criterion_name = criterion_name
        # Sorted list of (criterion_value, epoch, filepath) — best first
        self.tracked_checkpoints: list[tuple[float, int, str]] = []

    def save_checkpoint(
        self,
        checkpoint_data: dict,
        epoch: int,
        criterion_value: float,
        is_best: bool,
    ) -> str:
        """Save a checkpoint and prune old ones if exceeding keep_best_k.

        Always saves ``checkpoint_epoch_{epoch}.pt``. If ``is_best``, also
        saves/overwrites ``best_model.pt``. Then prunes the tracked list
        so only the top-K checkpoints (by criterion_value) remain on disk.

        Args:
            checkpoint_data: Dict containing model_state_dict, optimizer, etc.
            epoch: Current epoch number.
            criterion_value: Value of the selection metric (e.g. R@200).
            is_best: Whether this is a new overall best.

        Returns:
            Path to the saved checkpoint file.
        """
        checkpoint_path = os.path.join(
            self.checkpoints_directory, f'checkpoint_epoch_{epoch}.pt',
        )
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f'Saved checkpoint: {checkpoint_path}')

        # Track this checkpoint for pruning
        self.tracked_checkpoints.append(
            (criterion_value, epoch, checkpoint_path),
        )
        # Sort: best first. For 'max' mode, descending; for 'min', ascending.
        reverse = self.criterion_mode == 'max'
        self.tracked_checkpoints.sort(
            key=lambda entry: entry[0], reverse=reverse,
        )

        # Save best_model.pt as a copy of the overall best
        if is_best:
            best_path = os.path.join(
                self.checkpoints_directory, 'best_model.pt',
            )
            torch.save(checkpoint_data, best_path)
            logger.info(
                f'New best model '
                f'({self.criterion_name}={criterion_value:.5f})',
            )

        # Prune checkpoints beyond the top K
        self._prune_checkpoints()

        return checkpoint_path

    def _prune_checkpoints(self):
        """Delete tracked checkpoints that fall outside the top-K best.

        Skips pruning if keep_best_k is 0 (unlimited) or if the number
        of tracked checkpoints does not exceed the limit. Never deletes
        ``best_model.pt`` (it's not tracked in the list).
        """
        if self.keep_best_k <= 0:
            return

        while len(self.tracked_checkpoints) > self.keep_best_k:
            # Remove the worst (last in sorted list, since best are first)
            worst_value, worst_epoch, worst_path = (
                self.tracked_checkpoints.pop()
            )
            if os.path.exists(worst_path):
                os.remove(worst_path)
                logger.info(
                    f'Pruned checkpoint: epoch {worst_epoch} '
                    f'({self.criterion_name}={worst_value:.5f})',
                )
