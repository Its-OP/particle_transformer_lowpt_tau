"""Shared training utilities used by training scripts, diagnostics, and tests.

Extracted from train_trackfinder.py and pretrain_backbone.py to avoid fragile
cross-script imports. Both original scripts re-export these for backwards
compatibility.
"""
from __future__ import annotations

import importlib.util
import logging
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
        for k in k_values:
            top_k_indices = sorted_indices[batch_index, :k]
            found = torch.isin(gt_positions, top_k_indices).sum().item()
            recall_sums[k] += found / num_gt
            if found == num_gt:
                perfect_event_counts[k] += 1

        # GT pion ranks: batch gather, single CPU transfer
        event_gt_ranks = rank_lookup[batch_index, gt_positions]
        all_gt_ranks.extend(event_gt_ranks.cpu().tolist())

    metrics = {}
    for k in k_values:
        metrics[f'recall_at_{k}'] = recall_sums[k] / max(1, total_events_with_gt)
        metrics[f'perfect_at_{k}'] = perfect_event_counts[k] / max(1, total_events_with_gt)
    metrics['total_gt_tracks'] = total_gt_tracks

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

    # Median GT rank
    if all_gt_ranks:
        sorted_ranks = sorted(all_gt_ranks)
        midpoint = len(sorted_ranks) // 2
        if len(sorted_ranks) % 2 == 0:
            metrics['median_gt_rank'] = (
                sorted_ranks[midpoint - 1] + sorted_ranks[midpoint]
            ) / 2.0
        else:
            metrics['median_gt_rank'] = float(sorted_ranks[midpoint])
    else:
        metrics['median_gt_rank'] = float('inf')

    return metrics


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
