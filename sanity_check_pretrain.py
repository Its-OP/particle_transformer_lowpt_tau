"""Sanity checks for masked track reconstruction pretraining.

Loads a trained MaskedTrackPretrainer checkpoint and runs diagnostic tests
to verify the model genuinely reconstructs masked tracks rather than
exploiting shortcuts in the loss function.

Tests:
    1. Eta-phi visualization: plot masked GT vs reconstructed tracks with
       arrows showing Hungarian-matched correspondences.
    2. Per-feature scatter: predicted vs ground-truth for each of the 7
       features, with R² correlation.
    3. Baseline comparison: model loss vs zero/mean-prediction baselines.
    4. Mask-ratio sensitivity: Hungarian-matched loss vs random-assignment
       loss at different mask ratios (10%-90%). Diagnoses whether loss
       variation is a genuine model quality change or a combinatorial
       matching artifact (optimal assignment exploits more lucky pairs
       at higher track counts).
    5. Prediction diversity: pairwise cosine similarity among predictions
       vs ground truth. Detects mode collapse.

Usage:
    python sanity_check_pretrain.py \\
        --data-config data/low-pt/lowpt_tau_pretrain.yaml \\
        --data-dir data/low-pt/ \\
        --network networks/lowpt_tau_BackbonePretrain.py \\
        --checkpoint experiments/cosine_scheduler_0.089/checkpoints/best_model.pt \\
        --device cpu \\
        --output-dir experiments/cosine_scheduler_0.089/sanity_check
"""
import argparse
import importlib.util
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from weaver.utils.dataset import SimpleIterDataset
from weaver.nn.model.hungarian_matcher import hungarian_matcher

# Feature names in the order they appear in pf_features (indices 0-6)
FEATURE_NAMES = [
    'track_px', 'track_py', 'track_pz',
    'track_eta', 'track_phi',
    'track_charge', 'track_dxy_significance',
]
ETA_FEATURE_INDEX = 3
PHI_FEATURE_INDEX = 4


def load_network_module(network_path: str):
    """Load get_model() from the network wrapper Python file."""
    spec = importlib.util.spec_from_file_location('network', network_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unstandardize(
    standardized_values: torch.Tensor,
    variable_name: str,
    data_config,
) -> torch.Tensor:
    """Convert standardized feature values back to raw physical units.

    Weaver's auto-standardization applies:
        standardized = (raw - center) * scale
    where center = median, scale = 1 / IQR.

    Inverse: raw = standardized / scale + center

    Args:
        standardized_values: Tensor of standardized values.
        variable_name: Name of the variable in data_config.preprocess_params
            (e.g. 'track_eta').
        data_config: Weaver DataConfig with populated preprocess_params.

    Returns:
        Tensor of raw (un-standardized) values.
    """
    params = data_config.preprocess_params[variable_name]
    center = params['center']
    scale = params['scale']

    if center is None or scale is None:
        # Variable was not standardized (e.g. pf_mask)
        return standardized_values

    # raw = standardized / scale + center
    return standardized_values / scale + center


def trim_to_max_valid_tracks(
    inputs: list[torch.Tensor],
    mask_input_index: int,
) -> list[torch.Tensor]:
    """Trim padded tensors to the maximum number of valid tracks in the batch.

    Replicates the logic from pretrain_backbone.py to remove ~60-80% zero
    padding that wastes compute and corrupts BatchNorm statistics.

    Args:
        inputs: List of input tensors, each (B, C_i, P).
        mask_input_index: Index of the pf_mask tensor in the inputs list.

    Returns:
        List of trimmed tensors, each (B, C_i, P_trimmed).
    """
    mask = inputs[mask_input_index]  # (B, 1, P)
    max_valid_tracks = int(mask.sum(dim=2).max().item())
    max_valid_tracks = max(1, max_valid_tracks)
    return [tensor[:, :, :max_valid_tracks] for tensor in inputs]


def run_forward_pass(
    model,
    points: torch.Tensor,
    features: torch.Tensor,
    lorentz_vectors: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    """Run the pretraining forward pass step-by-step, returning intermediates.

    Follows the same sequence as MaskedTrackPretrainer.forward() but exposes
    all intermediate tensors for diagnostic visualization.

    Args:
        model: MaskedTrackPretrainer instance (eval mode).
        points: (B, 2, P) coordinates in (η, φ).
        features: (B, input_dim, P) standardized track features.
        lorentz_vectors: (B, 4, P) raw 4-vectors.
        mask: (B, 1, P) boolean validity mask.

    Returns:
        Dictionary with all intermediate tensors:
            - visible_mask, masked_mask: (B, 1, P) boolean masks
            - masked_ground_truth: (B, F, max_masked) GT features
            - masked_validity: (B, 1, max_masked) validity mask for GT
            - predicted_features: (B, F, max_masked) decoder predictions
            - matching_indices: (B, 2, K) Hungarian matching
            - points, features, mask: original inputs
            - num_masked_per_event: (B,) count of masked tracks
    """
    with torch.no_grad():
        # Step 1: Enrich all tracks
        enriched_features = model.backbone.enrich(
            points, features, lorentz_vectors, mask,
        )

        # Step 2: Create masking split
        visible_mask, masked_mask = model._create_random_mask(mask)

        num_visible_per_event = visible_mask.squeeze(1).sum(dim=1)
        num_masked_per_event = masked_mask.squeeze(1).sum(dim=1)
        max_visible = int(num_visible_per_event.max().item())
        max_masked = int(num_masked_per_event.max().item())

        # Step 3: Gather visible tracks (dense packing for compaction)
        visible_enriched, visible_validity = model._gather_tracks(
            enriched_features, visible_mask, max_visible,
        )
        visible_coordinates, _ = model._gather_tracks(
            points, visible_mask, max_visible,
        )

        # Step 4: Compact visible tracks → backbone tokens
        backbone_tokens, _ = model.backbone.compact(
            visible_coordinates, visible_enriched, visible_validity,
        )

        # Step 5: Gather ground truth for masked tracks
        masked_ground_truth, masked_validity = model._gather_tracks(
            features, masked_mask, max_masked,
        )

        # Also gather raw (η, φ) coordinates for masked tracks (for plotting)
        masked_raw_points, _ = model._gather_tracks(
            points, masked_mask, max_masked,
        )

        # Step 6: Decode
        predicted_features = model.decoder(backbone_tokens, max_masked)

        # Step 7: Compute cost matrix for Hungarian matching
        # cost[b, i, j] = ||pred[b, :, i] - true[b, :, j]||²
        # Using expansion: ||a-b||² = ||a||² + ||b||² - 2·a·b
        predicted_transposed = predicted_features.transpose(1, 2)  # (B, K, F)
        true_transposed = masked_ground_truth.transpose(1, 2)  # (B, K, F)

        predicted_squared = (predicted_transposed ** 2).sum(dim=2)  # (B, K)
        true_squared = (true_transposed ** 2).sum(dim=2)  # (B, K)
        cross_term = torch.bmm(
            predicted_transposed, true_transposed.transpose(1, 2),
        )  # (B, K, K)
        cost_matrix = (
            predicted_squared.unsqueeze(2)
            + true_squared.unsqueeze(1)
            - 2 * cross_term
        )  # (B, K, K)

        # Mask out invalid slots
        validity_flat = masked_validity.squeeze(1)  # (B, K)
        large_cost = 1e6
        cost_matrix = cost_matrix.masked_fill(
            ~validity_flat.unsqueeze(2), large_cost,
        )
        cost_matrix = cost_matrix.masked_fill(
            ~validity_flat.unsqueeze(1), large_cost,
        )

        # Hungarian matching
        matching_indices = hungarian_matcher(cost_matrix)

    return {
        'visible_mask': visible_mask,
        'masked_mask': masked_mask,
        'masked_ground_truth': masked_ground_truth,
        'masked_validity': masked_validity,
        'masked_raw_points': masked_raw_points,
        'predicted_features': predicted_features,
        'matching_indices': matching_indices,
        'points': points,
        'features': features,
        'mask': mask,
        'num_masked_per_event': num_masked_per_event,
    }


def plot_eta_phi(
    forward_results: dict,
    data_config,
    output_directory: str,
    num_events: int = 4,
):
    """Plot masked ground-truth vs reconstructed tracks in η-φ space.

    For each event, shows:
        - Blue dots: masked ground-truth track positions
        - Red dots: reconstructed track positions (un-standardized)
        - Thin gray arrows: from reconstruction to matched ground truth

    Only masked tracks are shown (no visible tracks) to avoid clutter.

    Args:
        forward_results: Output of run_forward_pass().
        data_config: Weaver DataConfig for un-standardization parameters.
        output_directory: Directory to save the plot.
        num_events: Number of events to display (arranged in 2-column grid).
    """
    masked_ground_truth = forward_results['masked_ground_truth']
    predicted_features = forward_results['predicted_features']
    masked_validity = forward_results['masked_validity']
    matching_indices = forward_results['matching_indices']
    masked_raw_points = forward_results['masked_raw_points']

    batch_size = masked_ground_truth.shape[0]
    num_events = min(num_events, batch_size)
    num_columns = 2
    num_rows = (num_events + num_columns - 1) // num_columns

    figure, axes = plt.subplots(
        num_rows, num_columns, figsize=(16, 8 * num_rows),
        squeeze=False,
    )

    for event_index in range(num_events):
        row = event_index // num_columns
        column = event_index % num_columns
        axis = axes[row, column]

        validity = masked_validity[event_index, 0].bool()  # (K,)
        num_valid = validity.sum().item()

        if num_valid == 0:
            axis.set_title(f'Event {event_index}: no masked tracks')
            continue

        # Ground truth: raw (η, φ) from pf_points (already un-standardized)
        ground_truth_eta = masked_raw_points[event_index, 0, :num_valid].cpu().numpy()
        ground_truth_phi = masked_raw_points[event_index, 1, :num_valid].cpu().numpy()

        # Predicted: standardized η, φ → un-standardize to raw coordinates
        predicted_eta_standardized = predicted_features[event_index, ETA_FEATURE_INDEX, :]
        predicted_phi_standardized = predicted_features[event_index, PHI_FEATURE_INDEX, :]

        predicted_eta_raw = unstandardize(
            predicted_eta_standardized, 'track_eta', data_config,
        ).cpu().numpy()
        predicted_phi_raw = unstandardize(
            predicted_phi_standardized, 'track_phi', data_config,
        ).cpu().numpy()

        # Hungarian matching: reorder predictions to match GT
        matched_prediction_indices = matching_indices[event_index, 0, :num_valid].long()
        matched_ground_truth_indices = matching_indices[event_index, 1, :num_valid].long()

        matched_predicted_eta = predicted_eta_raw[matched_prediction_indices.cpu().numpy()]
        matched_predicted_phi = predicted_phi_raw[matched_prediction_indices.cpu().numpy()]

        # GT in matched order
        matched_ground_truth_eta = ground_truth_eta[matched_ground_truth_indices.cpu().numpy()]
        matched_ground_truth_phi = ground_truth_phi[matched_ground_truth_indices.cpu().numpy()]

        # Plot arrows from reconstruction to matched ground truth
        for track_index in range(num_valid):
            delta_eta = matched_ground_truth_eta[track_index] - matched_predicted_eta[track_index]
            delta_phi = matched_ground_truth_phi[track_index] - matched_predicted_phi[track_index]
            axis.annotate(
                '', xy=(matched_ground_truth_eta[track_index],
                        matched_ground_truth_phi[track_index]),
                xytext=(matched_predicted_eta[track_index],
                        matched_predicted_phi[track_index]),
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.4),
            )

        # Plot ground truth and predictions
        axis.scatter(
            ground_truth_eta, ground_truth_phi,
            c='tab:blue', s=12, alpha=0.7, label='Ground truth', zorder=3,
        )
        axis.scatter(
            matched_predicted_eta, matched_predicted_phi,
            c='tab:red', s=12, alpha=0.7, label='Reconstructed', zorder=3,
            marker='x',
        )

        axis.set_xlabel('η')
        axis.set_ylabel('φ')
        axis.set_title(f'Event {event_index} ({num_valid} masked tracks)')
        axis.legend(fontsize=9, loc='upper right')
        axis.grid(True, alpha=0.2)

    # Hide unused subplots
    for unused_index in range(num_events, num_rows * num_columns):
        row = unused_index // num_columns
        column = unused_index % num_columns
        axes[row, column].set_visible(False)

    figure.suptitle('Masked Track Reconstruction in η-φ Space', fontsize=14)
    figure.tight_layout()
    output_path = os.path.join(output_directory, 'eta_phi_reconstruction.png')
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f'Saved eta-phi visualization: {output_path}')


def plot_per_feature_scatter(
    forward_results: dict,
    output_directory: str,
):
    """Scatter plot of predicted vs ground-truth for each feature channel.

    For each of the 7 features, plots all valid matched predictions against
    their ground-truth values. Computes Pearson R² and displays in the title.
    A perfect model produces points along the diagonal.

    Args:
        forward_results: Output of run_forward_pass().
        output_directory: Directory to save the plot.
    """
    masked_ground_truth = forward_results['masked_ground_truth']
    predicted_features = forward_results['predicted_features']
    masked_validity = forward_results['masked_validity']
    matching_indices = forward_results['matching_indices']

    batch_size = masked_ground_truth.shape[0]
    num_features = masked_ground_truth.shape[1]

    # Collect matched (prediction, ground_truth) pairs across all events
    all_predictions = {feature_index: [] for feature_index in range(num_features)}
    all_ground_truths = {feature_index: [] for feature_index in range(num_features)}

    for event_index in range(batch_size):
        validity = masked_validity[event_index, 0].bool()
        num_valid = validity.sum().item()
        if num_valid == 0:
            continue

        matched_prediction_indices = matching_indices[event_index, 0, :num_valid].long()
        matched_ground_truth_indices = matching_indices[event_index, 1, :num_valid].long()

        for feature_index in range(num_features):
            predicted_values = predicted_features[
                event_index, feature_index, matched_prediction_indices
            ].cpu()
            ground_truth_values = masked_ground_truth[
                event_index, feature_index, matched_ground_truth_indices
            ].cpu()
            all_predictions[feature_index].append(predicted_values)
            all_ground_truths[feature_index].append(ground_truth_values)

    # Create scatter plots
    num_columns = 4
    num_rows = (num_features + num_columns - 1) // num_columns
    figure, axes = plt.subplots(
        num_rows, num_columns, figsize=(5 * num_columns, 5 * num_rows),
        squeeze=False,
    )

    for feature_index in range(num_features):
        row = feature_index // num_columns
        column = feature_index % num_columns
        axis = axes[row, column]

        predictions_concatenated = torch.cat(all_predictions[feature_index])
        ground_truths_concatenated = torch.cat(all_ground_truths[feature_index])

        # Compute R² = 1 - SS_res / SS_tot
        # SS_res = Σ (gt - pred)², SS_tot = Σ (gt - mean(gt))²
        residual_sum_squares = ((ground_truths_concatenated - predictions_concatenated) ** 2).sum()
        total_sum_squares = ((ground_truths_concatenated - ground_truths_concatenated.mean()) ** 2).sum()
        r_squared = 1.0 - (residual_sum_squares / total_sum_squares).item()

        axis.scatter(
            predictions_concatenated.numpy(),
            ground_truths_concatenated.numpy(),
            s=1, alpha=0.1, c='tab:blue', rasterized=True,
        )

        # Add diagonal reference line
        combined_values = torch.cat([predictions_concatenated, ground_truths_concatenated])
        value_min, value_max = combined_values.min().item(), combined_values.max().item()
        axis.plot(
            [value_min, value_max], [value_min, value_max],
            'r--', lw=1, alpha=0.7, label='y = x',
        )

        axis.set_xlabel('Predicted (standardized)')
        axis.set_ylabel('Ground truth (standardized)')
        axis.set_title(f'{FEATURE_NAMES[feature_index]}\nR² = {r_squared:.4f}')
        axis.legend(fontsize=8)
        axis.set_aspect('equal', adjustable='box')
        axis.grid(True, alpha=0.2)

    # Hide unused subplots
    for unused_index in range(num_features, num_rows * num_columns):
        row = unused_index // num_columns
        column = unused_index % num_columns
        axes[row, column].set_visible(False)

    figure.suptitle('Per-Feature Reconstruction Quality', fontsize=14)
    figure.tight_layout()
    output_path = os.path.join(output_directory, 'per_feature_scatter.png')
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f'Saved per-feature scatter: {output_path}')


def compute_baselines(
    model,
    loader: DataLoader,
    data_config,
    mask_input_index: int,
    device: torch.device,
    num_batches: int = 10,
) -> dict:
    """Compute model loss and trivial baselines over multiple batches.

    Baselines:
        - zero_baseline: predict all zeros → loss = E[x²] for standardized data
        - batch_mean_baseline: predict per-batch mean of masked GT features
        - model_loss: actual decoder predictions matched via Hungarian

    Args:
        model: MaskedTrackPretrainer instance (eval mode).
        loader: DataLoader yielding (X, y, Z) tuples.
        data_config: Weaver DataConfig for input name ordering.
        mask_input_index: Index of pf_mask in data_config.input_names.
        device: Target device.
        num_batches: Number of batches to average over.

    Returns:
        Dictionary with 'model', 'zero', 'batch_mean' loss values.
    """
    model.eval()
    model_losses = []
    zero_losses = []
    batch_mean_losses = []

    with torch.no_grad():
        for batch_index, (batch_inputs, _, _) in enumerate(loader):
            if batch_index >= num_batches:
                break

            inputs = [batch_inputs[key].to(device) for key in data_config.input_names]
            inputs = trim_to_max_valid_tracks(inputs, mask_input_index)
            points, features, lorentz_vectors, mask = inputs

            # Run full forward pass with intermediates
            results = run_forward_pass(model, points, features, lorentz_vectors, mask)

            masked_ground_truth = results['masked_ground_truth']
            predicted_features = results['predicted_features']
            masked_validity = results['masked_validity']
            matching_indices = results['matching_indices']
            num_masked = results['num_masked_per_event'].float()
            num_feature_channels = features.shape[1]

            # Reorder predictions via matching
            matched_prediction_indices = matching_indices[:, 0, :]  # (B, K)
            matched_ground_truth_indices = matching_indices[:, 1, :]  # (B, K)

            prediction_gather = matched_prediction_indices.unsqueeze(1).expand(
                -1, num_feature_channels, -1,
            )
            ground_truth_gather = matched_ground_truth_indices.unsqueeze(1).expand(
                -1, num_feature_channels, -1,
            )

            matched_predictions = predicted_features.gather(2, prediction_gather)
            matched_targets = masked_ground_truth.gather(2, ground_truth_gather)

            matched_validity_mask = masked_validity.squeeze(1).gather(
                1, matched_prediction_indices,
            ).unsqueeze(1).float()

            # Model loss: MSE on matched pairs
            # L = (1 / N_valid / F) × Σ (pred - true)²
            model_error = (
                (matched_predictions - matched_targets).square()
                * matched_validity_mask
            )
            per_event_model_loss = model_error.sum(dim=(1, 2)) / (
                num_masked * num_feature_channels
            ).clamp(min=1.0)
            model_losses.append(per_event_model_loss.mean().item())

            # Zero baseline: predict zeros → loss = mean(true²)
            track_valid = masked_validity.float()
            zero_error = masked_ground_truth.square() * track_valid
            per_event_zero_loss = zero_error.sum(dim=(1, 2)) / (
                num_masked * num_feature_channels
            ).clamp(min=1.0)
            zero_losses.append(per_event_zero_loss.mean().item())

            # Batch-mean baseline: predict per-event mean of GT features
            # mean_per_feature[b, f] = mean over valid masked tracks
            mean_per_feature = (
                (masked_ground_truth * track_valid).sum(dim=2)
                / num_masked.unsqueeze(1).clamp(min=1.0)
            )  # (B, F)
            mean_prediction = mean_per_feature.unsqueeze(2).expand_as(
                masked_ground_truth,
            )
            mean_error = (
                (mean_prediction - masked_ground_truth).square() * track_valid
            )
            per_event_mean_loss = mean_error.sum(dim=(1, 2)) / (
                num_masked * num_feature_channels
            ).clamp(min=1.0)
            batch_mean_losses.append(per_event_mean_loss.mean().item())

    return {
        'model': sum(model_losses) / len(model_losses),
        'zero': sum(zero_losses) / len(zero_losses),
        'batch_mean': sum(batch_mean_losses) / len(batch_mean_losses),
    }


def plot_baseline_comparison(baselines: dict, output_directory: str):
    """Bar chart comparing model loss against trivial baselines.

    Args:
        baselines: Dict with 'model', 'zero', 'batch_mean' loss values.
        output_directory: Directory to save the plot.
    """
    figure, axis = plt.subplots(figsize=(8, 5))

    labels = ['Model', 'Zero prediction', 'Batch mean']
    values = [baselines['model'], baselines['zero'], baselines['batch_mean']]
    colors = ['tab:green', 'tab:red', 'tab:orange']

    bars = axis.bar(labels, values, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f'{value:.4f}', ha='center', va='bottom', fontsize=11,
        )

    axis.set_ylabel('MSE Loss (standardized features)')
    axis.set_title('Model vs Baseline Losses')
    axis.grid(True, alpha=0.2, axis='y')

    figure.tight_layout()
    output_path = os.path.join(output_directory, 'baseline_comparison.png')
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f'Saved baseline comparison: {output_path}')


def _compute_random_assignment_loss(
    predicted_features: torch.Tensor,
    true_features: torch.Tensor,
    validity_mask: torch.Tensor,
    num_valid_per_event: torch.Tensor,
    num_feature_channels: int,
    num_random_permutations: int = 5,
) -> torch.Tensor:
    """Compute MSE under random (non-optimal) assignment, averaged over
    multiple random permutations.

    This provides a baseline for the Hungarian-matched loss: if the
    Hungarian loss is much lower, the difference is due to the optimal
    matching exploiting accidental proximity, not better predictions.

    For each event, randomly permutes prediction indices and computes MSE
    against the (unpermuted) GT. Averages over `num_random_permutations`
    independent permutations to reduce variance.

    Args:
        predicted_features: (B, F, K) decoder predictions.
        true_features: (B, F, K) ground-truth masked features.
        validity_mask: (B, 1, K) boolean mask for valid slots.
        num_valid_per_event: (B,) count of valid tracks per event.
        num_feature_channels: F — number of feature channels.
        num_random_permutations: How many random permutations to average.

    Returns:
        per_event_loss: (B,) MSE loss per event under random assignment.
    """
    batch_size = predicted_features.shape[0]
    max_tracks = predicted_features.shape[2]
    validity_flat = validity_mask.squeeze(1)  # (B, K)

    # Accumulate loss over multiple random permutations
    accumulated_loss = torch.zeros(batch_size, device=predicted_features.device)

    for _ in range(num_random_permutations):
        # For each event, create a random permutation of valid prediction indices
        random_prediction_indices = torch.zeros(
            batch_size, max_tracks, dtype=torch.long, device=predicted_features.device,
        )
        for event_index in range(batch_size):
            num_valid = int(num_valid_per_event[event_index].item())
            if num_valid > 0:
                permutation = torch.randperm(num_valid, device=predicted_features.device)
                random_prediction_indices[event_index, :num_valid] = permutation

        # Gather predictions in random order
        prediction_gather = random_prediction_indices.unsqueeze(1).expand(
            -1, num_feature_channels, -1,
        )
        permuted_predictions = predicted_features.gather(2, prediction_gather)

        # L = (1 / N_valid / F) × Σ (permuted_pred - true)²
        feature_error = (
            (permuted_predictions - true_features).square()
            * validity_flat.unsqueeze(1).float()
        )
        per_event_loss = feature_error.sum(dim=(1, 2)) / (
            num_valid_per_event.float() * num_feature_channels
        ).clamp(min=1.0)
        accumulated_loss += per_event_loss

    return accumulated_loss / num_random_permutations


def compute_mask_ratio_sensitivity(
    model,
    loader: DataLoader,
    data_config,
    mask_input_index: int,
    device: torch.device,
    num_batches: int = 10,
) -> dict:
    """Compute model loss at different mask ratios with three metrics.

    At each mask ratio, computes:
        1. Hungarian-matched MSE — optimal 1-to-1 assignment (standard loss)
        2. Random-assignment MSE — random permutation baseline

    If the Hungarian loss decreases with mask ratio while random-assignment
    loss stays flat, the decrease is a combinatorial matching artifact:
    with more tracks, the optimal assignment has more "lucky nearby pairs"
    to exploit, reducing average MSE per matched pair.

    The model was trained with 50% masking.

    Args:
        model: MaskedTrackPretrainer instance (eval mode).
        loader: DataLoader yielding (X, y, Z) tuples.
        data_config: Weaver DataConfig.
        mask_input_index: Index of pf_mask in data_config.input_names.
        device: Target device.
        num_batches: Number of batches to average per ratio.

    Returns:
        Dict with:
            'ratios': list of tested mask ratios
            'hungarian_losses': list of Hungarian-matched MSE values
            'random_losses': list of random-assignment MSE values
    """
    original_mask_ratio = model.mask_ratio

    # The decoder has a fixed nn.Embedding(max_masked_tracks, ...).
    # High mask ratios can produce more masked tracks than the embedding
    # supports, causing IndexError.
    max_decoder_slots = model.decoder.query_embeddings.num_embeddings
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    hungarian_losses_per_ratio = []
    random_losses_per_ratio = []

    model.eval()

    for ratio in ratios:
        model.mask_ratio = ratio
        batch_hungarian_losses = []
        batch_random_losses = []

        with torch.no_grad():
            for batch_index, (batch_inputs, _, _) in enumerate(loader):
                if batch_index >= num_batches:
                    break

                inputs = [batch_inputs[key].to(device) for key in data_config.input_names]
                inputs = trim_to_max_valid_tracks(inputs, mask_input_index)
                points, features, lorentz_vectors, mask = inputs

                # Check if this ratio would exceed decoder capacity
                max_tracks_in_batch = int(mask.sum(dim=2).max().item())
                max_masked_expected = int(ratio * max_tracks_in_batch)
                if max_masked_expected > max_decoder_slots:
                    print(f'  mask_ratio={ratio:.1f}: SKIPPED '
                          f'(would need {max_masked_expected} slots, '
                          f'decoder supports {max_decoder_slots})')
                    break

                # Run step-by-step forward pass to get intermediates
                results = run_forward_pass(
                    model, points, features, lorentz_vectors, mask,
                )

                predicted = results['predicted_features']  # (B, F, K)
                ground_truth = results['masked_ground_truth']  # (B, F, K)
                validity = results['masked_validity']  # (B, 1, K)
                num_masked = results['num_masked_per_event']  # (B,)
                matching = results['matching_indices']  # (B, 2, K)
                num_feature_channels = features.shape[1]

                # --- Hungarian-matched loss ---
                matched_pred_indices = matching[:, 0, :]
                matched_true_indices = matching[:, 1, :]
                pred_gather = matched_pred_indices.unsqueeze(1).expand(
                    -1, num_feature_channels, -1,
                )
                true_gather = matched_true_indices.unsqueeze(1).expand(
                    -1, num_feature_channels, -1,
                )
                matched_preds = predicted.gather(2, pred_gather)
                matched_truths = ground_truth.gather(2, true_gather)
                matched_valid = validity.squeeze(1).gather(
                    1, matched_pred_indices,
                ).unsqueeze(1).float()

                # L = (1 / N_valid / F) × Σ (pred - true)²
                hungarian_error = (
                    (matched_preds - matched_truths).square() * matched_valid
                )
                hungarian_per_event = hungarian_error.sum(dim=(1, 2)) / (
                    num_masked.float() * num_feature_channels
                ).clamp(min=1.0)
                batch_hungarian_losses.append(hungarian_per_event.mean().item())

                # --- Random-assignment loss ---
                random_per_event = _compute_random_assignment_loss(
                    predicted, ground_truth, validity, num_masked,
                    num_feature_channels, num_random_permutations=5,
                )
                batch_random_losses.append(random_per_event.mean().item())

        if not batch_hungarian_losses:
            # All batches were skipped — stop testing higher ratios
            break

        avg_hungarian = sum(batch_hungarian_losses) / len(batch_hungarian_losses)
        avg_random = sum(batch_random_losses) / len(batch_random_losses)
        hungarian_losses_per_ratio.append(avg_hungarian)
        random_losses_per_ratio.append(avg_random)

        print(f'  mask_ratio={ratio:.1f}: '
              f'hungarian={avg_hungarian:.5f}, '
              f'random_assign={avg_random:.5f}')

    # Restore original mask ratio
    model.mask_ratio = original_mask_ratio

    successful_ratios = ratios[:len(hungarian_losses_per_ratio)]
    return {
        'ratios': successful_ratios,
        'hungarian_losses': hungarian_losses_per_ratio,
        'random_losses': random_losses_per_ratio,
    }


def plot_mask_ratio_sensitivity(sensitivity: dict, output_directory: str):
    """Line plot of loss vs mask ratio with Hungarian and random baselines.

    Shows two curves:
        - Hungarian-matched loss (optimal 1-to-1 assignment)
        - Random-assignment loss (random permutation baseline)

    If the Hungarian curve decreases while the random curve stays flat,
    the decrease is a combinatorial matching artifact, not a genuine
    model quality change.

    Args:
        sensitivity: Dict with 'ratios', 'hungarian_losses', 'random_losses'.
        output_directory: Directory to save the plot.
    """
    figure, axis = plt.subplots(figsize=(10, 6))

    ratios = sensitivity['ratios']
    hungarian_losses = sensitivity['hungarian_losses']
    random_losses = sensitivity['random_losses']

    axis.plot(
        ratios, hungarian_losses,
        'o-', color='tab:blue', linewidth=2, markersize=6,
        label='Hungarian-matched (optimal)',
    )
    axis.plot(
        ratios, random_losses,
        's--', color='tab:orange', linewidth=2, markersize=6,
        label='Random assignment',
    )

    # Mark training ratio
    training_ratio = 0.5
    axis.axvline(
        x=training_ratio, color='tab:red', linestyle=':', alpha=0.7,
        label=f'Training ratio ({training_ratio:.0%})',
    )

    # Shade the gap between random and Hungarian to highlight matching benefit
    axis.fill_between(
        ratios, hungarian_losses, random_losses,
        alpha=0.15, color='tab:green',
        label='Matching benefit',
    )

    axis.set_xlabel('Mask Ratio', fontsize=12)
    axis.set_ylabel('MSE Loss', fontsize=12)
    axis.set_title(
        'Reconstruction Loss vs Mask Ratio\n'
        '(Hungarian matching artifact diagnosis)',
        fontsize=13,
    )
    axis.legend(fontsize=10)
    axis.grid(True, alpha=0.3)
    axis.set_xticks(ratios)
    axis.set_xticklabels([f'{r:.0%}' for r in ratios])

    figure.tight_layout()
    output_path = os.path.join(output_directory, 'mask_ratio_sensitivity.png')
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f'Saved mask-ratio sensitivity: {output_path}')


def check_prediction_diversity(forward_results: dict):
    """Check whether predictions are diverse or collapsed to a prototype.

    Computes pairwise cosine similarity among predictions and among ground
    truth tracks within each event. If predictions have much higher cosine
    similarity than GT, the model has collapsed to predicting near-identical
    outputs (mean prediction / mode collapse).

    Args:
        forward_results: Output of run_forward_pass().
    """
    predicted_features = forward_results['predicted_features']
    masked_ground_truth = forward_results['masked_ground_truth']
    masked_validity = forward_results['masked_validity']

    batch_size = predicted_features.shape[0]

    prediction_cosine_similarities = []
    ground_truth_cosine_similarities = []

    for event_index in range(batch_size):
        validity = masked_validity[event_index, 0].bool()
        num_valid = validity.sum().item()
        if num_valid < 2:
            continue

        # Extract valid predictions and GT: (num_valid, F)
        predictions_valid = predicted_features[event_index, :, :num_valid].T  # (N, F)
        ground_truth_valid = masked_ground_truth[event_index, :, :num_valid].T  # (N, F)

        # Pairwise cosine similarity
        # cos_sim(a, b) = (a · b) / (||a|| × ||b||)
        predictions_normalized = torch.nn.functional.normalize(
            predictions_valid, dim=1,
        )
        ground_truth_normalized = torch.nn.functional.normalize(
            ground_truth_valid, dim=1,
        )

        prediction_cosine_matrix = predictions_normalized @ predictions_normalized.T
        ground_truth_cosine_matrix = ground_truth_normalized @ ground_truth_normalized.T

        # Extract upper triangle (exclude diagonal self-similarity)
        upper_triangle_mask = torch.triu(
            torch.ones(num_valid, num_valid, dtype=torch.bool), diagonal=1,
        )
        prediction_cosine_similarities.append(
            prediction_cosine_matrix[upper_triangle_mask].cpu()
        )
        ground_truth_cosine_similarities.append(
            ground_truth_cosine_matrix[upper_triangle_mask].cpu()
        )

    if not prediction_cosine_similarities:
        print('  No events with ≥ 2 valid masked tracks — skipping diversity check.')
        return

    all_prediction_cosines = torch.cat(prediction_cosine_similarities)
    all_ground_truth_cosines = torch.cat(ground_truth_cosine_similarities)

    print(f'\n{"="*60}')
    print('PREDICTION DIVERSITY CHECK')
    print(f'{"="*60}')
    print(f'  Ground truth pairwise cosine similarity:')
    print(f'    mean={all_ground_truth_cosines.mean():.4f}, '
          f'std={all_ground_truth_cosines.std():.4f}')
    print(f'  Prediction pairwise cosine similarity:')
    print(f'    mean={all_prediction_cosines.mean():.4f}, '
          f'std={all_prediction_cosines.std():.4f}')

    cosine_ratio = all_prediction_cosines.mean() / all_ground_truth_cosines.mean()
    print(f'  Ratio (pred/GT): {cosine_ratio:.4f}')

    if cosine_ratio > 2.0:
        print('  ⚠ WARNING: Predictions are significantly less diverse than GT.')
        print('    The model may be collapsing to a mean/prototype prediction.')
    elif cosine_ratio > 1.5:
        print('  ⚠ CAUTION: Predictions are somewhat less diverse than GT.')
        print('    Partial mode collapse may be occurring.')
    else:
        print('  ✓ Predictions have comparable diversity to ground truth.')


def main():
    parser = argparse.ArgumentParser(
        description='Sanity checks for masked track reconstruction pretraining',
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data-config', type=str, required=True,
                        help='Path to YAML data config')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing parquet files')
    parser.add_argument('--network', type=str, required=True,
                        help='Path to network wrapper Python file')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for inference (default: cpu)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for data loading')
    parser.add_argument('--num-events', type=int, default=4,
                        help='Number of events for eta-phi visualization')
    parser.add_argument('--num-batches', type=int, default=10,
                        help='Number of batches for baseline/sensitivity averaging')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save output plots')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load data ----
    print('Loading data...')
    parquet_files = sorted([
        os.path.join(args.data_dir, f)
        for f in os.listdir(args.data_dir) if f.endswith('.parquet')
    ])
    if not parquet_files:
        raise FileNotFoundError(f'No parquet files found in {args.data_dir}')

    file_dict = {'data': parquet_files}
    dataset = SimpleIterDataset(
        file_dict,
        data_config_file=args.data_config,
        for_training=True,
        load_range_and_fraction=((0.0, 0.8), 1.0),
        fetch_by_files=True,
        fetch_step=len(parquet_files),
        in_memory=True,
    )
    data_config = dataset.config
    mask_input_index = data_config.input_names.index('pf_mask')

    loader = DataLoader(
        dataset, batch_size=args.batch_size, drop_last=True,
    )

    # ---- Load model ----
    # Reconstruct model kwargs from checkpoint's saved training args so the
    # architecture matches (e.g. num_enrichment_layers, mask_ratio).
    print('Loading model...')
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get('args', {})

    model_kwargs = {}
    if 'mask_ratio' in checkpoint_args:
        model_kwargs['mask_ratio'] = checkpoint_args['mask_ratio']
    if 'num_enrichment_layers' in checkpoint_args:
        model_kwargs['num_enrichment_layers'] = checkpoint_args['num_enrichment_layers']
    if 'train_matcher' in checkpoint_args:
        model_kwargs['train_matcher'] = checkpoint_args['train_matcher']
    if 'num_decoder_layers' in checkpoint_args:
        model_kwargs['num_decoder_layers'] = checkpoint_args['num_decoder_layers']

    network_module = load_network_module(args.network)
    model, _ = network_module.get_model(data_config, **model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f'Loaded checkpoint: {args.checkpoint}')
    print(f'  Model kwargs from checkpoint: {model_kwargs}')

    # ---- Test 1: Eta-phi visualization ----
    print(f'\n{"="*60}')
    print('TEST 1: Eta-phi reconstruction visualization')
    print(f'{"="*60}')

    batch_inputs, _, _ = next(iter(loader))
    inputs = [batch_inputs[key].to(device) for key in data_config.input_names]
    inputs = trim_to_max_valid_tracks(inputs, mask_input_index)
    points, features, lorentz_vectors, mask = inputs

    forward_results = run_forward_pass(model, points, features, lorentz_vectors, mask)
    plot_eta_phi(forward_results, data_config, args.output_dir, num_events=args.num_events)

    # ---- Test 2: Per-feature scatter plots ----
    print(f'\n{"="*60}')
    print('TEST 2: Per-feature reconstruction quality')
    print(f'{"="*60}')
    plot_per_feature_scatter(forward_results, args.output_dir)

    # ---- Test 3: Baseline comparison ----
    print(f'\n{"="*60}')
    print('TEST 3: Baseline comparison')
    print(f'{"="*60}')

    # Need a fresh loader since we consumed one batch
    loader = DataLoader(dataset, batch_size=args.batch_size, drop_last=True)
    baselines = compute_baselines(
        model, loader, data_config, mask_input_index, device,
        num_batches=args.num_batches,
    )
    print(f'  Model loss:      {baselines["model"]:.5f}')
    print(f'  Zero baseline:   {baselines["zero"]:.5f}')
    print(f'  Batch mean:      {baselines["batch_mean"]:.5f}')
    improvement_over_zero = (1 - baselines['model'] / baselines['zero']) * 100
    improvement_over_mean = (1 - baselines['model'] / baselines['batch_mean']) * 100
    print(f'  Improvement over zero: {improvement_over_zero:.1f}%')
    print(f'  Improvement over batch mean: {improvement_over_mean:.1f}%')
    plot_baseline_comparison(baselines, args.output_dir)

    # ---- Test 4: Mask-ratio sensitivity ----
    print(f'\n{"="*60}')
    print('TEST 4: Mask-ratio sensitivity')
    print(f'{"="*60}')

    loader = DataLoader(dataset, batch_size=args.batch_size, drop_last=True)
    sensitivity = compute_mask_ratio_sensitivity(
        model, loader, data_config, mask_input_index, device,
        num_batches=args.num_batches,
    )
    plot_mask_ratio_sensitivity(sensitivity, args.output_dir)

    # ---- Test 5: Prediction diversity ----
    print(f'\n{"="*60}')
    print('TEST 5: Prediction diversity')
    print(f'{"="*60}')
    check_prediction_diversity(forward_results)

    print(f'\n{"="*60}')
    print(f'All plots saved to: {args.output_dir}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
