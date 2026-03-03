"""Diagnostic script for the Enrich-Compact backbone pretraining.

Runs a single forward pass through MaskedTrackPretrainer and prints
activation statistics at every stage:
  - Input features (after standardization by weaver)
  - Enrichment: node encoding → each MultiScaleEdgeConv layer → enriched
  - Masking split
  - Compaction: each CompactionStage → backbone tokens
  - Decoder: cross-attention → predictions vs ground truth

Additional diagnostic modes:
  --test-backbone-bypass: Replaces backbone tokens with random noise
      and compares loss. If the loss barely changes, the decoder is
      solving the task from positional encoding alone (bypassing the
      backbone). This is the key test for decoder shortcut detection.

Usage:
    python diagnose_pretraining.py \
        --data-config data/low-pt/lowpt_tau_pretrain.yaml \
        --data-dir data/low-pt/ \
        --network networks/lowpt_tau_BackbonePretrain.py \
        --batch-size 32 \
        --device cuda:0

    # Test backbone bypass (requires --checkpoint of a trained model):
    python diagnose_pretraining.py \
        --data-config data/low-pt/lowpt_tau_pretrain.yaml \
        --data-dir data/low-pt/ \
        --network networks/lowpt_tau_BackbonePretrain.py \
        --batch-size 32 \
        --device cpu \
        --checkpoint path/to/checkpoint.pt \
        --test-backbone-bypass
"""
import argparse
import importlib.util
import sys
import os

import torch
from torch.utils.data import DataLoader

from weaver.utils.dataset import SimpleIterDataset


def load_network_module(network_path: str):
    spec = importlib.util.spec_from_file_location('network', network_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_stats(tensor: torch.Tensor, name: str, mask: torch.Tensor | None = None):
    """Print statistics of a tensor, optionally masked."""
    if mask is not None:
        # Expand mask to match tensor shape for broadcasting
        while mask.dim() < tensor.dim():
            mask = mask.unsqueeze(1)
        mask = mask.expand_as(tensor).bool()
        values = tensor[mask]
    else:
        values = tensor.flatten()

    if values.numel() == 0:
        print(f"  {name}: EMPTY")
        return

    values = values.float()
    print(
        f"  {name}: "
        f"shape={list(tensor.shape)}, "
        f"mean={values.mean().item():.4f}, "
        f"std={values.std().item():.4f}, "
        f"min={values.min().item():.4f}, "
        f"max={values.max().item():.4f}, "
        f"|mean|={values.abs().mean().item():.4f}, "
        f"zeros%={100 * (values == 0).float().mean().item():.1f}"
    )


def check_token_diversity(tokens: torch.Tensor, name: str):
    """Check if tokens are diverse or collapsed to near-constant."""
    # tokens: (B, C, N)
    batch_size, channels, num_tokens = tokens.shape
    if num_tokens < 2:
        print(f"  {name} token diversity: only {num_tokens} token(s), skipping")
        return
    tokens_normed = torch.nn.functional.normalize(tokens.float(), dim=1)  # (B, C, N)
    # Pairwise cosine sim: (B, N, N)
    cosine_similarity = torch.bmm(tokens_normed.transpose(1, 2), tokens_normed)
    # Exclude diagonal (self-similarity = 1)
    mask_diagonal = ~torch.eye(num_tokens, device=tokens.device, dtype=torch.bool).unsqueeze(0)
    pairwise_cosine = cosine_similarity[mask_diagonal.expand(batch_size, -1, -1)]
    print(
        f"  {name} token diversity: "
        f"mean_pairwise_cosine={pairwise_cosine.mean().item():.4f}, "
        f"std={pairwise_cosine.std().item():.4f} "
        f"(1.0 = all identical, 0.0 = orthogonal)"
    )


def compute_loss_with_backbone_tokens(
    model,
    backbone_tokens: torch.Tensor,
    features: torch.Tensor,
    masked_mask: torch.Tensor,
    num_masked_per_event: torch.Tensor,
    max_masked: int,
) -> float:
    """Run the decoder with given backbone tokens and compute MSE loss.

    Args:
        model: MaskedTrackPretrainer instance.
        backbone_tokens: (B, C_backbone, M) tokens to feed to decoder.
        features: (B, F, P) standardized track features.
        masked_mask: (B, 1, P) mask indicating which tracks were masked.
        num_masked_per_event: (B,) count of masked tracks per event.
        max_masked: Maximum number of masked tracks in this batch.

    Returns:
        Scalar loss value (averaged over batch).
    """
    device = features.device

    # Gather ground truth for masked tracks
    masked_true_features, masked_validity = model._gather_tracks(
        features, masked_mask, max_masked
    )

    # Run decoder
    predicted_features = model.decoder(backbone_tokens, max_masked)

    # Per-event MSE loss: L = (1 / N_masked / F) * Σ (pred - true)²
    track_valid = masked_validity.float()
    feature_error = (predicted_features - masked_true_features).square() * track_valid
    per_event_loss = feature_error.sum(dim=(1, 2)) / (
        num_masked_per_event.float() * features.shape[1]
    ).clamp(min=1.0)

    return per_event_loss.mean().item()


def test_backbone_bypass(
    model,
    loader: DataLoader,
    data_config,
    mask_input_index: int,
    device: torch.device,
    num_batches: int = 10,
):
    """Test whether the decoder bypasses the backbone.

    For each batch, computes loss in three conditions:
      1. Normal: real backbone tokens from the trained model
      2. Random noise: backbone tokens replaced with Gaussian noise
         matched in mean/std to the real tokens
      3. Zeros: backbone tokens replaced with all zeros

    If loss barely changes between conditions, the decoder is solving the
    task from positional encoding + self-attention alone.
    """
    model.eval()

    losses_normal = []
    losses_random = []
    losses_zeros = []
    losses_zero_pred = []

    print(f"\n{'='*70}")
    print("BACKBONE BYPASS TEST")
    print(f"{'='*70}")
    print(f"Averaging over {num_batches} batches...")

    with torch.no_grad():
        for batch_idx, (X, y, _) in enumerate(loader):
            if batch_idx >= num_batches:
                break

            inputs = [X[k].to(device) for k in data_config.input_names]

            # Trim padding
            mask_tensor = inputs[mask_input_index]
            max_valid = int(mask_tensor.sum(dim=2).max().item())
            max_valid = max(1, max_valid)
            inputs = [tensor[:, :, :max_valid] for tensor in inputs]

            points, features, lorentz_vectors, mask = inputs

            # Step 1: Enrich all tracks
            enriched = model.backbone.enrich(
                points, features, lorentz_vectors, mask,
            )

            # Step 2: Create masking split (same for all conditions)
            visible_mask, masked_mask = model._create_random_mask(mask)
            num_visible_per_event = visible_mask.squeeze(1).sum(dim=1)
            num_masked_per_event = masked_mask.squeeze(1).sum(dim=1).float()
            max_visible = int(num_visible_per_event.max().item())
            max_masked = int(num_masked_per_event.max().item())

            if max_masked == 0:
                continue

            # Step 3: Gather visible tracks
            visible_enriched, visible_validity = model._gather_tracks(
                enriched, visible_mask, max_visible,
            )
            visible_coordinates, _ = model._gather_tracks(
                points, visible_mask, max_visible,
            )

            # Step 4: Compact
            backbone_tokens, _ = model.backbone.compact(
                visible_coordinates, visible_enriched, visible_validity,
            )

            # Condition 1: Normal (real backbone tokens)
            loss_normal = compute_loss_with_backbone_tokens(
                model, backbone_tokens, features,
                masked_mask, num_masked_per_event, max_masked,
            )
            losses_normal.append(loss_normal)

            # Condition 2: Random noise (matched statistics)
            random_tokens = torch.randn_like(backbone_tokens)
            random_tokens = (
                random_tokens * backbone_tokens.std() + backbone_tokens.mean()
            )
            loss_random = compute_loss_with_backbone_tokens(
                model, random_tokens, features,
                masked_mask, num_masked_per_event, max_masked,
            )
            losses_random.append(loss_random)

            # Condition 3: All zeros
            zero_tokens = torch.zeros_like(backbone_tokens)
            loss_zeros = compute_loss_with_backbone_tokens(
                model, zero_tokens, features,
                masked_mask, num_masked_per_event, max_masked,
            )
            losses_zeros.append(loss_zeros)

            # Baseline: zero-prediction loss (predict all zeros)
            masked_true_features, masked_validity = model._gather_tracks(
                features, masked_mask, max_masked,
            )
            track_valid = masked_validity.float()
            zero_pred_error = masked_true_features.square() * track_valid
            zero_pred_loss = zero_pred_error.sum(dim=(1, 2)) / (
                num_masked_per_event.float() * features.shape[1]
            ).clamp(min=1.0)
            losses_zero_pred.append(zero_pred_loss.mean().item())

            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx}: normal={loss_normal:.4f}, "
                      f"random={loss_random:.4f}, zeros={loss_zeros:.4f}")

    # Compute averages
    avg_normal = sum(losses_normal) / len(losses_normal)
    avg_random = sum(losses_random) / len(losses_random)
    avg_zeros = sum(losses_zeros) / len(losses_zeros)
    avg_zero_pred = sum(losses_zero_pred) / len(losses_zero_pred)

    print(f"\n--- RESULTS (averaged over {len(losses_normal)} batches) ---")
    print(f"  Zero-prediction baseline:     {avg_zero_pred:.4f}")
    print(f"  Normal (real backbone):        {avg_normal:.4f}")
    print(f"  Random noise backbone:         {avg_random:.4f}")
    print(f"  Zero backbone:                 {avg_zeros:.4f}")

    # Interpretation
    normal_vs_random_delta = avg_random - avg_normal
    normal_vs_random_pct = 100 * normal_vs_random_delta / avg_normal

    print(f"\n--- INTERPRETATION ---")
    print(f"  Loss increase (normal -> random): "
          f"{normal_vs_random_delta:+.4f} ({normal_vs_random_pct:+.1f}%)")
    print(f"  Loss increase (normal -> zeros):  "
          f"{avg_zeros - avg_normal:+.4f} "
          f"({100 * (avg_zeros - avg_normal) / avg_normal:+.1f}%)")

    if abs(normal_vs_random_pct) < 5:
        print(f"\n  WARNING: BACKBONE BYPASS DETECTED: Loss barely changes when "
              f"backbone tokens are replaced with noise.")
        print(f"  The decoder is not extracting meaningful information "
              f"from the backbone tokens.")
        print(f"  If model is freshly initialized, this is expected — "
              f"train longer and re-test.")
    elif abs(normal_vs_random_pct) < 15:
        print(f"\n  WARNING: PARTIAL BYPASS: Backbone contributes some information, "
              f"but the decoder can mostly function without it.")
    else:
        print(f"\n  OK: Backbone is providing meaningful information to the "
              f"decoder ({normal_vs_random_pct:+.1f}% loss increase without it).")

    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='Diagnose pretraining pipeline')
    parser.add_argument('--data-config', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--network', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Optional checkpoint to load (diagnoses trained model)')
    parser.add_argument('--test-backbone-bypass', action='store_true',
                        help='Test whether the decoder bypasses the backbone')
    parser.add_argument('--num-batches', type=int, default=10,
                        help='Number of batches for backbone bypass test '
                             '(averaged for stable estimates). Default: 10.')
    args = parser.parse_args()

    device = torch.device(args.device)

    # ---- Load data ----
    parquet_files = sorted([
        os.path.join(args.data_dir, f)
        for f in os.listdir(args.data_dir) if f.endswith('.parquet')
    ])
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
    loader = DataLoader(dataset, batch_size=args.batch_size, drop_last=True)

    # ---- Load model ----
    network_module = load_network_module(args.network)
    model, _ = network_module.get_model(data_config)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint: {args.checkpoint}")
    model = model.to(device)
    model.eval()

    mask_input_index = data_config.input_names.index('pf_mask')

    # ---- Get one batch ----
    X, y, _ = next(iter(loader))
    inputs = [X[k].to(device) for k in data_config.input_names]

    # Trim
    mask_tensor = inputs[mask_input_index]
    max_valid = int(mask_tensor.sum(dim=2).max().item())
    inputs = [tensor[:, :, :max_valid] for tensor in inputs]

    points, features, lorentz_vectors, mask = inputs

    print(f"\n{'='*70}")
    print("DIAGNOSTIC: Enrich-Compact Pretraining Pipeline")
    print(f"{'='*70}")
    print(f"Batch size: {features.shape[0]}")
    print(f"Sequence length (after trim): {features.shape[2]}")
    print(f"Valid tracks per event: min={mask.sum(dim=2).min().item():.0f}, "
          f"max={mask.sum(dim=2).max().item():.0f}, "
          f"mean={mask.sum(dim=2).mean().item():.0f}")

    # ---- Input data ----
    print(f"\n--- INPUT DATA ---")
    tensor_stats(features, "pf_features (standardized)", mask)
    tensor_stats(lorentz_vectors, "pf_vectors (raw)", mask)
    tensor_stats(points, "pf_points (eta, phi)", mask)

    # Per-feature stats
    print("\n  Per-feature statistics (valid tracks only):")
    flat_mask = mask.squeeze(1).bool()  # (B, P)
    for feature_index in range(features.shape[1]):
        feature_values = features[:, feature_index, :][flat_mask]
        print(
            f"    feature[{feature_index}]: "
            f"mean={feature_values.mean().item():.4f}, "
            f"std={feature_values.std().item():.4f}, "
            f"min={feature_values.min().item():.4f}, "
            f"max={feature_values.max().item():.4f}"
        )

    with torch.no_grad():
        backbone = model.backbone

        # ---- ENRICHMENT STAGE ----
        print(f"\n{'='*70}")
        print("STAGE 1: ENRICHMENT (ParticleNeXt MultiScaleEdgeConv)")
        print(f"{'='*70}")

        boolean_mask = mask.bool()
        null_positions = ~boolean_mask

        # Static graph computation (mirrors backbone.enrich logic)
        points_for_knn = points.clone()
        points_for_knn.masked_fill_(null_positions, 1e9)

        knn_indices = backbone.enrichment_knn(points_for_knn)
        print(f"  kNN indices: shape={list(knn_indices.shape)}")

        edge_inputs, _, lvs_neighbors, null_edge_positions = (
            backbone.enrichment_get_graph_feature(
                lvs=lorentz_vectors,
                mask=boolean_mask,
                edges=None,
                idx=knn_indices,
                null_edge_pos=None,
            )
        )
        tensor_stats(edge_inputs, "pairwise LV edge features")

        # Node encoding
        features_4d = features.unsqueeze(-1)
        encoded_features = backbone.node_encode(features_4d)
        tensor_stats(
            encoded_features.squeeze(-1), "node_encode output",
            mask
        )

        # Edge encoding
        encoded_edges = backbone.edge_encode(edge_inputs)
        tensor_stats(encoded_edges, "edge_encode output")

        # Each MultiScaleEdgeConv layer
        current_features = encoded_features
        for layer_index, layer in enumerate(backbone.enrichment_layers):
            _, current_features = layer(
                points=points_for_knn,
                features=current_features,
                lorentz_vectors=lorentz_vectors,
                mask=boolean_mask,
                edges=None,
                idx=knn_indices,
                null_edge_pos=null_edge_positions,
                edge_inputs=encoded_edges,
                lvs_ngbs=lvs_neighbors,
            )
            layer_output = current_features.squeeze(-1)
            print(f"\n  --- MultiScaleEdgeConv layer {layer_index} ---")
            tensor_stats(layer_output, f"  layer_{layer_index} output", mask)
            check_token_diversity(
                layer_output[:, :, :min(256, layer_output.shape[2])],
                f"  layer_{layer_index} (first 256 tracks)"
            )
            # LayerScale gamma
            print(f"    gamma: {layer.gamma.data.mean().item():.6f} "
                  f"(init_scale=1e-5, growth indicates learning)")

        # Post-enrichment
        enriched = backbone.enrichment_post(current_features).squeeze(-1)
        enriched = enriched * mask.float()
        print(f"\n  --- Enriched output ---")
        tensor_stats(enriched, "enriched_features", mask)
        check_token_diversity(
            enriched[:, :, :min(256, enriched.shape[2])],
            "enriched (first 256 tracks)"
        )

        # ---- MASKING ----
        print(f"\n{'='*70}")
        print("MASKING")
        print(f"{'='*70}")
        visible_mask, masked_mask = model._create_random_mask(mask)
        num_visible = visible_mask.squeeze(1).sum(dim=1).float()
        num_masked = masked_mask.squeeze(1).sum(dim=1).float()
        max_visible = int(num_visible.max().item())
        max_masked = int(num_masked.max().item())
        print(f"  Visible per event: mean={num_visible.mean().item():.0f}, "
              f"min={num_visible.min().item():.0f}")
        print(f"  Masked per event: mean={num_masked.mean().item():.0f}, "
              f"min={num_masked.min().item():.0f}")

        # Gather visible tracks
        visible_enriched, visible_validity = model._gather_tracks(
            enriched, visible_mask, max_visible,
        )
        visible_coordinates, _ = model._gather_tracks(
            points, visible_mask, max_visible,
        )
        print(f"  Gathered visible: {list(visible_enriched.shape)}, "
              f"validity: {visible_validity.sum().item():.0f}/{visible_validity.numel()} slots")

        # ---- COMPACTION STAGE ----
        print(f"\n{'='*70}")
        print("STAGE 2: COMPACTION (PointNet++ Set Abstraction)")
        print(f"{'='*70}")

        current_compact_features = visible_enriched
        current_compact_coordinates = visible_coordinates
        current_compact_mask = visible_validity

        for stage_index, stage in enumerate(backbone.compaction_stages):
            print(f"\n  --- CompactionStage {stage_index} ---")
            print(f"    Input: {list(current_compact_features.shape)}")
            current_compact_features, current_compact_coordinates, current_compact_mask = stage(
                current_compact_coordinates, current_compact_features,
                current_compact_mask,
            )
            tensor_stats(current_compact_features, f"  stage_{stage_index} output")
            check_token_diversity(current_compact_features, f"  stage_{stage_index}")
            print(f"    Output points: {current_compact_features.shape[2]}")

        backbone_tokens = current_compact_features
        print(f"\n  --- Backbone output tokens ---")
        tensor_stats(backbone_tokens, "backbone_tokens (final)")
        check_token_diversity(backbone_tokens, "backbone_tokens")

        # ---- DECODER ----
        print(f"\n{'='*70}")
        print("DECODER")
        print(f"{'='*70}")

        # Gather ground truth
        masked_true_features, masked_validity = model._gather_tracks(
            features, masked_mask, max_masked,
        )

        decoder = model.decoder
        batch_size = backbone_tokens.shape[0]

        # Project backbone tokens + LayerNorm
        memory = decoder.memory_norm(
            decoder.backbone_projection(backbone_tokens.transpose(1, 2))
        )
        tensor_stats(memory, "decoder memory (projected + normed)")

        # Build queries
        query_indices = torch.arange(max_masked, device=device)
        queries = decoder.query_norm(
            decoder.query_embeddings(query_indices)
        )
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1)
        tensor_stats(queries, "decoder queries (learned + normed)")

        check_token_diversity(
            queries[0].unsqueeze(0).transpose(1, 2), "query_embeddings"
        )

        # Cross-attention
        cross_attention_output, cross_attention_weights = decoder.cross_attention(
            query=queries, key=memory, value=memory
        )
        tensor_stats(cross_attention_output, "cross_attention output")
        num_keys = cross_attention_weights.shape[-1]
        entropy = -(cross_attention_weights * (cross_attention_weights + 1e-8).log()).sum(dim=-1).mean().item()
        uniform_entropy = torch.tensor(float(num_keys)).log().item()
        print(f"  cross_attention_weights: shape={list(cross_attention_weights.shape)}, "
              f"entropy={entropy:.4f} "
              f"(uniform={uniform_entropy:.4f}, "
              f"ratio={entropy / uniform_entropy:.4f})")

        queries = decoder.cross_attention_norm(queries + cross_attention_output)
        tensor_stats(queries, "after cross_attention + layernorm")

        # Output MLP
        predictions = decoder.output_mlp(queries).transpose(1, 2)
        tensor_stats(predictions, "decoder predictions")
        tensor_stats(masked_true_features, "ground truth (masked features)")

        # Check query output diversity
        check_token_diversity(predictions, "prediction")

        # ---- LOSS ANALYSIS ----
        print(f"\n{'='*70}")
        print("LOSS ANALYSIS")
        print(f"{'='*70}")
        track_valid = masked_validity.float()

        error = (predictions - masked_true_features).square() * track_valid
        per_feature_mse = error.sum(dim=2) / num_masked.unsqueeze(1).clamp(min=1.0)
        print("  Per-feature MSE (averaged over events):")
        for feature_index in range(per_feature_mse.shape[1]):
            print(f"    feature[{feature_index}]: "
                  f"{per_feature_mse[:, feature_index].mean().item():.4f}")

        total_loss = error.sum(dim=(1, 2)) / (
            num_masked * features.shape[1]
        ).clamp(min=1.0)
        print(f"\n  Total loss: {total_loss.mean().item():.4f}")

        # Baselines
        zero_error = masked_true_features.square() * track_valid
        zero_loss = zero_error.sum(dim=(1, 2)) / (
            num_masked * features.shape[1]
        ).clamp(min=1.0)
        print(f"  Zero-prediction baseline: {zero_loss.mean().item():.4f}")

        mean_per_feature = (
            (masked_true_features * track_valid).sum(dim=2) /
            num_masked.unsqueeze(1).clamp(min=1.0)
        )
        mean_prediction = mean_per_feature.unsqueeze(2).expand_as(
            masked_true_features
        )
        mean_error = (
            (mean_prediction - masked_true_features).square() * track_valid
        )
        mean_loss = mean_error.sum(dim=(1, 2)) / (
            num_masked * features.shape[1]
        ).clamp(min=1.0)
        print(f"  Mean-prediction baseline: {mean_loss.mean().item():.4f}")

        # Per-feature correlation between predictions and ground truth
        print("\n  Per-feature prediction-GT correlation:")
        for feature_index in range(features.shape[1]):
            pred_feature = predictions[:, feature_index, :][
                track_valid.squeeze(1).bool()
            ]
            gt_feature = masked_true_features[:, feature_index, :][
                track_valid.squeeze(1).bool()
            ]
            if pred_feature.numel() > 1:
                correlation = torch.corrcoef(
                    torch.stack([pred_feature, gt_feature])
                )[0, 1].item()
                print(f"    feature[{feature_index}]: r={correlation:.4f}")

        # Prediction stats
        valid_predictions = predictions[
            track_valid.expand_as(predictions).bool()
        ]
        print(f"\n  Prediction stats (valid positions only):")
        print(f"    mean={valid_predictions.mean().item():.4f}, "
              f"std={valid_predictions.std().item():.4f}, "
              f"abs_mean={valid_predictions.abs().mean().item():.4f}")
        gt_valid = masked_true_features[
            track_valid.expand_as(masked_true_features).bool()
        ]
        print(f"  GT stats (valid positions only):")
        print(f"    mean={gt_valid.mean().item():.4f}, "
              f"std={gt_valid.std().item():.4f}")
        print(f"  Pred/GT std ratio: "
              f"{valid_predictions.std().item() / gt_valid.std().item():.4f}")

    # ---- Optional: Backbone bypass test ----
    if args.test_backbone_bypass:
        loader = DataLoader(dataset, batch_size=args.batch_size, drop_last=True)
        test_backbone_bypass(
            model, loader, data_config, mask_input_index, device,
            num_batches=args.num_batches,
        )

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
