"""Diagnostic script to identify whether encoder or decoder is broken.

Runs a single forward pass through MaskedTrackPretrainer and prints
activation statistics at every stage:
  - Input features (after standardization by weaver)
  - After input embedding + BN
  - After each SetAbstraction stage
  - Backbone output tokens
  - Decoder cross-attention output
  - Decoder final predictions vs ground truth

Usage:
    python diagnose_pretraining.py \
        --data-config data/low-pt/lowpt_tau_pretrain.yaml \
        --data-dir data/low-pt/ \
        --network networks/lowpt_tau_BackbonePretrain.py \
        --batch-size 32 \
        --device cuda:0
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
    # Cosine similarity between all pairs of tokens within each event
    batch_size, channels, num_tokens = tokens.shape
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


def main():
    parser = argparse.ArgumentParser(description='Diagnose pretraining pipeline')
    parser.add_argument('--data-config', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--network', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Optional checkpoint to load (diagnoses trained model)')
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
    print("DIAGNOSTIC: Pretraining Pipeline Analysis")
    print(f"{'='*70}")
    print(f"Batch size: {features.shape[0]}")
    print(f"Sequence length (after trim): {features.shape[2]}")
    print(f"Valid tracks per event: min={mask.sum(dim=2).min().item():.0f}, "
          f"max={mask.sum(dim=2).max().item():.0f}, "
          f"mean={mask.sum(dim=2).mean().item():.0f}")

    # ---- Stage 1: Input data ----
    print(f"\n--- INPUT DATA ---")
    tensor_stats(features, "pf_features (standardized)", mask)
    tensor_stats(lorentz_vectors, "pf_vectors (raw)", mask)
    tensor_stats(points, "pf_points (eta, phi)", mask)

    # Check per-feature stats
    print("\n  Per-feature statistics (masked valid tracks only):")
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
        # ---- Stage 2: Masking ----
        print(f"\n--- MASKING ---")
        visible_mask, masked_mask = model._create_random_mask(mask)
        num_visible = visible_mask.squeeze(1).sum(dim=1)
        num_masked = masked_mask.squeeze(1).sum(dim=1)
        print(f"  Visible per event: mean={num_visible.mean().item():.0f}, "
              f"min={num_visible.min().item():.0f}")
        print(f"  Masked per event: mean={num_masked.mean().item():.0f}, "
              f"min={num_masked.min().item():.0f}")

        # ---- Stage 3: Input embedding ----
        print(f"\n--- BACKBONE: Input Embedding ---")
        visible_features = features * visible_mask.float()
        visible_lorentz_vectors = lorentz_vectors * visible_mask.float()
        visible_points = points * visible_mask.float()

        backbone = model.backbone
        embedded = backbone.input_embedding(visible_features) * visible_mask.float()
        tensor_stats(embedded, "after embed+BN+ReLU", visible_mask)
        check_token_diversity(embedded, "embedded")

        # ---- Stage 4: Each SetAbstraction stage ----
        current_features = embedded
        current_lv = visible_lorentz_vectors
        current_points = visible_points
        current_mask = visible_mask.float()

        for stage_index, stage in enumerate([backbone.stage1, backbone.stage2, backbone.stage3], 1):
            print(f"\n--- BACKBONE: Stage {stage_index} ---")
            current_features, current_lv, current_points, current_mask = stage(
                current_points, current_features, current_lv, current_mask
            )
            tensor_stats(current_features, f"stage{stage_index} features")
            check_token_diversity(current_features, f"stage{stage_index}")
            tensor_stats(current_lv, f"stage{stage_index} lorentz_vectors")

        backbone_tokens = current_features
        print(f"\n--- BACKBONE OUTPUT ---")
        tensor_stats(backbone_tokens, "backbone_tokens (final)")
        check_token_diversity(backbone_tokens, "backbone_tokens")

        # ---- Stage 5: Decoder ----
        print(f"\n--- DECODER ---")
        max_masked = num_masked.max().item()
        masked_coordinates = model._gather_masked_tracks(
            points, masked_mask, max_masked
        )
        masked_true_features = model._gather_masked_tracks(
            features, masked_mask, max_masked
        )

        decoder = model.decoder

        # Project backbone tokens
        memory = decoder.backbone_projection(backbone_tokens.transpose(1, 2))
        tensor_stats(memory, "decoder memory (projected backbone)")

        # Build queries
        batch_size = backbone_tokens.shape[0]
        queries = decoder.mask_token.expand(batch_size, -1, max_masked)
        position_encoding = decoder.positional_encoding(masked_coordinates)
        queries = queries + position_encoding
        queries = queries.transpose(1, 2)  # (B, N_masked, D)
        tensor_stats(queries, "decoder queries (mask_token + pos_enc)")

        # Cross-attention
        cross_attention_output, cross_attention_weights = decoder.cross_attention(
            query=queries, key=memory, value=memory
        )
        tensor_stats(cross_attention_output, "cross_attention output")
        print(f"  cross_attention_weights: shape={list(cross_attention_weights.shape)}, "
              f"entropy={-(cross_attention_weights * (cross_attention_weights + 1e-8).log()).sum(dim=-1).mean().item():.4f} "
              f"(uniform={torch.tensor(64.0).log().item():.4f})")

        queries = decoder.cross_attention_norm(queries + cross_attention_output)
        tensor_stats(queries, "after cross_attention + layernorm")

        # Self-attention layers
        for layer_index, (self_attention, self_attention_norm, feedforward, feedforward_norm) in enumerate(
            zip(decoder.self_attention_layers, decoder.self_attention_norms,
                decoder.feedforward_layers, decoder.feedforward_norms)
        ):
            self_attention_output, _ = self_attention(queries, queries, queries)
            queries = self_attention_norm(queries + self_attention_output)
            feedforward_output = feedforward(queries)
            queries = feedforward_norm(queries + feedforward_output)
            tensor_stats(queries, f"after self_attention_layer[{layer_index}]")

        # Output projection
        predictions = decoder.output_projection(queries).transpose(1, 2)
        tensor_stats(predictions, "decoder predictions")
        tensor_stats(masked_true_features, "ground truth (masked features)")

        # ---- Stage 6: Loss analysis ----
        print(f"\n--- LOSS ANALYSIS ---")
        # Build validity mask for gathered dense tensor
        track_valid = torch.zeros(batch_size, 1, max_masked, device=device)
        for batch_idx in range(batch_size):
            track_valid[batch_idx, :, :num_masked[batch_idx].long()] = 1.0

        error = (predictions - masked_true_features).square() * track_valid
        per_feature_mse = error.sum(dim=2) / num_masked.unsqueeze(1).clamp(min=1.0)
        print("  Per-feature MSE (averaged over events):")
        for feature_index in range(per_feature_mse.shape[1]):
            print(f"    feature[{feature_index}]: {per_feature_mse[:, feature_index].mean().item():.4f}")

        total_loss = error.sum(dim=(1, 2)) / (num_masked * features.shape[1]).clamp(min=1.0)
        print(f"\n  Total loss: {total_loss.mean().item():.4f}")

        # Baseline: what would zero-prediction give?
        zero_error = masked_true_features.square() * track_valid
        zero_loss = zero_error.sum(dim=(1, 2)) / (num_masked * features.shape[1]).clamp(min=1.0)
        print(f"  Zero-prediction baseline: {zero_loss.mean().item():.4f}")

        # Baseline: mean-prediction (predict batch mean per feature)
        mean_per_feature = (masked_true_features * track_valid).sum(dim=2) / num_masked.unsqueeze(1).clamp(min=1.0)
        mean_prediction = mean_per_feature.unsqueeze(2).expand_as(masked_true_features)
        mean_error = (mean_prediction - masked_true_features).square() * track_valid
        mean_loss = mean_error.sum(dim=(1, 2)) / (num_masked * features.shape[1]).clamp(min=1.0)
        print(f"  Mean-prediction baseline: {mean_loss.mean().item():.4f}")

        # Check: are predictions near-zero?
        pred_valid = predictions * track_valid
        print(f"\n  Prediction stats (valid positions only):")
        valid_predictions = predictions[track_valid.expand_as(predictions).bool()]
        print(f"    mean={valid_predictions.mean().item():.4f}, "
              f"std={valid_predictions.std().item():.4f}, "
              f"abs_mean={valid_predictions.abs().mean().item():.4f}")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
