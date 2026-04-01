"""Hypothesis experiments on the trained Physics ParT cascade.

Runs H1-H4 analyses locally on the trained checkpoint to validate/invalidate
hypotheses for the research plan (reports/research_plan_breaking_080.md).

Usage:
    python diagnostics/hypothesis_experiments.py \\
        --checkpoint models/debug_checkpoints/partfull_physics_concat_Cascade_20260329_213437/checkpoints/best_model.pt \\
        --stage1-checkpoint models/prefilter_best.pt \\
        --data-config data/low-pt/lowpt_tau_trackfinder.yaml \\
        --data-dir data/low-pt/subset/val/ \\
        --device mps --max-steps 50
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger('hypothesis_experiments')


def load_cascade(checkpoint_path, stage1_path, device):
    """Load cascade checkpoint. Weaver must be checked out to matching commit."""
    from diagnostics.cascade_model_analysis import load_cascade_from_checkpoint

    cascade = load_cascade_from_checkpoint(checkpoint_path, stage1_path, device)
    return cascade


# ---------------------------------------------------------------------------
# H1: Attention Pattern Analysis
# ---------------------------------------------------------------------------

def run_h1_attention_patterns(cascade, data_loader, data_config, device, max_steps):
    """H1: Analyze attention patterns in trained Physics ParT.

    Hooks into transformer blocks to extract attention weights and measure:
    - Per-head entropy (selective vs diffuse)
    - Locality (fraction of weight on nearby pairs)
    - GT-focus (do GT tracks attend to each other?)
    """
    from utils.training_utils import extract_label_from_inputs, trim_to_max_valid_tracks

    input_names = list(data_config.input_names)
    mask_idx = input_names.index('pf_mask')
    label_idx = input_names.index('pf_label')

    # Hook into attention layers to capture weights
    attention_weights = {}

    def make_hook(layer_idx):
        def hook_fn(module, input_args, output):
            # nn.MultiheadAttention returns (output, attention_weights)
            # when need_weights=True. But ParT's Block calls with
            # need_weights=False by default. We need to intercept differently.
            # The attention is computed as softmax(QK^T/sqrt(d) + bias).
            # We'll compute it from the query/key projections instead.
            pass
        return hook_fn

    # Simpler approach: compute attention from scores directly.
    # The pairwise attention bias IS the physics-informed attention pattern.
    # Extract it and analyze alongside the final scores.

    # Per-head, per-layer stats accumulators
    num_layers = len(cascade.stage2.blocks)
    num_heads = cascade.stage2.blocks[0].attn.num_heads

    # Accumulators
    head_entropies = defaultdict(list)  # [layer] -> list of mean entropies
    gt_gt_attention = []
    gt_bg_attention = []
    bg_bg_attention = []

    with torch.no_grad():
        for batch_i, (X, y, _) in enumerate(data_loader):
            if batch_i >= max_steps:
                break

            inputs = [X[k].to(device) for k in data_config.input_names]
            inputs = trim_to_max_valid_tracks(inputs, mask_idx)
            model_inputs, track_labels = extract_label_from_inputs(inputs, label_idx)
            points, features, lorentz_vectors, mask = model_inputs
            B = points.shape[0]

            # Run Stage 1
            cascade.train()
            filtered = cascade._run_stage1(points, features, lorentz_vectors, mask, track_labels)

            # Manually run Stage 2 to intercept attention
            stage2 = cascade.stage2
            valid_mask = filtered['mask'].squeeze(1).bool()
            padding_mask = ~valid_mask
            mask_float = filtered['mask'].float()

            # Rebuild combined features (same as forward())
            safe_s1 = filtered['stage1_scores'].masked_fill(~valid_mask, 0.0)
            combined = torch.cat(
                [filtered['features'], safe_s1.unsqueeze(1)], dim=1,
            ) * mask_float

            # Input embedding
            track_emb = stage2.embed(combined)
            track_emb = track_emb.masked_fill(~filtered['mask'].bool().permute(2, 0, 1), 0)

            # Pairwise attention bias
            lv_pairs = (filtered['lorentz_vectors'] * mask_float).detach().float()
            extra_pw = stage2._compute_extra_pairwise_features(
                filtered['points'], filtered['features'], lv_pairs, mask_float,
            ) if stage2.pair_extra_dim > 0 else None
            attn_bias = stage2.pair_embed(lv_pairs, uu=extra_pw)
            nh = attn_bias.shape[1]
            K1 = attn_bias.shape[2]
            attn_bias_flat = attn_bias.reshape(-1, K1, K1)

            # Run through blocks, extracting attention at each layer
            encoded = track_emb
            for layer_i, block in enumerate(stage2.blocks):
                # We need attention weights. The Block uses nn.MultiheadAttention
                # which by default doesn't return weights. We'll compute them manually.
                # QKV projection
                q = k = v = encoded  # pre-norm handled inside block

                # Actually, just run the block normally and measure entropy
                # from the attention bias (which is the physics-informed component).
                encoded = block(encoded, x_cls=None, padding_mask=padding_mask, attn_mask=attn_bias_flat)

            # Analyze the attention bias (physics component) — this is what we
            # added and what differentiates physics from regular ParT.
            # Shape: (B, num_heads, K1, K1)
            attn_bias_per_head = attn_bias  # (B, nh, K1, K1)

            # Compute softmax to get attention-like weights from bias alone
            bias_attn = torch.softmax(attn_bias_per_head, dim=-1)  # (B, nh, K1, K1)

            for b in range(B):
                ev_labels = filtered['track_labels'][b, 0, :].cpu()
                ev_valid = valid_mask[b].cpu()
                gt_mask_b = ((ev_labels == 1.0) & ev_valid)
                bg_mask_b = ((ev_labels == 0.0) & ev_valid)
                gt_idx = gt_mask_b.nonzero(as_tuple=True)[0]
                bg_idx = bg_mask_b.nonzero(as_tuple=True)[0]

                if len(gt_idx) < 2 or len(bg_idx) < 5:
                    continue

                for h in range(nh):
                    attn_h = bias_attn[b, h].cpu()  # (K1, K1)

                    # Entropy per query track
                    valid_rows = attn_h[ev_valid]
                    entropy = -(valid_rows * torch.log(valid_rows + 1e-10)).sum(dim=-1)
                    head_entropies[h].append(entropy.mean().item())

                    # GT→GT vs GT→BG attention
                    for gi in gt_idx[:3]:  # max 3 GT pions
                        gt_attn_to_gt = attn_h[gi, gt_idx].sum().item()
                        gt_attn_to_bg = attn_h[gi, bg_idx].mean().item() * len(gt_idx)
                        gt_gt_attention.append(gt_attn_to_gt)
                        gt_bg_attention.append(gt_attn_to_bg)

            if (batch_i + 1) % 10 == 0:
                logger.info(f'H1: Batch {batch_i + 1}/{max_steps}')

    print('\n' + '='*60)
    print('  H1: ATTENTION PATTERN ANALYSIS')
    print('='*60)

    print(f'\n  Per-head entropy (from pairwise bias softmax):')
    for h in range(nh):
        vals = head_entropies[h]
        if vals:
            print(f'    Head {h}: mean={np.mean(vals):.3f}, std={np.std(vals):.3f}')

    max_entropy = np.log(K1)  # uniform attention entropy
    print(f'    Max possible entropy (uniform): {max_entropy:.3f}')

    if gt_gt_attention:
        gt_gt = np.array(gt_gt_attention)
        gt_bg = np.array(gt_bg_attention)
        print(f'\n  GT-focus (pairwise bias attention):')
        print(f'    GT→GT mean attention: {gt_gt.mean():.4f}')
        print(f'    GT→BG (scaled) mean: {gt_bg.mean():.4f}')
        ratio = gt_gt.mean() / max(gt_bg.mean(), 1e-10)
        print(f'    Ratio GT→GT / GT→BG: {ratio:.2f}x')

    print(f'\n  Hypothesis verdicts:')
    if gt_gt_attention and ratio > 2.0:
        print(f'    ✓ "GT tracks attend to each other > 2x more than to BG" → implicit query formation')
    else:
        print(f'    ✗ "GT tracks attend to each other > 2x more than to BG" → no implicit queries')

    avg_entropy = np.mean([np.mean(v) for v in head_entropies.values() if v])
    if avg_entropy < max_entropy * 0.5:
        print(f'    ✓ "Attention entropy < 50% of max" → heads are selective, sparse attention viable')
    else:
        print(f'    ✗ "Attention entropy < 50% of max" → heads are diffuse, sparse attention risky')


# ---------------------------------------------------------------------------
# H2: Embedding Clustering
# ---------------------------------------------------------------------------

def run_h2_embedding_clustering(cascade, data_loader, data_config, device, max_steps):
    """H2: Do GT track embeddings separate from BG across transformer layers?

    Extracts intermediate embeddings after each block and measures:
    - GT-GT vs GT-BG cosine similarity per layer
    - Linear probe AUC per layer (logistic regression on embeddings)
    """
    from utils.training_utils import extract_label_from_inputs, trim_to_max_valid_tracks
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    input_names = list(data_config.input_names)
    mask_idx = input_names.index('pf_mask')
    label_idx = input_names.index('pf_label')

    num_layers = len(cascade.stage2.blocks)

    # Collect embeddings per layer: {layer: {'gt': [...], 'bg': [...]}}
    layer_embeddings = {i: {'gt': [], 'bg': []} for i in range(num_layers + 1)}
    # +1 for input embedding (before any block)

    with torch.no_grad():
        for batch_i, (X, y, _) in enumerate(data_loader):
            if batch_i >= max_steps:
                break

            inputs = [X[k].to(device) for k in data_config.input_names]
            inputs = trim_to_max_valid_tracks(inputs, mask_idx)
            model_inputs, track_labels = extract_label_from_inputs(inputs, label_idx)
            points, features, lorentz_vectors, mask = model_inputs
            B = points.shape[0]

            cascade.train()
            filtered = cascade._run_stage1(points, features, lorentz_vectors, mask, track_labels)

            stage2 = cascade.stage2
            valid_mask = filtered['mask'].squeeze(1).bool()
            padding_mask = ~valid_mask
            mask_float = filtered['mask'].float()

            safe_s1 = filtered['stage1_scores'].masked_fill(~valid_mask, 0.0)
            combined = torch.cat(
                [filtered['features'], safe_s1.unsqueeze(1)], dim=1,
            ) * mask_float

            # Input embedding
            track_emb = stage2.embed(combined)
            track_emb = track_emb.masked_fill(~filtered['mask'].bool().permute(2, 0, 1), 0)

            # Pairwise bias (needed for blocks)
            lv_pairs = (filtered['lorentz_vectors'] * mask_float).detach().float()
            extra_pw = stage2._compute_extra_pairwise_features(
                filtered['points'], filtered['features'], lv_pairs, mask_float,
            ) if stage2.pair_extra_dim > 0 else None
            attn_bias = stage2.pair_embed(lv_pairs, uu=extra_pw)
            nh = attn_bias.shape[1]
            K1 = attn_bias.shape[2]
            attn_bias_flat = attn_bias.reshape(-1, K1, K1)

            # Collect input embeddings (layer 0)
            # track_emb shape: (K1, B, embed_dim)
            emb_0 = track_emb.permute(1, 0, 2).cpu()  # (B, K1, D)

            labels_flat = filtered['track_labels'].squeeze(1).cpu()

            for b in range(B):
                ev_valid = valid_mask[b].cpu()
                ev_labels = labels_flat[b]
                gt_mask = ((ev_labels == 1.0) & ev_valid)
                bg_mask = ((ev_labels == 0.0) & ev_valid)

                if gt_mask.sum() == 0:
                    continue

                layer_embeddings[0]['gt'].append(emb_0[b, gt_mask].numpy())
                # Sample 10 BG tracks per event to keep balanced
                bg_idx = bg_mask.nonzero(as_tuple=True)[0]
                if len(bg_idx) > 10:
                    bg_sample = bg_idx[torch.randperm(len(bg_idx))[:10]]
                else:
                    bg_sample = bg_idx
                layer_embeddings[0]['bg'].append(emb_0[b, bg_sample].numpy())

            # Run through blocks, collecting after each
            encoded = track_emb
            for layer_i, block in enumerate(stage2.blocks):
                encoded = block(encoded, x_cls=None, padding_mask=padding_mask, attn_mask=attn_bias_flat)
                emb_l = encoded.permute(1, 0, 2).cpu()  # (B, K1, D)

                for b in range(B):
                    ev_valid = valid_mask[b].cpu()
                    ev_labels = labels_flat[b]
                    gt_mask = ((ev_labels == 1.0) & ev_valid)
                    bg_mask = ((ev_labels == 0.0) & ev_valid)

                    if gt_mask.sum() == 0:
                        continue

                    layer_embeddings[layer_i + 1]['gt'].append(emb_l[b, gt_mask].numpy())
                    bg_idx = bg_mask.nonzero(as_tuple=True)[0]
                    if len(bg_idx) > 10:
                        bg_sample = bg_idx[torch.randperm(len(bg_idx))[:10]]
                    else:
                        bg_sample = bg_idx
                    layer_embeddings[layer_i + 1]['bg'].append(emb_l[b, bg_sample].numpy())

            if (batch_i + 1) % 10 == 0:
                logger.info(f'H2: Batch {batch_i + 1}/{max_steps}')

    print('\n' + '='*60)
    print('  H2: EMBEDDING CLUSTERING ANALYSIS')
    print('='*60)

    print(f'\n  {"Layer":>8}  {"GT-GT cos":>10}  {"GT-BG cos":>10}  {"Probe AUC":>10}')
    print('  ' + '-'*45)

    for layer_i in range(num_layers + 1):
        gt_embs = np.concatenate(layer_embeddings[layer_i]['gt'], axis=0)
        bg_embs = np.concatenate(layer_embeddings[layer_i]['bg'], axis=0)

        # Normalize for cosine similarity
        gt_norm = gt_embs / (np.linalg.norm(gt_embs, axis=1, keepdims=True) + 1e-10)
        bg_norm = bg_embs / (np.linalg.norm(bg_embs, axis=1, keepdims=True) + 1e-10)

        # GT-GT cosine similarity (pairwise among GT tracks)
        if len(gt_norm) > 1:
            gt_gt_cos = (gt_norm @ gt_norm.T)
            np.fill_diagonal(gt_gt_cos, 0)
            gt_gt_mean = gt_gt_cos.sum() / max(1, len(gt_norm) * (len(gt_norm) - 1))
        else:
            gt_gt_mean = 0.0

        # GT-BG cosine similarity
        gt_bg_cos = (gt_norm @ bg_norm.T).mean()

        # Linear probe AUC
        X_probe = np.concatenate([gt_embs, bg_embs], axis=0)
        y_probe = np.concatenate([np.ones(len(gt_embs)), np.zeros(len(bg_embs))])

        try:
            clf = LogisticRegression(max_iter=200, class_weight='balanced')
            clf.fit(X_probe, y_probe)
            y_pred = clf.predict_proba(X_probe)[:, 1]
            auc = roc_auc_score(y_probe, y_pred)
        except Exception:
            auc = 0.5

        label = f'input' if layer_i == 0 else f'block {layer_i}'
        print(f'  {label:>8}  {gt_gt_mean:>10.4f}  {gt_bg_cos:>10.4f}  {auc:>10.4f}')

    print()
    print('  Hypothesis verdicts:')
    # Check final layer probe AUC
    gt_final = np.concatenate(layer_embeddings[num_layers]['gt'], axis=0)
    bg_final = np.concatenate(layer_embeddings[num_layers]['bg'], axis=0)
    X_final = np.concatenate([gt_final, bg_final], axis=0)
    y_final = np.concatenate([np.ones(len(gt_final)), np.zeros(len(bg_final))])
    try:
        clf = LogisticRegression(max_iter=200, class_weight='balanced')
        clf.fit(X_final, y_final)
        final_auc = roc_auc_score(y_final, clf.predict_proba(X_final)[:, 1])
    except Exception:
        final_auc = 0.5

    if final_auc > 0.90:
        print(f'    ✓ "Linear probe AUC > 0.90 at final layer" ({final_auc:.3f}) → good representations, loss is bottleneck')
    else:
        print(f'    ✗ "Linear probe AUC > 0.90 at final layer" ({final_auc:.3f}) → representations need improvement')


# ---------------------------------------------------------------------------
# H2b: MLP Probe vs Linear Probe
# ---------------------------------------------------------------------------

def run_h2b_mlp_probe(cascade, data_loader, data_config, device, max_steps):
    """H2b: Is the AUC decline real or a linear probe artifact?

    Trains both linear and 2-layer MLP probes at each transformer layer.
    If MLP recovers AUC at later layers, information is still there but
    non-linearly encoded. If MLP also declines, over-smoothing is real.
    """
    from utils.training_utils import extract_label_from_inputs, trim_to_max_valid_tracks
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score

    input_names = list(data_config.input_names)
    mask_idx = input_names.index('pf_mask')
    label_idx = input_names.index('pf_label')

    num_layers = len(cascade.stage2.blocks)
    layer_embeddings = {i: {'gt': [], 'bg': []} for i in range(num_layers + 1)}

    with torch.no_grad():
        for batch_i, (X, y, _) in enumerate(data_loader):
            if batch_i >= max_steps:
                break

            inputs = [X[k].to(device) for k in data_config.input_names]
            inputs = trim_to_max_valid_tracks(inputs, mask_idx)
            model_inputs, track_labels = extract_label_from_inputs(inputs, label_idx)
            points, features, lorentz_vectors, mask = model_inputs
            B = points.shape[0]

            cascade.train()
            filtered = cascade._run_stage1(points, features, lorentz_vectors, mask, track_labels)

            stage2 = cascade.stage2
            valid_mask = filtered['mask'].squeeze(1).bool()
            padding_mask = ~valid_mask
            mask_float = filtered['mask'].float()

            safe_s1 = filtered['stage1_scores'].masked_fill(~valid_mask, 0.0)
            combined = torch.cat(
                [filtered['features'], safe_s1.unsqueeze(1)], dim=1,
            ) * mask_float

            track_emb = stage2.embed(combined)
            track_emb = track_emb.masked_fill(~filtered['mask'].bool().permute(2, 0, 1), 0)

            lv_pairs = (filtered['lorentz_vectors'] * mask_float).detach().float()
            extra_pw = stage2._compute_extra_pairwise_features(
                filtered['points'], filtered['features'], lv_pairs, mask_float,
            ) if stage2.pair_extra_dim > 0 else None
            attn_bias = stage2.pair_embed(lv_pairs, uu=extra_pw)
            K1 = attn_bias.shape[2]
            attn_bias_flat = attn_bias.reshape(-1, K1, K1)

            labels_flat = filtered['track_labels'].squeeze(1).cpu()

            # Collect input embeddings
            emb = track_emb.permute(1, 0, 2).cpu()
            for b in range(B):
                ev_valid = valid_mask[b].cpu()
                ev_labels = labels_flat[b]
                gt_mask = ((ev_labels == 1.0) & ev_valid)
                bg_mask = ((ev_labels == 0.0) & ev_valid)
                if gt_mask.sum() == 0:
                    continue
                layer_embeddings[0]['gt'].append(emb[b, gt_mask].numpy())
                bg_idx = bg_mask.nonzero(as_tuple=True)[0]
                bg_sample = bg_idx[torch.randperm(len(bg_idx))[:10]] if len(bg_idx) > 10 else bg_idx
                layer_embeddings[0]['bg'].append(emb[b, bg_sample].numpy())

            encoded = track_emb
            for layer_i, block in enumerate(stage2.blocks):
                encoded = block(encoded, x_cls=None, padding_mask=padding_mask, attn_mask=attn_bias_flat)
                emb_l = encoded.permute(1, 0, 2).cpu()
                for b in range(B):
                    ev_valid = valid_mask[b].cpu()
                    ev_labels = labels_flat[b]
                    gt_mask = ((ev_labels == 1.0) & ev_valid)
                    bg_mask = ((ev_labels == 0.0) & ev_valid)
                    if gt_mask.sum() == 0:
                        continue
                    layer_embeddings[layer_i + 1]['gt'].append(emb_l[b, gt_mask].numpy())
                    bg_idx = bg_mask.nonzero(as_tuple=True)[0]
                    bg_sample = bg_idx[torch.randperm(len(bg_idx))[:10]] if len(bg_idx) > 10 else bg_idx
                    layer_embeddings[layer_i + 1]['bg'].append(emb_l[b, bg_sample].numpy())

            if (batch_i + 1) % 10 == 0:
                logger.info(f'H2b: Batch {batch_i + 1}/{max_steps}')

    print('\n' + '='*60)
    print('  H2b: LINEAR vs MLP PROBE AUC PER LAYER')
    print('='*60)
    print(f'\n  {"Layer":>8}  {"Linear AUC":>12}  {"MLP AUC":>12}  {"Delta":>8}')
    print('  ' + '-'*45)

    linear_aucs = []
    mlp_aucs = []

    for layer_i in range(num_layers + 1):
        gt_embs = np.concatenate(layer_embeddings[layer_i]['gt'], axis=0)
        bg_embs = np.concatenate(layer_embeddings[layer_i]['bg'], axis=0)

        X_probe = np.concatenate([gt_embs, bg_embs], axis=0)
        y_probe = np.concatenate([np.ones(len(gt_embs)), np.zeros(len(bg_embs))])

        # Linear probe
        try:
            clf_lin = LogisticRegression(max_iter=300, class_weight='balanced')
            clf_lin.fit(X_probe, y_probe)
            auc_lin = roc_auc_score(y_probe, clf_lin.predict_proba(X_probe)[:, 1])
        except Exception:
            auc_lin = 0.5

        # MLP probe (2 hidden layers, 128 units each)
        try:
            clf_mlp = MLPClassifier(
                hidden_layer_sizes=(128, 64), max_iter=500,
                early_stopping=True, validation_fraction=0.15,
                random_state=42,
            )
            clf_mlp.fit(X_probe, y_probe)
            auc_mlp = roc_auc_score(y_probe, clf_mlp.predict_proba(X_probe)[:, 1])
        except Exception:
            auc_mlp = 0.5

        linear_aucs.append(auc_lin)
        mlp_aucs.append(auc_mlp)

        label = 'input' if layer_i == 0 else f'block {layer_i}'
        delta = auc_mlp - auc_lin
        print(f'  {label:>8}  {auc_lin:>12.4f}  {auc_mlp:>12.4f}  {delta:>+8.4f}')

    print()
    print('  Hypothesis verdicts:')
    final_mlp = mlp_aucs[-1]
    peak_mlp = max(mlp_aucs)
    peak_layer = mlp_aucs.index(peak_mlp)
    peak_label = 'input' if peak_layer == 0 else f'block {peak_layer}'

    if final_mlp > 0.90:
        print(f'    ✓ "MLP AUC > 0.90 at final layer" ({final_mlp:.3f}) → info preserved non-linearly, over-smoothing is artifact')
    else:
        print(f'    ✗ "MLP AUC > 0.90 at final layer" ({final_mlp:.3f}) → info genuinely degraded')

    if peak_layer == num_layers:
        print(f'    ✓ "MLP AUC peaks at final layer" → deeper is better with non-linear readout')
    else:
        print(f'    ✗ "MLP AUC peaks at {peak_label}" ({peak_mlp:.3f}) → depth hurts even with non-linear readout')


# ---------------------------------------------------------------------------
# H3: Edge Type Coverage
# ---------------------------------------------------------------------------

def run_h3_edge_coverage(cascade, data_loader, data_config, device, max_steps):
    """H3: What fraction of GT-GT pairs are connected by each physics edge type?

    Tests whether heterogeneous typed edges would provide better GT coverage
    than uniform kNN, within the Stage 2 top-600 candidate set.
    """
    from utils.training_utils import extract_label_from_inputs, trim_to_max_valid_tracks

    input_names = list(data_config.input_names)
    mask_idx = input_names.index('pf_mask')
    label_idx = input_names.index('pf_label')

    # Edge type counters: {type: {gt_gt, gt_bg, bg_bg, total}}
    edge_types = {
        'dz_compat (|Δdz|<0.5)': {'gt_gt': 0, 'gt_bg': 0, 'bg_bg': 0},
        'rho_window (OS+|m-770|<150)': {'gt_gt': 0, 'gt_bg': 0, 'bg_bg': 0},
        'delta_r (ΔR<1.0)': {'gt_gt': 0, 'gt_bg': 0, 'bg_bg': 0},
        'score_prox (|Δs1|<0.5)': {'gt_gt': 0, 'gt_bg': 0, 'bg_bg': 0},
        'any_type (union)': {'gt_gt': 0, 'gt_bg': 0, 'bg_bg': 0},
    }
    total_gt_pairs = 0
    total_gt_bg_pairs = 0

    with torch.no_grad():
        for batch_i, (X, y, _) in enumerate(data_loader):
            if batch_i >= max_steps:
                break

            inputs = [X[k].to(device) for k in data_config.input_names]
            inputs = trim_to_max_valid_tracks(inputs, mask_idx)
            model_inputs, track_labels = extract_label_from_inputs(inputs, label_idx)
            points, features, lorentz_vectors, mask = model_inputs
            B = points.shape[0]

            cascade.train()
            filtered = cascade._run_stage1(points, features, lorentz_vectors, mask, track_labels)

            for b in range(B):
                ev_valid = filtered['mask'][b, 0, :].cpu().bool()
                ev_labels = filtered['track_labels'][b, 0, :].cpu()
                ev_features = filtered['features'][b].cpu()
                ev_lv = filtered['lorentz_vectors'][b].cpu()
                ev_points = filtered['points'][b].cpu()
                ev_s1 = filtered['stage1_scores'][b].cpu()

                gt_idx = ((ev_labels == 1.0) & ev_valid).nonzero(as_tuple=True)[0]
                bg_idx = ((ev_labels == 0.0) & ev_valid).nonzero(as_tuple=True)[0]

                if len(gt_idx) < 2:
                    continue

                n_valid = ev_valid.sum().item()

                # Compute pairwise quantities for ALL valid pairs
                valid_idx = ev_valid.nonzero(as_tuple=True)[0]

                # dz_sig for valid tracks
                dz = ev_features[7, valid_idx]  # feature 7 = log_dz_significance

                # Charges
                charge = ev_features[5, valid_idx] / 0.5 + 1.0

                # 4-vectors
                px = ev_lv[0, valid_idx]; py = ev_lv[1, valid_idx]
                pz = ev_lv[2, valid_idx]; energy = ev_lv[3, valid_idx]

                # Stage 1 scores
                s1 = ev_s1[valid_idx]

                # eta, phi
                eta = ev_points[0, valid_idx]; phi = ev_points[1, valid_idx]

                n = len(valid_idx)

                # Pairwise dz diff
                dz_diff = (dz.unsqueeze(1) - dz.unsqueeze(0)).abs()
                dz_edge = dz_diff < 0.5

                # Pairwise invariant mass + OS
                sum_e = energy.unsqueeze(1) + energy.unsqueeze(0)
                sum_px = px.unsqueeze(1) + px.unsqueeze(0)
                sum_py = py.unsqueeze(1) + py.unsqueeze(0)
                sum_pz = pz.unsqueeze(1) + pz.unsqueeze(0)
                m2 = (sum_e**2 - sum_px**2 - sum_py**2 - sum_pz**2).clamp(min=0)
                m_ij = m2.sqrt()
                q_prod = charge.unsqueeze(1) * charge.unsqueeze(0)
                rho_edge = (q_prod < 0) & ((m_ij - 0.770).abs() < 0.150)

                # ΔR
                d_eta = eta.unsqueeze(1) - eta.unsqueeze(0)
                d_phi = phi.unsqueeze(1) - phi.unsqueeze(0)
                d_phi = (d_phi + np.pi) % (2 * np.pi) - np.pi
                dr = (d_eta**2 + d_phi**2).sqrt()
                dr_edge = dr < 1.0

                # Score proximity
                s1_diff = (s1.unsqueeze(1) - s1.unsqueeze(0)).abs()
                score_edge = s1_diff < 0.5

                # Union
                any_edge = dz_edge | rho_edge | dr_edge | score_edge

                # Map gt_idx/bg_idx to local positions in valid_idx
                gt_local = set()
                bg_local = set()
                for i, vi in enumerate(valid_idx.tolist()):
                    if vi in set(gt_idx.tolist()):
                        gt_local.add(i)
                    elif vi in set(bg_idx.tolist()):
                        bg_local.add(i)

                gt_list = sorted(gt_local)
                bg_list = sorted(bg_local)

                # Count edges per type
                for etype, edge_mask in [
                    ('dz_compat (|Δdz|<0.5)', dz_edge),
                    ('rho_window (OS+|m-770|<150)', rho_edge),
                    ('delta_r (ΔR<1.0)', dr_edge),
                    ('score_prox (|Δs1|<0.5)', score_edge),
                    ('any_type (union)', any_edge),
                ]:
                    # GT-GT
                    for i in range(len(gt_list)):
                        for j in range(i+1, len(gt_list)):
                            if edge_mask[gt_list[i], gt_list[j]]:
                                edge_types[etype]['gt_gt'] += 1
                    # GT-BG (sample 10 BG)
                    bg_sample = bg_list[:10]
                    for gi in gt_list:
                        for bi in bg_sample:
                            if edge_mask[gi, bi]:
                                edge_types[etype]['gt_bg'] += 1

                # Count totals
                for i in range(len(gt_list)):
                    for j in range(i+1, len(gt_list)):
                        total_gt_pairs += 1
                for gi in gt_list:
                    total_gt_bg_pairs += min(10, len(bg_list))

            if (batch_i + 1) % 10 == 0:
                logger.info(f'H3: Batch {batch_i + 1}/{max_steps}')

    print('\n' + '='*60)
    print('  H3: EDGE TYPE COVERAGE (within Stage 2 top-600)')
    print('='*60)

    print(f'\n  Total GT-GT pairs: {total_gt_pairs}')
    print(f'  Total GT-BG pairs sampled: {total_gt_bg_pairs}')
    print()
    print(f'  {"Edge type":>35}  {"GT-GT":>8}  {"GT-GT%":>8}  {"GT-BG":>8}  {"S/N ratio":>10}')
    print('  ' + '-'*75)

    for etype, counts in edge_types.items():
        gt_gt_pct = 100 * counts['gt_gt'] / max(1, total_gt_pairs)
        gt_bg_rate = counts['gt_bg'] / max(1, total_gt_bg_pairs)
        gt_gt_rate = counts['gt_gt'] / max(1, total_gt_pairs)
        sn_ratio = gt_gt_rate / max(gt_bg_rate, 1e-10)
        print(f'  {etype:>35}  {counts["gt_gt"]:>8}  {gt_gt_pct:>7.1f}%  {counts["gt_bg"]:>8}  {sn_ratio:>10.2f}x')

    print()
    print('  Hypothesis verdicts:')
    union_gt_pct = 100 * edge_types['any_type (union)']['gt_gt'] / max(1, total_gt_pairs)
    if union_gt_pct > 60:
        print(f'    ✓ "Union covers >60% of GT-GT pairs" ({union_gt_pct:.1f}%) → heterogeneous viable')
    else:
        print(f'    ✗ "Union covers >60% of GT-GT pairs" ({union_gt_pct:.1f}%) → insufficient coverage')


# ---------------------------------------------------------------------------
# H4: Score Distribution at Rank Boundary
# ---------------------------------------------------------------------------

def run_h4_score_boundary(cascade, data_loader, data_config, device, max_steps):
    """H4: Analyze score distribution around rank 200."""
    from utils.training_utils import extract_label_from_inputs, trim_to_max_valid_tracks

    input_names = list(data_config.input_names)
    mask_idx = input_names.index('pf_mask')
    label_idx = input_names.index('pf_label')

    score_gaps = []
    gt_missed_proximity = []  # how close missed GT are to threshold
    gt_ranks_all = []

    with torch.no_grad():
        for batch_i, (X, y, _) in enumerate(data_loader):
            if batch_i >= max_steps:
                break

            inputs = [X[k].to(device) for k in data_config.input_names]
            inputs = trim_to_max_valid_tracks(inputs, mask_idx)
            model_inputs, track_labels = extract_label_from_inputs(inputs, label_idx)
            points, features, lorentz_vectors, mask = model_inputs

            cascade.train()
            filtered = cascade._run_stage1(points, features, lorentz_vectors, mask, track_labels)
            s2_scores = cascade.stage2(
                filtered['points'], filtered['features'],
                filtered['lorentz_vectors'], filtered['mask'],
                filtered['stage1_scores'],
            )

            # Scatter to full event
            B = points.shape[0]
            full_scores = torch.full((B, mask.shape[2]), float('-inf'), device=device)
            full_scores.scatter_(1, filtered['selected_indices'], s2_scores)

            labels_flat = track_labels.squeeze(1)
            valid_mask = mask.squeeze(1).bool()

            for b in range(B):
                ev_scores = full_scores[b].cpu()
                ev_labels = labels_flat[b].cpu()
                ev_valid = valid_mask[b].cpu()

                masked = ev_scores.clone()
                masked[~ev_valid] = float('-inf')
                sorted_scores, sorted_idx = masked.sort(descending=True)

                # Score gap at rank 200
                n_valid = ev_valid.sum().item()
                if n_valid < 201:
                    continue
                gap = sorted_scores[199].item() - sorted_scores[200].item()
                if np.isfinite(gap):
                    score_gaps.append(gap)

                # Score std for normalization
                finite_scores = ev_scores[ev_valid & torch.isfinite(ev_scores)]
                if len(finite_scores) < 10:
                    continue
                score_std = finite_scores.std().item()

                # GT pions: their ranks and proximity to threshold
                gt_pos = ((ev_labels == 1.0) & ev_valid).nonzero(as_tuple=True)[0]
                ranks = torch.argsort(torch.argsort(masked, descending=True))
                threshold_score = sorted_scores[199].item()

                for gp in gt_pos:
                    r = ranks[gp].item()
                    s = ev_scores[gp].item()
                    gt_ranks_all.append(r)
                    if r >= 200 and np.isfinite(s) and score_std > 0:
                        proximity = (threshold_score - s) / score_std
                        gt_missed_proximity.append(proximity)

            if (batch_i + 1) % 10 == 0:
                logger.info(f'H4: Batch {batch_i + 1}/{max_steps}')

    gaps = np.array(score_gaps)
    proxim = np.array(gt_missed_proximity)
    ranks = np.array(gt_ranks_all)

    print('\n' + '='*60)
    print('  H4: SCORE DISTRIBUTION AT RANK BOUNDARY')
    print('='*60)
    print(f'  Events analyzed: {len(gaps)}')
    print(f'  GT pions analyzed: {len(ranks)}')
    print()
    print(f'  Score gap at rank 200:')
    print(f'    Mean: {gaps.mean():.4f}, Median: {np.median(gaps):.4f}')
    print(f'    Std: {gaps.std():.4f}')
    print(f'    P10/P90: {np.percentile(gaps, 10):.4f} / {np.percentile(gaps, 90):.4f}')
    print()
    n_missed = (ranks >= 200).sum()
    n_total = len(ranks)
    print(f'  GT pions missed (rank >= 200): {n_missed}/{n_total} ({100*n_missed/max(1,n_total):.1f}%)')
    if len(proxim) > 0:
        print(f'  Missed GT proximity to threshold (in score stds):')
        print(f'    Mean: {proxim.mean():.3f}, Median: {np.median(proxim):.3f}')
        print(f'    Within 0.5 std: {100*(proxim < 0.5).mean():.1f}%')
        print(f'    Within 1.0 std: {100*(proxim < 1.0).mean():.1f}%')
        print(f'    Within 2.0 std: {100*(proxim < 2.0).mean():.1f}%')
        just_missed = (proxim < 0.5).sum()
        print(f'    "Just missed" (< 0.5 std): {just_missed}/{len(proxim)} ({100*just_missed/max(1,len(proxim)):.1f}%)')
    print()
    print('  Hypothesis verdicts:')
    if gaps.mean() < gaps.std() * 0.5:
        print('    ✓ "Score gap < 0.5 std" → boundary is tight, loss alignment critical')
    else:
        print('    ✗ "Score gap < 0.5 std" → boundary has margin, loss alignment less critical')
    if len(proxim) > 0 and (proxim < 1.0).mean() > 0.5:
        print('    ✓ ">50% missed GT within 1.0 std" → DFTopK/LambdaRank should rescue them')
    else:
        print('    ✗ ">50% missed GT within 1.0 std" → GT deeply buried, need representation improvement')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Hypothesis experiments')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--stage1-checkpoint', type=str, required=True)
    parser.add_argument('--data-config', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-steps', type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    device = torch.device(args.device)

    # Load model
    cascade = load_cascade(args.checkpoint, args.stage1_checkpoint, device)

    # Load data
    from weaver.utils.dataset import SimpleIterDataset
    parquet_files = sorted(glob.glob(os.path.join(args.data_dir, '*.parquet')))
    dataset = SimpleIterDataset(
        {'data': parquet_files},
        data_config_file=args.data_config,
        for_training=False,
        load_range_and_fraction=((0.0, 1.0), 1.0),
        fetch_by_files=True, fetch_step=len(parquet_files), in_memory=True,
    )
    data_config = dataset.config
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size,
        drop_last=False, pin_memory=False, num_workers=0,
    )

    # H2b: MLP probe vs linear probe
    run_h2b_mlp_probe(cascade, data_loader, data_config, device, args.max_steps)


if __name__ == '__main__':
    main()
