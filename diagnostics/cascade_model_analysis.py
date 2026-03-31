"""Comprehensive cascade model diagnostic analysis.

Loads two cascade checkpoints (regular ParT vs physics-pairwise ParT),
runs inference on validation data, and produces an exhaustive markdown
report comparing their performance across multiple dimensions.

Usage:
    python diagnostics/cascade_model_analysis.py \\
        --regular-checkpoint models/debug_checkpoints/part_large_Cascade_20260330_061319/checkpoints/best_model.pt \\
        --physics-checkpoint models/debug_checkpoints/partfull_physics_concat_Cascade_20260329_213437/checkpoints/best_model.pt \\
        --stage1-checkpoint models/prefilter_best.pt \\
        --data-config data/low-pt/lowpt_tau_trackfinder.yaml \\
        --data-dir data/low-pt/val/ \\
        --output reports/cascade_model_analysis.md \\
        --device mps
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger('cascade_analysis')


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_cascade_from_checkpoint(
    checkpoint_path: str,
    stage1_checkpoint_path: str,
    device: torch.device,
) -> torch.nn.Module:
    """Load a CascadeModel from checkpoint."""
    from weaver.nn.model.CascadeModel import CascadeModel
    from weaver.nn.model.CascadeReranker import CascadeReranker
    from weaver.nn.model.TrackPreFilter import TrackPreFilter

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False,
    )
    args = checkpoint.get('args', {})

    # Stage 1
    stage1_ckpt = torch.load(
        stage1_checkpoint_path, map_location=device, weights_only=False,
    )
    stage1 = TrackPreFilter(
        mode='mlp', input_dim=16, hidden_dim=192,
        num_message_rounds=2, num_neighbors=16,
    )
    stage1.load_state_dict(stage1_ckpt['model_state_dict'])

    # Stage 2
    pair_embed_dims_str = args.get('stage2_pair_embed_dims', '64,64')
    if isinstance(pair_embed_dims_str, str):
        pair_embed_dims = [int(x) for x in pair_embed_dims_str.split(',')]
    else:
        pair_embed_dims = pair_embed_dims_str

    stage2 = CascadeReranker(
        input_dim=16,
        embed_dim=args.get('stage2_embed_dim', 128),
        num_heads=args.get('stage2_num_heads', 4),
        num_layers=args.get('stage2_num_layers', 3),
        pair_input_dim=4,
        pair_extra_dim=args.get('stage2_pair_extra_dim', 0) or 0,
        pair_embed_dims=pair_embed_dims,
        pair_embed_mode=args.get('stage2_pair_embed_mode', 'sum') or 'sum',
        ffn_ratio=args.get('stage2_ffn_ratio', 4),
        dropout=args.get('stage2_dropout', 0.1),
    )

    cascade = CascadeModel(
        stage1=stage1, stage2=stage2,
        top_k1=args.get('top_k1', 600),
    )
    cascade.load_state_dict(checkpoint['model_state_dict'])
    cascade = cascade.to(device)

    epoch = checkpoint.get('epoch', '?')
    r200 = checkpoint.get('best_val_recall_at_200', 0)
    logger.info(
        f'Loaded cascade: {checkpoint_path} '
        f'(epoch {epoch}, R@200={r200:.4f}, '
        f'pair_extra_dim={args.get("stage2_pair_extra_dim", 0)})'
    )
    return cascade


def load_stage1_model(
    checkpoint_path: str,
    device: torch.device,
) -> torch.nn.Module:
    """Load standalone Stage 1 for baseline comparison."""
    from weaver.nn.model.TrackPreFilter import TrackPreFilter

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False,
    )
    model = TrackPreFilter(
        mode='mlp', input_dim=16, hidden_dim=192,
        num_message_rounds=2, num_neighbors=16,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.train()  # BatchNorm batch stats (stale running stats)
    return model


# ---------------------------------------------------------------------------
# Per-event data collection
# ---------------------------------------------------------------------------

class PerPionCollector:
    """Collects per-GT-pion data across all batches for later analysis."""

    def __init__(self):
        # Per GT pion: list of dicts with pT, dxy_sig, charge, rank, found, role
        self.pions: list[dict] = []
        # Per event: list of dicts with n_gt, n_found, gt_pion_indices
        self.events: list[dict] = []

    def update(
        self,
        full_scores: torch.Tensor,
        stage1_scores: torch.Tensor,
        track_labels: torch.Tensor,
        mask: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        features: torch.Tensor,
        top_k: int = 200,
    ) -> None:
        """Collect per-pion data from one batch."""
        batch_size = full_scores.shape[0]
        labels_flat = track_labels.squeeze(1)
        valid_mask = mask.squeeze(1).bool()

        # Rank lookup for cascade scores
        masked_scores = full_scores.clone()
        masked_scores[~valid_mask] = float('-inf')
        rank_lookup = torch.argsort(
            torch.argsort(masked_scores, dim=1, descending=True), dim=1,
        )

        # Rank lookup for stage1 scores
        s1_masked = stage1_scores.clone()
        s1_masked[~valid_mask] = float('-inf')
        s1_rank_lookup = torch.argsort(
            torch.argsort(s1_masked, dim=1, descending=True), dim=1,
        )

        for batch_index in range(batch_size):
            event_labels = labels_flat[batch_index]
            event_valid = valid_mask[batch_index]
            gt_positions = (
                (event_labels == 1.0) & event_valid
            ).nonzero(as_tuple=True)[0]

            if len(gt_positions) == 0:
                continue

            # Compute pT from lorentz vectors (px, py)
            px = lorentz_vectors[batch_index, 0, :]
            py = lorentz_vectors[batch_index, 1, :]
            pt_all = torch.sqrt(px ** 2 + py ** 2)

            # Get per-pion properties
            gt_pts = pt_all[gt_positions].cpu().numpy()
            gt_dxy = features[batch_index, 6, gt_positions].abs().cpu().numpy()
            gt_charge_std = features[batch_index, 5, gt_positions].cpu().numpy()
            gt_charge = gt_charge_std / 0.5 + 1.0  # recover raw

            gt_ranks = rank_lookup[batch_index, gt_positions].cpu().numpy()
            gt_s1_ranks = s1_rank_lookup[batch_index, gt_positions].cpu().numpy()
            gt_found = (gt_ranks < top_k).astype(int)

            # Sort by pT to assign roles: highest, middle, lowest
            pt_order = np.argsort(-gt_pts)  # descending
            roles = [''] * len(gt_positions)
            if len(gt_positions) == 3:
                roles[pt_order[0]] = 'highest_pt'
                roles[pt_order[1]] = 'middle_pt'
                roles[pt_order[2]] = 'lowest_pt'

            event_id = len(self.events)
            pion_start = len(self.pions)
            for i, gt_pos in enumerate(gt_positions):
                self.pions.append({
                    'event_id': event_id,
                    'pt': gt_pts[i],
                    'dxy_sig': gt_dxy[i],
                    'charge': gt_charge[i],
                    'cascade_rank': gt_ranks[i],
                    'stage1_rank': gt_s1_ranks[i],
                    'found_at_200': gt_found[i],
                    'role': roles[i],
                    'in_stage1_top600': int(gt_s1_ranks[i] < 600),
                })

            n_found = int(gt_found.sum())
            self.events.append({
                'n_gt': len(gt_positions),
                'n_found': n_found,
                'pion_indices': list(range(pion_start, len(self.pions))),
            })


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def extract_training_curves(log_path: str) -> dict[str, list]:
    """Extract epoch-by-epoch metrics from a training.log file."""
    import re
    curves = {
        'epoch': [], 'val_loss': [], 'val_r200': [],
        'train_loss': [], 'train_r200': [],
        'val_dprime': [], 's1_recall': [],
        'val_rank': [],
    }
    with open(log_path) as f:
        for line in f:
            # Validation line
            match = re.search(
                r'Epoch (\d+) val \| total: ([\d.]+).*R@200: ([\d.]+).*'
                r"d': ([\d.-]+).*rank: (\d+).*S1_R@K1: ([\d.]+)",
                line,
            )
            if match:
                epoch = int(match.group(1))
                curves['epoch'].append(epoch)
                curves['val_loss'].append(float(match.group(2)))
                curves['val_r200'].append(float(match.group(3)))
                curves['val_dprime'].append(float(match.group(4)))
                curves['val_rank'].append(int(match.group(5)))
                curves['s1_recall'].append(float(match.group(6)))

            # Train eval line
            match = re.search(
                r'Epoch (\d+) train_eval \| total: ([\d.]+).*R@200: ([\d.]+)',
                line,
            )
            if match:
                curves['train_loss'].append(float(match.group(2)))
                curves['train_r200'].append(float(match.group(3)))
    return curves


def generate_report(
    s1_metrics: dict,
    regular_metrics: dict,
    physics_metrics: dict,
    stage1_collector: PerPionCollector,
    regular_collector: PerPionCollector,
    physics_collector: PerPionCollector,
    regular_curves: dict | None,
    physics_curves: dict | None,
    regular_args: dict,
    physics_args: dict,
) -> str:
    """Generate the full markdown report."""
    lines = []

    def add(text=''):
        lines.append(text)

    add('# Cascade Model Diagnostic Analysis')
    add()
    add(f'**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    add()
    add('---')
    add()

    # ---- Section 1: R@K Sweep ----
    add('## 1. R@K Sweep')
    add()
    k_values = [10, 20, 30, 50, 100, 200, 300, 400, 500, 600]
    add(f'| K | Stage 1 R@K | Regular R@K | Physics R@K | Delta (P-R) |')
    add(f'|---|-----------|------------|------------|-------------|')
    for k in k_values:
        s1_r = s1_metrics.get(f'recall_at_{k}', 0)
        reg_r = regular_metrics.get(f'recall_at_{k}', 0)
        phy_r = physics_metrics.get(f'recall_at_{k}', 0)
        delta = phy_r - reg_r
        add(f'| {k} | {s1_r:.4f} | {reg_r:.4f} | {phy_r:.4f} | {delta:+.4f} |')

    add()
    add('### Perfect recall (P@K)')
    add()
    add(f'| K | Stage 1 | Regular | Physics | Delta |')
    add(f'|---|---------|---------|---------|-------|')
    for k in k_values:
        s1_p = s1_metrics.get(f'perfect_at_{k}', 0)
        reg_p = regular_metrics.get(f'perfect_at_{k}', 0)
        phy_p = physics_metrics.get(f'perfect_at_{k}', 0)
        delta = phy_p - reg_p
        add(f'| {k} | {s1_p:.4f} | {reg_p:.4f} | {phy_p:.4f} | {delta:+.4f} |')

    # ---- Section 2: Score Distribution ----
    add()
    add('---')
    add()
    add('## 2. Score Distribution Analysis')
    add()
    add('| Metric | Stage 1 | Regular | Physics |')
    add('|--------|---------|---------|---------|')
    for metric_name in ['d_prime', 'median_gt_rank', 'gt_rank_p75', 'gt_rank_p90', 'gt_rank_p95']:
        s1_v = s1_metrics.get(metric_name, 0)
        reg_v = regular_metrics.get(metric_name, 0)
        phy_v = physics_metrics.get(metric_name, 0)
        add(f'| {metric_name} | {s1_v:.3f} | {reg_v:.3f} | {phy_v:.3f} |')

    # Stage 2 vs Stage 1 score correlation
    add()
    add('### Stage 2 vs Stage 1 score correlation')
    add()
    for name, collector in [('Regular', regular_collector), ('Physics', physics_collector)]:
        pions = collector.pions
        if not pions:
            continue
        gt_cascade_ranks = np.array([p['cascade_rank'] for p in pions])
        gt_s1_ranks = np.array([p['stage1_rank'] for p in pions])
        corr = np.corrcoef(gt_cascade_ranks, gt_s1_ranks)[0, 1]
        add(f'- **{name}** — Pearson r (GT pion ranks): {corr:.3f}')
    add()

    # ---- Section 3: Per-Event Breakdown ----
    add('---')
    add()
    add('## 3. Per-Event Breakdown at K=200')
    add()
    add('| Found | Stage 1 | Regular | Physics |')
    add('|-------|---------|---------|---------|')
    for n_found in [0, 1, 2, 3]:
        s1_key = f'found_{n_found}_of_3_at_200'
        s1_v = s1_metrics.get(s1_key, 0) * 100
        reg_v = sum(1 for e in regular_collector.events if e['n_gt'] == 3 and e['n_found'] == n_found)
        phy_v = sum(1 for e in physics_collector.events if e['n_gt'] == 3 and e['n_found'] == n_found)
        reg_total = max(1, sum(1 for e in regular_collector.events if e['n_gt'] == 3))
        phy_total = max(1, sum(1 for e in physics_collector.events if e['n_gt'] == 3))
        add(f'| {n_found}/3 | {s1_v:.1f}% | {100*reg_v/reg_total:.1f}% | {100*phy_v/phy_total:.1f}% |')

    # "Just missed" analysis
    add()
    add('### "Just missed" GT pions (ranked 201+)')
    add()
    add('| Rank range | Regular | Physics |')
    add('|-----------|---------|---------|')
    for lo, hi, label in [(200, 300, '201-300'), (300, 500, '301-500'), (500, 9999, '500+')]:
        reg_count = sum(1 for p in regular_collector.pions if lo <= p['cascade_rank'] < hi)
        phy_count = sum(1 for p in physics_collector.pions if lo <= p['cascade_rank'] < hi)
        reg_total = max(1, len(regular_collector.pions))
        phy_total = max(1, len(physics_collector.pions))
        add(f'| {label} | {reg_count} ({100*reg_count/reg_total:.1f}%) | {phy_count} ({100*phy_count/phy_total:.1f}%) |')

    # ---- Section 4: Failure by Physics Properties ----
    add()
    add('---')
    add()
    add('## 4. Failure Analysis by Physics Properties')
    add()
    add('### Found rate by pT bin')
    add()
    add('| pT range (GeV) | Stage 1 | Regular | Physics |')
    add('|----------------|---------|---------|---------|')
    pt_bins = [(0, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 100)]
    for lo, hi in pt_bins:
        label = f'[{lo}, {hi})' if hi < 100 else f'[{lo}, +)'
        rates = {}
        for cname, collector in [('s1', stage1_collector), ('reg', regular_collector), ('phy', physics_collector)]:
            found = sum(1 for p in collector.pions if lo <= p['pt'] < hi and p['found_at_200'])
            total = max(1, sum(1 for p in collector.pions if lo <= p['pt'] < hi))
            rates[cname] = (found, total, found / total)
        add(f'| {label} | {rates["s1"][2]:.3f} (n={rates["s1"][1]}) | {rates["reg"][2]:.3f} | {rates["phy"][2]:.3f} |')

    add()
    add('### Found rate by |dxy_sig| bin')
    add()
    add('| |dxy_sig| range | Stage 1 | Regular | Physics |')
    add('|----------------|---------|---------|---------|')
    dxy_bins = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 100)]
    for lo, hi in dxy_bins:
        label = f'[{lo}, {hi})' if hi < 100 else f'[{lo}, +)'
        rates = {}
        for cname, collector in [('s1', stage1_collector), ('reg', regular_collector), ('phy', physics_collector)]:
            found = sum(1 for p in collector.pions if lo <= p['dxy_sig'] < hi and p['found_at_200'])
            total = max(1, sum(1 for p in collector.pions if lo <= p['dxy_sig'] < hi))
            rates[cname] = (found, total, found / total)
        add(f'| {label} | {rates["s1"][2]:.3f} (n={rates["s1"][1]}) | {rates["reg"][2]:.3f} | {rates["phy"][2]:.3f} |')

    add()
    add('### Uncanny valley (pT 0.3-0.5 AND |dxy_sig| < 0.5)')
    add()
    for name, collector in [('Stage 1', stage1_collector), ('Regular', regular_collector), ('Physics', physics_collector)]:
        uv = [p for p in collector.pions if 0.3 <= p['pt'] < 0.5 and p['dxy_sig'] < 0.5]
        found = sum(1 for p in uv if p['found_at_200'])
        total = max(1, len(uv))
        add(f'- **{name}:** {found}/{total} = {100*found/total:.1f}%')

    # ---- Section 5: Per-Pion-Role Analysis ----
    add()
    add('---')
    add()
    add('## 5. Per-Pion-Role Analysis (3-GT events only)')
    add()
    add('| Role | Stage 1 | Regular | Physics |')
    add('|------|---------|---------|---------|')
    for role in ['highest_pt', 'middle_pt', 'lowest_pt']:
        rates = {}
        for cname, collector in [('s1', stage1_collector), ('reg', regular_collector), ('phy', physics_collector)]:
            pions = [p for p in collector.pions if p['role'] == role]
            rate = sum(p['found_at_200'] for p in pions) / max(1, len(pions))
            rates[cname] = (rate, len(pions))
        add(f'| {role} | {rates["s1"][0]:.3f} (n={rates["s1"][1]}) | {rates["reg"][0]:.3f} | {rates["phy"][0]:.3f} |')

    add()
    add('Mean pT by role:')
    for name, collector in [('Regular', regular_collector), ('Physics', physics_collector)]:
        for role in ['highest_pt', 'middle_pt', 'lowest_pt']:
            pts = [p['pt'] for p in collector.pions if p['role'] == role]
            if pts:
                add(f'- {name} {role}: mean pT = {np.mean(pts):.3f} GeV')

    # ---- Section 6: Stage 1 → Stage 2 Pipeline ----
    add()
    add('---')
    add()
    add('## 6. Stage 1 → Stage 2 Pipeline Analysis')
    add()
    for name, collector in [('Regular', regular_collector), ('Physics', physics_collector)]:
        pions = collector.pions
        total = max(1, len(pions))
        dropped_by_s1 = sum(1 for p in pions if not p['in_stage1_top600'])
        passed_s1_missed_s2 = sum(1 for p in pions if p['in_stage1_top600'] and not p['found_at_200'])
        # Rescue: Stage 1 rank 200-600, cascade rank < 200
        rescued = sum(1 for p in pions if 200 <= p['stage1_rank'] < 600 and p['cascade_rank'] < 200)
        # Demoted: Stage 1 rank < 200, cascade rank >= 200
        demoted = sum(1 for p in pions if p['stage1_rank'] < 200 and p['cascade_rank'] >= 200)

        add(f'### {name}')
        add(f'- GT pions dropped by Stage 1 (rank >= 600): {dropped_by_s1}/{total} ({100*dropped_by_s1/total:.1f}%)')
        add(f'- Passed Stage 1 but missed by Stage 2 (in top-600, not in top-200): {passed_s1_missed_s2}/{total} ({100*passed_s1_missed_s2/total:.1f}%)')
        add(f'- **Rescued** (S1 rank 200-600 → cascade top-200): {rescued}/{total} ({100*rescued/total:.1f}%)')
        add(f'- **Demoted** (S1 rank < 200 → cascade rank >= 200): {demoted}/{total} ({100*demoted/total:.1f}%)')
        add(f'- Net rescue: {rescued - demoted:+d}')
        add()

    # ---- Section 7: Learning Dynamics ----
    add('---')
    add()
    add('## 7. Learning Dynamics')
    add()
    for name, curves in [('Regular', regular_curves), ('Physics', physics_curves)]:
        if curves is None or not curves['epoch']:
            add(f'### {name}: training log not available')
            continue
        add(f'### {name}')
        epochs = curves['epoch']
        val_r200 = curves['val_r200']
        best_idx = int(np.argmax(val_r200))
        best_epoch = epochs[best_idx]
        best_r200 = val_r200[best_idx]
        last_r200 = val_r200[-1]
        add(f'- Best R@200: {best_r200:.4f} at epoch {best_epoch}')
        add(f'- Final R@200: {last_r200:.4f} (epoch {epochs[-1]})')
        add(f'- Total epochs: {epochs[-1]}')
        if curves['val_loss']:
            add(f'- Final val loss: {curves["val_loss"][-1]:.4f}')
        if curves['train_loss'] and curves['val_loss']:
            gap = curves['val_loss'][-1] - curves['train_loss'][-1]
            add(f'- Train-val loss gap (last epoch): {gap:+.4f}')
        if curves['val_dprime']:
            add(f'- Final d\': {curves["val_dprime"][-1]:.3f}')
        add()

    # ---- Section 8: Head-to-Head ----
    add('---')
    add()
    add('## 8. Head-to-Head Event Comparison')
    add()

    # Match events between collectors (they process the same data in same order)
    n_events = min(len(regular_collector.events), len(physics_collector.events))
    physics_wins = 0
    regular_wins = 0
    ties = 0
    for i in range(n_events):
        reg_e = regular_collector.events[i]
        phy_e = physics_collector.events[i]
        if reg_e['n_gt'] != phy_e['n_gt']:
            continue  # sanity
        if phy_e['n_found'] > reg_e['n_found']:
            physics_wins += 1
        elif reg_e['n_found'] > phy_e['n_found']:
            regular_wins += 1
        else:
            ties += 1

    add(f'- Events compared: {n_events}')
    add(f'- Physics model finds MORE GT: {physics_wins} events ({100*physics_wins/max(1,n_events):.1f}%)')
    add(f'- Regular model finds MORE GT: {regular_wins} events ({100*regular_wins/max(1,n_events):.1f}%)')
    add(f'- Same performance: {ties} events ({100*ties/max(1,n_events):.1f}%)')
    add(f'- Net: physics wins {physics_wins - regular_wins:+d} events')

    # ---- Summary ----
    add()
    add('---')
    add()
    add('## Summary')
    add()
    add('| Metric | Stage 1 | Regular ParT | Physics ParT |')
    add('|--------|---------|-------------|-------------|')
    add(f'| R@200 | {s1_metrics.get("recall_at_200",0):.4f} | {regular_metrics.get("recall_at_200",0):.4f} | {physics_metrics.get("recall_at_200",0):.4f} |')
    add(f'| P@200 | {s1_metrics.get("perfect_at_200",0):.4f} | {regular_metrics.get("perfect_at_200",0):.4f} | {physics_metrics.get("perfect_at_200",0):.4f} |')
    add(f'| d\' | {s1_metrics.get("d_prime",0):.3f} | {regular_metrics.get("d_prime",0):.3f} | {physics_metrics.get("d_prime",0):.3f} |')
    add(f'| Median rank | {s1_metrics.get("median_gt_rank",0):.0f} | {regular_metrics.get("median_gt_rank",0):.0f} | {physics_metrics.get("median_gt_rank",0):.0f} |')
    add(f'| GT pions | {s1_metrics.get("total_gt_tracks",0)} | {regular_metrics.get("total_gt_tracks",0)} | {physics_metrics.get("total_gt_tracks",0)} |')
    add(f'| pair_extra_dim | — | 0 | 5 |')
    add(f'| embed_dim | — | {regular_args.get("stage2_embed_dim")} | {physics_args.get("stage2_embed_dim")} |')
    add(f'| num_layers | — | {regular_args.get("stage2_num_layers")} | {physics_args.get("stage2_num_layers")} |')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Parquet export
# ---------------------------------------------------------------------------

def _export_gt_pion_data(
    stage1_collector: PerPionCollector,
    regular_collector: PerPionCollector,
    physics_collector: PerPionCollector,
    output_path: str,
) -> None:
    """Export per-GT-pion data to parquet for downstream analysis.

    Each row is one GT pion. Columns include pT, dxy_sig, charge, role,
    and ranks from all three models (Stage 1, regular cascade, physics cascade).
    """
    import pandas as pd

    rows = []
    n_pions = len(stage1_collector.pions)
    for i in range(n_pions):
        s1 = stage1_collector.pions[i]
        reg = regular_collector.pions[i]
        phy = physics_collector.pions[i]
        rows.append({
            'event_id': s1['event_id'],
            'pt': s1['pt'],
            'dxy_sig': s1['dxy_sig'],
            'charge': s1['charge'],
            'role': s1['role'],
            'stage1_rank': s1['cascade_rank'],
            'regular_rank': reg['cascade_rank'],
            'physics_rank': phy['cascade_rank'],
            'stage1_found_at_200': s1['found_at_200'],
            'regular_found_at_200': reg['found_at_200'],
            'physics_found_at_200': phy['found_at_200'],
            'regular_in_stage1_top600': reg['in_stage1_top600'],
        })

    dataframe = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    dataframe.to_parquet(output_path, index=False)
    logger.info(
        f'Exported {len(dataframe)} GT pion ranks to {output_path}'
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive cascade model diagnostic analysis',
    )
    parser.add_argument('--regular-checkpoint', type=str, required=True)
    parser.add_argument('--physics-checkpoint', type=str, required=True)
    parser.add_argument('--stage1-checkpoint', type=str, required=True)
    parser.add_argument('--data-config', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='reports/cascade_model_analysis.md')
    parser.add_argument('--batch-size', type=int, default=96)
    parser.add_argument('--max-steps', type=int, default=900,
                        help='Max batches (default: 900, enough for ~84K val events at batch=96)')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='mps')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    device = torch.device(args.device)

    # Load models
    logger.info('Loading models...')
    stage1_model = load_stage1_model(args.stage1_checkpoint, device)
    regular_cascade = load_cascade_from_checkpoint(
        args.regular_checkpoint, args.stage1_checkpoint, device,
    )
    physics_cascade = load_cascade_from_checkpoint(
        args.physics_checkpoint, args.stage1_checkpoint, device,
    )

    # Load checkpoint args for the report
    regular_ckpt = torch.load(args.regular_checkpoint, map_location='cpu', weights_only=False)
    physics_ckpt = torch.load(args.physics_checkpoint, map_location='cpu', weights_only=False)
    regular_args = regular_ckpt.get('args', {})
    physics_args = physics_ckpt.get('args', {})

    # Load data
    from weaver.utils.dataset import SimpleIterDataset
    from utils.training_utils import (
        MetricsAccumulator,
        extract_label_from_inputs,
        trim_to_max_valid_tracks,
    )

    parquet_files = sorted(glob.glob(os.path.join(args.data_dir, '*.parquet')))
    logger.info(f'Found {len(parquet_files)} parquet files in {args.data_dir}')

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
        drop_last=False, pin_memory=False, num_workers=args.num_workers,
    )
    input_names = list(data_config.input_names)
    mask_input_index = input_names.index('pf_mask')
    label_input_index = input_names.index('pf_label')

    # Accumulators
    k_values = (10, 20, 30, 50, 100, 200, 300, 400, 500, 600)
    s1_accum = MetricsAccumulator(k_values=k_values)
    regular_accum = MetricsAccumulator(k_values=k_values)
    physics_accum = MetricsAccumulator(k_values=k_values)

    stage1_collector = PerPionCollector()
    regular_collector = PerPionCollector()
    physics_collector = PerPionCollector()

    # Run inference
    logger.info('Running inference...')
    with torch.no_grad():
        for batch_index, (X, y, _) in enumerate(data_loader):
            if args.max_steps is not None and batch_index >= args.max_steps:
                break

            inputs = [X[key].to(device) for key in data_config.input_names]
            inputs = trim_to_max_valid_tracks(inputs, mask_input_index)
            model_inputs, track_labels = extract_label_from_inputs(
                inputs, label_input_index,
            )
            points, features, lorentz_vectors, mask = model_inputs
            batch_size_actual = points.shape[0]

            # Stage 1 baseline
            stage1_model.train()
            s1_scores = stage1_model(points, features, lorentz_vectors, mask)
            s1_accum.update(s1_scores, track_labels, mask)
            stage1_collector.update(
                s1_scores.cpu(), s1_scores.cpu(), track_labels.cpu(),
                mask.cpu(), lorentz_vectors.cpu(), features.cpu(),
            )

            # Regular cascade
            regular_cascade.train()
            reg_filtered = regular_cascade._run_stage1(
                points, features, lorentz_vectors, mask, track_labels,
            )
            reg_s2_scores = regular_cascade.stage2(
                reg_filtered['points'], reg_filtered['features'],
                reg_filtered['lorentz_vectors'], reg_filtered['mask'],
                reg_filtered['stage1_scores'],
            )
            reg_full = torch.full(
                (batch_size_actual, mask.shape[2]), float('-inf'), device=device,
            )
            reg_full.scatter_(1, reg_filtered['selected_indices'], reg_s2_scores)
            regular_accum.update(reg_full, track_labels, mask)
            regular_collector.update(
                reg_full.cpu(), s1_scores.cpu(), track_labels.cpu(),
                mask.cpu(), lorentz_vectors.cpu(), features.cpu(),
            )

            # Physics cascade
            physics_cascade.train()
            phy_filtered = physics_cascade._run_stage1(
                points, features, lorentz_vectors, mask, track_labels,
            )
            phy_s2_scores = physics_cascade.stage2(
                phy_filtered['points'], phy_filtered['features'],
                phy_filtered['lorentz_vectors'], phy_filtered['mask'],
                phy_filtered['stage1_scores'],
            )
            phy_full = torch.full(
                (batch_size_actual, mask.shape[2]), float('-inf'), device=device,
            )
            phy_full.scatter_(1, phy_filtered['selected_indices'], phy_s2_scores)
            physics_accum.update(phy_full, track_labels, mask)
            physics_collector.update(
                phy_full.cpu(), s1_scores.cpu(), track_labels.cpu(),
                mask.cpu(), lorentz_vectors.cpu(), features.cpu(),
            )

            if (batch_index + 1) % 10 == 0:
                logger.info(
                    f'Batch {batch_index + 1}: '
                    f'{regular_accum.total_events_with_gt} events'
                )

            del inputs, model_inputs, track_labels
            del reg_filtered, phy_filtered, reg_s2_scores, phy_s2_scores
            del reg_full, phy_full, s1_scores

    # Compute metrics
    s1_metrics = s1_accum.compute()
    regular_metrics = regular_accum.compute()
    physics_metrics = physics_accum.compute()

    logger.info(
        f'Analysis complete: {regular_accum.total_events_with_gt} events, '
        f'{regular_accum.total_gt_tracks} GT tracks'
    )

    # Export per-GT-pion ranks to parquet for downstream analysis
    export_path = args.output.replace('.md', '_gt_pions.parquet')
    _export_gt_pion_data(
        stage1_collector, regular_collector, physics_collector, export_path,
    )

    # Extract training curves from logs
    regular_log = os.path.join(
        os.path.dirname(os.path.dirname(args.regular_checkpoint)),
        'training.log',
    )
    physics_log = os.path.join(
        os.path.dirname(os.path.dirname(args.physics_checkpoint)),
        'training.log',
    )
    regular_curves = extract_training_curves(regular_log) if os.path.exists(regular_log) else None
    physics_curves = extract_training_curves(physics_log) if os.path.exists(physics_log) else None

    # Generate report
    report = generate_report(
        s1_metrics, regular_metrics, physics_metrics,
        stage1_collector, regular_collector, physics_collector,
        regular_curves, physics_curves,
        regular_args, physics_args,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(report)

    logger.info(f'Report saved to {args.output}')

    # Print summary to console
    print()
    print(f'R@200:  S1={s1_metrics["recall_at_200"]:.4f}  '
          f'Regular={regular_metrics["recall_at_200"]:.4f}  '
          f'Physics={physics_metrics["recall_at_200"]:.4f}')
    print(f'd\':    S1={s1_metrics["d_prime"]:.3f}  '
          f'Regular={regular_metrics["d_prime"]:.3f}  '
          f'Physics={physics_metrics["d_prime"]:.3f}')
    print(f'Rank:  S1={s1_metrics["median_gt_rank"]:.0f}  '
          f'Regular={regular_metrics["median_gt_rank"]:.0f}  '
          f'Physics={physics_metrics["median_gt_rank"]:.0f}')


if __name__ == '__main__':
    main()
