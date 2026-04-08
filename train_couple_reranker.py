"""Training script for the per-couple reranker (Stage 3).

Loads a frozen 2-stage cascade from a checkpoint, builds a fresh
``CoupleReranker`` head, and trains the head with pairwise ranking loss
on per-couple feature vectors enumerated from the ParT top-50 (Filter A:
``m(ij) <= m_tau``).

Architecture and pilot recipe per
``reports/triplet_reranking/triplet_research_plan_20260408.md`` direction A:
- Frozen cascade: TrackPreFilter (Stage 1) + CascadeReranker / ParT (Stage 2)
- Trainable head: ``CoupleReranker`` — input projection (Conv1d 51→256) + 4
  ``ResidualBlock(256)`` + scoring head (Conv1d 256→128→1), ~580K params
- Loss: pairwise ranking with N=50 random negatives per positive (matches
  the prefilter and CascadeReranker convention)
- Best metric: ``couple_recall_at_100`` on val

Usage:
    python train_couple_reranker.py \\
        --data-config data/low-pt/lowpt_tau_trackfinder.yaml \\
        --data-dir data/low-pt/train/ \\
        --val-data-dir data/low-pt/val/ \\
        --network networks/lowpt_tau_CoupleReranker.py \\
        --cascade-checkpoint models/debug_checkpoints/cascade_soap_*/checkpoints/best_model.pt \\
        --top-k2 50 \\
        --epochs 50 --batch-size 16 --steps-per-epoch 500 \\
        --device cuda:0
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

import torch

torch.set_float32_matmul_precision('high')
from torch.utils.data import DataLoader

from weaver.utils.dataset import SimpleIterDataset

from pretrain_backbone import (
    WarmupThenCosineScheduler,
    WarmupThenPlateauScheduler,
    _TeeStream,
    build_experiment_directory,
    plot_loss_curves,
    save_loss_history,
)
from utils.training_utils import (
    CheckpointManager,
    CoupleMetricsAccumulator,
    extract_label_from_inputs,
    load_network_module,
    save_epoch_metrics,
    trim_to_max_valid_tracks,
)

logger = logging.getLogger('train_couple_reranker')


# ---------------------------------------------------------------------------
# Metric labels for the on-disk loss_history.json
# ---------------------------------------------------------------------------
#
# Maps loss-history dict keys to short human-readable descriptions. The
# saver wraps each metric as ``{'label': str, 'values': list[float]}`` so
# the JSON file is self-documenting and the user can read it manually
# without grepping the code for what each key means.

METRIC_LABELS: dict[str, str] = {
    'train': 'Train loss (couple ranking, mean per epoch)',
    'val': 'Validation loss (couple ranking)',
    'lr': 'Learning rate',
    'val_eligible_events':
        'Eligible events (val): events with ≥1 GT couple in candidate pool',
    'val_total_events':
        'Total events (val) seen during validation',
    'val_events_with_full_triplet':
        'Events (val) with all 3 GT pions in cascade Stage 1 top-K1',
    'val_mean_first_gt_rank_couples':
        'Mean rank of best GT couple in reranker output (1-indexed; '
        'lower is better; averaged over eligible events)',
}

# Per-K labels for D@K_tracks, C@K_couples, RC@K_couples are generated
# programmatically (one per K value) so adding new K values doesn't
# require touching this constant.
for _k in (30, 50, 75, 100, 200):
    METRIC_LABELS[f'val_d_at_{_k}_tracks'] = (
        f'D@{_k}_tracks: events with ≥2 GT pions in ParT top-{_k} tracks '
        f'(cascade duplet rate, fixed by checkpoint)'
    )
for _k in (50, 75, 100, 200):
    METRIC_LABELS[f'val_c_at_{_k}_couples'] = (
        f'C@{_k}_couples: events with ≥1 GT couple in top-{_k} of reranker '
        f'output (per-event binary)'
    )
    METRIC_LABELS[f'val_rc_at_{_k}_couples'] = (
        f'RC@{_k}_couples: C@{_k}_couples AND full triplet in cascade Stage 1 '
        f'top-K1=256'
    )
del _k


# ---------------------------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    data_config,
    epoch: int,
    tensorboard_writer: 'SummaryWriter | None',
    global_batch_count: int,
    steps_per_epoch: int,
    mask_input_index: int,
    label_input_index: int,
    grad_clip_max_norm: float = 1.0,
) -> tuple[dict[str, float], int]:
    """Train the CoupleReranker for one epoch (frozen cascade inside)."""
    model.train()
    loss_accumulators: dict[str, torch.Tensor] | None = None
    num_batches = 0
    start_time = time.time()

    for batch_index, (X, _, _) in enumerate(train_loader):
        if batch_index >= steps_per_epoch:
            break

        inputs = [X[k].to(device) for k in data_config.input_names]
        padded_length = inputs[0].shape[2]
        inputs = trim_to_max_valid_tracks(inputs, mask_input_index)

        if batch_index == 0:
            trimmed_length = inputs[0].shape[2]
            logger.info(
                f'Epoch {epoch} | Trim: {padded_length} → {trimmed_length} '
                f'({100 * (1 - trimmed_length / padded_length):.0f}% '
                f'padding removed)',
            )

        model_inputs, track_labels = extract_label_from_inputs(
            inputs, label_input_index,
        )
        points, features, lorentz_vectors, mask = model_inputs

        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask, track_labels,
        )
        # Drop the heavy metric tensors so the loss accumulator only sees
        # scalar loss components.
        loss_dict.pop('_scores', None)
        loss_dict.pop('_couple_labels', None)
        loss_dict.pop('_couple_mask', None)
        loss_dict.pop('_n_gt_in_top_k1', None)
        loss_dict.pop('_n_gt_in_top_k_tracks', None)
        loss = loss_dict['total_loss']

        if not torch.isfinite(loss).item():
            logger.warning(
                f'Epoch {epoch} | Batch {batch_index} | '
                f'Skipping batch with non-finite loss',
            )
            optimizer.zero_grad(set_to_none=True)
            global_batch_count += 1
            continue

        loss.backward()
        if grad_clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                grad_clip_max_norm,
            )
        optimizer.step()
        scheduler.step_batch()

        if loss_accumulators is None:
            loss_accumulators = {
                key: torch.zeros(1, device=loss.device) for key in loss_dict
            }
        for key in loss_accumulators:
            loss_accumulators[key] += loss_dict[key].detach()

        num_batches += 1
        global_batch_count += 1

        if batch_index % 20 == 0:
            elapsed = time.time() - start_time
            avg_loss = loss_accumulators['total_loss'].item() / num_batches
            logger.info(
                f'Epoch {epoch} | Batch {batch_index} | '
                f'Loss: {loss.item():.5f} | Avg: {avg_loss:.5f} | '
                f'LR: {scheduler.get_last_lr()[0]:.2e} | '
                f'Time: {elapsed:.1f}s',
            )

        del inputs, model_inputs, track_labels, loss_dict

    if loss_accumulators is None:
        loss_accumulators = {'total_loss': torch.zeros(1)}
    loss_averages = {
        key: value.item() / max(1, num_batches)
        for key, value in loss_accumulators.items()
    }
    logger.info(
        f'Epoch {epoch} train | total: {loss_averages["total_loss"]:.5f}',
    )
    return loss_averages, global_batch_count


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    data_config,
    mask_input_index: int,
    label_input_index: int,
    max_steps: int | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Validate and compute couple_recall@K metrics on the val set."""
    model.eval()
    loss_accumulators: dict[str, float] | None = None
    num_batches = 0
    couple_metrics_accumulator = CoupleMetricsAccumulator(
        k_values_couples=(50, 75, 100, 200),
        k_values_tracks=(30, 50, 75, 100, 200),
    )

    for batch_index, (X, _, _) in enumerate(val_loader):
        if max_steps is not None and batch_index >= max_steps:
            break

        inputs = [X[k].to(device) for k in data_config.input_names]
        inputs = trim_to_max_valid_tracks(inputs, mask_input_index)
        model_inputs, track_labels = extract_label_from_inputs(
            inputs, label_input_index,
        )
        points, features, lorentz_vectors, mask = model_inputs

        # Train mode for BatchNorm batch stats — same workaround as
        # train_cascade.py validate, since the cascade still has BN inside.
        model.train()
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask, track_labels,
        )
        model.eval()

        couple_scores = loss_dict.pop('_scores').detach()
        couple_labels = loss_dict.pop('_couple_labels').detach()
        couple_mask = loss_dict.pop('_couple_mask').detach()
        n_gt_in_top_k1 = loss_dict.pop('_n_gt_in_top_k1').detach()
        n_gt_in_top_k_tracks = loss_dict.pop('_n_gt_in_top_k_tracks').detach()

        couple_metrics_accumulator.update(
            couple_scores, couple_labels, couple_mask,
            n_gt_in_top_k1=n_gt_in_top_k1,
            n_gt_in_top_k_tracks=n_gt_in_top_k_tracks,
        )

        if loss_accumulators is None:
            loss_accumulators = {key: 0.0 for key in loss_dict}
        for key in loss_accumulators:
            loss_accumulators[key] += loss_dict[key].item()

        num_batches += 1
        del inputs, model_inputs, track_labels, loss_dict

    if loss_accumulators is None:
        loss_accumulators = {'total_loss': 0.0}
    loss_averages = {
        key: value / max(1, num_batches)
        for key, value in loss_accumulators.items()
    }
    metrics = couple_metrics_accumulator.compute()
    return loss_averages, metrics


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train CoupleReranker (Stage 3) on top of a frozen cascade',
    )
    parser.add_argument('--data-config', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--network', type=str, required=True)
    parser.add_argument('--cascade-checkpoint', type=str, required=True,
                        help='Path to a trained cascade checkpoint (Stage 1+2)')
    parser.add_argument('--top-k2', type=int, default=50,
                        help='Number of top tracks per event from which '
                             'couples are enumerated (default 50).')
    parser.add_argument('--model-name', type=str, default='CoupleReranker')
    parser.add_argument('--experiments-dir', type=str, default='experiments')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['plateau', 'cosine'])
    parser.add_argument('--warmup-fraction', type=float, default=0.05)
    parser.add_argument('--plateau-factor', type=float, default=0.5)
    parser.add_argument('--plateau-patience', type=int, default=5)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--train-fraction', type=float, default=0.8)
    parser.add_argument('--val-data-dir', type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--no-in-memory', action='store_true')
    parser.add_argument('--steps-per-epoch', type=int, default=None)
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--keep-best-k', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None)
    # CoupleReranker architecture knobs
    parser.add_argument('--couple-hidden-dim', type=int, default=256)
    parser.add_argument('--couple-num-residual-blocks', type=int, default=4)
    parser.add_argument('--couple-dropout', type=float, default=0.1)
    parser.add_argument('--couple-ranking-num-samples', type=int, default=50)
    parser.add_argument('--couple-ranking-temperature', type=float, default=1.0)
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = _build_parser()
    args = parser.parse_args()
    device = torch.device(args.device)

    # ---- Experiment directory ----
    resume_dir = None
    if args.resume is not None:
        resume_dir = os.path.dirname(os.path.dirname(args.resume))
    experiment_dir, checkpoints_dir, tensorboard_dir = build_experiment_directory(
        args.experiments_dir, args.model_name, resume_dir,
    )

    # ---- Logging ----
    log_file = os.path.join(experiment_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )
    log_file_handle = open(log_file, 'a')  # noqa: SIM115
    sys.stderr = _TeeStream(sys.stderr, log_file_handle)

    logger.info(f'Experiment directory: {experiment_dir}')
    logger.info(f'Arguments: {vars(args)}')

    # ---- Data loading ----
    import glob
    load_in_memory = not args.no_in_memory

    train_parquet_files = sorted(glob.glob(f'{args.data_dir}/*.parquet'))
    logger.info(f'Found {len(train_parquet_files)} train parquet files in {args.data_dir}')
    train_file_dict = {'data': train_parquet_files}
    num_train_files = len(train_parquet_files)

    if args.val_data_dir is not None:
        val_parquet_files = sorted(glob.glob(f'{args.val_data_dir}/*.parquet'))
        logger.info(f'Found {len(val_parquet_files)} val parquet files in {args.val_data_dir}')
        val_file_dict = {'data': val_parquet_files}
        num_val_files = len(val_parquet_files)
        train_range = ((0.0, 1.0), 1.0)
        val_range = ((0.0, 1.0), 1.0)
    else:
        val_file_dict = train_file_dict
        num_val_files = num_train_files
        train_range = ((0.0, args.train_fraction), 1.0)
        val_range = ((args.train_fraction, 1.0), 1.0)

    train_num_workers = min(args.num_workers, num_train_files)
    train_dataset = SimpleIterDataset(
        train_file_dict,
        data_config_file=args.data_config,
        for_training=True,
        load_range_and_fraction=train_range,
        fetch_by_files=True,
        fetch_step=num_train_files,
        in_memory=load_in_memory,
    )
    data_config = train_dataset.config

    val_dataset = SimpleIterDataset(
        val_file_dict,
        data_config_file=args.data_config,
        for_training=False,
        load_range_and_fraction=val_range,
        fetch_by_files=True,
        fetch_step=num_val_files,
        in_memory=load_in_memory,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=train_num_workers,
        persistent_workers=train_num_workers > 0,
    )
    val_num_workers = (
        min(max(1, train_num_workers // 2), num_val_files)
        if train_num_workers > 0
        else 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=val_num_workers,
        persistent_workers=val_num_workers > 0,
    )

    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch is None:
        steps_per_epoch = 100
        logger.warning(
            f'--steps-per-epoch not set, defaulting to {steps_per_epoch}.',
        )
    logger.info(f'Steps per epoch: {steps_per_epoch}')
    logger.info(f'DataLoader workers: train={train_num_workers}, val={val_num_workers}')

    # ---- Model ----
    network_module = load_network_module(args.network)
    model, model_info = network_module.get_model(
        data_config,
        cascade_checkpoint=args.cascade_checkpoint,
        top_k2=args.top_k2,
        couple_hidden_dim=args.couple_hidden_dim,
        couple_num_residual_blocks=args.couple_num_residual_blocks,
        couple_dropout=args.couple_dropout,
        couple_ranking_num_samples=args.couple_ranking_num_samples,
        couple_ranking_temperature=args.couple_ranking_temperature,
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(f'Total parameters: {total_params:,} | Trainable: {trainable_params:,}')

    input_names = list(data_config.input_names)
    mask_input_index = input_names.index('pf_mask')
    label_input_index = input_names.index('pf_label')

    # ---- Optimizer (only the trainable CoupleReranker params) ----
    trainable_parameter_iter = filter(
        lambda parameter: parameter.requires_grad, model.parameters(),
    )
    optimizer = torch.optim.AdamW(
        trainable_parameter_iter, lr=args.lr, weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * steps_per_epoch
    max_warmup_steps = 2000
    warmup_steps = min(int(args.warmup_fraction * total_steps), max_warmup_steps)

    if args.scheduler == 'cosine':
        warmup_epochs = math.ceil(warmup_steps / steps_per_epoch)
        num_post_warmup_epochs = max(1, args.epochs - warmup_epochs)
        logger.info(
            f'LR schedule: {warmup_steps} warmup steps, then '
            f'CosineAnnealingLR over {num_post_warmup_epochs} epochs',
        )
        scheduler = WarmupThenCosineScheduler(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_post_warmup_epochs=num_post_warmup_epochs,
            min_lr=args.min_lr,
        )
    else:
        scheduler = WarmupThenPlateauScheduler(
            optimizer,
            num_warmup_steps=warmup_steps,
            plateau_factor=args.plateau_factor,
            plateau_patience=args.plateau_patience,
            min_lr=args.min_lr,
        )

    checkpoint_manager = CheckpointManager(
        checkpoints_directory=checkpoints_dir,
        keep_best_k=args.keep_best_k,
        criterion_mode='max',
        criterion_name='C@100',
    )

    # ---- TensorBoard ----
    from torch.utils.tensorboard import SummaryWriter
    tensorboard_writer = SummaryWriter(tensorboard_dir)

    # ---- Training loop ----
    start_epoch = 1
    best_val_c_at_100 = 0.0
    best_val_epoch = 0
    global_batch_count = 0
    loss_history: dict[str, list] = {
        'train': [], 'val': [], 'lr': [],
    }

    if args.resume is not None:
        logger.info(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(
            args.resume, map_location=device, weights_only=False,
        )
        # Slim checkpoint format: only the trainable couple_reranker
        # weights are persisted; the frozen cascade is rebuilt from
        # `--cascade-checkpoint` at startup, so it does NOT belong in
        # per-epoch artifacts.
        model.couple_reranker.load_state_dict(
            checkpoint['couple_reranker_state_dict'],
        )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_c_at_100 = checkpoint.get('best_val_c_at_100', 0.0)
        best_val_epoch = checkpoint.get('best_val_epoch', 0)
        global_batch_count = checkpoint.get('global_batch_count', 0)
        logger.info(
            f'Resumed from epoch {start_epoch - 1}, '
            f'best C@100={best_val_c_at_100:.5f}',
        )

    logger.info(f'=== Training CoupleReranker (top_k2={args.top_k2}) ===')

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            logger.info(f'=== Epoch {epoch}/{args.epochs} ===')

            train_losses, global_batch_count = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                device, data_config, epoch,
                tensorboard_writer, global_batch_count,
                steps_per_epoch, mask_input_index, label_input_index,
                grad_clip_max_norm=args.grad_clip,
            )

            eval_steps = max(1, steps_per_epoch // 4)

            val_losses, val_metrics = validate(
                model, val_loader, device, data_config,
                mask_input_index, label_input_index,
                max_steps=eval_steps,
            )

            val_loss = val_losses['total_loss']
            val_c_at_100 = val_metrics.get('c_at_100_couples', 0.0)

            is_best = val_c_at_100 > best_val_c_at_100
            if is_best:
                best_val_c_at_100 = val_c_at_100
                best_val_epoch = epoch

            def _format_metrics(metrics: dict) -> str:
                parts = [
                    f'D@{k}t: {metrics.get(f"d_at_{k}_tracks", 0.0):.4f}'
                    for k in (30, 50, 75, 100, 200)
                ]
                parts += [
                    f'C@{k}c: {metrics.get(f"c_at_{k}_couples", 0.0):.4f}'
                    for k in (50, 75, 100, 200)
                ]
                parts += [
                    f'RC@{k}c: {metrics.get(f"rc_at_{k}_couples", 0.0):.4f}'
                    for k in (50, 75, 100, 200)
                ]
                if 'mean_first_gt_rank_couples' in metrics:
                    parts.append(
                        f'mean_rank: {metrics["mean_first_gt_rank_couples"]:.1f}',
                    )
                if 'eligible_events' in metrics:
                    parts.append(
                        f'eligible: {int(metrics["eligible_events"])}',
                    )
                if 'events_with_full_triplet' in metrics:
                    parts.append(
                        f'full_triplet: {int(metrics["events_with_full_triplet"])}',
                    )
                return ' | '.join(parts)

            val_summary = _format_metrics(val_metrics)
            if is_best:
                logger.info(
                    f'Epoch {epoch} val | total: {val_loss:.5f} '
                    f'C@100c: {val_c_at_100:.4f} ★ new best | {val_summary}',
                )
            else:
                epochs_since_best = epoch - best_val_epoch
                logger.info(
                    f'Epoch {epoch} val | total: {val_loss:.5f} '
                    f'(best C@100c: {best_val_c_at_100:.4f}, '
                    f'{epochs_since_best} epochs ago) | {val_summary}',
                )

            scheduler.step_epoch(val_loss)
            current_lr = scheduler.get_last_lr()[0]

            tensorboard_writer.add_scalar('Loss/train_epoch', train_losses['total_loss'], epoch)
            tensorboard_writer.add_scalar('Loss/val_epoch', val_loss, epoch)
            for metric_key, metric_value in val_metrics.items():
                tensorboard_writer.add_scalar(
                    f'Metrics/val_{metric_key}', metric_value, epoch,
                )
            tensorboard_writer.add_scalar('LR/epoch', current_lr, epoch)

            loss_history['train'].append(train_losses['total_loss'])
            loss_history['val'].append(val_loss)
            loss_history['lr'].append(current_lr)
            for metric_key, metric_value in val_metrics.items():
                history_key = f'val_{metric_key}'
                if history_key not in loss_history:
                    loss_history[history_key] = []
                loss_history[history_key].append(metric_value)
                if metric_key in loss_history:
                    loss_history[metric_key].append(metric_value)
            save_loss_history(
                loss_history, experiment_dir, metric_labels=METRIC_LABELS,
            )

            epoch_metrics = {
                'epoch': epoch,
                'train_loss': train_losses['total_loss'],
                'val_loss': val_loss,
                'lr': current_lr,
                'top_k2': args.top_k2,
            }
            for metric_key, metric_value in val_metrics.items():
                epoch_metrics[f'val_{metric_key}'] = metric_value
            save_epoch_metrics(epoch_metrics, experiment_dir, epoch)

            if epoch % args.save_every == 0 or is_best or epoch == args.epochs:
                # Slim checkpoint: save ONLY the trainable couple
                # reranker. The frozen cascade is reloaded from
                # `--cascade-checkpoint` at startup, so re-saving its
                # ~280 MB of weights every epoch would just bloat disk.
                checkpoint = {
                    'epoch': epoch,
                    'couple_reranker_state_dict':
                        model.couple_reranker.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_c_at_100': best_val_c_at_100,
                    'best_val_epoch': best_val_epoch,
                    'global_batch_count': global_batch_count,
                    'val_losses': val_losses,
                    'val_metrics': val_metrics,
                    'args': vars(args),
                }
                checkpoint_manager.save_checkpoint(
                    checkpoint, epoch, val_c_at_100, is_best,
                )

    except Exception:
        logger.error(f'Training failed:\n{traceback.format_exc()}')
        raise

    tensorboard_writer.close()
    plot_loss_curves(loss_history, experiment_dir)
    logger.info(f'Training complete. Best C@100: {best_val_c_at_100:.5f}')
    logger.info(f'Experiment: {experiment_dir}')


if __name__ == '__main__':
    main()
