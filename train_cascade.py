"""Training script for CascadeModel (Stage 1 pre-filter → Stage 2 reranker).

Reuses the same data pipeline and training loop structure as train_prefilter.py.
Key differences:
    - Loads Stage 1 from checkpoint (frozen, no gradients)
    - Only Stage 2 parameters are optimized
    - Logs both Stage 1 R@K1 and end-to-end Stage 2 metrics
    - No temperature/DRW scheduling (that's Stage 1 specific)

Usage:
    python train_cascade.py \\
        --data-config data/low-pt/lowpt_tau_trackfinder.yaml \\
        --data-dir data/low-pt/train/ \\
        --val-data-dir data/low-pt/val/ \\
        --network networks/lowpt_tau_CascadeReranker.py \\
        --stage1-checkpoint models/prefilter_best.pt \\
        --top-k1 600 \\
        --epochs 50 \\
        --batch-size 96 \\
        --device cuda:0 \\
        --amp
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import traceback
from datetime import datetime
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
    MetricsAccumulator,
    extract_label_from_inputs,
    load_network_module,
    save_epoch_metrics,
    trim_to_max_valid_tracks,
)

logger = logging.getLogger('train_cascade')


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    grad_scaler: torch.amp.GradScaler | None,
    device: torch.device,
    data_config,
    epoch: int,
    tensorboard_writer: SummaryWriter | None,
    global_batch_count: int,
    steps_per_epoch: int,
    mask_input_index: int,
    label_input_index: int,
    grad_clip_max_norm: float = 1.0,
) -> tuple[dict[str, float], int]:
    """Train Stage 2 for one epoch (Stage 1 is frozen inside CascadeModel)."""
    model.train()
    loss_accumulators: dict[str, float] | None = None
    num_batches = 0
    start_time = time.time()

    for batch_index, (X, y, _) in enumerate(train_loader):
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

        with torch.amp.autocast('cuda', enabled=grad_scaler is not None):
            loss_dict = model.compute_loss(
                points, features, lorentz_vectors, mask, track_labels,
            )
            loss_dict.pop('_scores', None)
            loss = loss_dict['total_loss']

        if not torch.isfinite(loss).item():
            logger.warning(
                f'Epoch {epoch} | Batch {batch_index} | '
                f'Skipping batch with non-finite loss',
            )
            optimizer.zero_grad(set_to_none=True)
            global_batch_count += 1
            continue

        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
        else:
            loss.backward()

        if grad_clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip_max_norm,
            )

        if grad_scaler is not None:
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            optimizer.step()

        scheduler.step_batch()

        if loss_accumulators is None:
            loss_accumulators = {
                key: torch.zeros(1, device=loss.device)
                for key in loss_dict
            }
        for key in loss_accumulators:
            loss_accumulators[key] += loss_dict[key].detach()

        num_batches += 1
        global_batch_count += 1

        if batch_index % 20 == 0:
            elapsed = time.time() - start_time
            avg_loss = loss_accumulators['total_loss'].item() / num_batches
            components = ' | '.join(
                f'{key.replace("_loss", "")}: '
                f'{value.item() / num_batches:.5f}'
                for key, value in loss_accumulators.items()
                if key != 'total_loss'
            )
            logger.info(
                f'Epoch {epoch} | Batch {batch_index} | '
                f'Loss: {loss.item():.5f} | Avg: {avg_loss:.5f} | '
                f'{components} | '
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

    components = ' | '.join(
        f'{key.replace("_loss", "")}: {value:.5f}'
        for key, value in loss_averages.items()
        if key != 'total_loss'
    )
    logger.info(
        f'Epoch {epoch} train | total: {loss_averages["total_loss"]:.5f} | '
        f'{components}',
    )

    return loss_averages, global_batch_count


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    data_config,
    mask_input_index: int,
    label_input_index: int,
    top_k1: int,
    max_steps: int | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Validate and compute recall@K metrics on the Stage 2 filtered set.

    Metrics are computed on the K1-track filtered set, measuring how well
    Stage 2 re-ranks within the candidates that Stage 1 selected.
    """
    model.eval()
    loss_accumulators: dict[str, float] | None = None
    num_batches = 0

    # Use smaller K values since we're ranking within K1 tracks, not 1100
    k_values = tuple(k for k in (10, 20, 30, 50, 100, 200) if k < top_k1)
    metrics_accumulator = MetricsAccumulator(k_values=k_values)

    # Track Stage 1 recall@K1 across batches
    stage1_recall_sum = 0.0
    stage1_recall_count = 0

    with torch.no_grad():
        for batch_index, (X, y, _) in enumerate(val_loader):
            if max_steps is not None and batch_index >= max_steps:
                break

            inputs = [X[k].to(device) for k in data_config.input_names]
            inputs = trim_to_max_valid_tracks(inputs, mask_input_index)
            model_inputs, track_labels = extract_label_from_inputs(
                inputs, label_input_index,
            )
            points, features, lorentz_vectors, mask = model_inputs

            model.train()
            loss_dict = model.compute_loss(
                points, features, lorentz_vectors, mask, track_labels,
            )
            model.eval()

            per_track_scores = loss_dict.pop('_scores').detach()

            # Track Stage 1 R@K1
            if 'stage1_recall_at_k1' in loss_dict:
                stage1_recall_sum += loss_dict.pop('stage1_recall_at_k1').item()
                stage1_recall_count += 1

            if loss_accumulators is None:
                loss_accumulators = {key: 0.0 for key in loss_dict}
            for key in loss_accumulators:
                loss_accumulators[key] += loss_dict[key].item()

            # Metrics on the filtered set (K1 tracks)
            # Need to get the filtered mask and labels from the cascade
            # The scores are (B, K1), we need matching mask and labels
            filtered = model._run_stage1(
                points, features, lorentz_vectors, mask, track_labels,
            )
            filtered_mask = filtered['mask']
            filtered_labels = filtered['track_labels']

            metrics_accumulator.update(
                per_track_scores, filtered_labels, filtered_mask,
            )

            num_batches += 1

            del inputs, model_inputs, track_labels, loss_dict
            del per_track_scores

    if loss_accumulators is None:
        loss_accumulators = {'total_loss': 0.0}
    loss_averages = {
        key: value / max(1, num_batches)
        for key, value in loss_accumulators.items()
    }

    metrics = metrics_accumulator.compute()

    # Add Stage 1 recall@K1
    if stage1_recall_count > 0:
        metrics['stage1_recall_at_k1'] = stage1_recall_sum / stage1_recall_count

    return loss_averages, metrics


def main():
    parser = argparse.ArgumentParser(
        description='Train CascadeModel (Stage 1 → Stage 2)',
    )
    parser.add_argument('--data-config', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--network', type=str, required=True)
    parser.add_argument('--stage1-checkpoint', type=str, required=True,
                        help='Path to trained Stage 1 (TrackPreFilter) checkpoint')
    parser.add_argument('--top-k1', type=int, default=600,
                        help='Number of tracks to pass from Stage 1 to Stage 2')
    parser.add_argument('--model-name', type=str, default='Cascade')
    parser.add_argument('--experiments-dir', type=str, default='experiments')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=96)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['plateau', 'cosine'])
    parser.add_argument('--warmup-fraction', type=float, default=0.05)
    parser.add_argument('--plateau-factor', type=float, default=0.5)
    parser.add_argument('--plateau-patience', type=int, default=5)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--train-fraction', type=float, default=0.8,
                        help='Fraction of data-dir for training (ignored if --val-data-dir set)')
    parser.add_argument('--val-data-dir', type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no-compile', action='store_true')
    parser.add_argument('--no-in-memory', action='store_true')
    parser.add_argument('--steps-per-epoch', type=int, default=None)
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--keep-best-k', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None)

    # Stage 2 architecture (passed through to network wrapper)
    parser.add_argument('--stage2-embed-dim', type=int, default=128,
                        help='Transformer embedding dimension (default: 128)')
    parser.add_argument('--stage2-num-heads', type=int, default=4,
                        help='Number of attention heads (default: 4)')
    parser.add_argument('--stage2-num-layers', type=int, default=3,
                        help='Number of transformer blocks (default: 3)')
    parser.add_argument('--stage2-pair-embed-dims', type=str, default='64,64',
                        help='Comma-separated pair embed MLP dims (default: 64,64)')
    parser.add_argument('--stage2-ffn-ratio', type=int, default=4,
                        help='Feed-forward expansion ratio (default: 4)')
    parser.add_argument('--stage2-dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')

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
    val_num_workers = min(
        max(1, train_num_workers // 2), num_val_files,
    ) if train_num_workers > 0 else 0
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
    pair_embed_dims = [int(x) for x in args.stage2_pair_embed_dims.split(',')]
    model, model_info = network_module.get_model(
        data_config,
        stage1_checkpoint=args.stage1_checkpoint,
        top_k1=args.top_k1,
        stage2_embed_dim=args.stage2_embed_dim,
        stage2_num_heads=args.stage2_num_heads,
        stage2_num_layers=args.stage2_num_layers,
        stage2_pair_embed_dims=pair_embed_dims,
        stage2_ffn_ratio=args.stage2_ffn_ratio,
        stage2_dropout=args.stage2_dropout,
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
    logger.info(f'Mask input index: {mask_input_index} | Label input index: {label_input_index}')

    # ---- torch.compile ----
    original_model = model
    use_compile = (
        not args.no_compile
        and device.type == 'cuda'
        and hasattr(torch, 'compile')
    )
    if use_compile:
        import logging as _logging
        _logging.getLogger('torch._inductor').setLevel(_logging.WARNING)
        _logging.getLogger('torch._dynamo').setLevel(_logging.WARNING)
        logger.info('Compiling model with torch.compile...')
        model = torch.compile(model, dynamic=True)
        logger.info('Model compiled.')
    else:
        logger.info('torch.compile disabled.')

    # ---- Optimizer (Stage 2 parameters only) ----
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * steps_per_epoch
    max_warmup_steps = 2000
    warmup_steps = min(
        int(args.warmup_fraction * total_steps), max_warmup_steps,
    )

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
        logger.info(
            f'LR schedule: {warmup_steps} warmup steps, then '
            f'ReduceLROnPlateau (factor={args.plateau_factor})',
        )
        scheduler = WarmupThenPlateauScheduler(
            optimizer,
            num_warmup_steps=warmup_steps,
            plateau_factor=args.plateau_factor,
            plateau_patience=args.plateau_patience,
            min_lr=args.min_lr,
        )

    grad_scaler = torch.amp.GradScaler('cuda') if args.amp else None

    checkpoint_manager = CheckpointManager(
        checkpoints_directory=checkpoints_dir,
        keep_best_k=args.keep_best_k,
        criterion_mode='max',
        criterion_name='R@200',
    )

    # ---- TensorBoard ----
    from torch.utils.tensorboard import SummaryWriter
    tensorboard_writer = SummaryWriter(tensorboard_dir)

    # ---- Training loop ----
    start_epoch = 1
    best_val_loss = float('inf')
    best_val_recall_at_200 = 0.0
    best_val_epoch = 0
    global_batch_count = 0
    loss_history = {
        'train': [], 'val': [], 'lr': [],
        'recall_at_10': [], 'recall_at_20': [], 'recall_at_30': [],
        'recall_at_50': [], 'recall_at_100': [], 'recall_at_200': [],
        'd_prime': [], 'median_gt_rank': [],
        'stage1_recall_at_k1': [],
    }

    if args.resume is not None:
        logger.info(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(
            args.resume, map_location=device, weights_only=False,
        )
        original_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_val_recall_at_200 = checkpoint.get(
            'best_val_recall_at_200', 0.0,
        )
        best_val_epoch = checkpoint.get('best_val_epoch', 0)
        global_batch_count = checkpoint.get('global_batch_count', 0)
        logger.info(
            f'Resumed from epoch {start_epoch - 1}, '
            f'best R@200={best_val_recall_at_200:.5f}',
        )

    logger.info(f'=== Training Cascade (K1={args.top_k1}) ===')

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            logger.info(f'=== Epoch {epoch}/{args.epochs} ===')

            train_losses, global_batch_count = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                grad_scaler, device, data_config, epoch,
                tensorboard_writer, global_batch_count,
                steps_per_epoch, mask_input_index, label_input_index,
                grad_clip_max_norm=args.grad_clip,
            )

            eval_steps = max(1, steps_per_epoch // 4)

            val_losses, val_metrics = validate(
                model, val_loader, device, data_config,
                mask_input_index, label_input_index,
                top_k1=args.top_k1,
                max_steps=eval_steps,
            )
            train_eval_losses, train_eval_metrics = validate(
                model, train_loader, device, data_config,
                mask_input_index, label_input_index,
                top_k1=args.top_k1,
                max_steps=eval_steps,
            )

            val_loss = val_losses['total_loss']
            val_recall_at_200 = val_metrics.get('recall_at_200', 0.0)
            stage1_recall = val_metrics.get('stage1_recall_at_k1', 0.0)

            is_best = val_recall_at_200 > best_val_recall_at_200
            if is_best:
                best_val_recall_at_200 = val_recall_at_200
                best_val_epoch = epoch
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            def _format_metrics(metrics):
                parts = []
                for k in [30, 50, 100, 200]:
                    key = f'recall_at_{k}'
                    if key in metrics:
                        parts.append(f'R@{k}: {metrics[key]:.4f}')
                if 'd_prime' in metrics:
                    parts.append(f"d': {metrics['d_prime']:.3f}")
                if 'median_gt_rank' in metrics:
                    rank_p90 = metrics.get('gt_rank_p90', 0.0)
                    parts.append(
                        f'rank: {metrics["median_gt_rank"]:.0f} '
                        f'(p90={rank_p90:.0f})'
                    )
                if 'stage1_recall_at_k1' in metrics:
                    parts.append(f'S1_R@K1: {metrics["stage1_recall_at_k1"]:.4f}')
                return ' | '.join(parts)

            train_summary = _format_metrics(train_eval_metrics)
            val_summary = _format_metrics(val_metrics)

            logger.info(
                f'Epoch {epoch} train_eval | '
                f'total: {train_eval_losses["total_loss"]:.5f} | '
                f'{train_summary}',
            )
            if is_best:
                logger.info(
                    f'Epoch {epoch} val | '
                    f'total: {val_loss:.5f} '
                    f'R@200: {val_recall_at_200:.4f} ★ new best | '
                    f'{val_summary}',
                )
            else:
                epochs_since_best = epoch - best_val_epoch
                logger.info(
                    f'Epoch {epoch} val | '
                    f'total: {val_loss:.5f} '
                    f'(best R@200: {best_val_recall_at_200:.4f}, '
                    f'{epochs_since_best} epochs ago) | '
                    f'{val_summary}',
                )

            previous_lr = scheduler.get_last_lr()[0]
            scheduler.step_epoch(val_loss)
            current_lr = scheduler.get_last_lr()[0]

            # TensorBoard
            tensorboard_writer.add_scalar(
                'Loss/train_epoch', train_losses['total_loss'], epoch,
            )
            tensorboard_writer.add_scalar('Loss/val_epoch', val_loss, epoch)
            for metric_key, metric_value in val_metrics.items():
                if metric_key == 'total_gt_tracks':
                    continue
                tensorboard_writer.add_scalar(
                    f'Metrics/val_{metric_key}', metric_value, epoch,
                )
            tensorboard_writer.add_scalar('LR/epoch', current_lr, epoch)

            # Loss history
            loss_history['train'].append(train_losses['total_loss'])
            loss_history['val'].append(val_loss)
            loss_history['lr'].append(current_lr)
            for prefix, metrics in [('val', val_metrics), ('train', train_eval_metrics)]:
                for metric_key, metric_value in metrics.items():
                    if metric_key == 'total_gt_tracks':
                        continue
                    history_key = f'{prefix}_{metric_key}'
                    if history_key not in loss_history:
                        loss_history[history_key] = []
                    loss_history[history_key].append(metric_value)
            save_loss_history(loss_history, experiment_dir)

            # Per-epoch metrics JSON
            epoch_metrics = {
                'epoch': epoch,
                'train_loss': train_losses['total_loss'],
                'val_loss': val_loss,
                'lr': current_lr,
                'top_k1': args.top_k1,
            }
            for metric_key, metric_value in val_metrics.items():
                epoch_metrics[f'val_{metric_key}'] = metric_value
            for metric_key, metric_value in train_eval_metrics.items():
                epoch_metrics[f'train_{metric_key}'] = metric_value
            save_epoch_metrics(epoch_metrics, experiment_dir, epoch)

            # Checkpointing
            if epoch % args.save_every == 0 or is_best or epoch == args.epochs:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': original_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_val_recall_at_200': best_val_recall_at_200,
                    'best_val_epoch': best_val_epoch,
                    'global_batch_count': global_batch_count,
                    'val_losses': val_losses,
                    'val_metrics': val_metrics,
                    'args': vars(args),
                }
                checkpoint_manager.save_checkpoint(
                    checkpoint, epoch, val_recall_at_200, is_best,
                )

    except Exception:
        logger.error(f'Training failed with exception:\n{traceback.format_exc()}')
        raise

    # ---- Final outputs ----
    tensorboard_writer.close()
    plot_loss_curves(loss_history, experiment_dir)
    logger.info(f'Training complete. Best R@200: {best_val_recall_at_200:.5f}')
    logger.info(f'Experiment: {experiment_dir}')


if __name__ == '__main__':
    main()
