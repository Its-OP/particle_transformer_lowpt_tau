"""Custom training script for tau-origin pion track finding.

Adapts pretrain_backbone.py for the TauTrackFinder model. Uses weaver's
dataset infrastructure for YAML parsing and parquet loading.

Key differences from pretraining:
    - 5 inputs (pf_points, pf_features, pf_vectors, pf_mask, pf_label)
      instead of 4. pf_label is extracted before passing to the model.
    - Model returns a loss dict during training, logits dict during inference.
    - Evaluation metrics: recall@10, recall@20, recall@30 via mask-based
      per-track scoring.
    - Pretrained backbone loading via --pretrained-backbone CLI arg.


Experiment directory layout:
    experiments/
    └── {model_name}_{timestamp}/
        ├── training.log
        ├── loss_history.json
        ├── loss_curves.png
        ├── checkpoints/
        │   ├── checkpoint_epoch_10.pt
        │   └── best_model.pt
        └── tensorboard/
            └── events.out.tfevents.*

Usage:
    python train_trackfinder.py \\
        --data-config data/low-pt/lowpt_tau_trackfinder.yaml \\
        --data-dir data/low-pt/ \\
        --network networks/lowpt_tau_TrackFinder.py \\
        --pretrained-backbone experiments/BackbonePretrain_.../checkpoints/backbone_best.pt \\
        --epochs 50 \\
        --batch-size 32 \\
        --lr 1e-4 \\
        --device cuda:0 \\
        --amp
"""
from __future__ import annotations

import argparse
import gc
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
import torch.nn.functional as functional

# Enable TensorFloat32 (TF32) for float32 matmul on Ampere+ GPUs (RTX 30xx,
# A100, RTX 40xx/50xx, H100). Uses tensor cores with ~same range as fp32 but
# reduced mantissa (10 bits vs 23). Negligible accuracy impact, significant
# speedup for all matrix multiplications (attention, linear layers, Conv1d).
torch.set_float32_matmul_precision('high')
from torch.utils.data import DataLoader

from weaver.utils.dataset import SimpleIterDataset

# Reuse infrastructure from pretrain_backbone.py
from pretrain_backbone import (
    WarmupThenCosineScheduler,
    WarmupThenPlateauScheduler,
    _TeeStream,
    build_experiment_directory,
    plot_loss_curves,
    save_loss_history,
)

# Shared utilities (canonical location: utils/training_utils.py)
from utils.training_utils import (
    CheckpointManager,
    compute_recall_at_k_metrics,
    extract_label_from_inputs,
    extract_per_track_scores,
    load_network_module,
    trim_to_max_valid_tracks,
)

logger = logging.getLogger('train_trackfinder')


def setup_logging(experiment_dir: str):
    """Configure logging to both stdout and a log file.

    Args:
        experiment_dir: Experiment root directory for the log file.
    """
    log_file = os.path.join(experiment_dir, 'training.log')
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    # Tee stderr → training.log
    log_file_handle = open(log_file, 'a')  # noqa: SIM115
    sys.stderr = _TeeStream(sys.stderr, log_file_handle)


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupThenPlateauScheduler | WarmupThenCosineScheduler,
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
    """Train for one epoch.

    Args:
        model: TauTrackFinder model.
        train_loader: DataLoader yielding (X, y, Z) tuples.
        optimizer: AdamW optimizer.
        scheduler: LR scheduler with step_batch() and step_epoch().
        grad_scaler: GradScaler for mixed precision, or None.
        device: Target device.
        data_config: Weaver DataConfig for input name ordering.
        epoch: Current epoch number (for logging).
        tensorboard_writer: Optional TensorBoard SummaryWriter.
        global_batch_count: Running batch counter across epochs.
        steps_per_epoch: Maximum number of batches per epoch.
        mask_input_index: Index of pf_mask in the inputs list.
        label_input_index: Index of pf_label in the inputs list.
        grad_clip_max_norm: Maximum gradient norm for clipping (0 to disable).

    Returns:
        Tuple of (loss_averages dict, updated global_batch_count).
    """
    model.train()
    loss_accumulators: dict[str, float] | None = None
    num_batches = 0
    start_time = time.time()

    for batch_index, (X, y, _) in enumerate(train_loader):
        if batch_index >= steps_per_epoch:
            break

        inputs = [X[k].to(device) for k in data_config.input_names]
        padded_length = inputs[0].shape[2]

        # Trim padding to max valid track count in batch
        inputs = trim_to_max_valid_tracks(inputs, mask_input_index)

        if batch_index == 0:
            trimmed_length = inputs[0].shape[2]
            logger.info(
                f'Epoch {epoch} | Trim: {padded_length} → {trimmed_length} '
                f'({100 * (1 - trimmed_length / padded_length):.0f}% '
                f'padding removed)',
            )

        # Extract pf_label from inputs before passing to model.
        # After extraction, model_inputs order is:
        # [pf_points, pf_features, pf_vectors, pf_mask]
        model_inputs, track_labels = extract_label_from_inputs(
            inputs, label_input_index,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=grad_scaler is not None):
            # model returns loss dict during training
            loss_dict = model(*model_inputs, track_labels=track_labels)
            # Remove cached logits (non-scalar) before loss accumulation
            loss_dict.pop('_per_track_logits', None)
            loss = loss_dict['total_loss']

        # Single GPU→CPU sync instead of two (isnan + isinf)
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
            if grad_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip_max_norm,
                )
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            if grad_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip_max_norm,
                )
            optimizer.step()

        scheduler.step_batch()

        # Accumulate losses on GPU to avoid per-batch GPU→CPU sync.
        # Only transfer to CPU at logging intervals.
        if loss_accumulators is None:
            loss_accumulators = {
                key: torch.zeros(1, device=loss.device)
                for key in loss_dict
            }
        for key in loss_accumulators:
            loss_accumulators[key] += loss_dict[key].detach()
        num_batches += 1
        global_batch_count += 1

        # TensorBoard + console logging at intervals only (reduces GPU→CPU syncs)
        if batch_index % 20 == 0:
            elapsed = time.time() - start_time
            current_lr = scheduler.get_last_lr()[0]
            avg_total = (
                loss_accumulators['total_loss'].item() / max(1, num_batches)
            )
            component_parts = ' | '.join(
                f'{key.replace("_loss", "")}: {value.item():.5f}'
                for key, value in loss_dict.items()
                if key != 'total_loss'
            )
            logger.info(
                f'Epoch {epoch} | Batch {batch_index} | '
                f'Loss: {loss.item():.5f} | '
                f'Avg: {avg_total:.5f} | '
                f'{component_parts} | '
                f'LR: {current_lr:.2e} | '
                f'Time: {elapsed:.1f}s',
            )

            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar(
                    'Loss/train_batch', loss.item(), global_batch_count,
                )
                for key, value in loss_dict.items():
                    if key != 'total_loss':
                        tensorboard_writer.add_scalar(
                            f'Loss/{key}_batch', value.item(),
                            global_batch_count,
                        )
                tensorboard_writer.add_scalar(
                    'LR/train', current_lr, global_batch_count,
                )

        # Free batch tensors to reduce peak memory between iterations
        del inputs, model_inputs, track_labels, loss_dict, loss

    # Average losses (transfer GPU tensors to CPU scalars)
    if loss_accumulators is None:
        loss_accumulators = {'total_loss': torch.zeros(1)}
    loss_averages = {
        key: value.item() / max(1, num_batches)
        for key, value in loss_accumulators.items()
    }
    return loss_averages, global_batch_count


def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    data_config,
    mask_input_index: int,
    label_input_index: int,
    max_steps: int | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Validate on held-out data and compute evaluation metrics.

    Args:
        model: TauTrackFinder model.
        val_loader: Validation DataLoader.
        device: Target device.
        data_config: Weaver DataConfig for input name ordering.
        mask_input_index: Index of pf_mask in the inputs list.
        label_input_index: Index of pf_label in the inputs list.
        max_steps: Maximum number of validation batches.

    Returns:
        Tuple of (loss_averages dict, metrics dict).
    """
    model.eval()
    loss_accumulators: dict[str, float] | None = None
    recall_accumulators: dict[str, float] | None = None
    num_batches = 0

    with torch.no_grad():
        for batch_index, (X, y, _) in enumerate(val_loader):
            if max_steps is not None and batch_index >= max_steps:
                break

            inputs = [X[k].to(device) for k in data_config.input_names]
            inputs = trim_to_max_valid_tracks(inputs, mask_input_index)
            model_inputs, track_labels = extract_label_from_inputs(
                inputs, label_input_index,
            )

            # Get loss and cached logits in a single forward pass.
            # The model returns '_per_track_logits' alongside loss terms
            # when track_labels is provided — avoids a second forward pass.
            model.train()
            loss_dict = model(*model_inputs, track_labels=track_labels)
            model.eval()

            per_track_scores = loss_dict.pop('_per_track_logits').detach()

            if loss_accumulators is None:
                loss_accumulators = {key: 0.0 for key in loss_dict}
            for key in loss_accumulators:
                loss_accumulators[key] += loss_dict[key].item()

            # Compute recall@K from cached logits
            mask_tensor = model_inputs[3]
            batch_metrics = compute_recall_at_k_metrics(
                per_track_scores, track_labels, mask_tensor,
            )

            if recall_accumulators is None:
                recall_accumulators = {key: 0 if key == 'total_gt_tracks' else 0.0
                                       for key in batch_metrics}
            for key in batch_metrics:
                recall_accumulators[key] += batch_metrics[key]

            num_batches += 1

            del inputs, model_inputs, track_labels, loss_dict
            del mask_tensor, batch_metrics, per_track_scores

    if loss_accumulators is None:
        loss_accumulators = {'total_loss': 0.0}
    loss_averages = {
        key: value / max(1, num_batches)
        for key, value in loss_accumulators.items()
    }

    if recall_accumulators is None:
        recall_accumulators = {'total_gt_tracks': 0}
    metrics = {}
    for key, value in recall_accumulators.items():
        if key == 'total_gt_tracks':
            metrics[key] = value
        else:
            metrics[key] = value / max(1, num_batches)

    return loss_averages, metrics


def main():
    parser = argparse.ArgumentParser(
        description='Tau-origin pion track finder training',
    )
    parser.add_argument('--data-config', type=str, required=True,
                        help='Path to YAML data config')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing parquet files')
    parser.add_argument('--val-data-dir', type=str, default=None,
                        help='Separate directory for validation parquet files. '
                             'When set, data-dir is used entirely for training '
                             'and val-data-dir entirely for validation.')
    parser.add_argument('--network', type=str, required=True,
                        help='Path to network wrapper Python file')
    parser.add_argument('--pretrained-backbone', type=str, default=None,
                        help='Path to pretrained backbone checkpoint '
                             '(backbone_best.pt or full pretrainer checkpoint)')
    parser.add_argument('--model-name', type=str, default='TrackFinder',
                        help='Short model name for experiment folder naming')
    parser.add_argument('--experiments-dir', type=str, default='experiments',
                        help='Root directory for all experiments')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (frozen backbone)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine'],
                        help='LR scheduler after warmup')
    parser.add_argument('--warmup-fraction', type=float, default=0.05,
                        help='Fraction of total steps for linear warmup '
                             '(capped at 2000 steps)')
    parser.add_argument('--plateau-factor', type=float, default=0.5,
                        help='LR reduction factor for plateau scheduler')
    parser.add_argument('--plateau-patience', type=int, default=5,
                        help='Patience for plateau scheduler')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Lower bound on learning rate')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Max gradient norm for clipping (0 to disable)')
    parser.add_argument('--train-fraction', type=float, default=0.8,
                        help='Fraction of data for training')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='DataLoader workers')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--amp', action='store_true',
                        help='Enable mixed precision training')
    parser.add_argument('--no-compile', action='store_true',
                        help='Disable torch.compile')
    parser.add_argument('--no-in-memory', action='store_true',
                        help='Stream data from disk instead of loading '
                             'entire dataset into memory. Slower but uses '
                             'much less RAM — use for local smoke tests.')
    parser.add_argument('--steps-per-epoch', type=int, default=None,
                        help='Training batches per epoch (required for '
                             'infinite SimpleIterDataset)')
    # ---- Head selection ----
    parser.add_argument('--oc', action='store_true',
                        help='Use Object Condensation head instead of DETR '
                             '(default: DETR mask-denoising head)')
    # ---- Backbone args ----
    parser.add_argument('--num-enrichment-layers', type=int, default=None,
                        help='Number of backbone enrichment layers')
    # ---- Head-specific args (only non-None values are passed to model) ----
    # OC head
    parser.add_argument('--focal-bce-weight', type=float, default=None,
                        help='[OC] Weight for focal BCE classification loss')
    parser.add_argument('--potential-loss-weight', type=float, default=None,
                        help='[OC] Weight for attractive + repulsive potential')
    parser.add_argument('--beta-loss-weight', type=float, default=None,
                        help='[OC] Weight for beta condensation+suppression')
    parser.add_argument('--clustering-dim', type=int, default=None,
                        help='[OC] Clustering space dimensionality')
    # DETR head
    parser.add_argument('--num-decoder-layers', type=int, default=None,
                        help='[DETR] Number of decoder layers')
    parser.add_argument('--num-queries', type=int, default=None,
                        help='[DETR] Number of learnable object queries')
    parser.add_argument('--mask-ce-loss-weight', type=float, default=None,
                        help='[DETR] Weight for mask cross-entropy loss')
    parser.add_argument('--confidence-loss-weight', type=float, default=None,
                        help='[DETR] Weight for confidence loss')
    parser.add_argument('--no-object-weight', type=float, default=None,
                        help='[DETR] Weight for no-object class in confidence')
    parser.add_argument('--per-track-loss-weight', type=float, default=None,
                        help='[DETR] Weight for per-track focal BCE auxiliary loss')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--keep-best-k', type=int, default=5,
                        help='Keep only the K best checkpoints by val loss. '
                             'Older checkpoints outside the top K are deleted '
                             'to save disk space. Set to 0 to keep all '
                             'checkpoints (default: 5)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # ---- Experiment directory setup ----
    resume_experiment_dir = None
    if args.resume is not None:
        resume_experiment_dir = os.path.dirname(os.path.dirname(
            os.path.abspath(args.resume),
        ))

    experiment_dir, checkpoints_dir, tensorboard_dir = (
        build_experiment_directory(
            experiments_base=args.experiments_dir,
            model_name=args.model_name,
            resume_dir=resume_experiment_dir,
        )
    )

    setup_logging(experiment_dir)
    device = torch.device(args.device)

    logger.info(f'Experiment directory: {experiment_dir}')
    logger.info(f'Arguments: {vars(args)}')

    # ---- TensorBoard ----
    from torch.utils.tensorboard import SummaryWriter
    tensorboard_writer = SummaryWriter(log_dir=tensorboard_dir)

    # ---- Data loading ----
    train_parquet_files = sorted([
        os.path.join(args.data_dir, f)
        for f in os.listdir(args.data_dir)
        if f.endswith('.parquet')
    ])
    if not train_parquet_files:
        raise FileNotFoundError(
            f'No parquet files found in {args.data_dir}',
        )
    logger.info(f'Found {len(train_parquet_files)} train parquet files in {args.data_dir}')

    train_file_dict = {'data': train_parquet_files}
    num_parquet_files = len(train_parquet_files)

    if args.val_data_dir is not None:
        # Separate train/val directories
        val_parquet_files = sorted([
            os.path.join(args.val_data_dir, f)
            for f in os.listdir(args.val_data_dir)
            if f.endswith('.parquet')
        ])
        logger.info(f'Found {len(val_parquet_files)} val parquet files in {args.val_data_dir}')
        val_file_dict = {'data': val_parquet_files}
        train_range = ((0.0, 1.0), 1.0)
        val_range = ((0.0, 1.0), 1.0)
    else:
        # Single directory — split by train_fraction
        val_file_dict = train_file_dict
        train_range = ((0.0, args.train_fraction), 1.0)
        val_range = ((args.train_fraction, 1.0), 1.0)

    # Weaver's SimpleIterDataset splits files across DataLoader workers.
    # num_workers must not exceed the number of parquet files, otherwise
    # workers with zero files crash: assert (len(new_files) > 0).
    train_num_workers = min(args.num_workers, num_parquet_files)
    if train_num_workers < args.num_workers:
        logger.warning(
            f'Reducing num_workers from {args.num_workers} to '
            f'{train_num_workers} (only {num_parquet_files} parquet files)',
        )

    load_in_memory = not args.no_in_memory
    if args.no_in_memory:
        logger.info('Streaming data from disk (--no-in-memory)')

    train_dataset = SimpleIterDataset(
        train_file_dict,
        data_config_file=args.data_config,
        for_training=True,
        load_range_and_fraction=train_range,
        fetch_by_files=True,
        fetch_step=num_parquet_files,
        in_memory=load_in_memory,
    )
    data_config = train_dataset.config

    val_dataset = SimpleIterDataset(
        val_file_dict,
        data_config_file=args.data_config,
        for_training=False,
        load_range_and_fraction=val_range,
        fetch_by_files=True,
        fetch_step=max(1, len(val_file_dict.get('data', []))),
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
    # Validation uses fewer workers than training: validation runs
    # infrequently and for fewer steps, so spawning the same number of
    # workers wastes memory and OS resources. drop_last=True avoids
    # a smaller final batch that can cause uneven GPU memory usage.
    val_num_workers = min(
        max(1, train_num_workers // 2), num_parquet_files,
    ) if train_num_workers > 0 else 0
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=val_num_workers,
        persistent_workers=val_num_workers > 0,
    )

    # ---- Steps per epoch ----
    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch is None:
        steps_per_epoch = 100
        logger.warning(
            f'--steps-per-epoch not set, defaulting to {steps_per_epoch}. '
            f'Set explicitly as floor(num_train_events / batch_size).',
        )
    logger.info(f'Steps per epoch: {steps_per_epoch}')
    logger.info(
        f'DataLoader workers: train={train_num_workers}, '
        f'val={val_num_workers}',
    )

    # ---- Model ----
    # Select network wrapper based on --oc flag (default: DETR)
    if args.oc:
        network_path = 'networks/lowpt_tau_TrackFinderOC.py'
        logger.info('Head: Object Condensation (--oc)')
    else:
        network_path = args.network
        logger.info(f'Head: DETR ({args.network})')
    network_module = load_network_module(network_path)

    model_kwargs = {}
    if args.pretrained_backbone is not None:
        model_kwargs['pretrained_backbone_path'] = args.pretrained_backbone

    # Pass only non-None head-specific args to the network wrapper.
    # Each wrapper's get_model() pops what it needs and ignores the rest.
    _head_arg_names = [
        # backbone
        'num_enrichment_layers',
        # OC
        'focal_bce_weight', 'potential_loss_weight', 'beta_loss_weight',
        'clustering_dim',
        # DETR
        'num_decoder_layers', 'num_queries',
        'mask_ce_loss_weight', 'confidence_loss_weight',
        'per_track_loss_weight', 'no_object_weight',
    ]
    for arg_name in _head_arg_names:
        value = getattr(args, arg_name, None)
        if value is not None:
            model_kwargs[arg_name] = value

    model, model_info = network_module.get_model(data_config, **model_kwargs)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(
        f'Total parameters: {total_params:,} | '
        f'Trainable: {trainable_params:,} | '
        f'Frozen: {total_params - trainable_params:,}',
    )
    logger.info(f'Input names: {data_config.input_names}')
    logger.info(f'Head kwargs: {model_kwargs}')

    # Find input indices for pf_mask and pf_label
    input_names = list(data_config.input_names)
    mask_input_index = input_names.index('pf_mask')
    label_input_index = input_names.index('pf_label')
    logger.info(
        f'Mask input index: {mask_input_index} | '
        f'Label input index: {label_input_index}',
    )

    # Keep reference to uncompiled model for state_dict access
    original_model = model

    # ---- torch.compile ----
    use_compile = (
        not args.no_compile
        and device.type == 'cuda'
        and hasattr(torch, 'compile')
    )
    if use_compile:
        # Silence verbose inductor/dynamo benchmarking logs during
        # max-autotune compilation (thousands of lines otherwise).
        import logging as _logging
        _logging.getLogger('torch._inductor').setLevel(_logging.WARNING)
        _logging.getLogger('torch._dynamo').setLevel(_logging.WARNING)

        logger.info(
            'Compiling model with torch.compile (dynamic=True)...',
        )
        model = torch.compile(model, dynamic=True, mode=None)
        logger.info('Model compiled.')
    else:
        logger.info('torch.compile disabled.')

    # ---- Optimizer (only trainable params) ----
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
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
            f'LR schedule: {warmup_steps} warmup steps '
            f'(~{warmup_epochs} epochs), then CosineAnnealingLR '
            f'over {num_post_warmup_epochs} epochs',
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
            f'ReduceLROnPlateau (factor={args.plateau_factor}, '
            f'patience={args.plateau_patience})',
        )
        scheduler = WarmupThenPlateauScheduler(
            optimizer,
            num_warmup_steps=warmup_steps,
            plateau_factor=args.plateau_factor,
            plateau_patience=args.plateau_patience,
            min_lr=args.min_lr,
        )

    # ---- Mixed precision ----
    grad_scaler = torch.amp.GradScaler('cuda') if args.amp else None

    # ---- Checkpoint manager ----
    checkpoint_manager = CheckpointManager(
        checkpoints_directory=checkpoints_dir,
        keep_best_k=args.keep_best_k,
    )
    logger.info(
        f'Checkpoint management: keep_best_k={args.keep_best_k} '
        f'({"unlimited" if args.keep_best_k == 0 else f"top {args.keep_best_k}"})',
    )

    # ---- Resume from checkpoint ----
    start_epoch = 1
    best_val_loss = float('inf')
    best_val_epoch = 0
    global_batch_count = 0
    loss_history = {
        'train': [], 'val': [], 'lr': [],
        'recall_at_10': [], 'recall_at_20': [], 'recall_at_30': [],
        'recall_at_100': [], 'd_prime': [], 'median_gt_rank': [],
    }

    if args.resume is not None:
        logger.info(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(
            args.resume, map_location=device, weights_only=False,
        )
        original_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_val_epoch = checkpoint.get('best_val_epoch', 0)
        global_batch_count = checkpoint.get('global_batch_count', 0)
        loss_history = checkpoint.get('loss_history', loss_history)
        logger.info(
            f'Resumed at epoch {start_epoch}, '
            f'best_val_loss={best_val_loss:.5f} (epoch {best_val_epoch})',
        )

    # ---- Phase 1: Main training ----
    logger.info('=== Phase 1: Training ===')

    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f'=== Epoch {epoch}/{args.epochs} ===')

        train_losses, global_batch_count = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            grad_scaler, device, data_config, epoch,
            tensorboard_writer, global_batch_count,
            steps_per_epoch=steps_per_epoch,
            mask_input_index=mask_input_index,
            label_input_index=label_input_index,
            grad_clip_max_norm=args.grad_clip,
        )
        train_components = ' | '.join(
            f'{key.replace("_loss", "")}: {value:.5f}'
            for key, value in train_losses.items()
            if key != 'total_loss'
        )
        logger.info(
            f'Epoch {epoch} train | '
            f'total: {train_losses["total_loss"]:.5f} | '
            f'{train_components}',
        )

        # Free training memory before validation to reduce peak usage.
        # gc.collect() releases Python-side references (e.g. cached autograd
        # graphs) so CUDA can reclaim the underlying device memory.
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Validation
        val_steps = max(1, steps_per_epoch // 4)
        val_losses, val_metrics = validate(
            model, val_loader, device, data_config,
            mask_input_index=mask_input_index,
            label_input_index=label_input_index,
            max_steps=val_steps,
        )
        val_loss = val_losses['total_loss']

        # Free validation memory before resuming training
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_val_epoch = epoch

        perfect_30 = val_metrics.get('perfect_at_30', 0.0)
        val_summary = (
            f'R@10: {val_metrics["recall_at_10"]:.4f} | '
            f'R@20: {val_metrics["recall_at_20"]:.4f} | '
            f'R@30: {val_metrics["recall_at_30"]:.4f} | '
            f'R@100: {val_metrics["recall_at_100"]:.4f} | '
            f'P@30: {perfect_30:.4f} | '
            f'd\': {val_metrics["d_prime"]:.3f} | '
            f'rank: {val_metrics["median_gt_rank"]:.0f}'
        )
        if is_best:
            logger.info(
                f'Epoch {epoch} val | '
                f'total: {val_loss:.5f} ★ new best | '
                f'{val_summary}',
            )
        else:
            epochs_since_best = epoch - best_val_epoch
            logger.info(
                f'Epoch {epoch} val | '
                f'total: {val_loss:.5f} '
                f'(best: {best_val_loss:.5f}, '
                f'{epochs_since_best} epochs ago) | '
                f'{val_summary}',
            )

        # LR scheduler step
        previous_lr = scheduler.get_last_lr()[0]
        scheduler.step_epoch(val_loss)
        current_lr = scheduler.get_last_lr()[0]
        if current_lr < previous_lr and args.scheduler == 'plateau':
            logger.info(
                f'ReduceLROnPlateau: LR {previous_lr:.2e} → {current_lr:.2e}',
            )

        # TensorBoard: per-epoch logging
        tensorboard_writer.add_scalar(
            'Loss/train_epoch', train_losses['total_loss'], epoch,
        )
        tensorboard_writer.add_scalar(
            'Loss/val_epoch', val_loss, epoch,
        )
        for key, value in val_losses.items():
            if key != 'total_loss':
                tensorboard_writer.add_scalar(f'Loss/val_{key}', value, epoch)
        for metric_key, metric_value in val_metrics.items():
            if metric_key == 'total_gt_tracks':
                continue
            tensorboard_writer.add_scalar(
                f'Metrics/{metric_key}', metric_value, epoch,
            )
        tensorboard_writer.add_scalar('LR/epoch', current_lr, epoch)

        # Loss history (head-specific keys added dynamically)
        loss_history['train'].append(train_losses['total_loss'])
        loss_history['val'].append(val_loss)
        loss_history['lr'].append(current_lr)
        for key, value in val_losses.items():
            if key == 'total_loss':
                continue
            short_key = key.replace('_loss', '')
            if short_key not in loss_history:
                loss_history[short_key] = []
            loss_history[short_key].append(value)
        for metric_key, metric_value in val_metrics.items():
            if metric_key == 'total_gt_tracks':
                continue
            if metric_key not in loss_history:
                loss_history[metric_key] = []
            loss_history[metric_key].append(metric_value)
        save_loss_history(loss_history, experiment_dir)

        # Checkpointing
        if epoch % args.save_every == 0 or is_best or epoch == args.epochs:
            checkpoint = {
                'epoch': epoch,
                'phase': 'frozen_backbone',
                'model_state_dict': original_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_losses['total_loss'],
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'best_val_epoch': best_val_epoch,
                'global_batch_count': global_batch_count,
                'loss_history': loss_history,
                'val_metrics': val_metrics,
                'args': vars(args),
            }

            checkpoint_manager.save_checkpoint(
                checkpoint, epoch, val_loss, is_best,
            )

    # ---- Final outputs ----
    tensorboard_writer.close()
    plot_loss_curves(loss_history, experiment_dir)
    logger.info(f'Training complete. Best val loss: {best_val_loss:.5f}')
    logger.info(f'Experiment: {experiment_dir}')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        logger.error(
            f'Training failed with exception:\n{traceback.format_exc()}',
        )
        raise
