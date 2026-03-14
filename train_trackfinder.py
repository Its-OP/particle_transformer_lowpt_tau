"""Custom training script for tau-origin pion track finding.

Adapts pretrain_backbone.py for the TauTrackFinder model. Uses weaver's
dataset infrastructure for YAML parsing and parquet loading.

Key differences from pretraining:
    - 5 inputs (pf_points, pf_features, pf_vectors, pf_mask, pf_label)
      instead of 4. pf_label is extracted before passing to the model.
    - Model returns a loss dict during training, logits dict during inference.
    - Evaluation metrics: track-finding efficiency, fake rate, perfect event
      rate, confidence accuracy.
    - Pretrained backbone loading via --pretrained-backbone CLI arg.
    - Optional two-phase training: frozen backbone → unfrozen fine-tuning.

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

import torch
import torch.nn.functional as functional

# Enable TensorFloat32 (TF32) for float32 matmul on Ampere+ GPUs (RTX 30xx,
# A100, RTX 40xx/50xx, H100). Uses tensor cores with ~same range as fp32 but
# reduced mantissa (10 bits vs 23). Negligible accuracy impact, significant
# speedup for all matrix multiplications (attention, linear layers, Conv1d).
torch.set_float32_matmul_precision('high')
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from weaver.utils.dataset import SimpleIterDataset

# Reuse infrastructure from pretrain_backbone.py
from pretrain_backbone import (
    WarmupThenCosineScheduler,
    WarmupThenPlateauScheduler,
    _TeeStream,
    build_experiment_directory,
    load_network_module,
    plot_loss_curves,
    save_loss_history,
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


@torch.no_grad()
def compute_evaluation_metrics(
    pointer_logits: torch.Tensor,
    confidence_logits: torch.Tensor,
    track_labels: torch.Tensor,
    mask: torch.Tensor,
    confidence_threshold: float = 0.0,
) -> dict[str, float]:
    """Compute track-finding evaluation metrics.

    Metrics:
        - track_finding_efficiency: fraction of GT tracks found by any
          confident query (i.e. recall).
        - fake_rate: fraction of confident queries not pointing to any
          GT track (i.e. 1 - precision among confident queries).
        - perfect_event_rate: fraction of events where all GT tracks are
          found exactly (no missed, no fakes).
        - confidence_accuracy: binary accuracy of exists/empty classification
          using the Hungarian-optimal assignment as ground truth.

    Args:
        pointer_logits: (B, num_queries, P) pointer scores.
        confidence_logits: (B, num_queries) confidence scores (pre-sigmoid).
        track_labels: (B, 1, P) binary labels (1.0 = tau track).
        mask: (B, 1, P) boolean mask (True = valid track).
        confidence_threshold: Sigmoid threshold for "confident" queries
            (default: 0.0 logit = 0.5 probability).

    Returns:
        Dict of metric name → float value.
    """
    batch_size = pointer_logits.shape[0]

    # Predicted track index for each query: argmax over valid tracks
    # (padded positions are -inf, so argmax ignores them)
    predicted_track_indices = pointer_logits.argmax(dim=2)  # (B, num_queries)

    # Confident queries: sigmoid(confidence_logits) > 0.5
    confident_mask = confidence_logits > confidence_threshold  # (B, num_queries)

    # Ground truth track indices
    labels_flat = track_labels.squeeze(1) * mask.squeeze(1).float()  # (B, P)

    total_gt_tracks = 0
    found_gt_tracks = 0
    total_confident_queries = 0
    fake_confident_queries = 0
    perfect_events = 0
    total_events_with_gt = 0

    for batch_index in range(batch_size):
        # GT track indices for this event
        gt_positions = labels_flat[batch_index].nonzero(as_tuple=True)[0]
        gt_set = set(gt_positions.tolist())
        num_gt = len(gt_set)

        # Confident query predictions for this event
        event_confident = confident_mask[batch_index]  # (num_queries,)
        event_predictions = predicted_track_indices[batch_index]  # (num_queries,)

        confident_predictions = event_predictions[event_confident].tolist()

        # Track-finding efficiency (recall)
        total_gt_tracks += num_gt
        found_set = set(confident_predictions) & gt_set
        found_gt_tracks += len(found_set)

        # Fake rate (among confident queries)
        num_confident = len(confident_predictions)
        total_confident_queries += num_confident
        num_fakes = sum(
            1 for prediction in confident_predictions
            if prediction not in gt_set
        )
        fake_confident_queries += num_fakes

        # Perfect event rate
        if num_gt > 0:
            total_events_with_gt += 1
            is_perfect = (
                found_set == gt_set  # all GT found
                and num_fakes == 0   # no fakes
            )
            if is_perfect:
                perfect_events += 1

    track_finding_efficiency = (
        found_gt_tracks / max(1, total_gt_tracks)
    )
    fake_rate = (
        fake_confident_queries / max(1, total_confident_queries)
    )
    perfect_event_rate = (
        perfect_events / max(1, total_events_with_gt)
    )

    return {
        'track_finding_efficiency': track_finding_efficiency,
        'fake_rate': fake_rate,
        'perfect_event_rate': perfect_event_rate,
        'total_gt_tracks': total_gt_tracks,
        'found_gt_tracks': found_gt_tracks,
        'total_confident_queries': total_confident_queries,
        'fake_confident_queries': fake_confident_queries,
    }


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
    loss_accumulators = {
        'total_loss': 0.0,
        'pointer_focal_loss': 0.0,
        'confidence_bce_loss': 0.0,
    }
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
            loss = loss_dict['total_loss']

        # Skip batches with NaN/Inf loss
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(
                f'Epoch {epoch} | Batch {batch_index} | '
                f'Skipping batch with '
                f'{"NaN" if torch.isnan(loss) else "Inf"} loss',
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

        # Accumulate losses
        for key in loss_accumulators:
            loss_accumulators[key] += loss_dict[key].item()
        num_batches += 1
        global_batch_count += 1

        # TensorBoard: per-batch logging
        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar(
                'Loss/train_batch', loss.item(), global_batch_count,
            )
            tensorboard_writer.add_scalar(
                'Loss/pointer_focal_batch',
                loss_dict['pointer_focal_loss'].item(),
                global_batch_count,
            )
            tensorboard_writer.add_scalar(
                'Loss/confidence_bce_batch',
                loss_dict['confidence_bce_loss'].item(),
                global_batch_count,
            )
            tensorboard_writer.add_scalar(
                'LR/train', scheduler.get_last_lr()[0], global_batch_count,
            )

        if batch_index % 20 == 0:
            elapsed = time.time() - start_time
            current_lr = scheduler.get_last_lr()[0]
            avg_total = loss_accumulators['total_loss'] / max(1, num_batches)
            logger.info(
                f'Epoch {epoch} | Batch {batch_index} | '
                f'Loss: {loss.item():.5f} | '
                f'Avg Loss: {avg_total:.5f} | '
                f'Ptr: {loss_dict["pointer_focal_loss"].item():.5f} | '
                f'Conf: {loss_dict["confidence_bce_loss"].item():.5f} | '
                f'LR: {current_lr:.2e} | '
                f'Time: {elapsed:.1f}s',
            )

        # Free batch tensors to reduce peak memory between iterations
        del inputs, model_inputs, track_labels, loss_dict, loss

    # Average losses
    loss_averages = {
        key: value / max(1, num_batches)
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
    loss_accumulators = {
        'total_loss': 0.0,
        'pointer_focal_loss': 0.0,
        'confidence_bce_loss': 0.0,
    }
    # Aggregate metrics across batches
    metrics_accumulators = {
        'total_gt_tracks': 0,
        'found_gt_tracks': 0,
        'total_confident_queries': 0,
        'fake_confident_queries': 0,
    }
    perfect_events = 0
    total_events_with_gt = 0
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

            # Get loss in training mode temporarily
            model.train()
            loss_dict = model(*model_inputs, track_labels=track_labels)
            model.eval()

            for key in loss_accumulators:
                loss_accumulators[key] += loss_dict[key].item()

            # Get logits for evaluation metrics
            logits_dict = model(*model_inputs)
            mask_tensor = model_inputs[
                # After label extraction, mask index may shift.
                # pf_mask is always at index 3 in model_inputs
                # (pf_points=0, pf_features=1, pf_vectors=2, pf_mask=3)
                3
            ]

            batch_metrics = compute_evaluation_metrics(
                logits_dict['pointer_logits'],
                logits_dict['confidence_logits'],
                track_labels,
                mask_tensor,
            )

            for key in ['total_gt_tracks', 'found_gt_tracks',
                        'total_confident_queries', 'fake_confident_queries']:
                metrics_accumulators[key] += batch_metrics[key]

            num_batches += 1

            # Free batch tensors to reduce peak memory between iterations
            del inputs, model_inputs, track_labels, loss_dict
            del logits_dict, mask_tensor, batch_metrics

    # Average losses
    loss_averages = {
        key: value / max(1, num_batches)
        for key, value in loss_accumulators.items()
    }

    # Aggregate metrics
    metrics = {
        'track_finding_efficiency': (
            metrics_accumulators['found_gt_tracks']
            / max(1, metrics_accumulators['total_gt_tracks'])
        ),
        'fake_rate': (
            metrics_accumulators['fake_confident_queries']
            / max(1, metrics_accumulators['total_confident_queries'])
        ),
        'total_gt_tracks': metrics_accumulators['total_gt_tracks'],
        'found_gt_tracks': metrics_accumulators['found_gt_tracks'],
    }

    return loss_averages, metrics


def main():
    parser = argparse.ArgumentParser(
        description='Tau-origin pion track finder training',
    )
    parser.add_argument('--data-config', type=str, required=True,
                        help='Path to YAML data config')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing parquet files')
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
    parser.add_argument('--finetune-epochs', type=int, default=0,
                        help='Number of fine-tuning epochs with unfrozen '
                             'backbone (0 = skip fine-tuning phase)')
    parser.add_argument('--finetune-lr-factor', type=float, default=0.1,
                        help='LR multiplier for fine-tuning phase '
                             '(applied to --lr)')
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
    parser.add_argument('--steps-per-epoch', type=int, default=None,
                        help='Training batches per epoch (required for '
                             'infinite SimpleIterDataset)')
    parser.add_argument('--num-enrichment-layers', type=int, default=None,
                        help='Number of backbone enrichment layers')
    parser.add_argument('--num-encoder-layers', type=int, default=None,
                        help='Number of compact token encoder layers')
    parser.add_argument('--num-decoder-layers', type=int, default=None,
                        help='Number of query decoder layers')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs')
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
    tensorboard_writer = SummaryWriter(log_dir=tensorboard_dir)

    # ---- Data loading ----
    parquet_files = sorted([
        os.path.join(args.data_dir, f)
        for f in os.listdir(args.data_dir)
        if f.endswith('.parquet')
    ])
    if not parquet_files:
        raise FileNotFoundError(
            f'No parquet files found in {args.data_dir}',
        )
    logger.info(f'Found {len(parquet_files)} parquet files')

    file_dict = {'data': parquet_files}
    num_parquet_files = len(parquet_files)

    # Weaver's SimpleIterDataset splits files across DataLoader workers.
    # num_workers must not exceed the number of parquet files, otherwise
    # workers with zero files crash: assert (len(new_files) > 0).
    train_num_workers = min(args.num_workers, num_parquet_files)
    if train_num_workers < args.num_workers:
        logger.warning(
            f'Reducing num_workers from {args.num_workers} to '
            f'{train_num_workers} (only {num_parquet_files} parquet files)',
        )

    train_dataset = SimpleIterDataset(
        file_dict,
        data_config_file=args.data_config,
        for_training=True,
        load_range_and_fraction=((0.0, args.train_fraction), 1.0),
        fetch_by_files=True,
        fetch_step=num_parquet_files,
        in_memory=True,
    )
    data_config = train_dataset.config

    val_dataset = SimpleIterDataset(
        file_dict,
        data_config_file=args.data_config,
        for_training=False,
        load_range_and_fraction=((args.train_fraction, 1.0), 1.0),
        fetch_by_files=True,
        fetch_step=num_parquet_files,
        in_memory=True,
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
    )
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
    network_module = load_network_module(args.network)
    model_kwargs = {}
    if args.pretrained_backbone is not None:
        model_kwargs['pretrained_backbone_path'] = args.pretrained_backbone
    if args.num_enrichment_layers is not None:
        model_kwargs['num_enrichment_layers'] = args.num_enrichment_layers
    if args.num_encoder_layers is not None:
        model_kwargs['num_encoder_layers'] = args.num_encoder_layers
    if args.num_decoder_layers is not None:
        model_kwargs['num_decoder_layers'] = args.num_decoder_layers

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
        logger.info(
            'Compiling model with torch.compile '
            '(mode="default", dynamic=True)...',
        )
        model = torch.compile(model, mode='max-autotune', dynamic=True)
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

    # ---- Resume from checkpoint ----
    start_epoch = 1
    best_val_loss = float('inf')
    best_val_epoch = 0
    global_batch_count = 0
    loss_history = {
        'train': [], 'val': [], 'lr': [],
        'pointer_focal': [], 'confidence_bce': [],
        'track_finding_efficiency': [], 'fake_rate': [],
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

    # ---- Phase 1: Training with frozen backbone ----
    logger.info('=== Phase 1: Training with frozen backbone ===')

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
        logger.info(
            f'Epoch {epoch} train | '
            f'total: {train_losses["total_loss"]:.5f} | '
            f'ptr: {train_losses["pointer_focal_loss"]:.5f} | '
            f'conf: {train_losses["confidence_bce_loss"]:.5f}',
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

        if is_best:
            logger.info(
                f'Epoch {epoch} val | '
                f'total: {val_loss:.5f} ★ new best | '
                f'efficiency: {val_metrics["track_finding_efficiency"]:.4f} | '
                f'fake_rate: {val_metrics["fake_rate"]:.4f}',
            )
        else:
            epochs_since_best = epoch - best_val_epoch
            logger.info(
                f'Epoch {epoch} val | '
                f'total: {val_loss:.5f} '
                f'(best: {best_val_loss:.5f}, '
                f'{epochs_since_best} epochs ago) | '
                f'efficiency: {val_metrics["track_finding_efficiency"]:.4f} | '
                f'fake_rate: {val_metrics["fake_rate"]:.4f}',
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
        tensorboard_writer.add_scalar(
            'Loss/val_pointer_focal', val_losses['pointer_focal_loss'], epoch,
        )
        tensorboard_writer.add_scalar(
            'Loss/val_confidence_bce', val_losses['confidence_bce_loss'], epoch,
        )
        tensorboard_writer.add_scalar(
            'Metrics/track_finding_efficiency',
            val_metrics['track_finding_efficiency'], epoch,
        )
        tensorboard_writer.add_scalar(
            'Metrics/fake_rate', val_metrics['fake_rate'], epoch,
        )
        tensorboard_writer.add_scalar('LR/epoch', current_lr, epoch)

        # Loss history
        loss_history['train'].append(train_losses['total_loss'])
        loss_history['val'].append(val_loss)
        loss_history['lr'].append(current_lr)
        loss_history['pointer_focal'].append(
            val_losses['pointer_focal_loss'],
        )
        loss_history['confidence_bce'].append(
            val_losses['confidence_bce_loss'],
        )
        loss_history['track_finding_efficiency'].append(
            val_metrics['track_finding_efficiency'],
        )
        loss_history['fake_rate'].append(val_metrics['fake_rate'])
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

            checkpoint_path = os.path.join(
                checkpoints_dir, f'checkpoint_epoch_{epoch}.pt',
            )
            torch.save(checkpoint, checkpoint_path)
            logger.info(f'Saved checkpoint: {checkpoint_path}')

            if is_best:
                best_path = os.path.join(checkpoints_dir, 'best_model.pt')
                torch.save(checkpoint, best_path)
                logger.info(f'New best model (val_loss={val_loss:.5f})')

    # ---- Phase 2: Fine-tuning with unfrozen backbone (optional) ----
    if args.finetune_epochs > 0:
        logger.info('=== Phase 2: Fine-tuning with unfrozen backbone ===')

        # Unfreeze backbone
        for param in original_model.backbone.parameters():
            param.requires_grad = True

        finetune_trainable = sum(
            p.numel() for p in original_model.parameters()
            if p.requires_grad
        )
        logger.info(f'Unfrozen backbone. Trainable params: {finetune_trainable:,}')

        # New optimizer with lower LR for backbone
        finetune_lr = args.lr * args.finetune_lr_factor
        optimizer = torch.optim.AdamW(
            [
                {
                    'params': [
                        p for p in original_model.backbone.parameters()
                        if p.requires_grad
                    ],
                    'lr': finetune_lr,
                },
                {
                    'params': [
                        p for p in original_model.head.parameters()
                        if p.requires_grad
                    ],
                    'lr': finetune_lr * 10,  # Head keeps higher LR
                },
            ],
            weight_decay=args.weight_decay,
        )

        finetune_total_steps = args.finetune_epochs * steps_per_epoch
        finetune_warmup_steps = min(
            int(0.05 * finetune_total_steps), 500,
        )

        scheduler = WarmupThenCosineScheduler(
            optimizer,
            num_warmup_steps=finetune_warmup_steps,
            num_post_warmup_epochs=args.finetune_epochs,
            min_lr=args.min_lr,
        )
        logger.info(
            f'Fine-tune LR: backbone={finetune_lr:.2e}, '
            f'head={finetune_lr * 10:.2e}, '
            f'{args.finetune_epochs} epochs',
        )

        # Recompile if using torch.compile
        if use_compile:
            model = torch.compile(
                original_model, mode='max-autotune', dynamic=True,
            )

        finetune_start = args.epochs + 1
        finetune_end = args.epochs + args.finetune_epochs

        for epoch in range(finetune_start, finetune_end + 1):
            logger.info(
                f'=== Fine-tune Epoch {epoch - args.epochs}/'
                f'{args.finetune_epochs} (global {epoch}) ===',
            )

            train_losses, global_batch_count = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                grad_scaler, device, data_config, epoch,
                tensorboard_writer, global_batch_count,
                steps_per_epoch=steps_per_epoch,
                mask_input_index=mask_input_index,
                label_input_index=label_input_index,
                grad_clip_max_norm=args.grad_clip,
            )
            logger.info(
                f'Epoch {epoch} train | '
                f'total: {train_losses["total_loss"]:.5f} | '
                f'ptr: {train_losses["pointer_focal_loss"]:.5f} | '
                f'conf: {train_losses["confidence_bce_loss"]:.5f}',
            )

            # Free training memory before validation
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

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

            logger.info(
                f'Epoch {epoch} val | '
                f'total: {val_loss:.5f}'
                f'{" ★ new best" if is_best else ""} | '
                f'efficiency: {val_metrics["track_finding_efficiency"]:.4f} | '
                f'fake_rate: {val_metrics["fake_rate"]:.4f}',
            )

            previous_lr = scheduler.get_last_lr()[0]
            scheduler.step_epoch(val_loss)
            current_lr = scheduler.get_last_lr()[0]

            # TensorBoard
            tensorboard_writer.add_scalar(
                'Loss/train_epoch', train_losses['total_loss'], epoch,
            )
            tensorboard_writer.add_scalar('Loss/val_epoch', val_loss, epoch)
            tensorboard_writer.add_scalar(
                'Metrics/track_finding_efficiency',
                val_metrics['track_finding_efficiency'], epoch,
            )
            tensorboard_writer.add_scalar(
                'Metrics/fake_rate', val_metrics['fake_rate'], epoch,
            )
            tensorboard_writer.add_scalar('LR/epoch', current_lr, epoch)

            # Loss history
            loss_history['train'].append(train_losses['total_loss'])
            loss_history['val'].append(val_loss)
            loss_history['lr'].append(current_lr)
            loss_history['pointer_focal'].append(
                val_losses['pointer_focal_loss'],
            )
            loss_history['confidence_bce'].append(
                val_losses['confidence_bce_loss'],
            )
            loss_history['track_finding_efficiency'].append(
                val_metrics['track_finding_efficiency'],
            )
            loss_history['fake_rate'].append(val_metrics['fake_rate'])
            save_loss_history(loss_history, experiment_dir)

            # Checkpointing
            if (epoch % args.save_every == 0
                    or is_best or epoch == finetune_end):
                checkpoint = {
                    'epoch': epoch,
                    'phase': 'finetune',
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

                checkpoint_path = os.path.join(
                    checkpoints_dir, f'checkpoint_epoch_{epoch}.pt',
                )
                torch.save(checkpoint, checkpoint_path)
                logger.info(f'Saved checkpoint: {checkpoint_path}')

                if is_best:
                    best_path = os.path.join(
                        checkpoints_dir, 'best_model.pt',
                    )
                    torch.save(checkpoint, best_path)
                    logger.info(
                        f'New best model (val_loss={val_loss:.5f})',
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
