"""Custom training script for backbone pretraining via masked track reconstruction.

Bypasses weaver's training loop to avoid fighting the regression mode interface.
Uses weaver's dataset infrastructure for YAML parsing and parquet loading.

Features:
    - torch.compile for optimized GPU kernels (enabled by default on CUDA)
    - Mixed precision (AMP) support
    - TensorBoard logging (per-batch and per-epoch scalars)
    - JSON loss history export (for plotting without TensorBoard)
    - File logging alongside stdout
    - Checkpointing with backbone-only weight extraction
    - Resume from checkpoint

Experiment directory layout:
    experiments/
    └── {model_name}_{timestamp}/
        ├── training.log          # full console output
        ├── loss_history.json     # per-epoch loss values
        ├── loss_curves.png       # generated after training
        ├── checkpoints/
        │   ├── checkpoint_epoch_10.pt
        │   ├── best_model.pt
        │   └── backbone_best.pt
        └── tensorboard/
            └── events.out.tfevents.*

Usage:
    python pretrain_backbone.py \\
        --data-config data/low-pt/lowpt_tau_pretrain.yaml \\
        --data-dir data/low-pt/ \\
        --network networks/lowpt_tau_BackbonePretrain.py \\
        --epochs 100 \\
        --batch-size 32 \\
        --lr 1e-3 \\
        --device cuda:0 \\
        --amp \\
        --no-compile  # optional: disable torch.compile
"""
import argparse
import importlib.util
import json
import logging
import math
import os
import sys
import time
import traceback
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from weaver.utils.dataset import SimpleIterDataset

logger = logging.getLogger('pretrain_backbone')


def build_experiment_directory(
    experiments_base: str,
    model_name: str,
    resume_dir: str | None,
) -> tuple[str, str, str]:
    """Create or resolve the experiment directory structure.

    Layout:
        {experiments_base}/{model_name}_{timestamp}/
            ├── checkpoints/
            └── tensorboard/

    When resuming, reuses the existing experiment directory.

    Args:
        experiments_base: Root experiments folder (e.g. 'experiments').
        model_name: Short model identifier (e.g. 'BackbonePretrain').
        resume_dir: If resuming, path to the existing experiment root.

    Returns:
        Tuple of (experiment_dir, checkpoints_dir, tensorboard_dir).
    """
    if resume_dir is not None:
        experiment_dir = resume_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f'{model_name}_{timestamp}'
        experiment_dir = os.path.join(experiments_base, experiment_name)

    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    tensorboard_dir = os.path.join(experiment_dir, 'tensorboard')

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    return experiment_dir, checkpoints_dir, tensorboard_dir


class _TeeStream:
    """Write stream that duplicates output to both original stream and a file.

    Used to redirect stderr so that Python warnings (e.g. RuntimeWarning from
    numpy overflow), DataLoader worker tracebacks, and any other stderr output
    are captured in training.log alongside the structured log messages.
    """

    def __init__(self, original_stream, log_file_handle):
        self.original_stream = original_stream
        self.log_file_handle = log_file_handle

    def write(self, message):
        self.original_stream.write(message)
        self.log_file_handle.write(message)
        self.log_file_handle.flush()

    def flush(self):
        self.original_stream.flush()
        self.log_file_handle.flush()

    def fileno(self):
        return self.original_stream.fileno()

    def isatty(self):
        return self.original_stream.isatty()


def setup_logging(experiment_dir: str):
    """Configure logging to both stdout and a log file in the experiment root.

    Sets up two output channels:
    1. Structured logging (logger.*) → both stdout and training.log
    2. stderr tee → training.log also captures Python warnings, tracebacks,
       and any other unstructured error output from libraries / subprocesses.

    Note: We configure the logger directly instead of using basicConfig(),
    because basicConfig() is silently ignored if any other library (e.g.
    numexpr) has already initialised the root logger.

    Args:
        experiment_dir: Experiment root directory for the log file.
    """
    log_file = os.path.join(experiment_dir, 'training.log')
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Configure our logger (not root) so it works regardless of import order
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    # Tee stderr → training.log so that Python warnings (e.g. RuntimeWarning),
    # DataLoader worker errors, and unhandled tracebacks are also captured.
    log_file_handle = open(log_file, 'a')  # noqa: SIM115 — kept open for process lifetime
    sys.stderr = _TeeStream(sys.stderr, log_file_handle)


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


def build_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine annealing with linear warmup.

    Learning rate schedule:
        - Linear warmup: lr scales from 0 to base_lr over num_warmup_steps
        - Cosine decay: lr follows cos(π × progress / 2) from base_lr to 0

    Args:
        optimizer: Optimizer to schedule.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.

    Returns:
        LambdaLR scheduler.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup: 0 → 1
            return current_step / max(1, num_warmup_steps)
        # Cosine decay: 1 → 0
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    grad_scaler: torch.amp.GradScaler | None,
    device: torch.device,
    data_config,
    epoch: int,
    tensorboard_writer: SummaryWriter | None,
    global_batch_count: int,
    grad_clip_max_norm: float = 1.0,
) -> tuple[float, int]:
    """Train for one epoch.

    Args:
        model: MaskedTrackPretrainer.
        train_loader: DataLoader yielding (X, y, Z) tuples.
        optimizer: AdamW optimizer.
        scheduler: Learning rate scheduler.
        grad_scaler: GradScaler for mixed precision, or None.
        device: Target device.
        data_config: Weaver DataConfig for input name ordering.
        epoch: Current epoch number (for logging).
        tensorboard_writer: Optional TensorBoard SummaryWriter.
        global_batch_count: Running batch counter across epochs.
        grad_clip_max_norm: Maximum gradient norm for clipping (0 to disable).

    Returns:
        Tuple of (average_loss, updated_global_batch_count).
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    for batch_idx, (X, y, _) in enumerate(train_loader):
        inputs = [X[k].to(device) for k in data_config.input_names]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=grad_scaler is not None):
            # model returns (B,) per-event losses
            per_event_loss = model(*inputs)
            loss = per_event_loss.mean()

        # Skip batches with NaN loss — prevents poisoning model weights
        # and the running average. Log the occurrence for debugging.
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(
                f'Epoch {epoch} | Batch {batch_idx} | '
                f'Skipping batch with {"NaN" if torch.isnan(loss) else "Inf"} loss'
            )
            optimizer.zero_grad(set_to_none=True)
            global_batch_count += 1
            continue

        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            # Unscale before clipping so clip threshold is in true gradient scale
            grad_scaler.unscale_(optimizer)
            if grad_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip_max_norm
                )
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            if grad_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip_max_norm
                )
            optimizer.step()

        scheduler.step()

        batch_loss = loss.item()
        total_loss += batch_loss
        num_batches += 1
        global_batch_count += 1

        # TensorBoard: per-batch logging
        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar(
                'Loss/train_batch', batch_loss, global_batch_count
            )
            tensorboard_writer.add_scalar(
                'LR/train', scheduler.get_last_lr()[0], global_batch_count
            )

        if batch_idx % 20 == 0:
            elapsed = time.time() - start_time
            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                f'Epoch {epoch} | Batch {batch_idx} | '
                f'Loss: {batch_loss:.5f} | '
                f'Avg Loss: {total_loss / num_batches:.5f} | '
                f'LR: {current_lr:.2e} | '
                f'Time: {elapsed:.1f}s'
            )

    avg_loss = total_loss / max(1, num_batches)
    return avg_loss, global_batch_count


def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    data_config,
) -> float:
    """Validate on held-out data.

    Args:
        model: MaskedTrackPretrainer.
        val_loader: Validation DataLoader.
        device: Target device.
        data_config: Weaver DataConfig for input name ordering.

    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for X, y, _ in val_loader:
            inputs = [X[k].to(device) for k in data_config.input_names]
            per_event_loss = model(*inputs)
            total_loss += per_event_loss.mean().item()
            num_batches += 1

    return total_loss / max(1, num_batches)


def save_loss_history(loss_history: dict, experiment_dir: str):
    """Save loss history to JSON for later plotting.

    Args:
        loss_history: Dict with 'train' and 'val' lists of per-epoch losses.
        experiment_dir: Experiment root directory for the JSON file.
    """
    output_path = os.path.join(experiment_dir, 'loss_history.json')
    with open(output_path, 'w') as file:
        json.dump(loss_history, file, indent=2)


def plot_loss_curves(loss_history: dict, experiment_dir: str):
    """Generate and save loss curve plots to the experiment root.

    Args:
        loss_history: Dict with 'train', 'val', and 'lr' lists.
        experiment_dir: Experiment root directory for the plot.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (axis_loss, axis_lr) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(loss_history['train']) + 1)

        # Loss curves
        axis_loss.plot(epochs, loss_history['train'], 'b-', label='Train')
        axis_loss.plot(epochs, loss_history['val'], 'r-', label='Validation')
        axis_loss.set_xlabel('Epoch')
        axis_loss.set_ylabel('Loss')
        axis_loss.set_title('Reconstruction Loss')
        axis_loss.legend()
        axis_loss.grid(True, alpha=0.3)

        # Learning rate
        axis_lr.plot(epochs, loss_history['lr'], 'g-')
        axis_lr.set_xlabel('Epoch')
        axis_lr.set_ylabel('Learning Rate')
        axis_lr.set_title('Learning Rate Schedule')
        axis_lr.grid(True, alpha=0.3)

        fig.tight_layout()
        output_path = os.path.join(experiment_dir, 'loss_curves.png')
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info(f'Saved loss curves: {output_path}')
    except ImportError:
        logger.warning('matplotlib not available, skipping loss curve plot')


def main():
    parser = argparse.ArgumentParser(
        description='Backbone pretraining via masked track reconstruction'
    )
    parser.add_argument('--data-config', type=str, required=True,
                        help='Path to YAML data config')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing parquet files')
    parser.add_argument('--network', type=str, required=True,
                        help='Path to network wrapper Python file')
    parser.add_argument('--model-name', type=str, default='BackbonePretrain',
                        help='Short model name for experiment folder naming')
    parser.add_argument('--experiments-dir', type=str, default='experiments',
                        help='Root directory for all experiments')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-fraction', type=float, default=0.05,
                        help='Fraction of total steps for linear warmup')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Max gradient norm for clipping (0 to disable)')
    parser.add_argument('--train-fraction', type=float, default=0.8,
                        help='Fraction of data for training (rest is validation)')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='DataLoader workers (must be <= number of parquet files)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--amp', action='store_true',
                        help='Enable mixed precision training')
    parser.add_argument('--no-compile', action='store_true',
                        help='Disable torch.compile (enabled by default on CUDA)')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # ---- Experiment directory setup ----
    # When resuming, reuse the existing experiment directory (checkpoint's grandparent).
    # Otherwise, create a new one with a timestamp.
    resume_experiment_dir = None
    if args.resume is not None:
        # Checkpoint lives at experiments/{name}/checkpoints/checkpoint_epoch_N.pt
        # So grandparent = experiment root
        resume_experiment_dir = os.path.dirname(os.path.dirname(
            os.path.abspath(args.resume)
        ))

    experiment_dir, checkpoints_dir, tensorboard_dir = build_experiment_directory(
        experiments_base=args.experiments_dir,
        model_name=args.model_name,
        resume_dir=resume_experiment_dir,
    )

    setup_logging(experiment_dir)
    device = torch.device(args.device)

    logger.info(f'Experiment directory: {experiment_dir}')
    logger.info(f'Arguments: {vars(args)}')

    # ---- TensorBoard ----
    tensorboard_writer = SummaryWriter(log_dir=tensorboard_dir)
    logger.info(f'TensorBoard logs: {tensorboard_dir}')

    # ---- Data loading via weaver's dataset infrastructure ----
    parquet_files = sorted([
        os.path.join(args.data_dir, f)
        for f in os.listdir(args.data_dir)
        if f.endswith('.parquet')
    ])
    if not parquet_files:
        raise FileNotFoundError(f'No parquet files found in {args.data_dir}')
    logger.info(f'Found {len(parquet_files)} parquet files')

    file_dict = {'data': parquet_files}

    # Training split: first train_fraction of each file.
    # in_memory=True loads the full dataset once and reshuffles indices each
    # epoch, avoiding the ~30s parquet re-read on every "Restarting DataIter".
    # fetch_step=0 ensures all data is loaded in a single initial fetch.
    train_dataset = SimpleIterDataset(
        file_dict,
        data_config_file=args.data_config,
        for_training=True,
        load_range_and_fraction=((0.0, args.train_fraction), 1.0),
        fetch_by_files=True,
        fetch_step=0,
        in_memory=True,
    )
    data_config = train_dataset.config

    # Validation split: remaining fraction (also in-memory)
    val_dataset = SimpleIterDataset(
        file_dict,
        data_config_file=args.data_config,
        for_training=False,
        load_range_and_fraction=((args.train_fraction, 1.0), 1.0),
        fetch_by_files=True,
        fetch_step=0,
        in_memory=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    # ---- Model ----
    network_module = load_network_module(args.network)
    model, model_info = network_module.get_model(data_config)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model parameters: {num_params:,}')
    logger.info(f'Input names: {data_config.input_names}')

    # Keep a reference to the original (uncompiled) model for state_dict access.
    # torch.compile wraps the module, so we need the original for clean
    # checkpoint saving and backbone weight extraction.
    original_model = model

    # ---- torch.compile ----
    use_compile = (
        not args.no_compile
        and device.type == 'cuda'
        and hasattr(torch, 'compile')
    )
    if use_compile:
        logger.info('Compiling model with torch.compile (mode="default")...')
        model = torch.compile(model, mode='default')
        logger.info('Model compiled.')
    else:
        logger.info('torch.compile disabled.')

    # ---- Optimizer and scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Estimate total steps (approximate — iterable dataset doesn't have len())
    estimated_steps_per_epoch = 100  # will be refined after first epoch
    total_steps = args.epochs * estimated_steps_per_epoch
    warmup_steps = int(args.warmup_fraction * total_steps)

    scheduler = build_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    # ---- Mixed precision ----
    grad_scaler = torch.amp.GradScaler('cuda') if args.amp else None

    # ---- Resume from checkpoint ----
    start_epoch = 1
    best_val_loss = float('inf')
    global_batch_count = 0
    loss_history = {'train': [], 'val': [], 'lr': []}

    if args.resume is not None:
        logger.info(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        # Load into the original (uncompiled) model — torch.compile wraps it,
        # but state_dict keys come from the unwrapped module.
        original_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        global_batch_count = checkpoint.get('global_batch_count', 0)
        loss_history = checkpoint.get('loss_history', loss_history)
        logger.info(
            f'Resumed at epoch {start_epoch}, '
            f'best_val_loss={best_val_loss:.5f}'
        )

    # ---- Training loop ----
    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f'=== Epoch {epoch}/{args.epochs} ===')

        train_loss, global_batch_count = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            grad_scaler, device, data_config, epoch,
            tensorboard_writer, global_batch_count,
            grad_clip_max_norm=args.grad_clip,
        )
        logger.info(f'Epoch {epoch} train loss: {train_loss:.5f}')

        val_loss = validate(model, val_loader, device, data_config)
        logger.info(f'Epoch {epoch} val loss: {val_loss:.5f}')

        current_lr = scheduler.get_last_lr()[0]

        # TensorBoard: per-epoch logging
        tensorboard_writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        tensorboard_writer.add_scalar('Loss/val_epoch', val_loss, epoch)
        tensorboard_writer.add_scalar('LR/epoch', current_lr, epoch)

        # Loss history for JSON export and plotting
        loss_history['train'].append(train_loss)
        loss_history['val'].append(val_loss)
        loss_history['lr'].append(current_lr)
        save_loss_history(loss_history, experiment_dir)

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        if epoch % args.save_every == 0 or is_best or epoch == args.epochs:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': original_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'global_batch_count': global_batch_count,
                'loss_history': loss_history,
                'args': vars(args),
            }

            checkpoint_path = os.path.join(
                checkpoints_dir, f'checkpoint_epoch_{epoch}.pt'
            )
            torch.save(checkpoint, checkpoint_path)
            logger.info(f'Saved checkpoint: {checkpoint_path}')

            if is_best:
                best_path = os.path.join(checkpoints_dir, 'best_model.pt')
                torch.save(checkpoint, best_path)
                logger.info(f'New best model (val_loss={val_loss:.5f})')

        # Save backbone-only weights (for downstream use)
        if is_best:
            backbone_state = {
                k.replace('backbone.', ''): v
                for k, v in original_model.state_dict().items()
                if k.startswith('backbone.')
            }
            backbone_path = os.path.join(checkpoints_dir, 'backbone_best.pt')
            torch.save(backbone_state, backbone_path)
            logger.info(f'Saved backbone weights: {backbone_path}')

    # ---- Final outputs ----
    tensorboard_writer.close()
    plot_loss_curves(loss_history, experiment_dir)
    logger.info(f'Training complete. Best val loss: {best_val_loss:.5f}')
    logger.info(f'Experiment: {experiment_dir}')
    logger.info(f'  - Log: training.log')
    logger.info(f'  - Loss history: loss_history.json')
    logger.info(f'  - Loss curves: loss_curves.png')
    logger.info(f'  - Checkpoints: checkpoints/')
    logger.info(f'  - TensorBoard: tensorboard/')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # Log the full traceback so it appears in training.log even if the
        # process crashes. The stderr tee will also capture it, but logging
        # it explicitly ensures it gets a proper timestamp.
        logger.error(f'Training failed with exception:\n{traceback.format_exc()}')
        raise
