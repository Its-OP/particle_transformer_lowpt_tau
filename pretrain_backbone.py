"""Custom training script for backbone pretraining via masked track reconstruction.

Bypasses weaver's training loop to avoid fighting the regression mode interface.
Uses weaver's dataset infrastructure for YAML parsing and parquet loading.

Usage:
    python pretrain_backbone.py \
        --data-config data/low-pt/lowpt_tau_pretrain.yaml \
        --data-dir data/low-pt/ \
        --network networks/lowpt_tau_BackbonePretrain.py \
        --epochs 100 \
        --batch-size 32 \
        --lr 1e-3 \
        --device cuda:0
"""
import argparse
import importlib.util
import logging
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from weaver.utils.dataset import SimpleIterDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('pretrain_backbone')


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
        - Cosine decay: lr follows cos(π * progress / 2) from base_lr to 0

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
) -> float:
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

    Returns:
        Average loss for the epoch.
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

        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()

        batch_loss = loss.item()
        total_loss += batch_loss
        num_batches += 1

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
    return avg_loss


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


def main():
    parser = argparse.ArgumentParser(description='Backbone pretraining via masked track reconstruction')
    parser.add_argument('--data-config', type=str, required=True,
                        help='Path to YAML data config')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing parquet files')
    parser.add_argument('--network', type=str, required=True,
                        help='Path to network wrapper Python file')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-fraction', type=float, default=0.05,
                        help='Fraction of total steps for linear warmup')
    parser.add_argument('--train-fraction', type=float, default=0.8,
                        help='Fraction of data for training (rest is validation)')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--amp', action='store_true',
                        help='Enable mixed precision training')
    parser.add_argument('--output-dir', type=str, default='checkpoints/pretrain',
                        help='Directory for saving checkpoints')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ---- Data loading via weaver's dataset infrastructure ----
    # Collect all parquet files in the data directory
    parquet_files = sorted([
        os.path.join(args.data_dir, f)
        for f in os.listdir(args.data_dir)
        if f.endswith('.parquet')
    ])
    if not parquet_files:
        raise FileNotFoundError(f'No parquet files found in {args.data_dir}')
    logger.info(f'Found {len(parquet_files)} parquet files')

    file_dict = {'data': parquet_files}

    # Training split: first train_fraction of each file
    train_dataset = SimpleIterDataset(
        file_dict,
        data_config_file=args.data_config,
        for_training=True,
        load_range_and_fraction=((0.0, args.train_fraction), 1.0),
        fetch_by_files=True,
        fetch_step=1,
    )
    data_config = train_dataset.config

    # Validation split: remaining fraction
    val_dataset = SimpleIterDataset(
        file_dict,
        data_config_file=args.data_config,
        for_training=False,
        load_range_and_fraction=((args.train_fraction, 1.0), 1.0),
        fetch_by_files=True,
        fetch_step=1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    # ---- Model ----
    network_module = load_network_module(args.network)
    model, model_info = network_module.get_model(data_config)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model parameters: {num_params:,}')
    logger.info(f'Input names: {data_config.input_names}')

    # ---- Optimizer and scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Estimate total steps (approximate — iterable dataset doesn't have len())
    # Use a rough estimate based on dataset size
    estimated_steps_per_epoch = 100  # will be refined after first epoch
    total_steps = args.epochs * estimated_steps_per_epoch
    warmup_steps = int(args.warmup_fraction * total_steps)

    scheduler = build_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    # ---- Mixed precision ----
    grad_scaler = torch.amp.GradScaler('cuda') if args.amp else None

    # ---- Training loop ----
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        logger.info(f'=== Epoch {epoch}/{args.epochs} ===')

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            grad_scaler, device, data_config, epoch,
        )
        logger.info(f'Epoch {epoch} train loss: {train_loss:.5f}')

        val_loss = validate(model, val_loader, device, data_config)
        logger.info(f'Epoch {epoch} val loss: {val_loss:.5f}')

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        if epoch % args.save_every == 0 or is_best or epoch == args.epochs:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'args': vars(args),
            }

            checkpoint_path = os.path.join(
                args.output_dir, f'checkpoint_epoch_{epoch}.pt'
            )
            torch.save(checkpoint, checkpoint_path)
            logger.info(f'Saved checkpoint: {checkpoint_path}')

            if is_best:
                best_path = os.path.join(args.output_dir, 'best_model.pt')
                torch.save(checkpoint, best_path)
                logger.info(f'New best model (val_loss={val_loss:.5f})')

        # Save backbone-only weights (for downstream use)
        if is_best:
            backbone_state = {
                k.replace('backbone.', ''): v
                for k, v in model.state_dict().items()
                if k.startswith('backbone.')
            }
            backbone_path = os.path.join(args.output_dir, 'backbone_best.pt')
            torch.save(backbone_state, backbone_path)
            logger.info(f'Saved backbone weights: {backbone_path}')

    logger.info(f'Training complete. Best val loss: {best_val_loss:.5f}')


if __name__ == '__main__':
    main()
