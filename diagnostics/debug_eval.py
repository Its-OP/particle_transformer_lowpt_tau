"""Evaluate both models using the EXACT validate() function from training.
Subset data only (M3-safe). Compares widened vs phase-a on the same data."""

import sys
import os
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from weaver.utils.dataset import SimpleIterDataset
from weaver.nn.model.TrackPreFilter import TrackPreFilter
from train_prefilter import validate


def make_loader(data_config_file, data_dir):
    parquet_files = sorted(glob.glob(f'{data_dir}/*.parquet'))
    dataset = SimpleIterDataset(
        {'data': parquet_files},
        data_config_file=data_config_file,
        for_training=False,
        load_range_and_fraction=((0.0, 1.0), 1.0),
        fetch_by_files=True,
        fetch_step=len(parquet_files),
        in_memory=True,
    )
    loader = DataLoader(
        dataset, batch_size=8, drop_last=True,
        pin_memory=False, num_workers=0,
    )
    return loader, dataset.config


def print_metrics(name, loss_avgs, metrics):
    print(f'\n=== {name} ===')
    print(f'  R@10:  {metrics.get("recall_at_10", 0):.4f}   '
          f'R@20:  {metrics.get("recall_at_20", 0):.4f}   '
          f'R@30:  {metrics.get("recall_at_30", 0):.4f}')
    print(f'  R@100: {metrics.get("recall_at_100", 0):.4f}   '
          f'R@200: {metrics.get("recall_at_200", 0):.4f}   '
          f'P@200: {metrics.get("perfect_at_200", 0):.4f}')
    print(f"  d':    {metrics.get('d_prime', 0):.4f}   "
          f'rank:  {metrics.get("median_gt_rank", 0):.1f}')
    loss_str = '  '.join(f'{k}: {v:.4f}' for k, v in loss_avgs.items())
    print(f'  {loss_str}')


device = torch.device('mps')
data_config_file = 'data/low-pt/lowpt_tau_trackfinder.yaml'
data_dir = 'data/low-pt/extended/subset/val/'
max_steps = 200  # 200 batches × 8 = 1600 events

# ---- Widened (hybrid) ----
print('Loading WIDENED (hybrid) model...')
model = TrackPreFilter(
    mode='hybrid', input_dim=13, hidden_dim=192, latent_dim=48,
    num_neighbors=16, num_message_rounds=2, ranking_num_samples=50,
).to(device)
ckpt = torch.load(
    'models/debug_checkpoints/widened/checkpoints/best_model.pt',
    map_location=device, weights_only=False,
)
model.load_state_dict(ckpt['model_state_dict'])
print(f'  Loaded epoch {ckpt.get("epoch", "?")}')

loader, data_config = make_loader(data_config_file, data_dir)
input_names = list(data_config.input_names)
mask_idx = input_names.index('pf_mask')
label_idx = input_names.index('pf_label')

print('Evaluating widened on subset...')
w_loss, w_metrics = validate(
    model, loader, device, data_config, mask_idx, label_idx, max_steps,
)
print_metrics('WIDENED on subset', w_loss, w_metrics)
del model
torch.mps.empty_cache()

# ---- Phase-A (mlp) ----
print('\nLoading PHASE-A (mlp) model...')
model = TrackPreFilter(
    mode='mlp', input_dim=13, hidden_dim=192,
    num_neighbors=16, num_message_rounds=2, ranking_num_samples=50,
    ranking_temperature_start=2.0, ranking_temperature_end=0.5,
    denoising_sigma_start=1.0, denoising_sigma_end=0.1,
    drw_warmup_fraction=0.3, drw_positive_weight=2.0,
).to(device)
ckpt = torch.load(
    'models/debug_checkpoints/phase-a/checkpoints/best_model.pt',
    map_location=device, weights_only=False,
)
model.load_state_dict(ckpt['model_state_dict'])
print(f'  Loaded epoch {ckpt.get("epoch", "?")}')

loader2, _ = make_loader(data_config_file, data_dir)
print('Evaluating phase-a on subset...')
pa_loss, pa_metrics = validate(
    model, loader2, device, data_config, mask_idx, label_idx, max_steps,
)
print_metrics('PHASE-A on subset', pa_loss, pa_metrics)

# ---- Comparison ----
print('\n' + '=' * 60)
print('COMPARISON (both on subset, same validate() code)')
print('=' * 60)
print(f'{"Metric":<12} {"Widened":>10} {"Phase-A":>10} {"Delta":>10}')
print('-' * 42)
for key, label in [
    ('recall_at_200', 'R@200'),
    ('perfect_at_200', 'P@200'),
    ('recall_at_100', 'R@100'),
    ('recall_at_30', 'R@30'),
    ('d_prime', "d'"),
    ('median_gt_rank', 'Rank'),
]:
    w_val = w_metrics.get(key, 0)
    pa_val = pa_metrics.get(key, 0)
    delta = pa_val - w_val
    sign = '+' if delta >= 0 else ''
    print(f'{label:<12} {w_val:>10.4f} {pa_val:>10.4f} {sign}{delta:>9.4f}')

print('\nTraining log reference (original val):')
print('  Widened ep22:  R@200=0.6228  P@200=0.3640  d\'=1.329  rank=112')
print('  Phase-A ep15:  R@200=0.5991  P@200=0.3362  d\'=1.301  rank=128')
print('  Phase-A ep23:  R@200=0.6157  P@200=0.3555  d\'=1.297  rank=118')
