"""Evaluate Phase-B model using the EXACT validate() function from training.
Subset data only (M3-safe). Also evaluates widened baseline for comparison."""

import sys
import os
import glob

# Add part/ (parent of diagnostics/) to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

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

models_to_eval = [
    {
        'name': 'PHASE-B (mlp, best R@200)',
        'checkpoint': 'models/debug_checkpoints/phase-b/checkpoints/best_model.pt',
        'config': dict(
            mode='mlp', input_dim=13, hidden_dim=192,
            num_neighbors=16, num_message_rounds=2, ranking_num_samples=50,
            ranking_temperature_start=2.0, ranking_temperature_end=0.5,
            denoising_sigma_start=1.0, denoising_sigma_end=0.1,
            drw_warmup_fraction=0.3, drw_positive_weight=2.0,
        ),
    },
    {
        'name': 'WIDENED (hybrid baseline)',
        'checkpoint': 'models/debug_checkpoints/widened/checkpoints/best_model.pt',
        'config': dict(
            mode='hybrid', input_dim=13, hidden_dim=192, latent_dim=48,
            num_neighbors=16, num_message_rounds=2, ranking_num_samples=50,
        ),
    },
]

all_results = {}

for model_spec in models_to_eval:
    print(f'\nLoading {model_spec["name"]}...')
    model = TrackPreFilter(**model_spec['config']).to(device)
    ckpt = torch.load(
        model_spec['checkpoint'], map_location=device, weights_only=False,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'  Loaded epoch {ckpt.get("epoch", "?")}')

    loader, data_config = make_loader(data_config_file, data_dir)
    input_names = list(data_config.input_names)
    mask_idx = input_names.index('pf_mask')
    label_idx = input_names.index('pf_label')

    print(f'Evaluating on subset ({max_steps} batches)...')
    losses, metrics = validate(
        model, loader, device, data_config, mask_idx, label_idx, max_steps,
    )
    print_metrics(model_spec['name'], losses, metrics)
    all_results[model_spec['name']] = (losses, metrics)

    del model
    torch.mps.empty_cache()

# ---- Comparison ----
names = list(all_results.keys())
print('\n' + '=' * 70)
print('COMPARISON (subset data, same validate() code)')
print('=' * 70)
header = f'{"Metric":<12}'
for name in names:
    short = name.split('(')[0].strip()
    header += f' {short:>12}'
header += f' {"Delta":>10}'
print(header)
print('-' * 60)

for key, label in [
    ('recall_at_10', 'R@10'), ('recall_at_20', 'R@20'),
    ('recall_at_30', 'R@30'), ('recall_at_100', 'R@100'),
    ('recall_at_200', 'R@200'), ('perfect_at_200', 'P@200'),
    ('d_prime', "d'"), ('median_gt_rank', 'Rank'),
]:
    row = f'{label:<12}'
    values = []
    for name in names:
        _, metrics = all_results[name]
        value = metrics.get(key, 0)
        values.append(value)
        row += f' {value:>12.4f}'
    delta = values[0] - values[1] if len(values) == 2 else 0
    sign = '+' if delta >= 0 else ''
    row += f' {sign}{delta:>9.4f}'
    print(row)

print('\nTraining log reference (original val, 84K events):')
print('  Widened ep22:   R@200=0.6228  P@200=0.3640  d\'=1.329  rank=112')
print('  Phase-B ep36:   R@200=0.6231  P@200=0.3680  d\'=1.315  rank=113')
print('  Phase-B ep39:   R@200=0.6226  P@200=0.3610  d\'=1.302  rank=111')
