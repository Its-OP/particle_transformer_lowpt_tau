"""Hyperparameter tuning for TrackPreFilter wide128+2rounds.

Varies one parameter at a time against the baseline.
"""
import sys
import torch
from torch.utils.data import DataLoader
from weaver.utils.dataset import SimpleIterDataset
from weaver.nn.model.TrackPreFilter import TrackPreFilter
from utils.training_utils import extract_label_from_inputs, trim_to_max_valid_tracks

sys.stdout.reconfigure(line_buffering=True)

dataset = SimpleIterDataset(
    {'data': ['data/low-pt/subset/lowpt_tau_trackorigin.parquet']},
    data_config_file='data/low-pt/lowpt_tau_trackfinder.yaml',
    for_training=False,
    load_range_and_fraction=((0.0, 1.0), 1.0),
    fetch_by_files=True, fetch_step=1, in_memory=False,
)
data_config = dataset.config
input_names = list(data_config.input_names)
label_idx = input_names.index('pf_label')
mask_idx = input_names.index('pf_mask')
mask_idx_adj = mask_idx if mask_idx < label_idx else mask_idx - 1

device = torch.device('mps')
STEPS = 50
EVAL_BATCHES = 20
TOP_K = 200

BASELINE = dict(mode='hybrid', input_dim=7, hidden_dim=128, latent_dim=16,
                num_neighbors=16, num_message_rounds=2, use_asl=True,
                ranking_num_samples=20)

CONFIGS = [
    ('baseline (k=16,r=2,h=128)', BASELINE, 20),
    ('k=32', {**BASELINE, 'num_neighbors': 32}, 20),
    ('k=8', {**BASELINE, 'num_neighbors': 8}, 20),
    ('3 rounds', {**BASELINE, 'num_message_rounds': 3}, 20),
    ('hidden=256', {**BASELINE, 'hidden_dim': 256}, 20),
    ('latent=32', {**BASELINE, 'latent_dim': 32}, 20),
    ('neg_samples=50', {**BASELINE, 'ranking_num_samples': 50}, 20),
]

results = {}

for config_name, config_kwargs, num_epochs in CONFIGS:
    model = TrackPreFilter(**config_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    total_params = sum(p.numel() for p in model.parameters())

    print(f'\n{"=" * 50}')
    print(f'{config_name} | {total_params:,} params | {num_epochs} epochs')
    print(f'{"=" * 50}', flush=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0
        num_batches = 0
        loader = DataLoader(dataset, batch_size=4, drop_last=True, num_workers=0)

        for batch_data in loader:
            inputs_dict = batch_data[0]
            flat_inputs = [inputs_dict[name].to(device) for name in input_names]
            model_inputs, track_labels = extract_label_from_inputs(flat_inputs, label_idx)
            model_inputs = list(trim_to_max_valid_tracks(model_inputs, mask_idx_adj))
            points, features, lorentz_vectors, mask = model_inputs
            track_labels = track_labels[:, :, :mask.shape[2]]

            loss_dict = model.compute_loss(points, features, lorentz_vectors, mask, track_labels)
            optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            optimizer.step()

            epoch_loss += loss_dict['total_loss'].item()
            num_batches += 1
            if num_batches >= STEPS:
                break

        model.eval()
        gt_kept = 0
        gt_total = 0
        with torch.no_grad():
            eval_loader = DataLoader(dataset, batch_size=4, drop_last=True, num_workers=0)
            for eval_idx, batch_data in enumerate(eval_loader):
                if eval_idx >= EVAL_BATCHES:
                    break
                inputs_dict = batch_data[0]
                flat_inputs = [inputs_dict[name].to(device) for name in input_names]
                model_inputs, track_labels = extract_label_from_inputs(flat_inputs, label_idx)
                model_inputs = list(trim_to_max_valid_tracks(model_inputs, mask_idx_adj))
                points, features, lorentz_vectors, mask = model_inputs
                track_labels = track_labels[:, :, :mask.shape[2]]

                filtered = model.filter_tracks(points, features, lorentz_vectors, mask, track_labels, top_k=TOP_K)
                gt_kept += filtered['track_labels'].sum().item()
                gt_total += (track_labels * mask[:, :, :track_labels.shape[2]].float()).sum().item()

        recall = gt_kept / max(1, gt_total)
        avg_loss = epoch_loss / num_batches

        if epoch in (5, 10, 15, 20):
            print(f'  Ep {epoch:2d} | loss={avg_loss:.4f} | R@200={recall:.3f} ({int(gt_kept)}/{int(gt_total)})', flush=True)

    results[config_name] = (total_params, recall)
    print()

print('=' * 50)
print('FINAL COMPARISON')
print('=' * 50)
for name, (params, recall) in results.items():
    print(f'  {name:<30s} | {params:>8,d} params | R@200={recall:.3f}')
