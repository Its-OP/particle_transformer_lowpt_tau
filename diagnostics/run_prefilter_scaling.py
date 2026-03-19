"""Run pre-filter scaling experiments one at a time.

Usage: python run_prefilter_scaling.py
"""
import sys
import torch
from torch.utils.data import DataLoader
from weaver.utils.dataset import SimpleIterDataset
from weaver.nn.model.TrackPreFilter import TrackPreFilter
from utils.training_utils import extract_label_from_inputs, trim_to_max_valid_tracks

# ---- Data setup ----
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
STEPS_PER_EPOCH = 50
EVAL_BATCHES = 20
TOP_K = 200


def run_experiment(config_name, config_kwargs, num_epochs):
    """Train and evaluate a single pre-filter configuration."""
    model = TrackPreFilter(**config_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    total_params = sum(parameter.numel() for parameter in model.parameters())

    print(f'\n{"=" * 60}')
    print(f'{config_name} | {total_params:,} params | {num_epochs} epochs')
    print(f'{"=" * 60}')

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0
        num_batches = 0
        loader = DataLoader(
            dataset, batch_size=4, drop_last=True, num_workers=0,
        )

        for batch_data in loader:
            inputs_dict = batch_data[0]
            flat_inputs = [
                inputs_dict[name].to(device) for name in input_names
            ]
            model_inputs, track_labels = extract_label_from_inputs(
                flat_inputs, label_idx,
            )
            model_inputs = list(
                trim_to_max_valid_tracks(model_inputs, mask_idx_adj),
            )
            points, features, lorentz_vectors, mask = model_inputs
            track_labels = track_labels[:, :, :mask.shape[2]]

            loss_dict = model.compute_loss(
                points, features, lorentz_vectors, mask, track_labels,
            )
            optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            optimizer.step()

            epoch_loss += loss_dict['total_loss'].item()
            num_batches += 1
            if num_batches >= STEPS_PER_EPOCH:
                break

        # Evaluate R@200
        model.eval()
        gt_kept = 0
        gt_total = 0
        with torch.no_grad():
            eval_loader = DataLoader(
                dataset, batch_size=4, drop_last=True, num_workers=0,
            )
            for eval_idx, batch_data in enumerate(eval_loader):
                if eval_idx >= EVAL_BATCHES:
                    break
                inputs_dict = batch_data[0]
                flat_inputs = [
                    inputs_dict[name].to(device) for name in input_names
                ]
                model_inputs, track_labels = extract_label_from_inputs(
                    flat_inputs, label_idx,
                )
                model_inputs = list(
                    trim_to_max_valid_tracks(model_inputs, mask_idx_adj),
                )
                points, features, lorentz_vectors, mask = model_inputs
                track_labels = track_labels[:, :, :mask.shape[2]]

                filtered = model.filter_tracks(
                    points, features, lorentz_vectors, mask, track_labels,
                    top_k=TOP_K,
                )
                gt_kept += filtered['track_labels'].sum().item()
                gt_total += (
                    track_labels * mask[:, :, :track_labels.shape[2]].float()
                ).sum().item()

        recall = gt_kept / max(1, gt_total)
        avg_loss = epoch_loss / num_batches

        if epoch in (1, 5, 10, 15, 20, 25, 30):
            print(
                f'  Ep {epoch:2d} | loss={avg_loss:.4f} | '
                f'R@200={recall:.3f} ({int(gt_kept)}/{int(gt_total)})',
            )

    return recall


# ---- Run experiments ----
results = {}

# 1. Baseline
results['baseline (22K)'] = run_experiment(
    'baseline hybrid+asl (hidden=64)',
    dict(mode='hybrid', input_dim=7, use_asl=True, hidden_dim=64),
    num_epochs=10,
)

# 2. Wide 128
results['wide128 (76K)'] = run_experiment(
    'wide128 hybrid+asl (hidden=128)',
    dict(mode='hybrid', input_dim=7, use_asl=True, hidden_dim=128),
    num_epochs=20,
)

# 3. Wide 256
results['wide256 (283K)'] = run_experiment(
    'wide256 hybrid+asl (hidden=256)',
    dict(mode='hybrid', input_dim=7, use_asl=True, hidden_dim=256),
    num_epochs=20,
)

# 4. Wide128 + Lorentz vectors (need implementation)
results['wide128+lv'] = run_experiment(
    'wide128+lv hybrid+asl',
    dict(mode='hybrid', input_dim=7, use_asl=True, hidden_dim=128,
         use_lorentz_vectors=True),
    num_epochs=20,
)

# 5. Wide128 + 2 message rounds (need implementation)
results['wide128+2rounds'] = run_experiment(
    'wide128+2rounds hybrid+asl',
    dict(mode='hybrid', input_dim=7, use_asl=True, hidden_dim=128,
         num_message_rounds=2),
    num_epochs=20,
)

# 6. Wide128 + GAPLayer MIA (need implementation)
results['wide128+gap'] = run_experiment(
    'wide128+gap hybrid+asl',
    dict(mode='hybrid', input_dim=7, use_asl=True, hidden_dim=128,
         use_gap_attention=True),
    num_epochs=20,
)

# 7. Full combination
results['full (all)'] = run_experiment(
    'full: wide128+lv+2rounds+gap',
    dict(mode='hybrid', input_dim=7, use_asl=True, hidden_dim=128,
         use_lorentz_vectors=True, num_message_rounds=2,
         use_gap_attention=True),
    num_epochs=30,
)

# ---- Summary ----
print('\n' + '=' * 60)
print('FINAL COMPARISON')
print('=' * 60)
for config_name, recall in results.items():
    print(f'  {config_name:<25s}: R@200 = {recall:.3f}')
