"""Run all pre-filter configurations and report R@200 comparison.

Usage: python run_prefilter_experiments.py
"""
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

CONFIGS = [
    ('mlp', dict(mode='mlp', input_dim=7)),
    ('mlp+asl', dict(mode='mlp', input_dim=7, use_asl=True)),
    ('two_tower_1proto', dict(mode='two_tower', input_dim=7)),
    ('two_tower_3proto', dict(mode='two_tower', input_dim=7, num_prototypes=3)),
    ('autoencoder', dict(mode='autoencoder', input_dim=7)),
    ('hybrid', dict(mode='hybrid', input_dim=7)),
    ('hybrid+asl', dict(mode='hybrid', input_dim=7, use_asl=True)),
]

NUM_EPOCHS = 10
STEPS_PER_EPOCH = 10
TOP_K = 200

print(f'{"Config":<22s} | {"Params":>7s} | {"Ep":>2s} | {"Loss":>8s} | {"R@200":>6s} | {"GT kept":>8s}')
print('-' * 75)

for config_name, config_kwargs in CONFIGS:
    model = TrackPreFilter(**config_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    total_params = sum(parameter.numel() for parameter in model.parameters())

    for epoch in range(1, NUM_EPOCHS + 1):
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

        # Evaluate R@200 on last batch
        model.eval()
        with torch.no_grad():
            filtered = model.filter_tracks(
                points, features, lorentz_vectors, mask, track_labels,
                top_k=TOP_K,
            )
            gt_in_filtered = filtered['track_labels'].sum().item()
            gt_total = (
                track_labels * mask[:, :, :track_labels.shape[2]].float()
            ).sum().item()
            recall_200 = gt_in_filtered / max(1, gt_total)

        avg_loss = epoch_loss / num_batches

        if epoch in (1, 5, 10):
            print(
                f'{config_name:<22s} | {total_params:>7,d} | {epoch:>2d} | '
                f'{avg_loss:>8.4f} | {recall_200:>6.3f} | '
                f'{int(gt_in_filtered):>3d}/{int(gt_total):>3d}',
            )

    print()
