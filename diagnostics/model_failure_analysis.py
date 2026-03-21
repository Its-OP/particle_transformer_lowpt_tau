"""Diagnose what the pre-filter excels at and where it fails.

Loads a trained model, runs inference on val data, and analyzes:
A. Success vs failure events (R@200 = 1.0 vs < 1.0)
B. Per-GT-track: found vs missed feature distributions
C. Score distributions across categories
D. kNN neighborhood GT density for found vs missed tracks
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch
from torch.utils.data import DataLoader
from weaver.utils.dataset import SimpleIterDataset
from weaver.nn.model.TrackPreFilter import TrackPreFilter
from weaver.nn.model.HierarchicalGraphBackbone import cross_set_knn
from utils.training_utils import trim_to_max_valid_tracks, extract_label_from_inputs


def main():
    parser = argparse.ArgumentParser(description='Pre-filter failure analysis')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-config', type=str,
                        default='data/low-pt/lowpt_tau_trackfinder.yaml')
    parser.add_argument('--data-dir', type=str, default='data/low-pt/subset/val/')
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--num-neighbors', type=int, default=16)
    args = parser.parse_args()

    device = torch.device(args.device)

    # ---- Load checkpoint and infer config ----
    print('Loading checkpoint...')
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = checkpoint.get('args', {})

    # Reconstruct model from checkpoint's training config
    ckpt_state = checkpoint['model_state_dict']
    # Infer input_dim from first layer weight shape
    first_key = [k for k in ckpt_state if 'track_mlp.0.weight' in k][0]
    input_dim = ckpt_state[first_key].shape[1]
    print(f'  Inferred input_dim={input_dim} from checkpoint')

    model = TrackPreFilter(
        mode='mlp', input_dim=input_dim, hidden_dim=192,
        num_neighbors=16, num_message_rounds=2, ranking_num_samples=50,
    )
    model.load_state_dict(ckpt_state)
    model = model.to(device)
    model.eval()
    print(f'  Loaded. {sum(p.numel() for p in model.parameters()):,} params')

    # ---- Load data ----
    import glob
    parquet_files = sorted(glob.glob(f'{args.data_dir}/*.parquet'))
    print(f'Loading data from {args.data_dir} ({len(parquet_files)} files)...')
    dataset = SimpleIterDataset(
        {'data': parquet_files},
        data_config_file=args.data_config,
        for_training=False,
        load_range_and_fraction=((0.0, 1.0), 1.0),
        fetch_by_files=True,
        fetch_step=len(parquet_files),
        in_memory=True,
    )
    data_config = dataset.config
    loader = DataLoader(
        dataset, batch_size=args.batch_size, drop_last=True,
        pin_memory=False, num_workers=0,
    )
    input_names = list(data_config.input_names)
    mask_idx = input_names.index('pf_mask')
    label_idx = input_names.index('pf_label')

    # ---- Collect per-event results ----
    print(f'Running inference (max {args.max_steps} batches)...')
    all_events = []

    with torch.no_grad():
        for batch_index, (X, y, _) in enumerate(loader):
            if batch_index >= args.max_steps:
                break

            inputs = [X[k].to(device) for k in data_config.input_names]
            inputs = trim_to_max_valid_tracks(inputs, mask_idx)
            model_inputs, track_labels = extract_label_from_inputs(
                inputs, label_idx,
            )
            points, features, lorentz_vectors, mask = model_inputs

            # Match training's validate() pattern: train mode for forward pass
            # (BatchNorm uses batch stats, matching how scores were computed
            # during training validation). Then back to eval.
            model.train()
            loss_dict = model.compute_loss(
                points, features, lorentz_vectors, mask, track_labels,
            )
            model.eval()
            scores = loss_dict['_scores'].detach()

            # kNN for neighborhood analysis
            neighbor_indices = cross_set_knn(
                points, points, args.num_neighbors, mask, None,
            )

            # Per-event analysis
            labels_flat = track_labels.squeeze(1) * mask.squeeze(1).float()
            valid_mask = mask.squeeze(1).bool()
            batch_size = scores.shape[0]

            for event_index in range(batch_size):
                event_valid = valid_mask[event_index].cpu()
                event_scores = scores[event_index].cpu()
                event_labels = labels_flat[event_index].cpu()
                event_features = features[event_index].cpu()
                event_neighbors = neighbor_indices[event_index].cpu()

                num_tracks = event_valid.sum().item()
                gt_mask = (event_labels == 1.0) & event_valid
                num_gt = gt_mask.sum().item()
                if num_gt == 0:
                    continue

                # Rank tracks by score
                masked_scores = event_scores.clone()
                masked_scores[~event_valid] = float('-inf')
                ranks = torch.argsort(torch.argsort(masked_scores, descending=True))

                gt_indices = gt_mask.nonzero(as_tuple=True)[0]
                gt_ranks = ranks[gt_indices].numpy()
                gt_scores_vals = event_scores[gt_indices].numpy()
                found = gt_ranks < 200
                recall_at_200 = found.mean()

                # GT track features (raw standardized)
                gt_features = event_features[:, gt_indices].numpy()  # (F, num_gt)

                # kNN GT neighbor count for each GT track
                gt_neighbor_gt_counts = []
                for gt_idx in gt_indices:
                    neighbors = event_neighbors[gt_idx.item()]  # (K,)
                    neighbor_is_gt = event_labels[neighbors] == 1.0
                    gt_neighbor_gt_counts.append(neighbor_is_gt.sum().item())

                # Top false positives
                noise_mask = (event_labels == 0.0) & event_valid
                noise_scores = event_scores[noise_mask]
                top_fp_scores = noise_scores.topk(min(5, len(noise_scores)))[0].numpy()

                all_events.append({
                    'num_tracks': num_tracks,
                    'num_gt': num_gt,
                    'recall_at_200': recall_at_200,
                    'perfect': recall_at_200 == 1.0,
                    'gt_ranks': gt_ranks,
                    'gt_scores': gt_scores_vals,
                    'gt_found': found,
                    'gt_features': gt_features,
                    'gt_neighbor_gt_counts': np.array(gt_neighbor_gt_counts),
                    'top_fp_scores': top_fp_scores,
                })

    print(f'Analyzed {len(all_events)} events with GT tracks\n')

    # ==== A. Success vs Failure ====
    print('=' * 60)
    print('A. SUCCESS vs FAILURE EVENTS')
    print('=' * 60)

    success = [e for e in all_events if e['perfect']]
    failure = [e for e in all_events if not e['perfect']]
    print(f'  Success (R@200=1.0): {len(success)} events ({100*len(success)/len(all_events):.1f}%)')
    print(f'  Failure (R@200<1.0): {len(failure)} events ({100*len(failure)/len(all_events):.1f}%)')

    if success and failure:
        print(f'\n  {"Metric":<25} {"Success":>10} {"Failure":>10}')
        print(f'  {"-"*45}')
        for name, key in [('Avg tracks', 'num_tracks'), ('Avg GT tracks', 'num_gt')]:
            s_val = np.mean([e[key] for e in success])
            f_val = np.mean([e[key] for e in failure])
            print(f'  {name:<25} {s_val:>10.1f} {f_val:>10.1f}')

        s_r200 = np.mean([e['recall_at_200'] for e in success])
        f_r200 = np.mean([e['recall_at_200'] for e in failure])
        print(f'  {"Avg R@200":<25} {s_r200:>10.4f} {f_r200:>10.4f}')

    # ==== B. Found vs Missed GT Tracks ====
    print(f'\n{"=" * 60}')
    print('B. FOUND vs MISSED GT TRACKS')
    print('=' * 60)

    found_ranks = []
    missed_ranks = []
    found_scores = []
    missed_scores = []
    found_features_all = []
    missed_features_all = []
    found_gt_neighbor_counts = []
    missed_gt_neighbor_counts = []

    for event in all_events:
        for track_index in range(len(event['gt_ranks'])):
            if event['gt_found'][track_index]:
                found_ranks.append(event['gt_ranks'][track_index])
                found_scores.append(event['gt_scores'][track_index])
                found_features_all.append(event['gt_features'][:, track_index])
                found_gt_neighbor_counts.append(
                    event['gt_neighbor_gt_counts'][track_index]
                )
            else:
                missed_ranks.append(event['gt_ranks'][track_index])
                missed_scores.append(event['gt_scores'][track_index])
                missed_features_all.append(event['gt_features'][:, track_index])
                missed_gt_neighbor_counts.append(
                    event['gt_neighbor_gt_counts'][track_index]
                )

    found_ranks = np.array(found_ranks)
    missed_ranks = np.array(missed_ranks)
    found_scores = np.array(found_scores)
    missed_scores = np.array(missed_scores)

    total_gt = len(found_ranks) + len(missed_ranks)
    print(f'  Found (rank < 200): {len(found_ranks)} ({100*len(found_ranks)/total_gt:.1f}%)')
    print(f'  Missed (rank >= 200): {len(missed_ranks)} ({100*len(missed_ranks)/total_gt:.1f}%)')

    if len(missed_ranks) > 0:
        print(f'\n  Missed GT rank distribution:')
        pcts = np.percentile(missed_ranks, [25, 50, 75, 90])
        print(f'    p25={pcts[0]:.0f}  p50={pcts[1]:.0f}  '
              f'p75={pcts[2]:.0f}  p90={pcts[3]:.0f}')
        near_miss = np.sum((missed_ranks >= 200) & (missed_ranks < 300))
        deep_miss = np.sum(missed_ranks >= 500)
        print(f'    Near-miss (200-300): {near_miss} ({100*near_miss/len(missed_ranks):.1f}%)')
        print(f'    Deep miss (>500): {deep_miss} ({100*deep_miss/len(missed_ranks):.1f}%)')

    # ==== C. Score Distributions ====
    print(f'\n{"=" * 60}')
    print('C. SCORE DISTRIBUTIONS')
    print('=' * 60)

    if len(found_scores) > 0 and len(missed_scores) > 0:
        print(f'  {"Category":<25} {"mean":>8} {"std":>8} {"min":>8} {"max":>8}')
        print(f'  {"-"*55}')
        for name, vals in [('Found GT', found_scores), ('Missed GT', missed_scores)]:
            print(f'  {name:<25} {vals.mean():>8.3f} {vals.std():>8.3f} '
                  f'{vals.min():>8.3f} {vals.max():>8.3f}')

        # d' between found and missed
        pooled_std = np.sqrt(0.5 * (found_scores.std()**2 + missed_scores.std()**2))
        d_prime = (found_scores.mean() - missed_scores.mean()) / max(pooled_std, 1e-8)
        print(f"\n  d' (found vs missed GT): {d_prime:.3f}")

        # Top false positives
        all_fp = np.concatenate([e['top_fp_scores'] for e in all_events])
        print(f'  Top false positive scores: mean={all_fp.mean():.3f}, '
              f'max={all_fp.max():.3f}')
        print(f'  Overlap: {np.sum(all_fp > missed_scores.mean())} FPs score above '
              f'missed GT mean ({missed_scores.mean():.3f})')

    # ==== D. kNN Neighborhood ====
    print(f'\n{"=" * 60}')
    print('D. kNN NEIGHBORHOOD GT DENSITY')
    print('=' * 60)

    found_counts = np.array(found_gt_neighbor_counts)
    missed_counts = np.array(missed_gt_neighbor_counts)

    if len(found_counts) > 0 and len(missed_counts) > 0:
        print(f'  GT neighbors among k={args.num_neighbors} kNN neighbors:')
        print(f'  {"Category":<25} {"mean":>8} {"0 GT":>8} {"1+ GT":>8} {"2+ GT":>8}')
        print(f'  {"-"*55}')
        for name, counts in [('Found GT', found_counts), ('Missed GT', missed_counts)]:
            pct_0 = 100 * np.mean(counts == 0)
            pct_1 = 100 * np.mean(counts >= 1)
            pct_2 = 100 * np.mean(counts >= 2)
            print(f'  {name:<25} {counts.mean():>8.2f} {pct_0:>7.1f}% {pct_1:>7.1f}% {pct_2:>7.1f}%')

        # Is having a GT neighbor predictive?
        found_has_gt_neighbor = np.mean(found_counts >= 1)
        missed_has_gt_neighbor = np.mean(missed_counts >= 1)
        print(f'\n  Having ≥1 GT neighbor:')
        print(f'    Found GT: {100*found_has_gt_neighbor:.1f}%')
        print(f'    Missed GT: {100*missed_has_gt_neighbor:.1f}%')

    # ==== E. Feature comparison ====
    print(f'\n{"=" * 60}')
    print('E. FEATURE COMPARISON (Found GT vs Missed GT)')
    print('=' * 60)

    feature_names = list(data_config.input_dicts['pf_features'])
    if found_features_all and missed_features_all:
        found_feat = np.stack(found_features_all)  # (N_found, F)
        missed_feat = np.stack(missed_features_all)  # (N_missed, F)

        print(f'  {"Feature":<35} {"Found mean":>10} {"Missed mean":>10} {"Delta":>8}')
        print(f'  {"-"*65}')
        for feature_index, name in enumerate(feature_names):
            f_mean = found_feat[:, feature_index].mean()
            m_mean = missed_feat[:, feature_index].mean()
            delta = m_mean - f_mean
            flag = ' ←' if abs(delta) > 0.3 else ''
            print(f'  {name:<35} {f_mean:>10.3f} {m_mean:>10.3f} {delta:>+8.3f}{flag}')

    print(f'\n{"=" * 60}')
    print('DONE')


if __name__ == '__main__':
    main()
