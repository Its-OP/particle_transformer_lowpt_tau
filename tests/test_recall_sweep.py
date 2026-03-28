"""Tests for diagnostics/recall_at_k_sweep.py — R@K diagnostic sweep.

Verifies that the sweep script can:
1. Load a checkpoint and construct the model
2. Run inference and compute R@K metrics
3. Return valid metrics dict with expected keys
"""

import os
import sys
import pytest
import torch

# Ensure part/ is on the import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def checkpoint_path():
    """Path to the best pre-filter checkpoint."""
    path = os.path.join(
        os.path.dirname(__file__), '..', 'models', 'prefilter_best.pt',
    )
    if not os.path.exists(path):
        pytest.skip('prefilter_best.pt not found')
    return path


@pytest.fixture
def data_config_path():
    """Path to the data config YAML."""
    return os.path.join(
        os.path.dirname(__file__), '..', 'data', 'low-pt',
        'lowpt_tau_trackfinder.yaml',
    )


@pytest.fixture
def val_data_dir():
    """Path to subset validation data for local testing."""
    path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'low-pt', 'subset', 'val',
    )
    if not os.path.exists(path):
        pytest.skip('Subset val data not found')
    return path


class TestLoadCheckpoint:
    """Test checkpoint loading and model construction."""

    def test_load_model_from_checkpoint(self, checkpoint_path):
        """Model loads from checkpoint and enters eval mode."""
        from diagnostics.recall_at_k_sweep import load_prefilter_from_checkpoint

        device = torch.device('cpu')
        model = load_prefilter_from_checkpoint(checkpoint_path, device)

        assert model is not None
        assert not model.training, 'Model should be in eval mode'

        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 100_000, (
            f'Expected > 100K params, got {total_params}'
        )

    def test_checkpoint_metadata(self, checkpoint_path):
        """Checkpoint contains expected metadata."""
        checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False,
        )
        assert 'model_state_dict' in checkpoint
        assert 'epoch' in checkpoint
        assert 'best_val_recall_at_200' in checkpoint
        assert checkpoint['best_val_recall_at_200'] > 0.5


class TestSweep:
    """Test the full R@K sweep on real subset data."""

    @pytest.mark.timeout(300)
    def test_sweep_produces_valid_metrics(
        self, checkpoint_path, data_config_path, val_data_dir,
    ):
        """Sweep returns metrics dict with all expected R@K values."""
        from diagnostics.recall_at_k_sweep import run_recall_sweep

        k_values = (10, 50, 100, 200, 300, 400, 500, 600, 800)
        metrics = run_recall_sweep(
            checkpoint_path=checkpoint_path,
            data_config_path=data_config_path,
            data_dir=val_data_dir,
            k_values=k_values,
            device=torch.device('cpu'),
            batch_size=4,
            max_steps=2,
            num_workers=0,
        )

        # Check all expected keys exist
        for k in k_values:
            assert f'recall_at_{k}' in metrics, f'Missing recall_at_{k}'
            assert f'perfect_at_{k}' in metrics, f'Missing perfect_at_{k}'

        # Check global metrics
        assert 'median_gt_rank' in metrics
        assert 'd_prime' in metrics
        assert 'total_events_with_gt' in metrics
        assert 'total_gt_tracks' in metrics

        # Values should be valid floats in [0, 1] for recall
        for k in k_values:
            recall = metrics[f'recall_at_{k}']
            assert 0.0 <= recall <= 1.0, (
                f'recall_at_{k}={recall} out of [0, 1]'
            )

        # R@K should be monotonically non-decreasing with K
        recalls = [metrics[f'recall_at_{k}'] for k in k_values]
        for i in range(1, len(recalls)):
            assert recalls[i] >= recalls[i - 1] - 1e-6, (
                f'R@{k_values[i]}={recalls[i]} < R@{k_values[i-1]}='
                f'{recalls[i-1]} — recall must be non-decreasing with K'
            )

    @pytest.mark.timeout(300)
    def test_sweep_events_counted(
        self, checkpoint_path, data_config_path, val_data_dir,
    ):
        """Sweep processes events and counts GT tracks."""
        from diagnostics.recall_at_k_sweep import run_recall_sweep

        metrics = run_recall_sweep(
            checkpoint_path=checkpoint_path,
            data_config_path=data_config_path,
            data_dir=val_data_dir,
            k_values=(200, 600),
            device=torch.device('cpu'),
            batch_size=4,
            max_steps=2,
            num_workers=0,
        )

        assert metrics['total_events_with_gt'] > 0
        assert metrics['total_gt_tracks'] > 0
