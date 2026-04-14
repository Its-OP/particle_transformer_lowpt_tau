"""Tests for NanSafeBatchNorm1d — a BatchNorm1d wrapper that skips
running stat updates when inputs contain NaN/Inf values.

Ensures that:
- Clean inputs behave identically to standard BatchNorm1d
- NaN inputs produce finite output (NaN elements replaced with 0)
- Running stats remain finite after processing NaN batches
- num_batches_tracked does not increment on NaN batches
- State dict is compatible with standard BatchNorm1d (same keys)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from weaver.nn.model.CoupleReranker import NanSafeBatchNorm1d


class TestNanSafeBatchNormCleanInputs:
    """Clean inputs should produce identical results to standard BN."""

    def test_output_matches_standard_batchnorm(self):
        torch.manual_seed(42)
        num_features = 16
        standard_batchnorm = nn.BatchNorm1d(num_features)
        nan_safe_batchnorm = NanSafeBatchNorm1d(num_features)

        # Copy weights so both start from identical state
        nan_safe_batchnorm.load_state_dict(standard_batchnorm.state_dict())

        x = torch.randn(8, num_features, 100)
        standard_batchnorm.train()
        nan_safe_batchnorm.train()

        out_standard = standard_batchnorm(x)
        out_safe = nan_safe_batchnorm(x)

        assert torch.allclose(out_standard, out_safe, atol=1e-6)

    def test_running_stats_match_after_clean_batches(self):
        torch.manual_seed(42)
        num_features = 8
        standard_batchnorm = nn.BatchNorm1d(num_features)
        nan_safe_batchnorm = NanSafeBatchNorm1d(num_features)
        nan_safe_batchnorm.load_state_dict(standard_batchnorm.state_dict())

        standard_batchnorm.train()
        nan_safe_batchnorm.train()

        for _ in range(10):
            x = torch.randn(4, num_features, 50)
            standard_batchnorm(x)
            nan_safe_batchnorm(x)

        assert torch.allclose(
            standard_batchnorm.running_mean,
            nan_safe_batchnorm.running_mean,
            atol=1e-6,
        )
        assert torch.allclose(
            standard_batchnorm.running_var,
            nan_safe_batchnorm.running_var,
            atol=1e-6,
        )
        assert (
            standard_batchnorm.num_batches_tracked
            == nan_safe_batchnorm.num_batches_tracked
        )


class TestNanSafeBatchNormNanInputs:
    """NaN inputs should produce finite output and not corrupt stats."""

    def test_output_is_finite_when_input_has_nan(self):
        num_features = 8
        batchnorm = NanSafeBatchNorm1d(num_features)
        batchnorm.train()

        # Feed a few clean batches first to build running stats
        for _ in range(5):
            batchnorm(torch.randn(4, num_features, 50))

        # Now feed a batch with NaN
        x = torch.randn(4, num_features, 50)
        x[0, 2, 10] = float('nan')
        x[1, 5, :] = float('nan')

        output = batchnorm(x)
        assert torch.isfinite(output).all(), (
            f'NaN in output: {(~torch.isfinite(output)).sum().item()} non-finite values'
        )

    def test_output_is_finite_when_input_has_inf(self):
        num_features = 8
        batchnorm = NanSafeBatchNorm1d(num_features)
        batchnorm.train()

        for _ in range(5):
            batchnorm(torch.randn(4, num_features, 50))

        x = torch.randn(4, num_features, 50)
        x[0, 0, 0] = float('inf')
        x[2, 3, 25] = float('-inf')

        output = batchnorm(x)
        assert torch.isfinite(output).all()

    def test_running_stats_stay_finite_after_nan_batch(self):
        num_features = 8
        batchnorm = NanSafeBatchNorm1d(num_features)
        batchnorm.train()

        # Build valid running stats
        for _ in range(5):
            batchnorm(torch.randn(4, num_features, 50))

        running_mean_before = batchnorm.running_mean.clone()
        running_var_before = batchnorm.running_var.clone()

        # Feed a NaN batch
        x = torch.randn(4, num_features, 50)
        x[0, :, :] = float('nan')
        batchnorm(x)

        # Running stats must not change
        assert torch.isfinite(batchnorm.running_mean).all()
        assert torch.isfinite(batchnorm.running_var).all()
        assert torch.equal(batchnorm.running_mean, running_mean_before)
        assert torch.equal(batchnorm.running_var, running_var_before)

    def test_num_batches_tracked_does_not_increment_on_nan(self):
        num_features = 8
        batchnorm = NanSafeBatchNorm1d(num_features)
        batchnorm.train()

        # 5 clean batches
        for _ in range(5):
            batchnorm(torch.randn(4, num_features, 50))
        assert batchnorm.num_batches_tracked == 5

        # 1 NaN batch — should NOT increment
        x = torch.randn(4, num_features, 50)
        x[1, :, :] = float('nan')
        batchnorm(x)
        assert batchnorm.num_batches_tracked == 5

        # 1 more clean batch — should increment
        batchnorm(torch.randn(4, num_features, 50))
        assert batchnorm.num_batches_tracked == 6

    def test_fully_nan_batch_does_not_corrupt(self):
        """Edge case: entire batch is NaN."""
        num_features = 4
        batchnorm = NanSafeBatchNorm1d(num_features)
        batchnorm.train()

        # Build valid running stats
        for _ in range(3):
            batchnorm(torch.randn(2, num_features, 20))

        x = torch.full((2, num_features, 20), float('nan'))
        output = batchnorm(x)

        assert torch.isfinite(batchnorm.running_mean).all()
        assert torch.isfinite(batchnorm.running_var).all()
        assert torch.isfinite(output).all()


class TestNanSafeBatchNormEvalMode:
    """In eval mode, NanSafeBatchNorm should use running stats normally."""

    def test_eval_mode_uses_running_stats(self):
        num_features = 8
        batchnorm = NanSafeBatchNorm1d(num_features)
        batchnorm.train()

        # Accumulate running stats
        for _ in range(10):
            batchnorm(torch.randn(4, num_features, 50))

        batchnorm.eval()
        x = torch.randn(4, num_features, 50)
        output = batchnorm(x)
        assert torch.isfinite(output).all()


class TestNanSafeBatchNormStateDict:
    """State dict must be compatible with standard BatchNorm1d."""

    def test_state_dict_keys_match_standard(self):
        num_features = 16
        standard_batchnorm = nn.BatchNorm1d(num_features)
        nan_safe_batchnorm = NanSafeBatchNorm1d(num_features)

        assert set(standard_batchnorm.state_dict().keys()) == set(
            nan_safe_batchnorm.state_dict().keys()
        )

    def test_can_load_standard_batchnorm_state(self):
        num_features = 16
        standard_batchnorm = nn.BatchNorm1d(num_features)
        nan_safe_batchnorm = NanSafeBatchNorm1d(num_features)

        # Train standard BN to get non-default running stats
        standard_batchnorm.train()
        for _ in range(5):
            standard_batchnorm(torch.randn(4, num_features, 50))

        # Load into NanSafe
        nan_safe_batchnorm.load_state_dict(standard_batchnorm.state_dict())
        assert torch.equal(
            standard_batchnorm.running_mean,
            nan_safe_batchnorm.running_mean,
        )

    def test_standard_can_load_nan_safe_state(self):
        """Reverse compatibility: NanSafe → standard."""
        num_features = 16
        nan_safe_batchnorm = NanSafeBatchNorm1d(num_features)
        standard_batchnorm = nn.BatchNorm1d(num_features)

        nan_safe_batchnorm.train()
        for _ in range(5):
            nan_safe_batchnorm(torch.randn(4, num_features, 50))

        standard_batchnorm.load_state_dict(nan_safe_batchnorm.state_dict())
        assert torch.equal(
            nan_safe_batchnorm.running_mean,
            standard_batchnorm.running_mean,
        )
