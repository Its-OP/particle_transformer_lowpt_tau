"""Unit tests for `weaver/weaver/nn/model/CoupleReranker.py`.

The CoupleReranker is the per-couple Stage 3 head designed in
`reports/triplet_reranking/triplet_research_plan_20260408.md`. It consumes a
flat 51-dim feature vector per couple (built by ``utils.couple_features``) and
produces a single scalar score per couple. Architecture: input projection
(Conv1d 51→256), 4 ResidualBlock(256), scoring head (Conv1d 256→128→1).

Tests cover the model contract: shape, padding mask, residual connectivity,
the pairwise ranking loss interface, gradient flow, and a synthetic
loss-decreases-during-overfitting check.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from weaver.nn.model.CoupleReranker import CoupleReranker, ResidualBlock


# ---------------------------------------------------------------------------
# Module construction
# ---------------------------------------------------------------------------

class TestCoupleRerankerConstruction:
    def test_default_input_dim_is_51(self):
        model = CoupleReranker()
        assert model.input_dim == 51

    def test_default_hidden_dim_is_256(self):
        model = CoupleReranker()
        assert model.hidden_dim == 256

    def test_default_num_residual_blocks_is_4(self):
        model = CoupleReranker()
        assert len(model.residual_blocks) == 4

    def test_param_count_in_expected_range(self):
        """At default settings, the model should have ~580K params."""
        model = CoupleReranker()
        n_params = sum(p.numel() for p in model.parameters())
        # Allow ±20% slack for BN counting and minor variations
        assert 450_000 < n_params < 800_000, f'Got {n_params} params'

    def test_custom_input_dim_accepted(self):
        model = CoupleReranker(input_dim=64)
        assert model.input_dim == 64

    def test_custom_hidden_dim_accepted(self):
        model = CoupleReranker(hidden_dim=128)
        assert model.hidden_dim == 128

    def test_custom_num_blocks_accepted(self):
        model = CoupleReranker(num_residual_blocks=2)
        assert len(model.residual_blocks) == 2


# ---------------------------------------------------------------------------
# ResidualBlock
# ---------------------------------------------------------------------------

class TestResidualBlock:
    def test_output_shape_matches_input(self):
        block = ResidualBlock(hidden_dim=256, dropout=0.1)
        x = torch.randn(2, 256, 10)
        y = block(x)
        assert y.shape == x.shape

    def test_residual_at_zero_weights_is_identity(self):
        """If we zero the conv weights inside the block, the output should
        be ReLU(x) (the residual passes through, the conv branch is zero,
        and the trailing ReLU is applied)."""
        block = ResidualBlock(hidden_dim=256, dropout=0.0)
        with torch.no_grad():
            for parameter in block.parameters():
                parameter.zero_()
            # BN running stats: mean 0, var 1, weight 0, bias 0 → BN(x) = 0
        block.eval()
        x = torch.randn(2, 256, 10)
        y = block(x)
        assert torch.allclose(y, torch.relu(x), atol=1e-6)

    def test_gradient_flows_through_skip(self):
        """Even with all conv weights zero, gradient should still propagate
        through the identity skip path."""
        block = ResidualBlock(hidden_dim=256, dropout=0.0)
        with torch.no_grad():
            for parameter in block.parameters():
                parameter.zero_()
        x = torch.randn(2, 256, 10, requires_grad=True)
        y = block(x).sum()
        y.backward()
        assert x.grad is not None
        assert (x.grad != 0).any()


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

class TestForward:
    def test_output_shape_is_batch_x_n_couples(self):
        model = CoupleReranker()
        # 2 events, 51 features per couple, 100 couples per event
        x = torch.randn(2, 51, 100)
        scores = model(x)
        assert scores.shape == (2, 100)

    def test_handles_single_event_batch(self):
        model = CoupleReranker()
        x = torch.randn(1, 51, 50)
        scores = model(x)
        assert scores.shape == (1, 50)

    def test_handles_variable_n_couples(self):
        """The Conv1d-1 architecture must work for any number of couples."""
        model = CoupleReranker()
        for n_couples in [1, 50, 200, 860, 1225]:
            x = torch.randn(2, 51, n_couples)
            scores = model(x)
            assert scores.shape == (2, n_couples)

    def test_scores_are_finite(self):
        model = CoupleReranker()
        model.eval()  # disable dropout for determinism
        x = torch.randn(4, 51, 100)
        scores = model(x)
        assert torch.isfinite(scores).all()

    def test_gradient_flows_to_all_params(self):
        model = CoupleReranker()
        x = torch.randn(2, 51, 50)
        loss = model(x).sum()
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f'{name} has no grad'
            # At least the input projection and the scoring head should
            # always receive non-zero gradient.
            if 'input_projection' in name or 'scorer' in name:
                assert parameter.grad.abs().sum() > 0, f'{name} grad is zero'


# ---------------------------------------------------------------------------
# Compute loss
# ---------------------------------------------------------------------------

class TestComputeLoss:
    def test_loss_dict_shape(self):
        model = CoupleReranker()
        couple_features = torch.randn(2, 51, 100)
        # 5 GT couples per event, marked at fixed positions
        couple_labels = torch.zeros(2, 100)
        couple_labels[:, [0, 1, 2, 3, 4]] = 1.0
        couple_mask = torch.ones(2, 100)
        loss_dict = model.compute_loss(couple_features, couple_labels, couple_mask)
        assert 'total_loss' in loss_dict
        assert 'ranking_loss' in loss_dict
        assert loss_dict['total_loss'].dim() == 0  # scalar
        assert torch.isfinite(loss_dict['total_loss'])

    def test_loss_handles_padded_couples(self):
        """Padded positions (mask = 0) must be ignored in the loss."""
        model = CoupleReranker()
        couple_features = torch.randn(2, 51, 100)
        couple_labels = torch.zeros(2, 100)
        couple_labels[:, 0] = 1.0  # 1 GT couple per event
        # Mask: only first 50 positions are valid; rest are padded
        couple_mask = torch.zeros(2, 100)
        couple_mask[:, :50] = 1.0
        loss_dict = model.compute_loss(couple_features, couple_labels, couple_mask)
        assert torch.isfinite(loss_dict['total_loss'])

    def test_loss_skips_events_with_no_gt(self):
        """Events with 0 GT couples should not contribute to the loss."""
        model = CoupleReranker()
        couple_features = torch.randn(2, 51, 100)
        # First event has 1 GT couple; second event has 0
        couple_labels = torch.zeros(2, 100)
        couple_labels[0, 0] = 1.0
        couple_mask = torch.ones(2, 100)
        loss_dict = model.compute_loss(couple_features, couple_labels, couple_mask)
        # Loss should be finite and based only on event 0
        assert torch.isfinite(loss_dict['total_loss'])

    def test_loss_zero_when_no_events_have_gt(self):
        model = CoupleReranker()
        couple_features = torch.randn(2, 51, 100)
        couple_labels = torch.zeros(2, 100)  # no GT anywhere
        couple_mask = torch.ones(2, 100)
        loss_dict = model.compute_loss(couple_features, couple_labels, couple_mask)
        # No positives → no loss to compute → returns zero
        assert loss_dict['total_loss'].item() == 0.0


# ---------------------------------------------------------------------------
# Synthetic overfit check
# ---------------------------------------------------------------------------

class TestOverfitsTinyTask:
    def test_loss_decreases_on_overfit(self):
        """The model should be able to overfit a tiny synthetic task in
        a few hundred steps. This is the most basic learnability sanity
        check — if it can't overfit 4 events, the architecture is broken.
        """
        torch.manual_seed(0)
        model = CoupleReranker(hidden_dim=64, num_residual_blocks=2)  # smaller for speed
        # 4 events, 51 features per couple, 30 couples per event
        # GT couples are at positions [0, 1, 2] in each event with high feature norm
        n_events = 4
        n_couples = 30
        couple_features = torch.randn(n_events, 51, n_couples) * 0.1
        couple_features[:, :, :3] *= 5.0  # GT couples are easily separable
        couple_labels = torch.zeros(n_events, n_couples)
        couple_labels[:, :3] = 1.0
        couple_mask = torch.ones(n_events, n_couples)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        initial_loss = None
        for step in range(200):
            optimizer.zero_grad()
            loss = model.compute_loss(couple_features, couple_labels, couple_mask)['total_loss']
            if step == 0:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()
        final_loss = loss.item()
        assert final_loss < initial_loss * 0.5, (
            f'Loss did not decrease enough: initial={initial_loss:.4f}, '
            f'final={final_loss:.4f}'
        )
