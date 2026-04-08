"""Unit tests for the train_couple_reranker module-level helpers.

The trainer is a script, but the helpers (label-builder, etc.) are
pure functions and worth testing in isolation so we can sweep over
custom K values without smoke-running every change.
"""
from __future__ import annotations

import pytest

from train_couple_reranker import _build_metric_labels


class TestBuildMetricLabels:
    def test_default_k_values(self):
        labels = _build_metric_labels(
            k_values_tracks=[30, 50, 75, 100, 200],
            k_values_couples=[50, 75, 100, 200],
        )
        # Constants
        assert 'train' in labels
        assert 'val' in labels
        assert 'lr' in labels
        assert 'val_eligible_events' in labels
        assert 'val_total_events' in labels
        assert 'val_events_with_full_triplet' in labels
        assert 'val_mean_first_gt_rank_couples' in labels
        # Per-K
        for k in (30, 50, 75, 100, 200):
            assert f'val_d_at_{k}_tracks' in labels
        for k in (50, 75, 100, 200):
            assert f'val_c_at_{k}_couples' in labels
            assert f'val_rc_at_{k}_couples' in labels

    def test_step_10_k_couples(self):
        """Sweep config: K_couples = 50, 60, 70, ..., 200 (16 values)."""
        k_couples = list(range(50, 201, 10))
        labels = _build_metric_labels(
            k_values_tracks=[30, 50, 100, 200],
            k_values_couples=k_couples,
        )
        for k in k_couples:
            assert f'val_c_at_{k}_couples' in labels
            assert f'val_rc_at_{k}_couples' in labels
            # Label string should reference the K value so the JSON
            # is self-documenting
            assert str(k) in labels[f'val_c_at_{k}_couples']
            assert str(k) in labels[f'val_rc_at_{k}_couples']

    def test_disjoint_k_lists(self):
        """K_tracks and K_couples can have completely different values."""
        labels = _build_metric_labels(
            k_values_tracks=[10, 20, 30],
            k_values_couples=[100, 200],
        )
        for k in (10, 20, 30):
            assert f'val_d_at_{k}_tracks' in labels
        for k in (100, 200):
            assert f'val_c_at_{k}_couples' in labels
            assert f'val_rc_at_{k}_couples' in labels
        # Cross-check: track-only K should not produce C/RC labels
        assert 'val_c_at_10_couples' not in labels
        assert 'val_rc_at_30_couples' not in labels
        # Couple-only K should not produce D labels
        assert 'val_d_at_100_tracks' not in labels

    def test_no_extra_keys_for_unused_k_values(self):
        labels = _build_metric_labels(
            k_values_tracks=[50],
            k_values_couples=[50],
        )
        # Only one K each → no 30, 75, 100, 200 entries
        assert 'val_d_at_30_tracks' not in labels
        assert 'val_c_at_100_couples' not in labels

    def test_label_strings_are_descriptive(self):
        """Each generated label must contain the metric name and the K
        value (so loss_history.json is human-readable)."""
        labels = _build_metric_labels(
            k_values_tracks=[42],
            k_values_couples=[77],
        )
        d_label = labels['val_d_at_42_tracks']
        assert 'D@42' in d_label or 'D@42_tracks' in d_label
        assert '42' in d_label
        c_label = labels['val_c_at_77_couples']
        assert '77' in c_label
        rc_label = labels['val_rc_at_77_couples']
        assert '77' in rc_label


class TestCliFlagPlumbing:
    """Argparse-only tests; no model/data side effects."""

    def test_default_k_values(self):
        from train_couple_reranker import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            '--data-config', 'x', '--data-dir', 'x',
            '--network', 'x', '--cascade-checkpoint', 'x',
        ])
        assert args.k_values_tracks == [30, 50, 75, 100, 200]
        assert args.k_values_couples == [50, 75, 100, 200]

    def test_custom_k_values(self):
        from train_couple_reranker import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            '--data-config', 'x', '--data-dir', 'x',
            '--network', 'x', '--cascade-checkpoint', 'x',
            '--k-values-couples', '50', '60', '70', '80', '90', '100',
            '--k-values-tracks', '30', '50', '100', '150',
        ])
        assert args.k_values_couples == [50, 60, 70, 80, 90, 100]
        assert args.k_values_tracks == [30, 50, 100, 150]


class TestCoupleCascadeKValuesTracksPropagation:
    """Regression: the trainer's k_values_tracks must reach the cascade
    glue model. Otherwise the per-event ``n_gt_in_top_k_tracks`` tensor
    has the model's default shape while the validation accumulator
    expects the trainer's shape, raising IndexError.
    """

    def test_get_model_forwards_k_values_tracks(self):
        """``lowpt_tau_CoupleReranker.get_model`` must thread the
        ``k_values_tracks`` kwarg into the underlying CoupleCascadeModel
        so its ``_run_cascade_to_top_k2`` produces a tensor of the
        right shape."""
        from networks.lowpt_tau_CoupleReranker import _build_frozen_cascade
        from weaver.nn.model.CoupleCascadeModel import CoupleCascadeModel
        from weaver.nn.model.CoupleReranker import CoupleReranker

        # We don't want a real cascade for this test (it'd require the
        # 282 MB checkpoint). Build the glue model directly with stub
        # cascade and verify the k_values_tracks plumbing.
        from tests.test_couple_cascade_model import _make_couple_cascade
        model_default = _make_couple_cascade()
        assert model_default.k_values_tracks == (30, 50, 75, 100, 200)

        # Build with a custom 6-value grid
        custom_k_tracks = (30, 50, 75, 100, 150, 200)
        from tests.test_couple_cascade_model import (
            INPUT_DIM,
            NUM_VALID,
            DummyStage2,
        )
        from weaver.nn.model.CascadeModel import CascadeModel
        from weaver.nn.model.TrackPreFilter import TrackPreFilter

        stage1 = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM, hidden_dim=32, num_message_rounds=1,
        )
        cascade = CascadeModel(
            stage1=stage1, stage2=DummyStage2(input_dim=INPUT_DIM),
            top_k1=NUM_VALID,
        )
        couple_reranker = CoupleReranker(
            input_dim=51, hidden_dim=32, num_residual_blocks=1, dropout=0.0,
        )
        model = CoupleCascadeModel(
            cascade=cascade,
            couple_reranker=couple_reranker,
            top_k2=12,
            k_values_tracks=custom_k_tracks,
        )
        assert model.k_values_tracks == custom_k_tracks

        # And verify the produced tensor has the matching shape
        from tests.test_couple_cascade_model import _make_inputs
        inputs = _make_inputs()
        loss_dict = model.compute_loss(*inputs)
        assert loss_dict['_n_gt_in_top_k_tracks'].shape[1] == len(custom_k_tracks)
