"""Unit tests for ``CoupleMetricsAccumulator`` (D@K_tracks, C@K_couples, RC@K_couples)."""
from __future__ import annotations

import pytest
import torch

from utils.training_utils import CoupleMetricsAccumulator


def _accumulator(
    k_values_couples=(50, 100, 200),
    k_values_tracks=(30, 50, 75, 100, 200),
):
    return CoupleMetricsAccumulator(
        k_values_couples=k_values_couples,
        k_values_tracks=k_values_tracks,
    )


# ---------------------------------------------------------------------------
# C@K_couples basics
# ---------------------------------------------------------------------------

class TestCCouples:
    def test_perfect_ranking_recall_at_50(self):
        accumulator = _accumulator(k_values_couples=(50, 100, 200))
        n_couples = 200
        scores = torch.zeros(1, n_couples)
        scores[0, 0] = 100.0
        labels = torch.zeros(1, n_couples)
        labels[0, 0] = 1.0
        mask = torch.ones(1, n_couples)
        accumulator.update(scores, labels, mask)
        metrics = accumulator.compute()
        assert metrics['c_at_50_couples'] == 1.0
        assert metrics['c_at_100_couples'] == 1.0
        assert metrics['c_at_200_couples'] == 1.0
        assert metrics['eligible_events'] == 1

    def test_gt_at_rank_75_only_in_top_100_and_200(self):
        accumulator = _accumulator(k_values_couples=(50, 75, 100, 200))
        n_couples = 200
        scores = torch.linspace(1.0, 0.0, n_couples).unsqueeze(0)
        labels = torch.zeros(1, n_couples)
        labels[0, 75] = 1.0
        mask = torch.ones(1, n_couples)
        accumulator.update(scores, labels, mask)
        metrics = accumulator.compute()
        assert metrics['c_at_50_couples'] == 0.0
        assert metrics['c_at_75_couples'] == 0.0
        assert metrics['c_at_100_couples'] == 1.0
        assert metrics['c_at_200_couples'] == 1.0

    def test_event_with_no_gt_excluded_from_c(self):
        accumulator = _accumulator(k_values_couples=(50, 100))
        scores = torch.randn(1, 200)
        labels = torch.zeros(1, 200)
        mask = torch.ones(1, 200)
        accumulator.update(scores, labels, mask)
        metrics = accumulator.compute()
        assert metrics['eligible_events'] == 0
        assert metrics['c_at_50_couples'] == 0.0

    def test_multiple_gt_couples_do_not_inflate_metric(self):
        """If an event has 3 GT couples and 2 are in top-50, C@50 should
        still contribute exactly 1.0 (binary per event)."""
        accumulator = _accumulator(k_values_couples=(50,))
        n_couples = 200
        scores = torch.linspace(1.0, 0.0, n_couples).unsqueeze(0)
        labels = torch.zeros(1, n_couples)
        labels[0, 5] = 1.0
        labels[0, 10] = 1.0
        labels[0, 80] = 1.0
        mask = torch.ones(1, n_couples)
        accumulator.update(scores, labels, mask)
        metrics = accumulator.compute()
        assert metrics['c_at_50_couples'] == 1.0


# ---------------------------------------------------------------------------
# RC@K_couples
# ---------------------------------------------------------------------------

class TestRcCouples:
    def test_rc_requires_full_triplet_in_top_k1(self):
        accumulator = _accumulator(k_values_couples=(50, 100))
        n_couples = 200
        scores = torch.zeros(1, n_couples)
        scores[0, 0] = 100.0
        labels = torch.zeros(1, n_couples)
        labels[0, 0] = 1.0
        mask = torch.ones(1, n_couples)
        n_gt_in_top_k1 = torch.tensor([2])
        accumulator.update(scores, labels, mask, n_gt_in_top_k1=n_gt_in_top_k1)
        metrics = accumulator.compute()
        assert metrics['c_at_50_couples'] == 1.0
        assert metrics['rc_at_50_couples'] == 0.0
        assert metrics['rc_at_100_couples'] == 0.0

    def test_rc_equals_c_when_full_triplet_present(self):
        accumulator = _accumulator(k_values_couples=(50,))
        n_couples = 200
        scores = torch.zeros(1, n_couples)
        scores[0, 0] = 100.0
        labels = torch.zeros(1, n_couples)
        labels[0, 0] = 1.0
        mask = torch.ones(1, n_couples)
        n_gt_in_top_k1 = torch.tensor([3])
        accumulator.update(scores, labels, mask, n_gt_in_top_k1=n_gt_in_top_k1)
        metrics = accumulator.compute()
        assert metrics['c_at_50_couples'] == 1.0
        assert metrics['rc_at_50_couples'] == 1.0

    def test_rc_zero_when_couple_not_in_top_k(self):
        accumulator = _accumulator(k_values_couples=(50,))
        n_couples = 200
        scores = torch.linspace(1.0, 0.0, n_couples).unsqueeze(0)
        labels = torch.zeros(1, n_couples)
        labels[0, 100] = 1.0
        mask = torch.ones(1, n_couples)
        n_gt_in_top_k1 = torch.tensor([3])
        accumulator.update(scores, labels, mask, n_gt_in_top_k1=n_gt_in_top_k1)
        metrics = accumulator.compute()
        assert metrics['c_at_50_couples'] == 0.0
        assert metrics['rc_at_50_couples'] == 0.0

    def test_rc_omitted_when_n_gt_in_top_k1_is_none(self):
        accumulator = _accumulator(k_values_couples=(50,))
        n_couples = 200
        scores = torch.zeros(1, n_couples)
        scores[0, 0] = 100.0
        labels = torch.zeros(1, n_couples)
        labels[0, 0] = 1.0
        mask = torch.ones(1, n_couples)
        accumulator.update(scores, labels, mask)
        metrics = accumulator.compute()
        assert metrics['c_at_50_couples'] == 1.0
        assert metrics['rc_at_50_couples'] == 0.0
        assert metrics['events_with_full_triplet'] == 0


# ---------------------------------------------------------------------------
# D@K_tracks
# ---------------------------------------------------------------------------

class TestDTracks:
    def test_d_zero_with_no_n_gt_in_top_k_tracks(self):
        accumulator = _accumulator(k_values_tracks=(30, 50))
        scores = torch.zeros(2, 100)
        labels = torch.zeros(2, 100)
        mask = torch.ones(2, 100)
        # No n_gt_in_top_k_tracks → D@K stays at 0
        accumulator.update(scores, labels, mask)
        metrics = accumulator.compute()
        assert metrics['d_at_30_tracks'] == 0.0
        assert metrics['d_at_50_tracks'] == 0.0
        # total_events should still be incremented (denominator works)
        assert metrics['total_events'] == 2

    def test_d_per_event_threshold_is_geq_2(self):
        """D@K = events with at least 2 GT in top-K tracks. 2 → 1, 1 → 0, 3 → 1."""
        accumulator = _accumulator(k_values_tracks=(50,))
        n_couples = 100
        scores = torch.zeros(3, n_couples)
        labels = torch.zeros(3, n_couples)
        mask = torch.ones(3, n_couples)
        # n_gt_in_top_50_tracks per event:
        # event 0: 1 (below threshold)
        # event 1: 2 (at threshold → counts)
        # event 2: 3 (above threshold → counts)
        n_gt_in_top_k_tracks = torch.tensor([[1], [2], [3]])
        accumulator.update(
            scores, labels, mask,
            n_gt_in_top_k_tracks=n_gt_in_top_k_tracks,
        )
        metrics = accumulator.compute()
        # 2 of 3 events have ≥2 GT pions in top-50 tracks
        assert abs(metrics['d_at_50_tracks'] - 2.0 / 3.0) < 1e-6
        assert metrics['total_events'] == 3

    def test_d_at_multiple_k_values(self):
        accumulator = _accumulator(k_values_tracks=(30, 50, 100))
        n_couples = 100
        scores = torch.zeros(2, n_couples)
        labels = torch.zeros(2, n_couples)
        mask = torch.ones(2, n_couples)
        # event 0: 1 GT in top-30, 2 in top-50, 3 in top-100
        # event 1: 0 GT in top-30, 1 in top-50, 2 in top-100
        n_gt_in_top_k_tracks = torch.tensor([
            [1, 2, 3],
            [0, 1, 2],
        ])
        accumulator.update(
            scores, labels, mask,
            n_gt_in_top_k_tracks=n_gt_in_top_k_tracks,
        )
        metrics = accumulator.compute()
        # D@30: 0/2 events have ≥2 GT in top-30
        assert metrics['d_at_30_tracks'] == 0.0
        # D@50: 1/2 events (event 0) have ≥2 GT in top-50
        assert metrics['d_at_50_tracks'] == 0.5
        # D@100: 2/2 events have ≥2 GT in top-100
        assert metrics['d_at_100_tracks'] == 1.0

    def test_d_denominator_is_total_events_not_eligible(self):
        """D's denominator counts ALL events (it's a property of the
        cascade, independent of whether the event has a GT couple in the
        candidate pool)."""
        accumulator = _accumulator(k_values_tracks=(50,), k_values_couples=(50,))
        n_couples = 100
        # Two events: neither has any GT couple (so eligible_events = 0)
        scores = torch.zeros(2, n_couples)
        labels = torch.zeros(2, n_couples)  # no GT couples
        mask = torch.ones(2, n_couples)
        # But both events have ≥2 GT pions in top-50 tracks
        n_gt_in_top_k_tracks = torch.tensor([[3], [3]])
        accumulator.update(
            scores, labels, mask,
            n_gt_in_top_k_tracks=n_gt_in_top_k_tracks,
        )
        metrics = accumulator.compute()
        assert metrics['eligible_events'] == 0  # no eligible events for C/RC
        assert metrics['total_events'] == 2     # but D's denominator is full
        assert metrics['d_at_50_tracks'] == 1.0
        assert metrics['c_at_50_couples'] == 0.0   # no eligible → 0 (max(1, 0))
        assert metrics['rc_at_50_couples'] == 0.0


# ---------------------------------------------------------------------------
# Padding mask handling
# ---------------------------------------------------------------------------

class TestPaddingMask:
    def test_padded_couples_pushed_to_bottom_of_ranking(self):
        accumulator = _accumulator(k_values_couples=(50,))
        n_couples = 200
        scores = torch.zeros(1, n_couples)
        scores[0, 100:] = 1000.0
        scores[0, 0] = 1.0
        labels = torch.zeros(1, n_couples)
        labels[0, 0] = 1.0
        mask = torch.zeros(1, n_couples)
        mask[0, :100] = 1.0
        accumulator.update(scores, labels, mask)
        metrics = accumulator.compute()
        assert metrics['c_at_50_couples'] == 1.0


# ---------------------------------------------------------------------------
# Multi-event averaging
# ---------------------------------------------------------------------------

class TestMultiEvent:
    def test_average_over_eligible_events(self):
        accumulator = _accumulator(k_values_couples=(50,))
        n_couples = 200
        scores = torch.zeros(4, n_couples)
        labels = torch.zeros(4, n_couples)
        mask = torch.ones(4, n_couples)
        scores[0, 0] = 100.0
        labels[0, 0] = 1.0
        scores[1, 0] = 100.0
        labels[1, 0] = 1.0
        scores[2] = torch.linspace(1.0, 0.0, n_couples)
        labels[2, 100] = 1.0
        # Event 3: no GT couple → excluded from C/RC
        n_gt_in_top_k1 = torch.tensor([3, 3, 3, 3])
        accumulator.update(scores, labels, mask, n_gt_in_top_k1=n_gt_in_top_k1)
        metrics = accumulator.compute()
        assert metrics['eligible_events'] == 3
        assert metrics['events_with_full_triplet'] == 3
        assert abs(metrics['c_at_50_couples'] - 2.0 / 3.0) < 1e-6
        assert abs(metrics['rc_at_50_couples'] - 2.0 / 3.0) < 1e-6

    def test_rc_smaller_than_c_when_some_have_partial_triplet(self):
        accumulator = _accumulator(k_values_couples=(50,))
        n_couples = 200
        scores = torch.zeros(4, n_couples)
        scores[:, 0] = 100.0
        labels = torch.zeros(4, n_couples)
        labels[:, 0] = 1.0
        mask = torch.ones(4, n_couples)
        n_gt_in_top_k1 = torch.tensor([3, 3, 3, 2])
        accumulator.update(scores, labels, mask, n_gt_in_top_k1=n_gt_in_top_k1)
        metrics = accumulator.compute()
        assert metrics['c_at_50_couples'] == 1.0
        assert metrics['rc_at_50_couples'] == 0.75
        assert metrics['events_with_full_triplet'] == 3


# ---------------------------------------------------------------------------
# Cross-batch accumulation
# ---------------------------------------------------------------------------

class TestCrossBatch:
    def test_accumulates_across_multiple_update_calls(self):
        accumulator = _accumulator(k_values_couples=(50,))
        n_couples = 200
        scores_1 = torch.zeros(1, n_couples)
        scores_1[0, 0] = 100.0
        labels_1 = torch.zeros(1, n_couples)
        labels_1[0, 0] = 1.0
        accumulator.update(
            scores_1, labels_1, torch.ones(1, n_couples),
            n_gt_in_top_k1=torch.tensor([3]),
        )
        scores_2 = torch.linspace(1.0, 0.0, n_couples).unsqueeze(0)
        labels_2 = torch.zeros(1, n_couples)
        labels_2[0, 100] = 1.0
        accumulator.update(
            scores_2, labels_2, torch.ones(1, n_couples),
            n_gt_in_top_k1=torch.tensor([3]),
        )
        metrics = accumulator.compute()
        assert metrics['eligible_events'] == 2
        assert metrics['c_at_50_couples'] == 0.5
        assert metrics['rc_at_50_couples'] == 0.5
