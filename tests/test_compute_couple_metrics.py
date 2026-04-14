"""Tests for diagnostics.compute_couple_metrics — post-hoc metric
computation on eval parquet output joined with input GT labels."""
from __future__ import annotations

import pytest

from diagnostics.compute_couple_metrics import (
    compute_event_metrics,
    aggregate_metrics,
)


# ---------------------------------------------------------------------------
# compute_event_metrics: per-event logic
# ---------------------------------------------------------------------------

class TestComputeEventMetrics:
    def test_gt_couple_at_rank_1(self):
        """GT couple is the top-ranked couple → best_rank=1."""
        gt_pion_pt = [0.5, 1.0, 2.0]
        couples = [[0.5, 1.0], [0.3, 0.7], [0.1, 0.2]]
        remaining = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0]
        result = compute_event_metrics(gt_pion_pt, couples, remaining)
        assert result['best_rank'] == 1
        assert result['eligible'] is True
        assert result['full_triplet'] is True

    def test_gt_couple_at_rank_5(self):
        """GT couple at position 5 → best_rank=5, C@3=0, C@5=1."""
        gt_pion_pt = [0.5, 1.0, 2.0]
        couples = [[0.1, 0.2], [0.3, 0.4], [0.6, 0.7], [0.8, 0.9],
                   [0.5, 1.0], [1.1, 1.2]]
        remaining = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                     1.0, 1.1, 1.2, 2.0]
        result = compute_event_metrics(gt_pion_pt, couples, remaining)
        assert result['best_rank'] == 5

    def test_no_gt_couple_in_output(self):
        """No GT couple found in the output → best_rank=None."""
        gt_pion_pt = [0.5, 1.0, 2.0]
        couples = [[0.1, 0.2], [0.3, 0.4], [0.6, 0.7]]
        remaining = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0, 2.0]
        result = compute_event_metrics(gt_pion_pt, couples, remaining)
        assert result['best_rank'] is None
        assert result['eligible'] is True  # GT tracks ARE in remaining

    def test_rc_requires_full_triplet(self):
        """GT couple found, but only 2 of 3 GT pions in remaining
        → full_triplet=False, RC@K should be 0."""
        gt_pion_pt = [0.5, 1.0, 2.0]
        couples = [[0.5, 1.0], [0.1, 0.2]]
        remaining = [0.1, 0.2, 0.5, 1.0]  # missing GT pion 2.0
        result = compute_event_metrics(gt_pion_pt, couples, remaining)
        assert result['best_rank'] == 1
        assert result['full_triplet'] is False
        assert result['eligible'] is True  # couple {0.5,1.0} IS in remaining

    def test_eligible_requires_gt_couple_in_remaining(self):
        """If no GT couple has BOTH tracks in remaining → not eligible."""
        gt_pion_pt = [0.5, 1.0, 2.0]
        couples = [[0.1, 0.2], [0.3, 0.4]]
        remaining = [0.1, 0.2, 0.3, 0.4, 0.5]  # only GT pion 0.5, no pair
        result = compute_event_metrics(gt_pion_pt, couples, remaining)
        assert result['eligible'] is False
        assert result['best_rank'] is None

    def test_zero_gt_pions(self):
        """Event with 0 GT pions → not eligible, no rank."""
        gt_pion_pt = []
        couples = [[0.1, 0.2], [0.3, 0.4]]
        remaining = [0.1, 0.2, 0.3, 0.4]
        result = compute_event_metrics(gt_pion_pt, couples, remaining)
        assert result['eligible'] is False
        assert result['best_rank'] is None
        assert result['full_triplet'] is False

    def test_two_gt_pions(self):
        """Event with only 2 GT pions → 1 GT couple possible."""
        gt_pion_pt = [0.5, 1.0]
        couples = [[0.5, 1.0], [0.1, 0.2]]
        remaining = [0.1, 0.2, 0.5, 1.0]
        result = compute_event_metrics(gt_pion_pt, couples, remaining)
        assert result['best_rank'] == 1
        assert result['eligible'] is True
        # full_triplet requires 3 GT pions; with only 2, it's False
        assert result['full_triplet'] is False

    def test_couple_order_doesnt_matter(self):
        """GT couple [1.0, 0.5] should match even though GT is {0.5, 1.0}."""
        gt_pion_pt = [0.5, 1.0, 2.0]
        couples = [[0.1, 0.2], [1.0, 0.5], [0.3, 0.4]]
        remaining = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0]
        result = compute_event_metrics(gt_pion_pt, couples, remaining)
        assert result['best_rank'] == 2

    def test_multiple_gt_couples_picks_best(self):
        """3 GT pions → 3 GT couples. Best rank is the earliest one."""
        gt_pion_pt = [0.5, 1.0, 2.0]
        # GT couples: {0.5,1.0}, {0.5,2.0}, {1.0,2.0}
        # {1.0,2.0} appears at rank 2, {0.5,1.0} at rank 4
        couples = [[0.1, 0.2], [1.0, 2.0], [0.3, 0.4], [0.5, 1.0],
                   [0.5, 2.0]]
        remaining = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0]
        result = compute_event_metrics(gt_pion_pt, couples, remaining)
        assert result['best_rank'] == 2  # {1.0,2.0} found first


# ---------------------------------------------------------------------------
# aggregate_metrics: multi-event aggregation
# ---------------------------------------------------------------------------

class TestAggregateMetrics:
    def test_c_at_k_from_ranks(self):
        """3 events with best_ranks [1, 5, None] → C@1=1/3, C@5=2/3."""
        event_results = [
            {'best_rank': 1, 'eligible': True, 'full_triplet': True},
            {'best_rank': 5, 'eligible': True, 'full_triplet': True},
            {'best_rank': None, 'eligible': False, 'full_triplet': False},
        ]
        metrics = aggregate_metrics(event_results, k_values=[1, 5, 10])
        assert abs(metrics['c_at_1'] - 1.0 / 3) < 1e-6
        assert abs(metrics['c_at_5'] - 2.0 / 3) < 1e-6
        assert abs(metrics['c_at_10'] - 2.0 / 3) < 1e-6

    def test_rc_uses_full_triplet(self):
        """RC@K = C@K AND full_triplet."""
        event_results = [
            {'best_rank': 1, 'eligible': True, 'full_triplet': True},
            {'best_rank': 2, 'eligible': True, 'full_triplet': False},
            {'best_rank': None, 'eligible': False, 'full_triplet': False},
        ]
        metrics = aggregate_metrics(event_results, k_values=[5])
        assert abs(metrics['c_at_5'] - 2.0 / 3) < 1e-6
        # Only the first event has full_triplet
        assert abs(metrics['rc_at_5'] - 1.0 / 3) < 1e-6

    def test_rc_leq_c_always(self):
        event_results = [
            {'best_rank': 1, 'eligible': True, 'full_triplet': True},
            {'best_rank': 3, 'eligible': True, 'full_triplet': False},
        ]
        metrics = aggregate_metrics(event_results, k_values=[1, 5, 10])
        for k in [1, 5, 10]:
            assert metrics[f'rc_at_{k}'] <= metrics[f'c_at_{k}']

    def test_monotonically_nondecreasing_in_k(self):
        event_results = [
            {'best_rank': 3, 'eligible': True, 'full_triplet': True},
            {'best_rank': 7, 'eligible': True, 'full_triplet': True},
            {'best_rank': 50, 'eligible': True, 'full_triplet': True},
        ]
        metrics = aggregate_metrics(event_results, k_values=[1, 5, 10, 50, 100])
        for i in range(len([1, 5, 10, 50, 100]) - 1):
            k_small = [1, 5, 10, 50, 100][i]
            k_large = [1, 5, 10, 50, 100][i + 1]
            assert metrics[f'c_at_{k_large}'] >= metrics[f'c_at_{k_small}']

    def test_mean_rank_over_eligible(self):
        event_results = [
            {'best_rank': 2, 'eligible': True, 'full_triplet': True},
            {'best_rank': 8, 'eligible': True, 'full_triplet': True},
            {'best_rank': None, 'eligible': False, 'full_triplet': False},
        ]
        metrics = aggregate_metrics(event_results, k_values=[10])
        # mean_rank = (2 + 8) / 2 = 5.0 (only eligible events)
        assert metrics['mean_rank'] == 5.0
        assert metrics['eligible'] == 2
        assert metrics['total'] == 3

    def test_empty_events(self):
        metrics = aggregate_metrics([], k_values=[10])
        assert metrics['c_at_10'] == 0.0
        assert metrics['total'] == 0
