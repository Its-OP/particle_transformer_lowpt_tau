"""Unit tests for curriculum negative subsampling (Phase 1.2).

TDD: Written before implementation. Tests cover:
    - subsample_negatives() preserves all GT tracks
    - subsample_negatives() returns correct negative count
    - subsample_negatives() handles edge cases (no GT, all GT, empty)
    - get_curriculum_num_negatives() follows the A/B/C phase schedule
"""
import math

import pytest
import torch

from utils.training_utils import (
    get_curriculum_num_negatives,
    subsample_negatives,
)


# ---- Shared test data ----

BATCH_SIZE = 4
NUM_TRACKS = 200


def _make_curriculum_inputs(
    batch_size=BATCH_SIZE,
    num_tracks=NUM_TRACKS,
    num_gt_per_event=3,
    num_padding=30,
):
    """Create mask and labels for curriculum testing."""
    mask = torch.ones(batch_size, 1, num_tracks)
    mask[:, :, -num_padding:] = 0.0

    track_labels = torch.zeros(batch_size, 1, num_tracks)
    for batch_index in range(batch_size):
        for gt_offset in range(num_gt_per_event):
            track_labels[batch_index, 0, 10 + gt_offset * 10] = 1.0

    return mask, track_labels


# ---- subsample_negatives() ----

class TestSubsampleNegatives:
    """Test the negative subsampling function."""

    def test_preserves_all_gt_tracks(self):
        """All GT tracks must remain in the subsampled mask."""
        mask, track_labels = _make_curriculum_inputs()
        subsampled_mask = subsample_negatives(
            mask, track_labels, num_negatives=30,
        )
        gt_positions = track_labels.squeeze(1) == 1.0
        gt_still_valid = subsampled_mask.squeeze(1)[gt_positions]
        assert gt_still_valid.all(), (
            'Some GT tracks were removed by subsampling'
        )

    def test_correct_negative_count(self):
        """After subsampling, each event should have exactly N negatives."""
        mask, track_labels = _make_curriculum_inputs(
            num_gt_per_event=3, num_padding=30,
        )
        num_negatives = 30
        subsampled_mask = subsample_negatives(
            mask, track_labels, num_negatives=num_negatives,
        )
        for event_index in range(BATCH_SIZE):
            event_mask = subsampled_mask[event_index, 0].bool()
            event_labels = track_labels[event_index, 0]
            num_gt = ((event_labels == 1.0) & event_mask).sum().item()
            actual_negatives = event_mask.sum().item() - num_gt
            assert actual_negatives == num_negatives, (
                f'Event {event_index}: expected {num_negatives} negatives, '
                f'got {actual_negatives}'
            )

    def test_does_not_modify_original_mask(self):
        """Subsampling should return a new tensor, not modify in-place."""
        mask, track_labels = _make_curriculum_inputs()
        original_mask_sum = mask.sum().item()
        _ = subsample_negatives(mask, track_labels, num_negatives=30)
        assert mask.sum().item() == original_mask_sum

    def test_padding_stays_masked(self):
        """Padding tracks (originally masked out) should stay masked."""
        mask, track_labels = _make_curriculum_inputs(num_padding=30)
        subsampled_mask = subsample_negatives(
            mask, track_labels, num_negatives=30,
        )
        assert (subsampled_mask[:, :, -30:] == 0.0).all()

    def test_num_negatives_larger_than_available(self):
        """When requesting more negatives than available, keep all negatives."""
        mask, track_labels = _make_curriculum_inputs(
            num_gt_per_event=3, num_padding=30,
        )
        # 200 - 30 padding - 3 GT = 167 negatives available
        subsampled_mask = subsample_negatives(
            mask, track_labels, num_negatives=500,
        )
        for event_index in range(BATCH_SIZE):
            num_valid = subsampled_mask[event_index, 0].bool().sum().item()
            assert num_valid == 170

    def test_no_gt_tracks(self):
        """Events with no GT should still subsample negatives correctly."""
        mask = torch.ones(1, 1, NUM_TRACKS)
        mask[:, :, -30:] = 0.0
        track_labels = torch.zeros(1, 1, NUM_TRACKS)

        subsampled_mask = subsample_negatives(
            mask, track_labels, num_negatives=20,
        )
        num_valid = subsampled_mask[0, 0].bool().sum().item()
        assert num_valid == 20

    def test_output_shape_matches_input(self):
        """Subsampled mask should have the same shape as input mask."""
        mask, track_labels = _make_curriculum_inputs()
        subsampled_mask = subsample_negatives(
            mask, track_labels, num_negatives=30,
        )
        assert subsampled_mask.shape == mask.shape

    def test_output_device_matches_input(self):
        """Subsampled mask should be on the same device as input."""
        mask, track_labels = _make_curriculum_inputs()
        subsampled_mask = subsample_negatives(
            mask, track_labels, num_negatives=30,
        )
        assert subsampled_mask.device == mask.device


# ---- get_curriculum_num_negatives() ----

class TestCurriculumSchedule:
    """Test the curriculum schedule for negative subsampling.

    Schedule (3 phases):
        Phase A (epochs 0 to E/3):     fixed at min_negatives (easy)
        Phase B (epochs E/3 to 2E/3):  cosine increase from min to full
        Phase C (epochs 2E/3 to E):    full tracks (no subsampling)
    """

    def test_phase_a_returns_min_negatives(self):
        """Phase A should return min_negatives for all epochs in [0, E/3)."""
        total_epochs = 30
        min_negatives = 30
        for epoch in range(0, 10):
            result = get_curriculum_num_negatives(
                epoch, total_epochs, min_negatives=min_negatives,
            )
            assert result == min_negatives, (
                f'Epoch {epoch}: expected {min_negatives}, got {result}'
            )

    def test_phase_b_increases_monotonically(self):
        """Phase B should increase monotonically from min to None (full)."""
        total_epochs = 30
        min_negatives = 30
        previous_value = 0
        for epoch in range(10, 20):
            result = get_curriculum_num_negatives(
                epoch, total_epochs, min_negatives=min_negatives,
            )
            if result is not None:
                assert result >= previous_value, (
                    f'Epoch {epoch}: value {result} < previous {previous_value}'
                )
                previous_value = result

    def test_phase_c_returns_none(self):
        """Phase C should return None (no subsampling = full tracks)."""
        total_epochs = 30
        for epoch in range(20, 30):
            result = get_curriculum_num_negatives(
                epoch, total_epochs, min_negatives=30,
            )
            assert result is None, (
                f'Epoch {epoch}: expected None (full tracks), got {result}'
            )

    def test_phase_b_cosine_midpoint(self):
        """At the midpoint of Phase B, negatives should be roughly halfway.

        Cosine schedule: n(t) = min + (max - min) × 0.5 × (1 - cos(πt))
        At t=0.5: n = min + (max - min) × 0.5
        """
        total_epochs = 30
        min_negatives = 30
        result = get_curriculum_num_negatives(
            15, total_epochs, min_negatives=min_negatives,
            max_negatives=170,
        )
        # Cosine midpoint: 30 + (170 - 30) * 0.5 = 100
        assert result is not None
        assert 80 <= result <= 120, (
            f'Phase B midpoint: expected ~100, got {result}'
        )

    def test_phase_b_start_equals_min(self):
        """First epoch of Phase B should equal min_negatives."""
        result = get_curriculum_num_negatives(
            10, 30, min_negatives=30, max_negatives=170,
        )
        assert result == 30

    def test_boundary_epoch_e_third(self):
        """Epoch exactly at E/3 boundary should be Phase B start."""
        result = get_curriculum_num_negatives(
            10, 30, min_negatives=30,
        )
        assert result is not None

    def test_boundary_epoch_two_thirds(self):
        """Epoch exactly at 2E/3 boundary should be Phase C."""
        result = get_curriculum_num_negatives(
            20, 30, min_negatives=30,
        )
        assert result is None

    def test_short_training_run(self):
        """Schedule should work for very short runs (e.g. 3 epochs)."""
        assert get_curriculum_num_negatives(0, 3, min_negatives=30) == 30
        assert get_curriculum_num_negatives(2, 3, min_negatives=30) is None
