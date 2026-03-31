"""Tests for diagnostics/graph_space_diagnostic.py — GT-neighbor connectivity.

Verifies the diagnostic computes signal-signal connectivity metrics
across different kNN feature spaces and k values.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

BATCH_SIZE = 4
NUM_TRACKS = 200
INPUT_DIM = 16


def _make_inputs(seed=42):
    """Create inputs matching the data pipeline shape conventions."""
    generator = torch.Generator().manual_seed(seed)

    eta = torch.randn(BATCH_SIZE, 1, NUM_TRACKS, generator=generator) * 1.5
    phi = (
        torch.rand(BATCH_SIZE, 1, NUM_TRACKS, generator=generator)
        * 2 * 3.14159 - 3.14159
    )
    points = torch.cat([eta, phi], dim=1)  # (B, 2, P)

    features = torch.randn(
        BATCH_SIZE, INPUT_DIM, NUM_TRACKS, generator=generator,
    )
    # Feature 5 = charge: set to +1 or -1
    features[:, 5, :] = (
        torch.randint(0, 2, (BATCH_SIZE, NUM_TRACKS), generator=generator)
        * 2.0 - 1.0
    )

    mask = torch.ones(BATCH_SIZE, 1, NUM_TRACKS)
    mask[:, :, -30:] = 0.0

    track_labels = torch.zeros(BATCH_SIZE, 1, NUM_TRACKS)
    for batch_index in range(BATCH_SIZE):
        track_labels[batch_index, 0, 10] = 1.0
        track_labels[batch_index, 0, 20] = 1.0
        track_labels[batch_index, 0, 30] = 1.0

    return points, features, mask, track_labels


class TestKnnL2:
    """Test the general-purpose L2 kNN utility."""

    def test_output_shape(self):
        from diagnostics.graph_space_diagnostic import knn_l2

        coords = torch.randn(BATCH_SIZE, 2, NUM_TRACKS)
        mask = torch.ones(BATCH_SIZE, 1, NUM_TRACKS)
        indices = knn_l2(coords, num_neighbors=8, mask=mask)
        assert indices.shape == (BATCH_SIZE, NUM_TRACKS, 8)

    def test_excludes_self(self):
        from diagnostics.graph_space_diagnostic import knn_l2

        coords = torch.randn(BATCH_SIZE, 2, NUM_TRACKS)
        mask = torch.ones(BATCH_SIZE, 1, NUM_TRACKS)
        indices = knn_l2(coords, num_neighbors=8, mask=mask)
        # No track should be its own neighbor
        track_range = torch.arange(NUM_TRACKS).unsqueeze(0).unsqueeze(-1)
        assert not (indices == track_range).any()

    def test_respects_mask(self):
        from diagnostics.graph_space_diagnostic import knn_l2

        coords = torch.randn(BATCH_SIZE, 2, NUM_TRACKS)
        mask = torch.ones(BATCH_SIZE, 1, NUM_TRACKS)
        mask[:, :, -50:] = 0.0  # Last 50 are invalid
        indices = knn_l2(coords, num_neighbors=8, mask=mask)
        # No neighbor should point to an invalid track
        assert (indices < NUM_TRACKS - 50).all()


class TestGtNeighborMetrics:
    """Test GT-neighbor connectivity computation."""

    def test_returns_expected_keys(self):
        from diagnostics.graph_space_diagnostic import compute_gt_neighbor_metrics

        points, features, mask, labels = _make_inputs()
        neighbor_indices = torch.randint(
            0, NUM_TRACKS - 30, (BATCH_SIZE, NUM_TRACKS, 16),
        )
        metrics = compute_gt_neighbor_metrics(
            neighbor_indices, labels, mask,
        )
        assert 'gt_with_at_least_1_gt_neighbor' in metrics
        assert 'gt_with_at_least_2_gt_neighbors' in metrics
        assert 'mean_gt_neighbors_per_gt' in metrics

    def test_values_in_valid_range(self):
        from diagnostics.graph_space_diagnostic import compute_gt_neighbor_metrics

        points, features, mask, labels = _make_inputs()
        neighbor_indices = torch.randint(
            0, NUM_TRACKS - 30, (BATCH_SIZE, NUM_TRACKS, 16),
        )
        metrics = compute_gt_neighbor_metrics(
            neighbor_indices, labels, mask,
        )
        assert 0.0 <= metrics['gt_with_at_least_1_gt_neighbor'] <= 1.0
        assert 0.0 <= metrics['gt_with_at_least_2_gt_neighbors'] <= 1.0
        assert metrics['mean_gt_neighbors_per_gt'] >= 0.0

    def test_perfect_connectivity(self):
        """If all GT tracks are each other's neighbors, fraction should be 1.0."""
        from diagnostics.graph_space_diagnostic import compute_gt_neighbor_metrics

        _, _, mask, labels = _make_inputs()
        # Make neighbors such that GT tracks (10, 20, 30) always include each other
        neighbor_indices = torch.zeros(
            BATCH_SIZE, NUM_TRACKS, 4, dtype=torch.long,
        )
        # Every track's neighbors include positions 10, 20, 30, 0
        neighbor_indices[:, :, 0] = 10
        neighbor_indices[:, :, 1] = 20
        neighbor_indices[:, :, 2] = 30
        neighbor_indices[:, :, 3] = 0

        metrics = compute_gt_neighbor_metrics(
            neighbor_indices, labels, mask,
        )
        assert metrics['gt_with_at_least_1_gt_neighbor'] == 1.0
        assert metrics['gt_with_at_least_2_gt_neighbors'] == 1.0


class TestSpaceBuilders:
    """Test that each kNN space builder produces valid indices."""

    def test_eta_phi_space(self):
        from diagnostics.graph_space_diagnostic import build_knn_eta_phi

        points, features, mask, _ = _make_inputs()
        indices = build_knn_eta_phi(points, mask, num_neighbors=8)
        assert indices.shape == (BATCH_SIZE, NUM_TRACKS, 8)

    def test_dz_sig_space(self):
        from diagnostics.graph_space_diagnostic import build_knn_dz_sig

        points, features, mask, _ = _make_inputs()
        indices = build_knn_dz_sig(features, mask, num_neighbors=8)
        assert indices.shape == (BATCH_SIZE, NUM_TRACKS, 8)

    def test_vertex_proxy_space(self):
        from diagnostics.graph_space_diagnostic import build_knn_vertex_proxy

        points, features, mask, _ = _make_inputs()
        indices = build_knn_vertex_proxy(points, features, mask, num_neighbors=8)
        assert indices.shape == (BATCH_SIZE, NUM_TRACKS, 8)

    def test_opposite_sign_space(self):
        from diagnostics.graph_space_diagnostic import build_knn_opposite_sign

        points, features, mask, _ = _make_inputs()
        indices = build_knn_opposite_sign(
            points, features, mask, num_neighbors=8,
        )
        assert indices.shape == (BATCH_SIZE, NUM_TRACKS, 8)

    def test_logpt_eta_phi_space(self):
        from diagnostics.graph_space_diagnostic import build_knn_logpt_eta_phi

        points, features, mask, _ = _make_inputs()
        indices = build_knn_logpt_eta_phi(
            points, features, mask, num_neighbors=8,
        )
        assert indices.shape == (BATCH_SIZE, NUM_TRACKS, 8)

    def test_composite_union(self):
        from diagnostics.graph_space_diagnostic import build_composite_graph

        points, features, mask, _ = _make_inputs()
        indices = build_composite_graph(points, features, mask)
        # Should have more neighbors than any single space
        assert indices.shape[2] > 16
