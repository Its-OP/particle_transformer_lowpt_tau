"""Unit tests for TripletScorer (anchor-based triplet enumeration and scoring).

TDD: Written before implementation.

Tests cover:
    - Triplet enumeration with physics cuts (mass, charge, ΔR)
    - Triplet scoring MLP
    - Boost propagation from triplet scores to per-track scores
    - Integration with TrackPreFilter
"""
import pytest
import torch
import math

from weaver.nn.model.TripletScorer import TripletScorer


BATCH_SIZE = 2
NUM_TRACKS = 100
PION_MASS = 0.13957


def _make_physical_lorentz_vectors(batch_size, num_tracks, seed=42):
    """Create physical Lorentz 4-vectors (px, py, pz, E)."""
    generator = torch.Generator().manual_seed(seed)
    eta = torch.randn(batch_size, num_tracks, generator=generator) * 1.5
    phi = torch.rand(batch_size, num_tracks, generator=generator) * 2 * math.pi - math.pi
    pt = torch.rand(batch_size, num_tracks, generator=generator) * 5 + 0.5

    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    energy = torch.sqrt(px**2 + py**2 + pz**2 + PION_MASS**2)

    # (B, 4, P) format
    lorentz_vectors = torch.stack([px, py, pz, energy], dim=1)
    points = torch.stack([eta, phi], dim=1)

    # Features: 7 dims, charge at index 5
    features = torch.randn(batch_size, 7, num_tracks, generator=generator)
    features[:, 5, :] = torch.sign(torch.randn(batch_size, num_tracks, generator=generator))  # ±1 charge

    mask = torch.ones(batch_size, 1, num_tracks)
    mask[:, :, -10:] = 0.0  # last 10 padded

    return points, features, lorentz_vectors, mask


# ---- Triplet Enumeration Tests ----

class TestTripletEnumeration:
    """Test physics-based triplet construction."""

    def test_pairwise_mass_computation(self):
        """Invariant mass of two pions with known momenta."""
        scorer = TripletScorer()

        # Two pions at rest: m = 2 * m_pi ≈ 0.279 GeV
        lv = torch.zeros(1, 4, 2)
        lv[0, 0, 0] = 0.0   # px1
        lv[0, 1, 0] = 0.0   # py1
        lv[0, 2, 0] = 0.0   # pz1
        lv[0, 3, 0] = PION_MASS  # E1
        lv[0, 0, 1] = 0.0   # px2
        lv[0, 1, 1] = 0.0   # py2
        lv[0, 2, 1] = 0.0   # pz2
        lv[0, 3, 1] = PION_MASS  # E2

        mass = scorer.compute_invariant_mass(lv, [0], [1])
        expected = 2 * PION_MASS
        assert abs(mass.item() - expected) < 0.001

    def test_triplet_mass_computation(self):
        """Invariant mass of three pions at rest."""
        scorer = TripletScorer()

        lv = torch.zeros(1, 4, 3)
        for i in range(3):
            lv[0, 3, i] = PION_MASS  # All at rest

        mass = scorer.compute_triplet_mass(lv, [0], [1], [2])
        expected = 3 * PION_MASS  # ~0.419 GeV
        assert abs(mass.item() - expected) < 0.001

    def test_physics_cuts_reduce_candidates(self):
        """Mass + charge + ΔR cuts should reduce triplet count."""
        scorer = TripletScorer(num_neighbors=50)
        points, features, lorentz_vectors, mask = _make_physical_lorentz_vectors(
            1, NUM_TRACKS,
        )
        scores = torch.randn(1, NUM_TRACKS)
        scores[:, -10:] = float('-inf')

        triplets, triplet_features = scorer.enumerate_triplets(
            scores, points, features, lorentz_vectors, mask,
        )

        # Should have fewer triplets than C(50, 2) = 1225 per anchor
        # (physics cuts reject most)
        if triplets is not None:
            assert triplets.shape[1] == 3  # Each triplet has 3 indices
            assert triplets.shape[0] < 200 * 1225  # Much fewer than unconstrained

    def test_charge_sum_constraint(self):
        """All surviving triplets should have charge sum = ±1."""
        scorer = TripletScorer(num_neighbors=50)
        points, features, lorentz_vectors, mask = _make_physical_lorentz_vectors(
            1, NUM_TRACKS,
        )
        scores = torch.randn(1, NUM_TRACKS)
        scores[:, -10:] = float('-inf')

        triplets, triplet_features = scorer.enumerate_triplets(
            scores, points, features, lorentz_vectors, mask,
        )

        if triplets is not None and len(triplets) > 0:
            # Check charge sum feature (should be ±1)
            charge_sums = triplet_features[:, scorer.CHARGE_SUM_IDX]
            assert torch.all(
                (charge_sums.abs() - 1.0).abs() < 0.1
            ), 'All triplets should have |charge_sum| = 1'


# ---- Triplet Scoring Tests ----

class TestTripletScoring:
    """Test the triplet scoring MLP."""

    def test_scorer_output_shape(self):
        """Scorer should produce one score per triplet."""
        scorer = TripletScorer()
        # Fake triplet features: (num_triplets, num_features)
        triplet_features = torch.randn(50, scorer.num_triplet_features)

        triplet_scores = scorer.score_triplets(triplet_features)
        assert triplet_scores.shape == (50,)

    def test_scorer_finite(self):
        """Scores should be finite."""
        scorer = TripletScorer()
        triplet_features = torch.randn(50, scorer.num_triplet_features)

        triplet_scores = scorer.score_triplets(triplet_features)
        assert torch.isfinite(triplet_scores).all()

    def test_scorer_gradients_flow(self):
        """Backward through scorer should produce gradients."""
        scorer = TripletScorer()
        triplet_features = torch.randn(50, scorer.num_triplet_features)

        triplet_scores = scorer.score_triplets(triplet_features)
        triplet_scores.sum().backward()

        params_with_grad = sum(
            1 for _, p in scorer.triplet_mlp.named_parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert params_with_grad > 0


# ---- Boost Propagation Tests ----

class TestBoostPropagation:
    """Test score boosting from triplet scores to per-track scores."""

    def test_boost_changes_scores(self):
        """Tracks in high-scoring triplets should get boosted."""
        scorer = TripletScorer()
        original_scores = torch.zeros(1, NUM_TRACKS)

        # Fake: triplet (5, 10, 15) scored high
        triplet_indices = torch.tensor([[5, 10, 15]])
        triplet_scores = torch.tensor([10.0])

        boosted = scorer.propagate_boost(
            original_scores, triplet_indices, triplet_scores,
            boost_weight=1.0,
        )

        assert boosted[0, 5] > original_scores[0, 5]
        assert boosted[0, 10] > original_scores[0, 10]
        assert boosted[0, 15] > original_scores[0, 15]
        # Other tracks unchanged
        assert boosted[0, 0] == original_scores[0, 0]

    def test_boost_preserves_padded(self):
        """Padded tracks should stay at -inf after boosting."""
        scorer = TripletScorer()
        original_scores = torch.zeros(1, NUM_TRACKS)
        original_scores[0, -10:] = float('-inf')

        triplet_indices = torch.tensor([[5, 10, 15]])
        triplet_scores = torch.tensor([10.0])

        boosted = scorer.propagate_boost(
            original_scores, triplet_indices, triplet_scores,
            boost_weight=1.0,
        )

        assert torch.all(boosted[0, -10:] == float('-inf'))

    def test_no_boost_when_no_triplets(self):
        """Without valid triplets, scores should be unchanged."""
        scorer = TripletScorer()
        original_scores = torch.randn(1, NUM_TRACKS)

        boosted = scorer.propagate_boost(
            original_scores, None, None, boost_weight=1.0,
        )

        torch.testing.assert_close(boosted, original_scores)
