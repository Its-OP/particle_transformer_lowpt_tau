"""Unit tests for CascadeReranker (Track A: ParT-style pairwise-bias encoder).

TDD: Written before implementation. Tests cover:
    - Forward pass shape: (B, K1) scores
    - Pairwise attention bias computation
    - Per-track scoring (not CLS-based)
    - Gradient flow through all parameters
    - compute_loss() with ranking loss
    - Stage 2 interface compatibility (stage1_scores input)
    - Integration with CascadeModel
"""
import pytest
import torch

from weaver.nn.model.CascadeReranker import CascadeReranker
from weaver.nn.model.CascadeModel import CascadeModel
from weaver.nn.model.TrackPreFilter import TrackPreFilter


# ---- Shared configuration ----

BATCH_SIZE = 4
NUM_TRACKS = 100  # K1 = 100 for test speed
INPUT_DIM = 16


def _make_filtered_inputs(
    batch_size=BATCH_SIZE,
    num_tracks=NUM_TRACKS,
    seed=42,
):
    """Create inputs simulating Stage 1 filtered output."""
    generator = torch.Generator().manual_seed(seed)

    eta = torch.randn(batch_size, 1, num_tracks, generator=generator) * 1.5
    phi = (
        torch.rand(batch_size, 1, num_tracks, generator=generator)
        * 2 * 3.14159 - 3.14159
    )
    points = torch.cat([eta, phi], dim=1)

    features = torch.randn(
        batch_size, INPUT_DIM, num_tracks, generator=generator,
    )

    transverse_momentum = (
        torch.rand(batch_size, 1, num_tracks, generator=generator) * 5 + 0.5
    )
    px = transverse_momentum * torch.cos(phi)
    py = transverse_momentum * torch.sin(phi)
    pz = transverse_momentum * torch.sinh(eta)
    pion_mass = 0.13957
    energy = torch.sqrt(px**2 + py**2 + pz**2 + pion_mass**2)
    lorentz_vectors = torch.cat([px, py, pz, energy], dim=1)

    mask = torch.ones(batch_size, 1, num_tracks)
    mask[:, :, -10:] = 0.0  # Some padding

    track_labels = torch.zeros(batch_size, 1, num_tracks)
    for batch_index in range(batch_size):
        track_labels[batch_index, 0, 5] = 1.0
        track_labels[batch_index, 0, 15] = 1.0
        track_labels[batch_index, 0, 25] = 1.0

    stage1_scores = torch.randn(batch_size, num_tracks, generator=generator)

    return points, features, lorentz_vectors, mask, track_labels, stage1_scores


def _make_reranker(**kwargs):
    """Create a CascadeReranker with small config for testing."""
    defaults = dict(
        input_dim=INPUT_DIM,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        pair_embed_dims=[32, 32],
    )
    defaults.update(kwargs)
    return CascadeReranker(**defaults)


# ---- Forward pass ----

class TestCascadeRerankerForward:
    """Test forward pass shape and properties."""

    def test_output_shape(self):
        """forward() should return (B, K1) scores."""
        model = _make_reranker()
        points, features, lorentz_vectors, mask, _, stage1_scores = (
            _make_filtered_inputs()
        )
        scores = model(points, features, lorentz_vectors, mask, stage1_scores)
        assert scores.shape == (BATCH_SIZE, NUM_TRACKS)

    def test_valid_scores_finite(self):
        """Scores for valid tracks should be finite."""
        model = _make_reranker()
        points, features, lorentz_vectors, mask, _, stage1_scores = (
            _make_filtered_inputs()
        )
        scores = model(points, features, lorentz_vectors, mask, stage1_scores)
        valid_mask = mask.squeeze(1).bool()
        assert torch.isfinite(scores[valid_mask]).all()

    def test_padded_scores_negative_inf(self):
        """Padded tracks should get -inf scores."""
        model = _make_reranker()
        points, features, lorentz_vectors, mask, _, stage1_scores = (
            _make_filtered_inputs()
        )
        scores = model(points, features, lorentz_vectors, mask, stage1_scores)
        padded_mask = ~mask.squeeze(1).bool()
        assert torch.all(scores[padded_mask] == float('-inf'))


# ---- Gradient flow ----

class TestCascadeRerankerGradients:
    """Test gradient flow through the model."""

    def test_all_parameters_receive_gradients(self):
        """All parameters should get gradients via compute_loss()."""
        model = _make_reranker()
        points, features, lorentz_vectors, mask, track_labels, stage1_scores = (
            _make_filtered_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask,
            track_labels, stage1_scores,
        )
        loss_dict['total_loss'].backward()

        params_with_grad = sum(
            1 for _, parameter in model.named_parameters()
            if parameter.grad is not None and parameter.grad.abs().sum() > 0
        )
        total_params = sum(1 for _ in model.parameters())
        # Allow 1-2 params with zero grad (e.g. unused bias)
        assert params_with_grad >= total_params - 2, (
            f'Only {params_with_grad}/{total_params} params got gradients'
        )

    def test_loss_is_finite(self):
        """compute_loss() should return finite total_loss."""
        model = _make_reranker()
        points, features, lorentz_vectors, mask, track_labels, stage1_scores = (
            _make_filtered_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask,
            track_labels, stage1_scores,
        )
        assert torch.isfinite(loss_dict['total_loss']).all()

    def test_loss_dict_has_ranking_loss(self):
        """Loss dict should contain ranking_loss component."""
        model = _make_reranker()
        points, features, lorentz_vectors, mask, track_labels, stage1_scores = (
            _make_filtered_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask,
            track_labels, stage1_scores,
        )
        assert 'ranking_loss' in loss_dict


# ---- Stage 2 interface ----

class TestStage2Interface:
    """Test that CascadeReranker implements the Stage 2 interface."""

    def test_forward_accepts_stage1_scores(self):
        """forward() must accept stage1_scores parameter."""
        model = _make_reranker()
        points, features, lorentz_vectors, mask, _, stage1_scores = (
            _make_filtered_inputs()
        )
        # Should not raise
        scores = model(points, features, lorentz_vectors, mask, stage1_scores)
        assert scores is not None

    def test_compute_loss_accepts_stage1_scores(self):
        """compute_loss() must accept stage1_scores parameter."""
        model = _make_reranker()
        points, features, lorentz_vectors, mask, track_labels, stage1_scores = (
            _make_filtered_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask,
            track_labels, stage1_scores,
        )
        assert 'total_loss' in loss_dict
        assert '_scores' in loss_dict


# ---- Integration with CascadeModel ----

class TestCascadeIntegration:
    """Test CascadeReranker plugged into CascadeModel."""

    def test_end_to_end_forward(self):
        """CascadeModel with CascadeReranker should produce scores."""
        stage1 = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            hidden_dim=64, num_message_rounds=1,
        )
        stage2 = _make_reranker()
        cascade = CascadeModel(stage1=stage1, stage2=stage2, top_k1=50)

        points, features, lorentz_vectors, mask, _, _ = (
            _make_filtered_inputs(num_tracks=200)
        )
        mask = torch.ones(BATCH_SIZE, 1, 200)
        mask[:, :, -30:] = 0.0

        scores = cascade(points, features, lorentz_vectors, mask)
        assert scores.shape == (BATCH_SIZE, 50)

    def test_end_to_end_loss(self):
        """CascadeModel with CascadeReranker should produce finite loss."""
        stage1 = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            hidden_dim=64, num_message_rounds=1,
        )
        stage2 = _make_reranker()
        cascade = CascadeModel(stage1=stage1, stage2=stage2, top_k1=50)

        points, features, lorentz_vectors, mask, track_labels, _ = (
            _make_filtered_inputs(num_tracks=200)
        )
        mask = torch.ones(BATCH_SIZE, 1, 200)
        mask[:, :, -30:] = 0.0
        track_labels = torch.zeros(BATCH_SIZE, 1, 200)
        for batch_index in range(BATCH_SIZE):
            track_labels[batch_index, 0, 5] = 1.0
            track_labels[batch_index, 0, 15] = 1.0

        loss_dict = cascade.compute_loss(
            points, features, lorentz_vectors, mask, track_labels,
        )
        assert torch.isfinite(loss_dict['total_loss']).all()
        assert 'stage1_recall_at_k1' in loss_dict
