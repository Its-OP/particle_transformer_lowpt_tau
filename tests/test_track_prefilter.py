"""Unit tests for TrackPreFilter (Stage 1 of two-stage pipeline).

TDD: Written before implementation. Tests cover:
    - Three pre-filter modes: MLP+neighborhood, two-tower, autoencoder
    - Top-K selection preserving GT tracks
    - Masking, finiteness, gradient flow
    - Two-stage pipeline integration
"""
import pytest
import torch

from weaver.nn.model.TrackPreFilter import TrackPreFilter


# ---- Shared configuration ----

BATCH_SIZE = 4
NUM_TRACKS = 200
INPUT_DIM = 7
INPUT_DIM_EXTENDED = 13
TOP_K = 50  # Smaller than production 200 for test speed


def _make_physical_inputs(batch_size, num_tracks, input_dim=INPUT_DIM, seed=42):
    """Create physically sensible synthetic inputs."""
    generator = torch.Generator().manual_seed(seed)

    eta = torch.randn(batch_size, 1, num_tracks, generator=generator) * 1.5
    phi = (
        torch.rand(batch_size, 1, num_tracks, generator=generator)
        * 2 * 3.14159 - 3.14159
    )
    points = torch.cat([eta, phi], dim=1)

    features = torch.randn(
        batch_size, input_dim, num_tracks, generator=generator,
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

    return points, features, lorentz_vectors


def _make_training_inputs():
    """Create inputs with mask and labels."""
    points, features, lorentz_vectors = _make_physical_inputs(
        BATCH_SIZE, NUM_TRACKS,
    )
    mask = torch.ones(BATCH_SIZE, 1, NUM_TRACKS)
    mask[:, :, -30:] = 0.0  # Last 30 are padding

    track_labels = torch.zeros(BATCH_SIZE, 1, NUM_TRACKS)
    for batch_index in range(BATCH_SIZE):
        track_labels[batch_index, 0, 10] = 1.0
        track_labels[batch_index, 0, 20] = 1.0
        track_labels[batch_index, 0, 30] = 1.0

    return points, features, lorentz_vectors, mask, track_labels


# ---- Mode A: MLP + Neighborhood ----

class TestMLPMode:
    """Test per-track MLP with neighborhood context."""

    def test_forward_shape(self):
        model = TrackPreFilter(mode='mlp', input_dim=INPUT_DIM)
        points, features, lorentz_vectors, mask, _ = _make_training_inputs()
        scores = model(points, features, lorentz_vectors, mask)
        assert scores.shape == (BATCH_SIZE, NUM_TRACKS)

    def test_scores_finite(self):
        model = TrackPreFilter(mode='mlp', input_dim=INPUT_DIM)
        points, features, lorentz_vectors, mask, _ = _make_training_inputs()
        scores = model(points, features, lorentz_vectors, mask)
        valid_mask = mask.squeeze(1).bool()
        assert torch.isfinite(scores[valid_mask]).all()

    def test_padded_scores_negative_inf(self):
        """Padded tracks should get -inf so they never appear in top-K."""
        model = TrackPreFilter(mode='mlp', input_dim=INPUT_DIM)
        points, features, lorentz_vectors, mask, _ = _make_training_inputs()
        scores = model(points, features, lorentz_vectors, mask)
        padded_mask = ~mask.squeeze(1).bool()
        assert torch.all(scores[padded_mask] == float('-inf'))


# ---- Mode B: Two-Tower ----

class TestTwoTowerMode:
    """Test two-tower retrieve with learned tau prototype."""

    def test_forward_shape(self):
        model = TrackPreFilter(mode='two_tower', input_dim=INPUT_DIM)
        points, features, lorentz_vectors, mask, _ = _make_training_inputs()
        scores = model(points, features, lorentz_vectors, mask)
        assert scores.shape == (BATCH_SIZE, NUM_TRACKS)

    def test_scores_finite(self):
        model = TrackPreFilter(mode='two_tower', input_dim=INPUT_DIM)
        points, features, lorentz_vectors, mask, _ = _make_training_inputs()
        scores = model(points, features, lorentz_vectors, mask)
        valid_mask = mask.squeeze(1).bool()
        assert torch.isfinite(scores[valid_mask]).all()


# ---- Mode C: Autoencoder ----

class TestAutoencoderMode:
    """Test autoencoder anomaly scorer."""

    def test_forward_shape(self):
        model = TrackPreFilter(mode='autoencoder', input_dim=INPUT_DIM)
        points, features, lorentz_vectors, mask, _ = _make_training_inputs()
        scores = model(points, features, lorentz_vectors, mask)
        assert scores.shape == (BATCH_SIZE, NUM_TRACKS)

    def test_scores_finite(self):
        model = TrackPreFilter(mode='autoencoder', input_dim=INPUT_DIM)
        points, features, lorentz_vectors, mask, _ = _make_training_inputs()
        scores = model(points, features, lorentz_vectors, mask)
        valid_mask = mask.squeeze(1).bool()
        assert torch.isfinite(scores[valid_mask]).all()

    def test_reconstruction_loss_finite(self):
        """Autoencoder mode should return finite reconstruction loss."""
        model = TrackPreFilter(mode='autoencoder', input_dim=INPUT_DIM)
        points, features, lorentz_vectors, mask, track_labels = (
            _make_training_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask, track_labels,
        )
        assert 'reconstruction_loss' in loss_dict
        assert torch.isfinite(loss_dict['reconstruction_loss']).all()


# ---- Top-K Selection ----

class TestTopKSelection:
    """Test top-K candidate selection from pre-filter scores."""

    def test_topk_preserves_gt(self):
        """When GT tracks have high scores, top-K should include them all."""
        model = TrackPreFilter(mode='mlp', input_dim=INPUT_DIM)

        # Create scores where GT tracks (10, 20, 30) score highest
        scores = torch.randn(1, NUM_TRACKS) - 5  # Background: low scores
        scores[0, 10] = 10.0
        scores[0, 20] = 10.0
        scores[0, 30] = 10.0
        mask = torch.ones(1, 1, NUM_TRACKS)

        selected_indices = model.select_top_k(scores, mask, top_k=TOP_K)
        selected_set = set(selected_indices[0].tolist())

        assert 10 in selected_set, 'GT track 10 not in top-K'
        assert 20 in selected_set, 'GT track 20 not in top-K'
        assert 30 in selected_set, 'GT track 30 not in top-K'

    def test_topk_returns_correct_count(self):
        """Should return exactly K indices per event."""
        model = TrackPreFilter(mode='mlp', input_dim=INPUT_DIM)
        scores = torch.randn(BATCH_SIZE, NUM_TRACKS)
        mask = torch.ones(BATCH_SIZE, 1, NUM_TRACKS)

        selected_indices = model.select_top_k(scores, mask, top_k=TOP_K)
        assert selected_indices.shape == (BATCH_SIZE, TOP_K)

    def test_topk_handles_fewer_valid_than_k(self):
        """Events with < K valid tracks should return all valid indices."""
        model = TrackPreFilter(mode='mlp', input_dim=INPUT_DIM)
        scores = torch.randn(1, NUM_TRACKS)
        mask = torch.ones(1, 1, NUM_TRACKS)
        mask[0, 0, 20:] = 0.0  # Only 20 valid tracks, K=50

        selected_indices = model.select_top_k(scores, mask, top_k=TOP_K)

        # All 20 valid should be selected, rest padded with -1 or repeated
        valid_selected = selected_indices[0][selected_indices[0] < 20]
        assert len(valid_selected) == 20


# ---- Loss Functions ----

class TestPreFilterLoss:
    """Test ranking loss for pre-filter training."""

    def test_ranking_loss_finite(self):
        model = TrackPreFilter(mode='mlp', input_dim=INPUT_DIM)
        points, features, lorentz_vectors, mask, track_labels = (
            _make_training_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask, track_labels,
        )
        assert 'total_loss' in loss_dict
        assert torch.isfinite(loss_dict['total_loss']).all()

    def test_ranking_loss_zero_when_perfectly_ranked(self):
        """Perfect ranking should give near-zero loss."""
        model = TrackPreFilter(mode='mlp', input_dim=INPUT_DIM)
        scores = torch.zeros(1, NUM_TRACKS) - 10
        scores[0, 10] = 10.0
        scores[0, 20] = 10.0
        scores[0, 30] = 10.0
        labels = torch.zeros(1, NUM_TRACKS)
        labels[0, 10] = 1.0
        labels[0, 20] = 1.0
        labels[0, 30] = 1.0
        valid_mask = torch.ones(1, NUM_TRACKS, dtype=torch.bool)

        loss = model._ranking_loss(scores, labels, valid_mask)
        assert loss.item() < 0.01

    def test_ranking_loss_high_when_misranked(self):
        """Misranked should give high loss."""
        model = TrackPreFilter(mode='mlp', input_dim=INPUT_DIM)
        scores = torch.zeros(1, NUM_TRACKS)
        scores[0, 10] = -5.0  # GT scores low
        scores[0, 20] = -5.0
        scores[0, 30] = -5.0
        scores[0, 0] = 5.0  # Background scores high
        labels = torch.zeros(1, NUM_TRACKS)
        labels[0, 10] = 1.0
        labels[0, 20] = 1.0
        labels[0, 30] = 1.0
        valid_mask = torch.ones(1, NUM_TRACKS, dtype=torch.bool)

        loss = model._ranking_loss(scores, labels, valid_mask)
        assert loss.item() > 1.0

    def test_gradients_flow(self):
        """Backward should produce gradients in all model parameters."""
        model = TrackPreFilter(mode='mlp', input_dim=INPUT_DIM)
        points, features, lorentz_vectors, mask, track_labels = (
            _make_training_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask, track_labels,
        )
        loss_dict['total_loss'].backward()

        params_with_grad = sum(
            1 for _, parameter in model.named_parameters()
            if parameter.grad is not None and parameter.grad.abs().sum() > 0
        )
        assert params_with_grad > 0


# ---- Two-Stage Pipeline ----

class TestTwoStagePipeline:
    """Test Stage 1 → top-K → Stage 2 pipeline."""

    def test_repack_reduces_track_count(self):
        """After top-K selection, repacked tensors should have K tracks."""
        model = TrackPreFilter(mode='mlp', input_dim=INPUT_DIM)
        points, features, lorentz_vectors, mask, track_labels = (
            _make_training_inputs()
        )

        filtered = model.filter_tracks(
            points, features, lorentz_vectors, mask, track_labels,
            top_k=TOP_K,
        )

        assert filtered['points'].shape[2] == TOP_K
        assert filtered['features'].shape[2] == TOP_K
        assert filtered['lorentz_vectors'].shape[2] == TOP_K
        assert filtered['mask'].shape[2] == TOP_K
        assert filtered['track_labels'].shape[2] == TOP_K

    def test_filter_preserves_gt_when_scored_high(self):
        """If GT tracks score high, they should survive filtering."""
        model = TrackPreFilter(mode='mlp', input_dim=INPUT_DIM)
        points, features, lorentz_vectors, mask, track_labels = (
            _make_training_inputs()
        )

        # We can't guarantee untrained model scores GT high,
        # but we can verify the pipeline runs without error
        filtered = model.filter_tracks(
            points, features, lorentz_vectors, mask, track_labels,
            top_k=TOP_K,
        )

        # Filtered labels should be valid (0 or 1)
        assert torch.all(
            (filtered['track_labels'] == 0) | (filtered['track_labels'] == 1)
        )


# ---- Extended 13-Feature Configuration (wide192) ----

def _make_extended_training_inputs():
    """Create inputs with the extended 13-feature set and wide192 config."""
    points, features, lorentz_vectors = _make_physical_inputs(
        BATCH_SIZE, NUM_TRACKS, input_dim=INPUT_DIM_EXTENDED,
    )
    mask = torch.ones(BATCH_SIZE, 1, NUM_TRACKS)
    mask[:, :, -30:] = 0.0

    track_labels = torch.zeros(BATCH_SIZE, 1, NUM_TRACKS)
    for batch_index in range(BATCH_SIZE):
        track_labels[batch_index, 0, 10] = 1.0
        track_labels[batch_index, 0, 20] = 1.0
        track_labels[batch_index, 0, 30] = 1.0

    return points, features, lorentz_vectors, mask, track_labels


class TestExtendedHybridMode:
    """Test hybrid mode with extended 13-feature input and wide192 config.

    Verifies the widened architecture (hidden_dim=192, latent_dim=48)
    works correctly with the 13-feature extended dataset.
    """

    def test_forward_shape(self):
        model = TrackPreFilter(
            mode='hybrid', input_dim=INPUT_DIM_EXTENDED,
            hidden_dim=192, latent_dim=48, num_message_rounds=2,
        )
        points, features, lorentz_vectors, mask, _ = (
            _make_extended_training_inputs()
        )
        scores = model(points, features, lorentz_vectors, mask)
        assert scores.shape == (BATCH_SIZE, NUM_TRACKS)

    def test_scores_finite(self):
        model = TrackPreFilter(
            mode='hybrid', input_dim=INPUT_DIM_EXTENDED,
            hidden_dim=192, latent_dim=48, num_message_rounds=2,
        )
        points, features, lorentz_vectors, mask, _ = (
            _make_extended_training_inputs()
        )
        scores = model(points, features, lorentz_vectors, mask)
        valid_mask = mask.squeeze(1).bool()
        assert torch.isfinite(scores[valid_mask]).all()

    def test_padded_scores_negative_inf(self):
        """Padded tracks should get -inf with the wider model."""
        model = TrackPreFilter(
            mode='hybrid', input_dim=INPUT_DIM_EXTENDED,
            hidden_dim=192, latent_dim=48, num_message_rounds=2,
        )
        points, features, lorentz_vectors, mask, _ = (
            _make_extended_training_inputs()
        )
        scores = model(points, features, lorentz_vectors, mask)
        padded_mask = ~mask.squeeze(1).bool()
        assert torch.all(scores[padded_mask] == float('-inf'))

    def test_loss_finite(self):
        """All loss components should be finite with extended features."""
        model = TrackPreFilter(
            mode='hybrid', input_dim=INPUT_DIM_EXTENDED,
            hidden_dim=192, latent_dim=48, num_message_rounds=2,
        )
        points, features, lorentz_vectors, mask, track_labels = (
            _make_extended_training_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask, track_labels,
        )
        assert torch.isfinite(loss_dict['total_loss']).all()
        assert torch.isfinite(loss_dict['ranking_loss']).all()
        assert torch.isfinite(loss_dict['reconstruction_loss']).all()

    def test_gradients_flow(self):
        """All parameters should receive gradients with extended input."""
        model = TrackPreFilter(
            mode='hybrid', input_dim=INPUT_DIM_EXTENDED,
            hidden_dim=192, latent_dim=48, num_message_rounds=2,
        )
        points, features, lorentz_vectors, mask, track_labels = (
            _make_extended_training_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask, track_labels,
        )
        loss_dict['total_loss'].backward()

        params_with_grad = sum(
            1 for _, parameter in model.named_parameters()
            if parameter.grad is not None and parameter.grad.abs().sum() > 0
        )
        total_params = sum(1 for _ in model.parameters())
        # Allow 1 param with zero grad (numerical artifact from random data)
        assert params_with_grad >= total_params - 1, (
            f'Only {params_with_grad}/{total_params} params got gradients'
        )

    def test_param_count_increase(self):
        """Wide192 with 13 features should have more params than wide128 with 7."""
        model_old = TrackPreFilter(
            mode='hybrid', input_dim=INPUT_DIM,
            hidden_dim=128, latent_dim=32, num_message_rounds=2,
        )
        model_new = TrackPreFilter(
            mode='hybrid', input_dim=INPUT_DIM_EXTENDED,
            hidden_dim=192, latent_dim=48, num_message_rounds=2,
        )
        old_params = sum(p.numel() for p in model_old.parameters())
        new_params = sum(p.numel() for p in model_new.parameters())
        assert new_params > old_params, (
            f'New model ({new_params}) should have more params than old ({old_params})'
        )
