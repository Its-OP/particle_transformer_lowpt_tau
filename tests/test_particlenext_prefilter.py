"""Unit tests for ParticleNeXtPreFilter (ParticleNeXt backbone for pre-filtering).

Tests cover:
    - Forward pass: shape, masking, finiteness
    - Loss computation: keys, finiteness, gradient flow
    - Scheduling: temperature, sigma, DRW
    - Top-K selection and track filtering
    - Backward compatibility: TrackPreFilter still works independently
"""
import pytest
import torch

from weaver.nn.model.ParticleNeXtPreFilter import ParticleNeXtPreFilter
from weaver.nn.model.TrackPreFilter import TrackPreFilter


# ---- Shared configuration ----

BATCH_SIZE = 2
NUM_TRACKS = 100
INPUT_DIM = 13
TOP_K = 30

# Smaller config for fast tests (fewer params, smaller k)
SMALL_CONFIG = dict(
    feature_input_dim=INPUT_DIM,
    node_dim=16,
    edge_dim=4,
    layer_params=[
        (16, 64, [(4, 1), (2, 1), (1, 1)], 32),
        (16, 64, [(4, 1), (2, 1), (1, 1)], 32),
    ],
    fc_params=[(64, 0.0)],
    edge_aggregation='attn4',
    use_rel_lv_fts=True,
    ranking_num_samples=10,
)


def _make_physical_inputs(batch_size=BATCH_SIZE, num_tracks=NUM_TRACKS, seed=42):
    """Create physically sensible synthetic inputs with Lorentz vectors."""
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

    # Physical Lorentz vectors: (px, py, pz, E)
    # E = sqrt(px^2 + py^2 + pz^2 + m_pi^2)
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
    mask[:, :, -20:] = 0.0  # Last 20 are padding

    track_labels = torch.zeros(batch_size, 1, num_tracks)
    for batch_index in range(batch_size):
        track_labels[batch_index, 0, 5] = 1.0
        track_labels[batch_index, 0, 15] = 1.0
        track_labels[batch_index, 0, 25] = 1.0

    return points, features, lorentz_vectors, mask, track_labels


# ---- Forward pass ----

class TestForwardPass:
    """Test ParticleNeXtPreFilter forward pass."""

    def test_output_shape(self):
        """Output should be (B, P) per-track scores."""
        model = ParticleNeXtPreFilter(**SMALL_CONFIG)
        points, features, lorentz_vectors, mask, _ = _make_physical_inputs()
        model.eval()
        with torch.no_grad():
            scores = model(points, features, lorentz_vectors, mask)
        assert scores.shape == (BATCH_SIZE, NUM_TRACKS)

    def test_masked_tracks_are_negative_infinity(self):
        """Padded tracks should have -inf scores."""
        model = ParticleNeXtPreFilter(**SMALL_CONFIG)
        points, features, lorentz_vectors, mask, _ = _make_physical_inputs()
        model.eval()
        with torch.no_grad():
            scores = model(points, features, lorentz_vectors, mask)
        # Last 20 tracks are masked
        assert (scores[:, -20:] == float('-inf')).all()

    def test_valid_scores_finite(self):
        """Valid tracks should have finite scores."""
        model = ParticleNeXtPreFilter(**SMALL_CONFIG)
        points, features, lorentz_vectors, mask, _ = _make_physical_inputs()
        model.eval()
        with torch.no_grad():
            scores = model(points, features, lorentz_vectors, mask)
        valid_scores = scores[:, :-20]
        assert valid_scores.isfinite().all()

    def test_different_inputs_different_scores(self):
        """Model should produce different outputs for different inputs."""
        model = ParticleNeXtPreFilter(**SMALL_CONFIG)
        model.eval()
        inputs_a = _make_physical_inputs(seed=42)
        inputs_b = _make_physical_inputs(seed=99)
        with torch.no_grad():
            scores_a = model(*inputs_a[:4])
            scores_b = model(*inputs_b[:4])
        assert not torch.allclose(scores_a, scores_b)


# ---- Loss computation ----

class TestComputeLoss:
    """Test compute_loss method."""

    def test_loss_dict_keys(self):
        """Loss dict should contain expected keys."""
        model = ParticleNeXtPreFilter(**SMALL_CONFIG)
        model.train()
        inputs = _make_physical_inputs()
        loss_dict = model.compute_loss(*inputs)
        assert 'total_loss' in loss_dict
        assert 'ranking_loss' in loss_dict
        assert 'denoising_loss' in loss_dict
        assert '_scores' in loss_dict

    def test_losses_finite(self):
        """All losses should be finite."""
        model = ParticleNeXtPreFilter(**SMALL_CONFIG)
        model.train()
        inputs = _make_physical_inputs()
        loss_dict = model.compute_loss(*inputs)
        assert loss_dict['total_loss'].isfinite()
        assert loss_dict['ranking_loss'].isfinite()
        assert loss_dict['denoising_loss'].isfinite()

    def test_scores_shape_in_loss_dict(self):
        """Cached scores should have shape (B, P)."""
        model = ParticleNeXtPreFilter(**SMALL_CONFIG)
        model.train()
        inputs = _make_physical_inputs()
        loss_dict = model.compute_loss(*inputs)
        assert loss_dict['_scores'].shape == (BATCH_SIZE, NUM_TRACKS)

    def test_no_reconstruction_loss(self):
        """ParticleNeXt pre-filter should not have reconstruction loss."""
        model = ParticleNeXtPreFilter(**SMALL_CONFIG)
        model.train()
        inputs = _make_physical_inputs()
        loss_dict = model.compute_loss(*inputs)
        assert 'reconstruction_loss' not in loss_dict

    def test_denoising_disabled_when_not_training(self):
        """Denoising loss should not appear in eval mode."""
        model = ParticleNeXtPreFilter(**SMALL_CONFIG)
        model.eval()
        inputs = _make_physical_inputs()
        with torch.no_grad():
            loss_dict = model.compute_loss(*inputs)
        assert 'denoising_loss' not in loss_dict


# ---- Gradient flow ----

class TestGradientFlow:
    """Test that gradients flow through the backbone."""

    def test_backbone_receives_gradients(self):
        """All backbone parameters should receive gradients."""
        model = ParticleNeXtPreFilter(**SMALL_CONFIG)
        model.train()
        inputs = _make_physical_inputs()
        loss_dict = model.compute_loss(*inputs)
        loss_dict['total_loss'].backward()

        total_params = sum(1 for p in model.parameters())
        params_with_gradient = sum(
            1 for p in model.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        # Allow a few params to have zero grad (BatchNorm running stats
        # on synthetic data, gamma init_scale params, etc.)
        assert params_with_gradient >= total_params - 5, (
            f'Only {params_with_gradient}/{total_params} params have gradients'
        )


# ---- Scheduling ----

class TestScheduling:
    """Test temperature, sigma, and DRW scheduling."""

    def test_temperature_interpolation(self):
        """Temperature should interpolate linearly between start and end."""
        model = ParticleNeXtPreFilter(
            **SMALL_CONFIG,
            ranking_temperature_start=2.0,
            ranking_temperature_end=0.5,
        )
        model.set_temperature_progress(0.0)
        assert model.current_ranking_temperature == pytest.approx(2.0)

        model.set_temperature_progress(0.5)
        assert model.current_ranking_temperature == pytest.approx(1.25)

        model.set_temperature_progress(1.0)
        assert model.current_ranking_temperature == pytest.approx(0.5)

    def test_sigma_interpolation(self):
        """Denoising sigma should interpolate linearly."""
        model = ParticleNeXtPreFilter(
            **SMALL_CONFIG,
            denoising_sigma_start=1.0,
            denoising_sigma_end=0.1,
        )
        model.set_temperature_progress(0.0)
        assert model.current_denoising_sigma == pytest.approx(1.0)

        model.set_temperature_progress(1.0)
        assert model.current_denoising_sigma == pytest.approx(0.1)

    def test_drw_activation(self):
        """DRW should be toggleable."""
        model = ParticleNeXtPreFilter(**SMALL_CONFIG, drw_positive_weight=2.0)
        assert not model._drw_active

        model.set_drw_active(True)
        assert model._drw_active

        model.set_drw_active(False)
        assert not model._drw_active

    def test_drw_increases_loss(self):
        """DRW should scale the ranking loss upward."""
        config = {**SMALL_CONFIG, 'ranking_num_samples': 1}
        model = ParticleNeXtPreFilter(
            **config,
            drw_positive_weight=3.0,
        )
        model.eval()
        inputs = _make_physical_inputs(seed=42)
        points, features, lorentz_vectors, mask, track_labels = inputs

        with torch.no_grad():
            scores = model(points, features, lorentz_vectors, mask)
        valid_mask = mask.squeeze(1).bool()
        labels_flat = track_labels.squeeze(1) * valid_mask.float()

        model._drw_active = False
        loss_without_drw = model._ranking_loss(scores, labels_flat, valid_mask)

        model._drw_active = True
        loss_with_drw = model._ranking_loss(scores, labels_flat, valid_mask)

        assert loss_with_drw > loss_without_drw


# ---- Top-K selection and track filtering ----

class TestFilterTracks:
    """Test top-K selection and track filtering."""

    def test_select_top_k_shape(self):
        """select_top_k should return (B, K) indices."""
        model = ParticleNeXtPreFilter(**SMALL_CONFIG)
        model.eval()
        inputs = _make_physical_inputs()
        with torch.no_grad():
            scores = model(*inputs[:4])
        indices = model.select_top_k(scores, inputs[3], top_k=TOP_K)
        assert indices.shape == (BATCH_SIZE, TOP_K)

    def test_filter_tracks_output_shape(self):
        """filter_tracks should return dict with P=top_k."""
        model = ParticleNeXtPreFilter(**SMALL_CONFIG)
        model.eval()
        inputs = _make_physical_inputs()
        with torch.no_grad():
            filtered = model.filter_tracks(*inputs, top_k=TOP_K)
        assert filtered['points'].shape == (BATCH_SIZE, 2, TOP_K)
        assert filtered['features'].shape == (BATCH_SIZE, INPUT_DIM, TOP_K)
        assert filtered['lorentz_vectors'].shape == (BATCH_SIZE, 4, TOP_K)
        assert filtered['mask'].shape == (BATCH_SIZE, 1, TOP_K)
        assert filtered['track_labels'].shape == (BATCH_SIZE, 1, TOP_K)

    def test_top_k_excludes_padded_tracks(self):
        """Top-K should never select padded (masked) tracks."""
        model = ParticleNeXtPreFilter(**SMALL_CONFIG)
        model.eval()
        inputs = _make_physical_inputs()
        with torch.no_grad():
            scores = model(*inputs[:4])
        indices = model.select_top_k(scores, inputs[3], top_k=TOP_K)
        # Padded tracks are at indices 80-99 (last 20)
        assert (indices < 80).all(), 'Top-K selected padded tracks'


# ---- Param count ----

class TestParamCount:
    """Test that the model has expected parameter count."""

    def test_default_config_params(self):
        """Default config should have ~550K params (more than Phase-B's 225K)."""
        model = ParticleNeXtPreFilter(feature_input_dim=13)
        total = sum(p.numel() for p in model.parameters())
        assert total > 300_000, f'Too few params: {total:,}'
        assert total < 2_000_000, f'Too many params: {total:,}'

    def test_small_config_params(self):
        """Small test config should have reasonable param count."""
        model = ParticleNeXtPreFilter(**SMALL_CONFIG)
        total = sum(p.numel() for p in model.parameters())
        assert total > 10_000
        assert total < 200_000


# ---- Backward compatibility ----

class TestBackwardCompatibility:
    """Verify TrackPreFilter still works independently."""

    def test_track_prefilter_unaffected(self):
        """TrackPreFilter MLP mode should still produce valid results."""
        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM, hidden_dim=64,
            num_neighbors=8, num_message_rounds=1,
        )
        points, features, lorentz_vectors, mask, labels = _make_physical_inputs()
        model.train()
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask, labels,
        )
        assert loss_dict['total_loss'].isfinite()
        assert loss_dict['_scores'].shape == (BATCH_SIZE, NUM_TRACKS)
