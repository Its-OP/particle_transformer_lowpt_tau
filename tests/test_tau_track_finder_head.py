"""Unit tests for TauTrackFinderHead (decoder + mask scoring + confidence).

Tests cover:
    - Output tensor shapes
    - Masking: padded track positions get -inf in mask logits
    - Learned temperature affects logit scale
    - Gradient flow through all parameters
    - Configurable decoder depth
"""
import pytest
import torch

from weaver.nn.model.TauTrackFinderHead import TauTrackFinderHead


# ---- Fixtures ----

BATCH_SIZE = 4
NUM_TRACKS = 200
BACKBONE_DIM = 256
NUM_QUERIES = 15
DECODER_DIM = 256
MASK_DIM = 128


@pytest.fixture
def default_head():
    """Create a TauTrackFinderHead with default hyperparameters."""
    return TauTrackFinderHead(
        backbone_dim=BACKBONE_DIM,
        decoder_dim=DECODER_DIM,
        mask_dim=MASK_DIM,
        num_queries=NUM_QUERIES,
        num_heads=8,
        num_decoder_layers=2,
        dropout=0.0,
    )


@pytest.fixture
def sample_inputs():
    """Create synthetic inputs matching backbone output shapes."""
    enriched_features = torch.randn(BATCH_SIZE, BACKBONE_DIM, NUM_TRACKS)
    mask = torch.ones(BATCH_SIZE, 1, NUM_TRACKS)
    mask[:, :, -50:] = 0.0  # Last 50 tracks are padding
    points = torch.randn(BATCH_SIZE, 2, NUM_TRACKS)
    return enriched_features, mask, points


# ---- Shape Tests ----

class TestOutputShapes:

    def test_mask_logits_shape(self, default_head, sample_inputs):
        enriched_features, mask, points = sample_inputs
        mask_logits, confidence_logits = default_head(enriched_features, mask, points)
        assert mask_logits.shape == (BATCH_SIZE, NUM_QUERIES, NUM_TRACKS)

    def test_confidence_logits_shape(self, default_head, sample_inputs):
        enriched_features, mask, points = sample_inputs
        mask_logits, confidence_logits = default_head(enriched_features, mask, points)
        assert confidence_logits.shape == (BATCH_SIZE, NUM_QUERIES)

    def test_different_batch_sizes(self, default_head):
        for batch_size in [1, 2, 8]:
            enriched = torch.randn(batch_size, BACKBONE_DIM, NUM_TRACKS)
            mask = torch.ones(batch_size, 1, NUM_TRACKS)
            points = torch.randn(batch_size, 2, NUM_TRACKS)
            mask_logits, confidence_logits = default_head(enriched, mask, points)
            assert mask_logits.shape == (batch_size, NUM_QUERIES, NUM_TRACKS)
            assert confidence_logits.shape == (batch_size, NUM_QUERIES)

    def test_different_track_counts(self, default_head):
        for num_tracks in [50, 500, 1200]:
            enriched = torch.randn(2, BACKBONE_DIM, num_tracks)
            mask = torch.ones(2, 1, num_tracks)
            points = torch.randn(2, 2, num_tracks)
            mask_logits, confidence_logits = default_head(enriched, mask, points)
            assert mask_logits.shape == (2, NUM_QUERIES, num_tracks)
            assert confidence_logits.shape == (2, NUM_QUERIES)


# ---- Masking Tests ----

class TestMasking:

    def test_padded_positions_are_negative_infinity(self, default_head, sample_inputs):
        enriched_features, mask, points = sample_inputs
        mask_logits, _ = default_head(enriched_features, mask, points)
        padded_logits = mask_logits[:, :, -50:]
        assert torch.all(padded_logits == float('-inf'))

    def test_valid_positions_are_finite(self, default_head, sample_inputs):
        enriched_features, mask, points = sample_inputs
        mask_logits, _ = default_head(enriched_features, mask, points)
        valid_logits = mask_logits[:, :, :150]
        assert torch.all(torch.isfinite(valid_logits))

    def test_all_valid_mask(self, default_head):
        enriched = torch.randn(2, BACKBONE_DIM, 100)
        mask = torch.ones(2, 1, 100)
        points = torch.randn(2, 2, 100)
        mask_logits, _ = default_head(enriched, mask, points)
        assert torch.all(torch.isfinite(mask_logits))


# ---- Temperature Tests ----

class TestTemperature:

    def test_temperature_scales_logits(self, default_head, sample_inputs):
        enriched_features, mask, points = sample_inputs

        mask_logits_default, _ = default_head(enriched_features, mask, points)
        valid_default = mask_logits_default[:, :, :150]
        std_default = valid_default.std().item()

        with torch.no_grad():
            default_head.temperature.fill_(0.5)
        mask_logits_scaled, _ = default_head(enriched_features, mask, points)
        valid_scaled = mask_logits_scaled[:, :, :150]
        std_scaled = valid_scaled.std().item()

        ratio = std_scaled / max(std_default, 1e-8)
        assert 1.5 < ratio < 3.0, f"Expected ~2x, got {ratio:.2f}"

        with torch.no_grad():
            default_head.temperature.fill_(1.0)

    def test_temperature_is_learnable(self, default_head):
        assert default_head.temperature.requires_grad


# ---- Gradient Flow Tests ----

class TestGradientFlow:

    def test_all_head_params_receive_gradients(self, default_head, sample_inputs):
        enriched_features, mask, points = sample_inputs
        mask_logits, confidence_logits = default_head(enriched_features, mask, points)
        loss = mask_logits[:, :, :150].sum() + confidence_logits.sum()
        loss.backward()

        params_without_grad = []
        for name, param in default_head.named_parameters():
            if param.requires_grad and param.grad is None:
                params_without_grad.append(name)
        assert len(params_without_grad) == 0, f"No gradients: {params_without_grad}"

    def test_outputs_are_finite(self, default_head, sample_inputs):
        enriched_features, mask, points = sample_inputs
        mask_logits, confidence_logits = default_head(enriched_features, mask, points)
        assert torch.all(torch.isfinite(mask_logits[:, :, :150]))
        assert torch.all(torch.isfinite(confidence_logits))

    def test_no_nan_in_backward(self, default_head, sample_inputs):
        enriched_features, mask, points = sample_inputs
        mask_logits, confidence_logits = default_head(enriched_features, mask, points)
        loss = mask_logits[:, :, :150].mean() + confidence_logits.mean()
        loss.backward()
        for name, param in default_head.named_parameters():
            if param.grad is not None:
                assert torch.all(torch.isfinite(param.grad)), f"NaN grad: {name}"


# ---- Configurable Architecture Tests ----

class TestConfigurableArchitecture:

    def test_single_decoder_layer(self):
        head = TauTrackFinderHead(
            backbone_dim=BACKBONE_DIM, decoder_dim=DECODER_DIM,
            mask_dim=MASK_DIM, num_queries=NUM_QUERIES,
            num_heads=8, num_decoder_layers=1, dropout=0.0,
        )
        enriched = torch.randn(2, BACKBONE_DIM, 100)
        mask = torch.ones(2, 1, 100)
        points = torch.randn(2, 2, 100)
        mask_logits, confidence_logits = head(enriched, mask, points)
        assert mask_logits.shape == (2, NUM_QUERIES, 100)

    def test_many_decoder_layers(self):
        head = TauTrackFinderHead(
            backbone_dim=BACKBONE_DIM, decoder_dim=DECODER_DIM,
            mask_dim=MASK_DIM, num_queries=NUM_QUERIES,
            num_heads=8, num_decoder_layers=8, dropout=0.0,
        )
        enriched = torch.randn(2, BACKBONE_DIM, 100)
        mask = torch.ones(2, 1, 100)
        points = torch.randn(2, 2, 100)
        mask_logits, confidence_logits = head(enriched, mask, points)
        assert mask_logits.shape == (2, NUM_QUERIES, 100)
        assert confidence_logits.shape == (2, NUM_QUERIES)
