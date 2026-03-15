"""Unit tests for TauTrackFinderHead (encoder + decoder + mask scoring + confidence).

Tests cover:
    - Output tensor shapes for all subcomponents
    - Masking: padded track positions get -inf in mask logits
    - Learned temperature affects logit scale
    - Gradient flow: only head parameters receive gradients
    - Configurable architecture: encoder/decoder layer counts
"""
import pytest
import torch

from weaver.nn.model.TauTrackFinderHead import TauTrackFinderHead


# ---- Fixtures ----

BATCH_SIZE = 4
NUM_TRACKS = 200  # Simulated track count (smaller than real ~1130 for speed)
BACKBONE_DIM = 256
NUM_COMPACT_TOKENS = 128
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
        num_encoder_layers=2,  # Fewer layers for test speed
        num_decoder_layers=2,
        dropout=0.0,
    )


@pytest.fixture
def sample_inputs():
    """Create synthetic inputs matching backbone output shapes.

    TauTrackFinderHead.forward() requires:
        enriched_features: (B, backbone_dim, P)
        compact_tokens: (B, backbone_dim, 128)
        mask: (B, 1, P)
        points: (B, 2, P) — (eta, phi) coordinates for FPS query init
    """
    enriched_features = torch.randn(BATCH_SIZE, BACKBONE_DIM, NUM_TRACKS)
    compact_tokens = torch.randn(BATCH_SIZE, BACKBONE_DIM, NUM_COMPACT_TOKENS)
    mask = torch.ones(BATCH_SIZE, 1, NUM_TRACKS)
    mask[:, :, -50:] = 0.0  # Last 50 tracks are padding
    points = torch.randn(BATCH_SIZE, 2, NUM_TRACKS)  # (eta, phi)
    return enriched_features, compact_tokens, mask, points


def _unpack_last_layer(head_output):
    """Extract last decoder layer's logits from head output.

    Head returns (list[mask_logits], list[confidence_logits]) with one
    entry per decoder layer for auxiliary losses. Tests typically only
    need the last layer.
    """
    all_layer_mask_logits, all_layer_confidence_logits = head_output
    return all_layer_mask_logits[-1], all_layer_confidence_logits[-1]


# ---- Shape Tests ----

class TestOutputShapes:
    """Verify output tensor shapes for all head outputs."""

    def test_mask_logits_shape(self, default_head, sample_inputs):
        enriched_features, compact_tokens, mask, points = sample_inputs
        mask_logits, confidence_logits = _unpack_last_layer(
            default_head(enriched_features, compact_tokens, mask, points),
        )
        # mask_logits: (B, num_queries, P)
        assert mask_logits.shape == (BATCH_SIZE, NUM_QUERIES, NUM_TRACKS)

    def test_confidence_logits_shape(self, default_head, sample_inputs):
        enriched_features, compact_tokens, mask, points = sample_inputs
        mask_logits, confidence_logits = _unpack_last_layer(
            default_head(enriched_features, compact_tokens, mask, points),
        )
        # confidence_logits: (B, num_queries)
        assert confidence_logits.shape == (BATCH_SIZE, NUM_QUERIES)

    def test_auxiliary_outputs_per_layer(self, default_head, sample_inputs):
        """Head should return outputs at every decoder layer."""
        enriched_features, compact_tokens, mask, points = sample_inputs
        all_mask, all_conf = default_head(
            enriched_features, compact_tokens, mask, points,
        )
        # 2 decoder layers → 2 sets of outputs
        assert len(all_mask) == 2
        assert len(all_conf) == 2
        for layer_mask, layer_conf in zip(all_mask, all_conf):
            assert layer_mask.shape == (BATCH_SIZE, NUM_QUERIES, NUM_TRACKS)
            assert layer_conf.shape == (BATCH_SIZE, NUM_QUERIES)

    def test_different_batch_sizes(self, default_head):
        """Head should handle arbitrary batch sizes."""
        for batch_size in [1, 2, 8]:
            enriched = torch.randn(batch_size, BACKBONE_DIM, NUM_TRACKS)
            compact = torch.randn(batch_size, BACKBONE_DIM, NUM_COMPACT_TOKENS)
            mask = torch.ones(batch_size, 1, NUM_TRACKS)
            points = torch.randn(batch_size, 2, NUM_TRACKS)
            mask_logits, confidence_logits = _unpack_last_layer(
                default_head(enriched, compact, mask, points),
            )
            assert mask_logits.shape == (batch_size, NUM_QUERIES, NUM_TRACKS)
            assert confidence_logits.shape == (batch_size, NUM_QUERIES)

    def test_different_track_counts(self, default_head):
        """Head should handle variable sequence lengths."""
        for num_tracks in [50, 500, 1200]:
            enriched = torch.randn(2, BACKBONE_DIM, num_tracks)
            compact = torch.randn(2, BACKBONE_DIM, NUM_COMPACT_TOKENS)
            mask = torch.ones(2, 1, num_tracks)
            points = torch.randn(2, 2, num_tracks)
            mask_logits, confidence_logits = _unpack_last_layer(
                default_head(enriched, compact, mask, points),
            )
            assert mask_logits.shape == (2, NUM_QUERIES, num_tracks)
            assert confidence_logits.shape == (2, NUM_QUERIES)


# ---- Masking Tests ----

class TestMasking:
    """Verify that padded track positions get -inf in mask logits."""

    def test_padded_positions_are_negative_infinity(self, default_head, sample_inputs):
        enriched_features, compact_tokens, mask, points = sample_inputs
        mask_logits, _ = _unpack_last_layer(
            default_head(enriched_features, compact_tokens, mask, points),
        )

        # Last 50 tracks are padded (mask=0) → logits should be -inf
        padded_logits = mask_logits[:, :, -50:]
        assert torch.all(padded_logits == float('-inf')), (
            "Padded track positions should have -inf logits"
        )

    def test_valid_positions_are_finite(self, default_head, sample_inputs):
        enriched_features, compact_tokens, mask, points = sample_inputs
        mask_logits, _ = _unpack_last_layer(
            default_head(enriched_features, compact_tokens, mask, points),
        )

        # First 150 tracks are valid → logits should be finite
        valid_logits = mask_logits[:, :, :150]
        assert torch.all(torch.isfinite(valid_logits)), (
            "Valid track positions should have finite logits"
        )

    def test_all_valid_mask(self, default_head):
        """When all tracks are valid, no -inf should appear."""
        enriched = torch.randn(2, BACKBONE_DIM, 100)
        compact = torch.randn(2, BACKBONE_DIM, NUM_COMPACT_TOKENS)
        mask = torch.ones(2, 1, 100)
        points = torch.randn(2, 2, 100)
        mask_logits, _ = _unpack_last_layer(
            default_head(enriched, compact, mask, points),
        )
        assert torch.all(torch.isfinite(mask_logits))


# ---- Temperature Tests ----

class TestTemperature:
    """Verify that learned temperature affects mask logit scale."""

    def test_temperature_scales_logits(self, default_head, sample_inputs):
        enriched_features, compact_tokens, mask, points = sample_inputs

        # Get logits with default temperature (tau=1.0)
        mask_logits_default, _ = _unpack_last_layer(
            default_head(enriched_features, compact_tokens, mask, points),
        )
        valid_logits_default = mask_logits_default[:, :, :150]
        std_default = valid_logits_default.std().item()

        # Manually set temperature to 0.5 → logits should be 2x larger
        with torch.no_grad():
            default_head.temperature.fill_(0.5)

        mask_logits_scaled, _ = _unpack_last_layer(
            default_head(enriched_features, compact_tokens, mask, points),
        )
        valid_logits_scaled = mask_logits_scaled[:, :, :150]
        std_scaled = valid_logits_scaled.std().item()

        # Logits should be roughly 2x larger with tau=0.5 vs tau=1.0
        ratio = std_scaled / max(std_default, 1e-8)
        assert 1.5 < ratio < 3.0, (
            f"Temperature halving should roughly double logit std. "
            f"Got ratio={ratio:.2f}"
        )

        # Restore temperature
        with torch.no_grad():
            default_head.temperature.fill_(1.0)

    def test_temperature_is_learnable(self, default_head):
        """Temperature should be a learnable parameter."""
        assert default_head.temperature.requires_grad


# ---- Gradient Flow Tests ----

class TestGradientFlow:
    """Verify gradient flow through the head."""

    def test_all_head_params_receive_gradients(self, default_head, sample_inputs):
        enriched_features, compact_tokens, mask, points = sample_inputs
        mask_logits, confidence_logits = _unpack_last_layer(
            default_head(enriched_features, compact_tokens, mask, points),
        )

        # Create a dummy loss combining both outputs
        loss = mask_logits[:, :, :150].sum() + confidence_logits.sum()
        loss.backward()

        # All head parameters should have gradients
        params_without_grad = []
        for name, param in default_head.named_parameters():
            if param.requires_grad and param.grad is None:
                params_without_grad.append(name)

        assert len(params_without_grad) == 0, (
            f"Parameters without gradients: {params_without_grad}"
        )

    def test_outputs_are_finite(self, default_head, sample_inputs):
        enriched_features, compact_tokens, mask, points = sample_inputs
        mask_logits, confidence_logits = _unpack_last_layer(
            default_head(enriched_features, compact_tokens, mask, points),
        )

        # Valid logits should be finite
        valid_logits = mask_logits[:, :, :150]
        assert torch.all(torch.isfinite(valid_logits))
        assert torch.all(torch.isfinite(confidence_logits))

    def test_no_nan_in_backward(self, default_head, sample_inputs):
        enriched_features, compact_tokens, mask, points = sample_inputs
        mask_logits, confidence_logits = _unpack_last_layer(
            default_head(enriched_features, compact_tokens, mask, points),
        )

        loss = mask_logits[:, :, :150].mean() + confidence_logits.mean()
        loss.backward()

        for name, param in default_head.named_parameters():
            if param.grad is not None:
                assert torch.all(torch.isfinite(param.grad)), (
                    f"NaN/Inf gradient in {name}"
                )


# ---- Configurable Architecture Tests ----

class TestConfigurableArchitecture:
    """Verify that encoder/decoder layers are independently configurable."""

    def test_encoder_only(self):
        head = TauTrackFinderHead(
            backbone_dim=BACKBONE_DIM,
            decoder_dim=DECODER_DIM,
            mask_dim=MASK_DIM,
            num_queries=NUM_QUERIES,
            num_heads=8,
            num_encoder_layers=4,
            num_decoder_layers=1,
            dropout=0.0,
        )
        enriched = torch.randn(2, BACKBONE_DIM, 100)
        compact = torch.randn(2, BACKBONE_DIM, NUM_COMPACT_TOKENS)
        mask = torch.ones(2, 1, 100)
        points = torch.randn(2, 2, 100)
        mask_logits, confidence_logits = _unpack_last_layer(
            head(enriched, compact, mask, points),
        )
        assert mask_logits.shape == (2, NUM_QUERIES, 100)

    def test_decoder_only(self):
        head = TauTrackFinderHead(
            backbone_dim=BACKBONE_DIM,
            decoder_dim=DECODER_DIM,
            mask_dim=MASK_DIM,
            num_queries=NUM_QUERIES,
            num_heads=8,
            num_encoder_layers=1,
            num_decoder_layers=4,
            dropout=0.0,
        )
        enriched = torch.randn(2, BACKBONE_DIM, 100)
        compact = torch.randn(2, BACKBONE_DIM, NUM_COMPACT_TOKENS)
        mask = torch.ones(2, 1, 100)
        points = torch.randn(2, 2, 100)
        mask_logits, confidence_logits = _unpack_last_layer(
            head(enriched, compact, mask, points),
        )
        assert mask_logits.shape == (2, NUM_QUERIES, 100)

    def test_many_layers(self):
        """6+6 = 12 layers (matching DETR) should work without issues."""
        head = TauTrackFinderHead(
            backbone_dim=BACKBONE_DIM,
            decoder_dim=DECODER_DIM,
            mask_dim=MASK_DIM,
            num_queries=NUM_QUERIES,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dropout=0.0,
        )
        enriched = torch.randn(2, BACKBONE_DIM, 100)
        compact = torch.randn(2, BACKBONE_DIM, NUM_COMPACT_TOKENS)
        mask = torch.ones(2, 1, 100)
        points = torch.randn(2, 2, 100)
        mask_logits, confidence_logits = _unpack_last_layer(
            head(enriched, compact, mask, points),
        )
        assert mask_logits.shape == (2, NUM_QUERIES, 100)
        assert confidence_logits.shape == (2, NUM_QUERIES)
