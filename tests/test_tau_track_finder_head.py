"""Unit tests for TauTrackFinderHead (encoder + decoder + pointer + confidence).

Tests cover:
    - Output tensor shapes for all subcomponents
    - Masking: padded track positions get -inf in pointer logits
    - Learned temperature affects logit scale
    - Gradient flow: only head parameters receive gradients
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
POINTER_DIM = 128


@pytest.fixture
def default_head():
    """Create a TauTrackFinderHead with default hyperparameters."""
    return TauTrackFinderHead(
        backbone_dim=BACKBONE_DIM,
        decoder_dim=DECODER_DIM,
        pointer_dim=POINTER_DIM,
        num_queries=NUM_QUERIES,
        num_heads=8,
        num_encoder_layers=2,  # Fewer layers for test speed
        num_decoder_layers=2,
        dropout=0.0,
    )


@pytest.fixture
def sample_inputs():
    """Create synthetic inputs matching backbone output shapes."""
    # Enriched per-track features from backbone.enrich()
    enriched_features = torch.randn(BATCH_SIZE, BACKBONE_DIM, NUM_TRACKS)
    # Compact tokens from backbone.compact()
    compact_tokens = torch.randn(BATCH_SIZE, BACKBONE_DIM, NUM_COMPACT_TOKENS)
    # Track mask: last 50 tracks are padding (invalid)
    mask = torch.ones(BATCH_SIZE, 1, NUM_TRACKS)
    mask[:, :, -50:] = 0.0
    return enriched_features, compact_tokens, mask


# ---- Shape Tests ----

class TestOutputShapes:
    """Verify output tensor shapes for all head outputs."""

    def test_pointer_logits_shape(self, default_head, sample_inputs):
        enriched_features, compact_tokens, mask = sample_inputs
        pointer_logits, confidence_logits = default_head(
            enriched_features, compact_tokens, mask,
        )
        # pointer_logits: (B, num_queries, P)
        assert pointer_logits.shape == (BATCH_SIZE, NUM_QUERIES, NUM_TRACKS)

    def test_confidence_logits_shape(self, default_head, sample_inputs):
        enriched_features, compact_tokens, mask = sample_inputs
        pointer_logits, confidence_logits = default_head(
            enriched_features, compact_tokens, mask,
        )
        # confidence_logits: (B, num_queries)
        assert confidence_logits.shape == (BATCH_SIZE, NUM_QUERIES)

    def test_different_batch_sizes(self, default_head):
        """Head should handle arbitrary batch sizes."""
        for batch_size in [1, 2, 8]:
            enriched = torch.randn(batch_size, BACKBONE_DIM, NUM_TRACKS)
            compact = torch.randn(batch_size, BACKBONE_DIM, NUM_COMPACT_TOKENS)
            mask = torch.ones(batch_size, 1, NUM_TRACKS)
            pointer_logits, confidence_logits = default_head(
                enriched, compact, mask,
            )
            assert pointer_logits.shape == (batch_size, NUM_QUERIES, NUM_TRACKS)
            assert confidence_logits.shape == (batch_size, NUM_QUERIES)

    def test_different_track_counts(self, default_head):
        """Head should handle variable sequence lengths."""
        for num_tracks in [50, 500, 1200]:
            enriched = torch.randn(2, BACKBONE_DIM, num_tracks)
            compact = torch.randn(2, BACKBONE_DIM, NUM_COMPACT_TOKENS)
            mask = torch.ones(2, 1, num_tracks)
            pointer_logits, confidence_logits = default_head(
                enriched, compact, mask,
            )
            assert pointer_logits.shape == (2, NUM_QUERIES, num_tracks)
            assert confidence_logits.shape == (2, NUM_QUERIES)


# ---- Masking Tests ----

class TestMasking:
    """Verify that padded track positions get -inf in pointer logits."""

    def test_padded_positions_are_negative_infinity(self, default_head, sample_inputs):
        enriched_features, compact_tokens, mask = sample_inputs
        pointer_logits, _ = default_head(enriched_features, compact_tokens, mask)

        # Last 50 tracks are padded (mask=0) → logits should be -inf
        padded_logits = pointer_logits[:, :, -50:]
        assert torch.all(padded_logits == float('-inf')), (
            "Padded track positions should have -inf logits"
        )

    def test_valid_positions_are_finite(self, default_head, sample_inputs):
        enriched_features, compact_tokens, mask = sample_inputs
        pointer_logits, _ = default_head(enriched_features, compact_tokens, mask)

        # First 150 tracks are valid → logits should be finite
        valid_logits = pointer_logits[:, :, :150]
        assert torch.all(torch.isfinite(valid_logits)), (
            "Valid track positions should have finite logits"
        )

    def test_all_valid_mask(self, default_head):
        """When all tracks are valid, no -inf should appear."""
        enriched = torch.randn(2, BACKBONE_DIM, 100)
        compact = torch.randn(2, BACKBONE_DIM, NUM_COMPACT_TOKENS)
        mask = torch.ones(2, 1, 100)
        pointer_logits, _ = default_head(enriched, compact, mask)
        assert torch.all(torch.isfinite(pointer_logits))


# ---- Temperature Tests ----

class TestTemperature:
    """Verify that learned temperature affects pointer logit scale."""

    def test_temperature_scales_logits(self, default_head, sample_inputs):
        enriched_features, compact_tokens, mask = sample_inputs

        # Get logits with default temperature (τ=1.0)
        pointer_logits_default, _ = default_head(
            enriched_features, compact_tokens, mask,
        )
        valid_logits_default = pointer_logits_default[:, :, :150]
        std_default = valid_logits_default.std().item()

        # Manually set temperature to 0.5 → logits should be 2× larger
        with torch.no_grad():
            default_head.temperature.fill_(0.5)

        pointer_logits_scaled, _ = default_head(
            enriched_features, compact_tokens, mask,
        )
        valid_logits_scaled = pointer_logits_scaled[:, :, :150]
        std_scaled = valid_logits_scaled.std().item()

        # Logits should be roughly 2× larger with τ=0.5 vs τ=1.0
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
        enriched_features, compact_tokens, mask = sample_inputs
        pointer_logits, confidence_logits = default_head(
            enriched_features, compact_tokens, mask,
        )

        # Create a dummy loss combining both outputs
        loss = pointer_logits[:, :, :150].sum() + confidence_logits.sum()
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
        enriched_features, compact_tokens, mask = sample_inputs
        pointer_logits, confidence_logits = default_head(
            enriched_features, compact_tokens, mask,
        )

        # Valid logits should be finite
        valid_logits = pointer_logits[:, :, :150]
        assert torch.all(torch.isfinite(valid_logits))
        assert torch.all(torch.isfinite(confidence_logits))

    def test_no_nan_in_backward(self, default_head, sample_inputs):
        enriched_features, compact_tokens, mask = sample_inputs
        pointer_logits, confidence_logits = default_head(
            enriched_features, compact_tokens, mask,
        )

        loss = pointer_logits[:, :, :150].mean() + confidence_logits.mean()
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
            pointer_dim=POINTER_DIM,
            num_queries=NUM_QUERIES,
            num_heads=8,
            num_encoder_layers=4,
            num_decoder_layers=1,
            dropout=0.0,
        )
        enriched = torch.randn(2, BACKBONE_DIM, 100)
        compact = torch.randn(2, BACKBONE_DIM, NUM_COMPACT_TOKENS)
        mask = torch.ones(2, 1, 100)
        pointer_logits, confidence_logits = head(enriched, compact, mask)
        assert pointer_logits.shape == (2, NUM_QUERIES, 100)

    def test_decoder_only(self):
        head = TauTrackFinderHead(
            backbone_dim=BACKBONE_DIM,
            decoder_dim=DECODER_DIM,
            pointer_dim=POINTER_DIM,
            num_queries=NUM_QUERIES,
            num_heads=8,
            num_encoder_layers=1,
            num_decoder_layers=4,
            dropout=0.0,
        )
        enriched = torch.randn(2, BACKBONE_DIM, 100)
        compact = torch.randn(2, BACKBONE_DIM, NUM_COMPACT_TOKENS)
        mask = torch.ones(2, 1, 100)
        pointer_logits, confidence_logits = head(enriched, compact, mask)
        assert pointer_logits.shape == (2, NUM_QUERIES, 100)

    def test_many_layers(self):
        """6+6 = 12 layers (matching DETR) should work without issues."""
        head = TauTrackFinderHead(
            backbone_dim=BACKBONE_DIM,
            decoder_dim=DECODER_DIM,
            pointer_dim=POINTER_DIM,
            num_queries=NUM_QUERIES,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dropout=0.0,
        )
        enriched = torch.randn(2, BACKBONE_DIM, 100)
        compact = torch.randn(2, BACKBONE_DIM, NUM_COMPACT_TOKENS)
        mask = torch.ones(2, 1, 100)
        pointer_logits, confidence_logits = head(enriched, compact, mask)
        assert pointer_logits.shape == (2, NUM_QUERIES, 100)
        assert confidence_logits.shape == (2, NUM_QUERIES)
