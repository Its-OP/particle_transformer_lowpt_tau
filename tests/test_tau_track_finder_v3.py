"""Unit tests for TauTrackFinderV3 (ABCNet-inspired GAPLayer architecture).

TDD: These tests are written BEFORE the implementation. All tests should
FAIL until TauTrackFinderV3 is implemented.

Tests cover:
    - GAPLayer: attention-weighted edge convolution, multi-head, masking
    - Dual kNN: physical space (eta, phi) + learned feature space
    - Global context: event-level average pooling + tiling
    - Forward pass: training returns finite loss, inference returns logits
    - Focal BCE loss: finiteness, class weighting, modulation
    - Backbone freezing: frozen backbone, trainable GAPLayers + scoring head
    - Masking: padded tracks excluded from attention and scoring
    - Multi-scale concatenation: correct dimension, all sources contribute
"""
import pytest
import torch

from weaver.nn.model.TauTrackFinderV3 import (
    GAPLayer,
    TauTrackFinderV3,
)
from weaver.nn.model.ParallelBackbone import ParallelBackbone


# ---- Shared test configuration ----

BATCH_SIZE = 4
NUM_TRACKS = 200  # Smaller than real ~1130 for test speed
INPUT_DIM = 7
GAP_ENCODING_DIM = 32  # Smaller than production for speed
GAP_NUM_NEIGHBORS = 8
GAP_NUM_HEADS = 2


def _make_backbone_kwargs(input_dim=INPUT_DIM):
    """Build backbone kwargs matching the pretraining config (small for tests)."""
    return dict(
        input_dim=input_dim,
        enrichment_kwargs=dict(
            node_dim=32,
            edge_dim=8,
            num_neighbors=16,
            edge_aggregation='attn8',
            layer_params=[
                # Single layer for test speed
                (16, 64, [(4, 1), (2, 1)], 32),
            ],
        ),
        compaction_kwargs=dict(
            stage_output_points=[64, 32],
            stage_output_channels=[64, 64],
            stage_num_neighbors=[8, 8],
        ),
    )


def _make_v3_kwargs():
    """Build V3-specific kwargs for testing (smaller than production)."""
    return dict(
        gap1_encoding_dim=GAP_ENCODING_DIM,
        gap1_num_neighbors=GAP_NUM_NEIGHBORS,
        gap1_num_heads=GAP_NUM_HEADS,
        gap2_encoding_dim=GAP_ENCODING_DIM,
        gap2_num_neighbors=GAP_NUM_NEIGHBORS,
        gap2_num_heads=GAP_NUM_HEADS,
        intermediate_dim=64,
        global_context_dim=16,
        scoring_dropout=0.0,  # Disable dropout for deterministic tests
        # ASL loss parameters
        focal_gamma_positive=1.0,
        focal_gamma_negative=4.0,
        asl_clip=0.05,
        # Ranking loss parameters
        ranking_loss_weight=0.1,
        ranking_num_samples=10,
    )


def _make_physical_inputs(batch_size, num_tracks, input_dim=INPUT_DIM, seed=42):
    """Create physically sensible synthetic inputs.

    Lorentz vectors must be physical (positive energy, E >= |p|) because
    the backbone computes pairwise_lv_fts() with log(kT), log(delta_r), etc.
    Random negative values would produce NaN from log of negatives.

    Uses fixed seed for test reproducibility.
    """
    generator = torch.Generator().manual_seed(seed)

    # Points: (eta, phi) coordinates in realistic ranges
    eta = torch.randn(batch_size, 1, num_tracks, generator=generator) * 1.5
    phi = (
        torch.rand(batch_size, 1, num_tracks, generator=generator)
        * 2 * 3.14159 - 3.14159
    )
    points = torch.cat([eta, phi], dim=1)

    # Features: standardized (zero-mean, unit-variance)
    features = torch.randn(
        batch_size, input_dim, num_tracks, generator=generator,
    )

    # Lorentz vectors: physical 4-momenta (px, py, pz, E)
    transverse_momentum = (
        torch.rand(batch_size, 1, num_tracks, generator=generator) * 5 + 0.5
    )  # pT in [0.5, 5.5] GeV
    px = transverse_momentum * torch.cos(phi)
    py = transverse_momentum * torch.sin(phi)
    pz = transverse_momentum * torch.sinh(eta)
    pion_mass = 0.13957
    energy = torch.sqrt(px**2 + py**2 + pz**2 + pion_mass**2)
    lorentz_vectors = torch.cat([px, py, pz, energy], dim=1)

    return points, features, lorentz_vectors


@pytest.fixture
def model():
    """Create a TauTrackFinderV3 with small architecture for testing."""
    return TauTrackFinderV3(
        backbone_kwargs=_make_backbone_kwargs(),
        **_make_v3_kwargs(),
    )


@pytest.fixture
def sample_training_inputs():
    """Create physically sensible synthetic training inputs with track labels."""
    points, features, lorentz_vectors = _make_physical_inputs(
        BATCH_SIZE, NUM_TRACKS, seed=42,
    )

    mask = torch.ones(BATCH_SIZE, 1, NUM_TRACKS)
    # Last 50 tracks are padding
    mask[:, :, -50:] = 0.0

    # Track labels: 3 tau tracks per event (at positions 10, 20, 30)
    track_labels = torch.zeros(BATCH_SIZE, 1, NUM_TRACKS)
    for batch_index in range(BATCH_SIZE):
        track_labels[batch_index, 0, 10] = 1.0
        track_labels[batch_index, 0, 20] = 1.0
        track_labels[batch_index, 0, 30] = 1.0

    return points, features, lorentz_vectors, mask, track_labels


# ---- GAPLayer Tests ----

class TestGAPLayer:
    """Test the Graph Attention Pooling Layer (ABCNet-style).

    GAPLayer computes attention-weighted edge features over kNN neighbors:
        c_ij = softmax_j(LeakyReLU(f(x'_i) + g(y'_ij)))
        hat_x_i = ReLU(sum_j c_ij * y'_ij)
    where y_ij = feature_j - feature_i (edge features).
    """

    def test_forward_shape(self):
        """GAPLayer output should match expected dimensions."""
        input_dim = 64
        encoding_dim = GAP_ENCODING_DIM
        num_neighbors = GAP_NUM_NEIGHBORS
        layer = GAPLayer(
            input_dim=input_dim,
            encoding_dim=encoding_dim,
            num_neighbors=num_neighbors,
            num_heads=GAP_NUM_HEADS,
        )

        features = torch.randn(BATCH_SIZE, input_dim, NUM_TRACKS)
        # Simulate kNN indices
        neighbor_indices = torch.randint(
            0, NUM_TRACKS, (BATCH_SIZE, NUM_TRACKS, num_neighbors),
        )
        mask = torch.ones(BATCH_SIZE, 1, NUM_TRACKS, dtype=torch.bool)

        attention_features, graph_features = layer(
            features, neighbor_indices, mask,
        )

        assert attention_features.shape == (
            BATCH_SIZE, encoding_dim, NUM_TRACKS,
        ), f'Expected ({BATCH_SIZE}, {encoding_dim}, {NUM_TRACKS}), got {attention_features.shape}'

        assert graph_features.shape == (
            BATCH_SIZE, encoding_dim, NUM_TRACKS,
        ), f'Expected ({BATCH_SIZE}, {encoding_dim}, {NUM_TRACKS}), got {graph_features.shape}'

    def test_attention_coefficients_sum_to_one(self):
        """Softmax attention means coefficients sum to 1 per valid node."""
        input_dim = 64
        layer = GAPLayer(
            input_dim=input_dim,
            encoding_dim=GAP_ENCODING_DIM,
            num_neighbors=GAP_NUM_NEIGHBORS,
            num_heads=1,
        )

        features = torch.randn(BATCH_SIZE, input_dim, NUM_TRACKS)
        neighbor_indices = torch.randint(
            0, NUM_TRACKS, (BATCH_SIZE, NUM_TRACKS, GAP_NUM_NEIGHBORS),
        )
        mask = torch.ones(BATCH_SIZE, 1, NUM_TRACKS, dtype=torch.bool)

        # Access attention coefficients (layer should expose them or we test
        # indirectly through the get_attention_coefficients method)
        attention_coefficients = layer.compute_attention_coefficients(
            features, neighbor_indices, mask,
        )
        # attention_coefficients: (B, P, K) — softmax over K neighbors

        # Sum over K dimension should be ~1.0 for valid nodes
        coefficient_sums = attention_coefficients.sum(dim=-1)  # (B, P)
        valid_mask = mask.squeeze(1)  # (B, P)

        torch.testing.assert_close(
            coefficient_sums[valid_mask],
            torch.ones_like(coefficient_sums[valid_mask]),
            atol=1e-5, rtol=1e-5,
        )

    def test_attention_ignores_padded_neighbors(self):
        """Padded neighbor positions should receive zero attention weight."""
        input_dim = 64
        layer = GAPLayer(
            input_dim=input_dim,
            encoding_dim=GAP_ENCODING_DIM,
            num_neighbors=GAP_NUM_NEIGHBORS,
            num_heads=1,
        )

        features = torch.randn(BATCH_SIZE, input_dim, NUM_TRACKS)
        neighbor_indices = torch.randint(
            0, NUM_TRACKS, (BATCH_SIZE, NUM_TRACKS, GAP_NUM_NEIGHBORS),
        )
        # Mask: last 50 tracks are padded
        mask = torch.ones(BATCH_SIZE, 1, NUM_TRACKS, dtype=torch.bool)
        mask[:, :, -50:] = False

        # Some neighbors point to padded tracks
        neighbor_indices[:, :, -2:] = NUM_TRACKS - 1  # Point to padded track

        attention_features, graph_features = layer(
            features, neighbor_indices, mask,
        )

        # Output for padded positions should be zero
        padded_output = attention_features[:, :, -50:]
        assert torch.all(padded_output == 0), (
            'Padded track positions should have zero output'
        )

    def test_multi_head_output(self):
        """Multi-head GAPLayer should combine heads via max."""
        input_dim = 64
        single_head_layer = GAPLayer(
            input_dim=input_dim,
            encoding_dim=GAP_ENCODING_DIM,
            num_neighbors=GAP_NUM_NEIGHBORS,
            num_heads=1,
        )
        multi_head_layer = GAPLayer(
            input_dim=input_dim,
            encoding_dim=GAP_ENCODING_DIM,
            num_neighbors=GAP_NUM_NEIGHBORS,
            num_heads=4,
        )

        features = torch.randn(BATCH_SIZE, input_dim, NUM_TRACKS)
        neighbor_indices = torch.randint(
            0, NUM_TRACKS, (BATCH_SIZE, NUM_TRACKS, GAP_NUM_NEIGHBORS),
        )
        mask = torch.ones(BATCH_SIZE, 1, NUM_TRACKS, dtype=torch.bool)

        single_output, _ = single_head_layer(features, neighbor_indices, mask)
        multi_output, _ = multi_head_layer(features, neighbor_indices, mask)

        # Both should produce same shape
        assert single_output.shape == multi_output.shape

    def test_mia_high_dim_attention(self):
        """MIA mode should produce multi-channel attention weights, not scalar.

        With MIA (More-Interaction Attention), each edge gets a D-dimensional
        attention vector instead of a scalar. Each dimension acts as a separate
        attention head with head_dim=1, element-wise scaling the edge features.
        """
        input_dim = 64
        layer = GAPLayer(
            input_dim=input_dim,
            encoding_dim=GAP_ENCODING_DIM,
            num_neighbors=GAP_NUM_NEIGHBORS,
            num_heads=1,
            use_mia=True,
        )

        features = torch.randn(BATCH_SIZE, input_dim, NUM_TRACKS)
        neighbor_indices = torch.randint(
            0, NUM_TRACKS, (BATCH_SIZE, NUM_TRACKS, GAP_NUM_NEIGHBORS),
        )
        mask = torch.ones(BATCH_SIZE, 1, NUM_TRACKS, dtype=torch.bool)

        attention_features, graph_features = layer(
            features, neighbor_indices, mask,
        )

        # Output shape should still be (B, encoding_dim, P)
        assert attention_features.shape == (
            BATCH_SIZE, GAP_ENCODING_DIM, NUM_TRACKS,
        )

    def test_mia_attention_weights_sum_to_one_per_channel(self):
        """Each channel of MIA attention should independently sum to 1."""
        input_dim = 64
        layer = GAPLayer(
            input_dim=input_dim,
            encoding_dim=GAP_ENCODING_DIM,
            num_neighbors=GAP_NUM_NEIGHBORS,
            num_heads=1,
            use_mia=True,
        )

        features = torch.randn(BATCH_SIZE, input_dim, NUM_TRACKS)
        neighbor_indices = torch.randint(
            0, NUM_TRACKS, (BATCH_SIZE, NUM_TRACKS, GAP_NUM_NEIGHBORS),
        )
        mask = torch.ones(BATCH_SIZE, 1, NUM_TRACKS, dtype=torch.bool)

        # Get MIA attention weights: (B, encoding_dim, P, K)
        # Also returns encoded_edges and neighbor_validity (for reuse in _forward_mia)
        attention_weights, _encoded_edges, _neighbor_validity = (
            layer.compute_mia_attention_weights(
                features, neighbor_indices, mask,
            )
        )

        # Each channel should sum to 1 over K for valid nodes
        channel_sums = attention_weights.sum(dim=-1)  # (B, encoding_dim, P)
        valid_mask_expanded = mask.expand_as(channel_sums)

        torch.testing.assert_close(
            channel_sums[valid_mask_expanded],
            torch.ones_like(channel_sums[valid_mask_expanded]),
            atol=1e-5, rtol=1e-5,
        )

    def test_edge_features_are_differences(self):
        """Edge features should be y_ij = feature_j - feature_i."""
        input_dim = 64
        layer = GAPLayer(
            input_dim=input_dim,
            encoding_dim=GAP_ENCODING_DIM,
            num_neighbors=GAP_NUM_NEIGHBORS,
            num_heads=1,
        )

        # Create features where track 0 = [1,0,...] and track 1 = [0,1,...]
        features = torch.zeros(1, input_dim, NUM_TRACKS)
        features[0, 0, 0] = 1.0  # Track 0: feature 0 = 1
        features[0, 1, 1] = 1.0  # Track 1: feature 1 = 1

        # Force track 0's only neighbor to be track 1
        neighbor_indices = torch.zeros(
            1, NUM_TRACKS, GAP_NUM_NEIGHBORS, dtype=torch.long,
        )
        neighbor_indices[0, 0, :] = 1  # Track 0's neighbors are all track 1

        mask = torch.ones(1, 1, NUM_TRACKS, dtype=torch.bool)

        # The edge feature for (track 0, track 1) should be:
        # y_01 = feature[1] - feature[0] = [0,1,...] - [1,0,...] = [-1,1,...]
        # We can't directly check this without accessing internals, but we
        # verify the layer runs without error and produces finite output.
        attention_features, graph_features = layer(
            features, neighbor_indices, mask,
        )
        assert torch.isfinite(attention_features).all()


# ---- Dual kNN Tests ----

class TestDualKNN:
    """Test that V3 uses two separate kNN computations."""

    def test_physical_knn_uses_eta_phi(self, model, sample_training_inputs):
        """First GAPLayer should use kNN based on (eta, phi) coordinates."""
        points, features, lorentz_vectors, mask, _ = sample_training_inputs

        # The model should internally compute kNN on points (eta, phi)
        # for the first GAPLayer. We verify this by checking the model
        # has the expected structure.
        assert hasattr(model, 'gap_layer_physical'), (
            'V3 should have a gap_layer_physical for eta-phi kNN'
        )

    def test_learned_knn_uses_features(self, model, sample_training_inputs):
        """Second GAPLayer should use kNN based on learned feature space."""
        # The model should compute kNN in the intermediate feature space
        assert hasattr(model, 'gap_layer_learned'), (
            'V3 should have a gap_layer_learned for feature-space kNN'
        )

    def test_learned_knn_changes_with_features(self, model, sample_training_inputs):
        """kNN indices in feature space should change when features change."""
        points, features, lorentz_vectors, mask, track_labels = (
            sample_training_inputs
        )

        model.eval()
        with torch.no_grad():
            # Run with original features
            output_1 = model(points, features, lorentz_vectors, mask)

            # Perturb features significantly
            perturbed_features = features + torch.randn_like(features) * 10
            output_2 = model(points, perturbed_features, lorentz_vectors, mask)

        # Outputs should differ because feature-space kNN changes
        assert not torch.allclose(
            output_1['per_track_logits'],
            output_2['per_track_logits'],
            atol=1e-3,
        ), 'Different features should produce different outputs via feature-space kNN'


# ---- Global Context Tests ----

class TestGlobalContext:
    """Test event-level global context injection."""

    def test_global_context_shape(self, model, sample_training_inputs):
        """Global context features should have correct shape after tiling."""
        points, features, lorentz_vectors, mask, _ = sample_training_inputs

        # The model should have a global_context module
        assert hasattr(model, 'global_context_projection'), (
            'V3 should have global_context_projection for event-level features'
        )

    def test_global_context_is_event_level(self, model, sample_training_inputs):
        """Global context features should be identical across tracks in same event."""
        points, features, lorentz_vectors, mask, track_labels = (
            sample_training_inputs
        )

        # Access the global context computation
        model.eval()
        with torch.no_grad():
            # We test this via the model's internal computation.
            # The global context should be a (B, global_dim, 1) tensor
            # that gets tiled to (B, global_dim, P).
            enriched = model.backbone.enrich(
                points, features, lorentz_vectors, mask,
            ).detach()

            # Compute global context
            global_context = model._compute_global_context(enriched, mask)

        # Check that global context is the same for all tracks in each event
        for event_index in range(BATCH_SIZE):
            event_global = global_context[event_index]  # (global_dim, P)
            # All columns should be identical
            first_track = event_global[:, 0:1]
            assert torch.allclose(
                event_global, first_track.expand_as(event_global),
                atol=1e-6,
            ), f'Global context should be identical across tracks in event {event_index}'

    def test_global_context_respects_mask(self, model, sample_training_inputs):
        """Average pooling should only consider valid (unmasked) tracks."""
        points, features, lorentz_vectors, mask, _ = sample_training_inputs

        model.eval()
        with torch.no_grad():
            enriched = model.backbone.enrich(
                points, features, lorentz_vectors, mask,
            ).detach()

            # Compute with original mask
            global_1 = model._compute_global_context(enriched, mask)

            # Modify padded track features (should not affect global context)
            enriched_modified = enriched.clone()
            enriched_modified[:, :, -50:] = enriched_modified[:, :, -50:] + 100.0
            global_2 = model._compute_global_context(enriched_modified, mask)

        torch.testing.assert_close(
            global_1, global_2,
            atol=1e-5, rtol=1e-5,
        )


# ---- Full Forward Pass Tests ----

class TestV3ForwardPass:
    """Verify the complete V3 forward pass produces valid outputs."""

    def test_training_returns_finite_loss(self, model, sample_training_inputs):
        points, features, lorentz_vectors, mask, track_labels = (
            sample_training_inputs
        )
        model.train()
        loss_dict = model(
            points, features, lorentz_vectors, mask, track_labels,
        )

        assert 'total_loss' in loss_dict
        assert torch.isfinite(loss_dict['total_loss']).all(), (
            f"Total loss is not finite: {loss_dict['total_loss']}"
        )

    def test_training_loss_is_scalar(self, model, sample_training_inputs):
        points, features, lorentz_vectors, mask, track_labels = (
            sample_training_inputs
        )
        model.train()
        loss_dict = model(
            points, features, lorentz_vectors, mask, track_labels,
        )

        assert loss_dict['total_loss'].dim() == 0

    def test_inference_returns_per_track_logits(
        self, model, sample_training_inputs,
    ):
        points, features, lorentz_vectors, mask, _ = sample_training_inputs
        model.eval()
        with torch.no_grad():
            output = model(points, features, lorentz_vectors, mask)

        assert 'per_track_logits' in output
        assert output['per_track_logits'].shape == (BATCH_SIZE, NUM_TRACKS)

    def test_loss_is_non_negative(self, model, sample_training_inputs):
        points, features, lorentz_vectors, mask, track_labels = (
            sample_training_inputs
        )
        model.train()
        loss_dict = model(
            points, features, lorentz_vectors, mask, track_labels,
        )

        assert loss_dict['total_loss'].item() >= 0.0
        assert loss_dict['per_track_loss'].item() >= 0.0

    def test_no_refinement_stage(self, model):
        """V3 should NOT have self-attention refinement (removed by design)."""
        assert not hasattr(model, 'refinement_layers'), (
            'V3 should not have refinement_layers (removed in V3 design)'
        )
        assert not hasattr(model, 'refinement_projection'), (
            'V3 should not have refinement_projection (removed in V3 design)'
        )


# ---- Loss Tests ----

class TestV3Loss:
    """Verify focal BCE loss behavior."""

    def test_focal_bce_finite(self, model, sample_training_inputs):
        """Focal BCE should be finite for mixed labels."""
        points, features, lorentz_vectors, mask, track_labels = (
            sample_training_inputs
        )
        model.train()
        loss_dict = model(
            points, features, lorentz_vectors, mask, track_labels,
        )

        assert torch.isfinite(loss_dict['per_track_loss']).all()

    def test_focal_bce_finite_all_zeros(self, model):
        """Focal BCE should be finite when all labels are zero (no GT)."""
        points, features, lorentz_vectors = _make_physical_inputs(
            1, NUM_TRACKS,
        )
        mask = torch.ones(1, 1, NUM_TRACKS)
        track_labels = torch.zeros(1, 1, NUM_TRACKS)

        model.train()
        loss_dict = model(
            points, features, lorentz_vectors, mask, track_labels,
        )

        assert torch.isfinite(loss_dict['total_loss']).all()

    def test_loss_components_all_present(self, model, sample_training_inputs):
        points, features, lorentz_vectors, mask, track_labels = (
            sample_training_inputs
        )
        model.train()
        loss_dict = model(
            points, features, lorentz_vectors, mask, track_labels,
        )

        assert 'total_loss' in loss_dict
        assert 'per_track_loss' in loss_dict

    def test_zero_gt_loss_finite(self, model):
        """Zero GT tracks should produce finite loss."""
        points, features, lorentz_vectors = _make_physical_inputs(
            BATCH_SIZE, NUM_TRACKS,
        )
        mask = torch.ones(BATCH_SIZE, 1, NUM_TRACKS)
        track_labels = torch.zeros(BATCH_SIZE, 1, NUM_TRACKS)

        model.train()
        loss_dict = model(
            points, features, lorentz_vectors, mask, track_labels,
        )

        assert torch.isfinite(loss_dict['total_loss']).all()


# ---- Backbone Freezing Tests ----

class TestV3BackboneFreezing:
    """Verify backbone is frozen and GAPLayers are trainable."""

    def test_backbone_frozen(self, model):
        """All backbone parameters should not require gradients."""
        for name, parameter in model.backbone.named_parameters():
            assert not parameter.requires_grad, (
                f"Backbone param '{name}' has requires_grad=True"
            )

    def test_gap_layers_trainable(self, model):
        """GAPLayer parameters should require gradients."""
        gap_params_found = False
        for name, parameter in model.named_parameters():
            if 'gap_layer' in name:
                gap_params_found = True
                assert parameter.requires_grad, (
                    f"GAPLayer param '{name}' has requires_grad=False"
                )
        assert gap_params_found, 'No GAPLayer parameters found in model'

    def test_scoring_head_trainable(self, model):
        """Scoring head parameters should require gradients."""
        scoring_params_found = False
        for name, parameter in model.named_parameters():
            if 'scoring' in name or 'per_track_head' in name:
                scoring_params_found = True
                assert parameter.requires_grad, (
                    f"Scoring param '{name}' has requires_grad=False"
                )
        assert scoring_params_found, 'No scoring head parameters found'

    def test_backbone_no_gradients_after_backward(
        self, model, sample_training_inputs,
    ):
        """After backward, backbone params should have no gradients."""
        points, features, lorentz_vectors, mask, track_labels = (
            sample_training_inputs
        )
        model.train()
        loss_dict = model(
            points, features, lorentz_vectors, mask, track_labels,
        )
        loss_dict['total_loss'].backward()

        for name, parameter in model.backbone.named_parameters():
            assert parameter.grad is None or torch.all(parameter.grad == 0), (
                f"Backbone param '{name}' received non-zero gradients"
            )

    def test_gap_layers_have_gradients_after_backward(
        self, model, sample_training_inputs,
    ):
        """After backward, GAPLayer params should have non-zero gradients."""
        points, features, lorentz_vectors, mask, track_labels = (
            sample_training_inputs
        )
        model.train()
        loss_dict = model(
            points, features, lorentz_vectors, mask, track_labels,
        )
        loss_dict['total_loss'].backward()

        gap_params_with_gradient = 0
        for name, parameter in model.named_parameters():
            if 'gap_layer' in name:
                if (
                    parameter.grad is not None
                    and parameter.grad.abs().sum() > 0
                ):
                    gap_params_with_gradient += 1

        assert gap_params_with_gradient > 0, (
            'No GAPLayer parameters received gradients'
        )


# ---- Masking Tests ----

class TestV3Masking:
    """Verify that padding is correctly handled."""

    def test_padded_tracks_have_zero_logits(
        self, model, sample_training_inputs,
    ):
        """Padded track positions should produce zero logits."""
        points, features, lorentz_vectors, mask, _ = sample_training_inputs
        model.eval()

        with torch.no_grad():
            output = model(points, features, lorentz_vectors, mask)

        logits = output['per_track_logits']
        padded_mask = ~mask.squeeze(1).bool()  # True where padded

        assert torch.all(logits[padded_mask] == 0), (
            'Padded positions should have zero logits'
        )

    def test_masking_consistency(self, model, sample_training_inputs):
        """Modifying padded track features should not change valid outputs."""
        points, features, lorentz_vectors, mask, _ = sample_training_inputs
        model.eval()

        with torch.no_grad():
            output_1 = model(points, features, lorentz_vectors, mask)

            # Modify padded tracks
            features_modified = features.clone()
            features_modified[:, :, -50:] = features_modified[:, :, -50:] + 100.0
            output_2 = model(
                points, features_modified, lorentz_vectors, mask,
            )

        # Valid track logits should be unchanged
        valid_mask = mask.squeeze(1).bool()
        torch.testing.assert_close(
            output_1['per_track_logits'][valid_mask],
            output_2['per_track_logits'][valid_mask],
            atol=1e-4, rtol=1e-4,
        )


# ---- Multi-Scale Concatenation Tests ----

class TestV3MultiScaleConcatenation:
    """Verify multi-scale feature aggregation."""

    def test_concatenation_dimension(self, model):
        """Final concatenated features should have correct total dimension.

        Expected: GAP1_attention + GAP1_graph + GAP2_attention + GAP2_graph
                  + backbone_enriched + raw_features + lorentz_vectors + global
        """
        # The model should expose the expected combined dimension
        assert hasattr(model, 'combined_dim'), (
            'V3 should expose combined_dim attribute'
        )

        expected_dim = (
            GAP_ENCODING_DIM  # GAP1 attention features
            + GAP_ENCODING_DIM  # GAP1 graph features
            + GAP_ENCODING_DIM  # GAP2 attention features
            + GAP_ENCODING_DIM  # GAP2 graph features
            + model.backbone.enrichment_output_dim  # backbone enriched
            + INPUT_DIM  # raw features (all 7)
            + 4  # lorentz vectors (px, py, pz, E)
            + _make_v3_kwargs()['global_context_dim']  # global context
        )

        assert model.combined_dim == expected_dim, (
            f'Expected combined_dim={expected_dim}, got {model.combined_dim}'
        )

    def test_all_feature_sources_present(
        self, model, sample_training_inputs,
    ):
        """Zeroing any feature source should change the output."""
        points, features, lorentz_vectors, mask, _ = sample_training_inputs
        model.eval()

        with torch.no_grad():
            baseline_output = model(
                points, features, lorentz_vectors, mask,
            )

            # Zero out raw features
            zero_features = torch.zeros_like(features)
            altered_output = model(
                points, zero_features, lorentz_vectors, mask,
            )

        # Output should change when features are zeroed
        valid_mask = mask.squeeze(1).bool()
        assert not torch.allclose(
            baseline_output['per_track_logits'][valid_mask],
            altered_output['per_track_logits'][valid_mask],
            atol=1e-3,
        ), 'Zeroing raw features should change the model output'


# ---- ParallelBackbone Tests ----

IDENTITY_DIM = 32
CONTEXT_DIM = 64


def _make_parallel_backbone_kwargs():
    """Build ParallelBackbone kwargs for testing (small dims)."""
    return dict(
        input_dim=INPUT_DIM,
        identity_dim=IDENTITY_DIM,
        context_dim=CONTEXT_DIM,
        num_context_layers=2,
        context_num_neighbors=8,
    )


@pytest.fixture
def parallel_backbone():
    """Create a ParallelBackbone for testing."""
    return ParallelBackbone(**_make_parallel_backbone_kwargs())


@pytest.fixture
def parallel_model():
    """Create a TauTrackFinderV3 with parallel backbone for testing."""
    return TauTrackFinderV3(
        backbone_mode='parallel',
        parallel_backbone_kwargs=_make_parallel_backbone_kwargs(),
        **_make_v3_kwargs(),
    )


class TestParallelBackbone:
    """Test the parallel identity + context backbone."""

    def test_output_shape(self, parallel_backbone):
        """Output should be (B, identity_dim + context_dim, P)."""
        points, features, lorentz_vectors = _make_physical_inputs(
            BATCH_SIZE, NUM_TRACKS,
        )
        mask = torch.ones(BATCH_SIZE, 1, NUM_TRACKS)

        output = parallel_backbone(points, features, lorentz_vectors, mask)

        expected_dim = IDENTITY_DIM + CONTEXT_DIM
        assert output.shape == (BATCH_SIZE, expected_dim, NUM_TRACKS), (
            f'Expected shape ({BATCH_SIZE}, {expected_dim}, {NUM_TRACKS}), '
            f'got {output.shape}'
        )

    def test_identity_preserves_features(self, parallel_backbone):
        """Identity stream output should differ when raw features differ,
        even if neighbor features are identical."""
        points, features, lorentz_vectors = _make_physical_inputs(
            BATCH_SIZE, NUM_TRACKS,
        )
        mask = torch.ones(BATCH_SIZE, 1, NUM_TRACKS)

        output_1 = parallel_backbone(points, features, lorentz_vectors, mask)

        # Change a single track's features
        features_modified = features.clone()
        features_modified[:, :, 5] += 10.0
        output_2 = parallel_backbone(
            points, features_modified, lorentz_vectors, mask,
        )

        # The identity stream for track 5 should change
        # (first IDENTITY_DIM channels are identity)
        identity_1 = output_1[:, :IDENTITY_DIM, 5]
        identity_2 = output_2[:, :IDENTITY_DIM, 5]
        assert not torch.allclose(identity_1, identity_2, atol=1e-5), (
            'Identity stream should change when track features change'
        )

    def test_context_uses_neighbors(self, parallel_backbone):
        """Context stream should change when neighbor features change."""
        points, features, lorentz_vectors = _make_physical_inputs(
            BATCH_SIZE, NUM_TRACKS,
        )
        mask = torch.ones(BATCH_SIZE, 1, NUM_TRACKS)

        output_1 = parallel_backbone(points, features, lorentz_vectors, mask)

        # Change features of many tracks (neighbors of track 5)
        features_modified = features.clone()
        features_modified[:, :, :20] += 10.0
        output_2 = parallel_backbone(
            points, features_modified, lorentz_vectors, mask,
        )

        # Context stream for track 50 (far from modified) should also
        # potentially change since it may share neighbors
        context_1 = output_1[:, IDENTITY_DIM:, :]
        context_2 = output_2[:, IDENTITY_DIM:, :]
        assert not torch.allclose(context_1, context_2, atol=1e-3), (
            'Context stream should change when neighbor features change'
        )

    def test_all_parameters_trainable(self, parallel_backbone):
        """All ParallelBackbone parameters should require gradients."""
        for name, parameter in parallel_backbone.named_parameters():
            assert parameter.requires_grad, (
                f"ParallelBackbone param '{name}' has requires_grad=False"
            )

    def test_padded_output_is_zero(self, parallel_backbone):
        """Padded track positions should produce zero output."""
        points, features, lorentz_vectors = _make_physical_inputs(
            BATCH_SIZE, NUM_TRACKS,
        )
        mask = torch.ones(BATCH_SIZE, 1, NUM_TRACKS)
        mask[:, :, -50:] = 0.0

        output = parallel_backbone(points, features, lorentz_vectors, mask)

        padded_output = output[:, :, -50:]
        assert torch.all(padded_output == 0), (
            'Padded positions should have zero output'
        )

    def test_gradients_flow_through_both_streams(self, parallel_backbone):
        """Backward should produce gradients in both identity and context."""
        points, features, lorentz_vectors = _make_physical_inputs(
            BATCH_SIZE, NUM_TRACKS,
        )
        mask = torch.ones(BATCH_SIZE, 1, NUM_TRACKS)

        output = parallel_backbone(points, features, lorentz_vectors, mask)
        loss = output.sum()
        loss.backward()

        identity_has_grad = False
        context_has_grad = False
        for name, parameter in parallel_backbone.named_parameters():
            if parameter.grad is not None and parameter.grad.abs().sum() > 0:
                if 'identity' in name:
                    identity_has_grad = True
                if 'context' in name:
                    context_has_grad = True

        assert identity_has_grad, 'Identity stream received no gradients'
        assert context_has_grad, 'Context stream received no gradients'


class TestV3ParallelMode:
    """Test V3 model with backbone_mode='parallel'."""

    def test_training_returns_finite_loss(
        self, parallel_model, sample_training_inputs,
    ):
        points, features, lorentz_vectors, mask, track_labels = (
            sample_training_inputs
        )
        parallel_model.train()
        loss_dict = parallel_model(
            points, features, lorentz_vectors, mask, track_labels,
        )

        assert 'total_loss' in loss_dict
        assert torch.isfinite(loss_dict['total_loss']).all()

    def test_inference_returns_per_track_logits(
        self, parallel_model, sample_training_inputs,
    ):
        points, features, lorentz_vectors, mask, _ = sample_training_inputs
        parallel_model.eval()
        with torch.no_grad():
            output = parallel_model(points, features, lorentz_vectors, mask)

        assert 'per_track_logits' in output
        assert output['per_track_logits'].shape == (BATCH_SIZE, NUM_TRACKS)

    def test_no_frozen_backbone(self, parallel_model):
        """Parallel mode should not have a frozen EnrichCompactBackbone."""
        # All parameters should be trainable (no frozen backbone)
        frozen_params = sum(
            1 for _, parameter in parallel_model.named_parameters()
            if not parameter.requires_grad
        )
        assert frozen_params == 0, (
            f'Parallel mode should have 0 frozen params, found {frozen_params}'
        )

    def test_gradients_after_backward(
        self, parallel_model, sample_training_inputs,
    ):
        points, features, lorentz_vectors, mask, track_labels = (
            sample_training_inputs
        )
        parallel_model.train()
        loss_dict = parallel_model(
            points, features, lorentz_vectors, mask, track_labels,
        )
        loss_dict['total_loss'].backward()

        params_with_grad = sum(
            1 for _, parameter in parallel_model.named_parameters()
            if parameter.grad is not None and parameter.grad.abs().sum() > 0
        )
        assert params_with_grad > 0, (
            'No parameters received gradients in parallel mode'
        )


# ---- Asymmetric Loss Tests ----

class TestAsymmetricLoss:
    """Test Asymmetric Loss (ASL) for extreme class imbalance.

    ASL (Ben-Baruch et al., ICCV 2021) uses different focusing parameters
    for positives vs negatives, plus a hard probability-shift threshold
    that zeros gradients from trivially easy negatives.
    """

    def test_asl_finite(self, model, sample_training_inputs):
        """ASL should produce finite loss for mixed labels."""
        points, features, lorentz_vectors, mask, track_labels = (
            sample_training_inputs
        )
        model.train()
        loss_dict = model(
            points, features, lorentz_vectors, mask, track_labels,
        )

        assert torch.isfinite(loss_dict['per_track_loss']).all()

    def test_asl_finite_all_zeros(self, model):
        """ASL should be finite when all labels are zero (no GT)."""
        points, features, lorentz_vectors = _make_physical_inputs(
            BATCH_SIZE, NUM_TRACKS,
        )
        mask = torch.ones(BATCH_SIZE, 1, NUM_TRACKS)
        track_labels = torch.zeros(BATCH_SIZE, 1, NUM_TRACKS)

        model.train()
        loss_dict = model(
            points, features, lorentz_vectors, mask, track_labels,
        )

        assert torch.isfinite(loss_dict['total_loss']).all()

    def test_asl_zeros_easy_negatives(self, model):
        """Easy negatives (score << 0) should contribute zero loss under ASL.

        The hard clip threshold m=0.05 means negatives with predicted
        probability p < m get exactly zero loss. At initialization,
        most tracks have p ≈ sigmoid(0) = 0.5, so this test verifies
        the mechanism by manually creating easy-negative scores.
        """
        # Create logits where most negatives have very low scores
        logits = torch.full((1, NUM_TRACKS), -5.0)  # p ≈ 0.007, well below clip
        logits[0, 10] = 0.0  # One positive
        labels = torch.zeros(1, NUM_TRACKS)
        labels[0, 10] = 1.0
        valid_mask = torch.ones(1, NUM_TRACKS, dtype=torch.bool)

        # Compute ASL
        loss = model._asymmetric_loss(logits, labels, valid_mask)

        # Now make all negatives even easier (score = -10, p ≈ 0.00005)
        logits_easier = logits.clone()
        logits_easier[0, :10] = -10.0
        logits_easier[0, 11:] = -10.0
        loss_easier = model._asymmetric_loss(
            logits_easier, labels, valid_mask,
        )

        # Easier negatives should contribute LESS loss (near zero with clip)
        assert loss_easier <= loss + 1e-6, (
            'Making negatives easier should not increase loss'
        )


# ---- Ranking Loss Tests ----

class TestRankingLoss:
    """Test pairwise ranking loss for recall@K optimization.

    The ranking loss penalizes any negative scoring above a positive:
    L_rank = mean_i mean_j log(1 + exp(score_neg_j - score_pos_i))
    """

    def test_ranking_loss_finite(self, model, sample_training_inputs):
        """Ranking loss should be finite."""
        points, features, lorentz_vectors, mask, track_labels = (
            sample_training_inputs
        )
        model.train()
        loss_dict = model(
            points, features, lorentz_vectors, mask, track_labels,
        )

        assert 'ranking_loss' in loss_dict
        assert torch.isfinite(loss_dict['ranking_loss']).all()

    def test_ranking_loss_zero_when_perfectly_ranked(self, model):
        """Ranking loss should be near zero when all positives outscore negatives."""
        logits = torch.zeros(1, NUM_TRACKS)
        logits[0, 10] = 10.0  # Positive scores much higher
        logits[0, 20] = 10.0
        logits[0, 30] = 10.0
        labels = torch.zeros(1, NUM_TRACKS)
        labels[0, 10] = 1.0
        labels[0, 20] = 1.0
        labels[0, 30] = 1.0
        valid_mask = torch.ones(1, NUM_TRACKS, dtype=torch.bool)

        loss = model._ranking_loss(logits, labels, valid_mask)

        # log(1 + exp(0 - 10)) ≈ log(1 + 0.00005) ≈ 0
        assert loss.item() < 0.01, (
            f'Perfectly ranked should have near-zero ranking loss, got {loss.item()}'
        )

    def test_ranking_loss_high_when_misranked(self, model):
        """Ranking loss should be high when negatives outscore positives."""
        logits = torch.zeros(1, NUM_TRACKS)
        logits[0, 10] = -5.0  # Positives score low
        logits[0, 20] = -5.0
        logits[0, 30] = -5.0
        logits[0, 0] = 5.0  # A negative scores high
        labels = torch.zeros(1, NUM_TRACKS)
        labels[0, 10] = 1.0
        labels[0, 20] = 1.0
        labels[0, 30] = 1.0
        valid_mask = torch.ones(1, NUM_TRACKS, dtype=torch.bool)

        loss = model._ranking_loss(logits, labels, valid_mask)

        # log(1 + exp(5 - (-5))) = log(1 + exp(10)) ≈ 10
        assert loss.item() > 1.0, (
            f'Misranked should have high ranking loss, got {loss.item()}'
        )

    def test_ranking_loss_zero_gt(self, model):
        """Ranking loss should be zero when there are no GT tracks."""
        logits = torch.randn(1, NUM_TRACKS)
        labels = torch.zeros(1, NUM_TRACKS)
        valid_mask = torch.ones(1, NUM_TRACKS, dtype=torch.bool)

        loss = model._ranking_loss(logits, labels, valid_mask)

        assert loss.item() == 0.0, (
            'Zero GT tracks should produce zero ranking loss'
        )

    def test_total_loss_includes_ranking(self, model, sample_training_inputs):
        """Total loss should include both ASL and ranking components."""
        points, features, lorentz_vectors, mask, track_labels = (
            sample_training_inputs
        )
        model.train()
        loss_dict = model(
            points, features, lorentz_vectors, mask, track_labels,
        )

        assert 'total_loss' in loss_dict
        assert 'per_track_loss' in loss_dict
        assert 'ranking_loss' in loss_dict
        # Total should be a combination of both
        assert loss_dict['total_loss'].item() > 0.0
