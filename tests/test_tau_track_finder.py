"""Unit tests for TauTrackFinder (top-level module: backbone + head + loss).

Tests cover:
    - Forward pass smoke test: random input → finite loss, correct shapes
    - Hungarian matching: cost matrix shape, matched indices validity
    - Loss components: mask CE loss and confidence BCE are finite and non-negative
    - GT extraction: correct extraction of up to 6 tau-track indices from labels
    - Zero GT event: all confidence targets = 0, no mask loss
    - Backbone freezing: backbone params have requires_grad=False
    - Architecture: decoder post-norm, mask scoring, confidence head dims
    - Configuration: no_object_weight configurability
    - Checkpoint management: rolling top-K best checkpoint pruning
"""
import os
import tempfile

import pytest
import torch

from utils.training_utils import CheckpointManager
from weaver.nn.model.TauTrackFinder import TauTrackFinder


# ---- Fixtures ----

BATCH_SIZE = 4
NUM_TRACKS = 200  # Smaller than real ~1130 for test speed
INPUT_DIM = 7
NUM_QUERIES = 15
MAX_GT_TRACKS = 6


def _make_backbone_kwargs(input_dim=INPUT_DIM):
    """Build backbone kwargs matching the pretraining config."""
    return dict(
        input_dim=input_dim,
        enrichment_kwargs=dict(
            node_dim=32,
            edge_dim=8,
            num_neighbors=16,  # Smaller than real 32 for speed
            edge_aggregation='attn8',
            layer_params=[
                # Single layer for test speed
                (16, 64, [(4, 1), (2, 1)], 32),
            ],
        ),
        compaction_kwargs=dict(
            stage_output_points=[64, 32],  # Smaller for speed
            stage_output_channels=[64, 64],
            stage_num_neighbors=[8, 8],
        ),
    )


def _make_decoder_kwargs():
    """Build decoder kwargs for testing."""
    return dict(
        num_queries=NUM_QUERIES,
        max_gt_tracks=MAX_GT_TRACKS,
        decoder_dim=64,  # Smaller for speed
        mask_dim=32,
        num_heads=4,
        num_decoder_layers=1,
        dropout=0.0,
    )


@pytest.fixture
def model():
    """Create a TauTrackFinder with small architecture for testing."""
    return TauTrackFinder(
        backbone_kwargs=_make_backbone_kwargs(),
        decoder_kwargs=_make_decoder_kwargs(),
    )


def _make_physical_inputs(batch_size, num_tracks, input_dim=INPUT_DIM, seed=42):
    """Create physically sensible synthetic inputs.

    Lorentz vectors must be physical (positive energy, E >= |p|) because
    the backbone computes pairwise_lv_fts() with log(kT), log(ΔR), etc.
    Random negative values would produce NaN from log of negatives.

    Uses fixed seed for test reproducibility.
    """
    generator = torch.Generator().manual_seed(seed)

    # Points: (η, φ) coordinates in realistic ranges
    eta = torch.randn(batch_size, 1, num_tracks, generator=generator) * 1.5
    phi = torch.rand(batch_size, 1, num_tracks, generator=generator) * 2 * 3.14159 - 3.14159
    points = torch.cat([eta, phi], dim=1)

    # Features: standardized (zero-mean, unit-variance)
    features = torch.randn(batch_size, input_dim, num_tracks, generator=generator)

    # Lorentz vectors: physical 4-momenta (px, py, pz, E)
    pt = torch.rand(batch_size, 1, num_tracks, generator=generator) * 5 + 0.5  # pT ∈ [0.5, 5.5] GeV
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    pion_mass = 0.13957
    energy = torch.sqrt(px**2 + py**2 + pz**2 + pion_mass**2)
    lorentz_vectors = torch.cat([px, py, pz, energy], dim=1)

    return points, features, lorentz_vectors


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


# ---- Forward Pass Smoke Tests ----

class TestForwardPass:
    """Verify the complete forward pass produces valid outputs."""

    def test_training_returns_finite_loss(self, model, sample_training_inputs):
        points, features, lorentz_vectors, mask, track_labels = sample_training_inputs
        model.train()
        loss_dict = model(points, features, lorentz_vectors, mask, track_labels)

        assert 'total_loss' in loss_dict
        assert torch.isfinite(loss_dict['total_loss']).all(), (
            f"Total loss is not finite: {loss_dict['total_loss']}"
        )

    def test_training_loss_is_scalar(self, model, sample_training_inputs):
        points, features, lorentz_vectors, mask, track_labels = sample_training_inputs
        model.train()
        loss_dict = model(points, features, lorentz_vectors, mask, track_labels)

        # Loss should be a scalar (0-dim) tensor
        assert loss_dict['total_loss'].dim() == 0

    def test_inference_returns_logits(self, model, sample_training_inputs):
        points, features, lorentz_vectors, mask, _ = sample_training_inputs
        model.eval()
        with torch.no_grad():
            output = model(points, features, lorentz_vectors, mask)

        assert 'mask_logits' in output
        assert 'confidence_logits' in output
        assert output['mask_logits'].shape == (
            BATCH_SIZE, NUM_QUERIES, NUM_TRACKS,
        )
        assert output['confidence_logits'].shape == (BATCH_SIZE, NUM_QUERIES)

    def test_loss_is_non_negative(self, model, sample_training_inputs):
        points, features, lorentz_vectors, mask, track_labels = sample_training_inputs
        model.train()
        loss_dict = model(points, features, lorentz_vectors, mask, track_labels)

        assert loss_dict['total_loss'].item() >= 0.0
        assert loss_dict['mask_ce_loss'].item() >= 0.0
        assert loss_dict['confidence_loss'].item() >= 0.0


# ---- Loss Component Tests ----

class TestLossComponents:
    """Verify individual loss components."""

    def test_all_loss_components_present(self, model, sample_training_inputs):
        points, features, lorentz_vectors, mask, track_labels = sample_training_inputs
        model.train()
        loss_dict = model(points, features, lorentz_vectors, mask, track_labels)

        assert 'mask_ce_loss' in loss_dict
        assert 'confidence_loss' in loss_dict
        assert 'per_track_loss' in loss_dict
        assert 'total_loss' in loss_dict

    def test_loss_components_are_finite(self, model, sample_training_inputs):
        points, features, lorentz_vectors, mask, track_labels = sample_training_inputs
        model.train()
        loss_dict = model(points, features, lorentz_vectors, mask, track_labels)

        for key, value in loss_dict.items():
            assert torch.isfinite(value).all(), f"{key} is not finite: {value}"

    def test_total_loss_is_weighted_sum(self, model, sample_training_inputs):
        points, features, lorentz_vectors, mask, track_labels = sample_training_inputs
        model.train()
        loss_dict = model(points, features, lorentz_vectors, mask, track_labels)

        expected_total = (
            model.mask_ce_loss_weight * loss_dict['mask_ce_loss']
            + model.confidence_loss_weight * loss_dict['confidence_loss']
            + model.per_track_loss_weight * loss_dict['per_track_loss']
        )
        torch.testing.assert_close(
            loss_dict['total_loss'], expected_total, rtol=1e-4, atol=1e-6,
        )


# ---- Ground Truth Extraction Tests ----

class TestGroundTruthExtraction:
    """Verify GT track index extraction from labels."""

    def test_extract_correct_count(self, model):
        """Should extract the right number of GT indices per event."""
        # 3 tau tracks at positions 5, 15, 25
        labels = torch.zeros(2, 1, 100)
        labels[0, 0, 5] = 1.0
        labels[0, 0, 15] = 1.0
        labels[0, 0, 25] = 1.0
        # 1 tau track at position 40
        labels[1, 0, 40] = 1.0

        mask = torch.ones(2, 1, 100)

        gt_indices, gt_count = model._extract_ground_truth_indices(labels, mask)

        assert gt_count[0].item() == 3
        assert gt_count[1].item() == 1
        assert gt_indices.shape == (2, MAX_GT_TRACKS)

    def test_max_6_gt_tracks(self, model):
        """Should clamp to max 6 even if more are labeled."""
        labels = torch.zeros(1, 1, 100)
        # 8 tau tracks (more than max 6)
        for position in [5, 10, 15, 20, 25, 30, 35, 40]:
            labels[0, 0, position] = 1.0

        mask = torch.ones(1, 1, 100)
        gt_indices, gt_count = model._extract_ground_truth_indices(labels, mask)

        assert gt_count[0].item() == 6  # Clamped to max
        assert gt_indices.shape == (1, MAX_GT_TRACKS)

    def test_zero_gt_tracks(self, model):
        """Should handle events with no tau tracks."""
        labels = torch.zeros(1, 1, 100)
        mask = torch.ones(1, 1, 100)

        gt_indices, gt_count = model._extract_ground_truth_indices(labels, mask)

        assert gt_count[0].item() == 0

    def test_padding_not_counted_as_gt(self, model):
        """GT labels on padded tracks should be ignored."""
        labels = torch.zeros(1, 1, 100)
        labels[0, 0, 95] = 1.0  # Labeled track in padding zone
        mask = torch.ones(1, 1, 100)
        mask[0, 0, 90:] = 0.0  # Last 10 tracks are padding

        gt_indices, gt_count = model._extract_ground_truth_indices(labels, mask)

        assert gt_count[0].item() == 0  # Padded label should be ignored


# ---- Zero GT Event Tests ----

class TestZeroGroundTruth:
    """Verify behavior when an event has no GT tracks."""

    def test_zero_gt_loss_is_finite(self, model):
        """Loss should be finite even with 0 GT tracks."""
        points, features, lorentz_vectors = _make_physical_inputs(1, NUM_TRACKS)
        mask = torch.ones(1, 1, NUM_TRACKS)
        track_labels = torch.zeros(1, 1, NUM_TRACKS)  # No GT tracks

        model.train()
        loss_dict = model(points, features, lorentz_vectors, mask, track_labels)

        assert torch.isfinite(loss_dict['total_loss']).all()
        # Mask CE loss should be 0 (no matched queries)
        assert loss_dict['mask_ce_loss'].item() == 0.0
        # Confidence loss should still be computed (all targets = 0)
        assert loss_dict['confidence_loss'].item() >= 0.0


# ---- Backbone Freezing Tests ----

class TestBackboneFreezing:
    """Verify backbone parameter freezing."""

    def test_backbone_params_frozen_by_default(self, model):
        """Backbone parameters should not require gradients."""
        for name, param in model.backbone.named_parameters():
            assert not param.requires_grad, (
                f"Backbone param '{name}' has requires_grad=True"
            )

    def test_head_params_require_grad(self, model):
        """Head parameters should require gradients."""
        for name, param in model.head.named_parameters():
            assert param.requires_grad, (
                f"Head param '{name}' has requires_grad=False"
            )

    def test_backbone_no_gradients_after_backward(
        self, model, sample_training_inputs
    ):
        """After backward, backbone params should have no gradients."""
        points, features, lorentz_vectors, mask, track_labels = sample_training_inputs
        model.train()
        loss_dict = model(points, features, lorentz_vectors, mask, track_labels)
        loss_dict['total_loss'].backward()

        for name, param in model.backbone.named_parameters():
            assert param.grad is None or torch.all(param.grad == 0), (
                f"Backbone param '{name}' received non-zero gradients"
            )

    def test_head_has_gradients_after_backward(
        self, model, sample_training_inputs
    ):
        """After backward, head params should have non-zero gradients."""
        points, features, lorentz_vectors, mask, track_labels = sample_training_inputs
        model.train()
        loss_dict = model(points, features, lorentz_vectors, mask, track_labels)
        loss_dict['total_loss'].backward()

        params_with_grad = 0
        for name, param in model.head.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                params_with_grad += 1

        assert params_with_grad > 0, "No head parameters received gradients"


# ---- Architecture Tests ----

class TestArchitecture:
    """Verify architecture changes for decoder and confidence head."""

    def test_decoder_uses_post_norm(self, model):
        """Decoder should use post-norm (norm_first=False) to prevent query
        norm explosion that makes cross-attention contributions negligible.

        Original DETR (Carion et al., ECCV 2020) uses post-norm. Pre-norm
        caused query norms to grow from ~1 to ~598 across 6 layers, making
        decoded queries input-independent (cosine sim = 1.0000 across events).
        """
        # DecoderLayer uses explicit post-norm (LayerNorm after residual).
        decoder_layer = model.head.decoder_layers[0]
        assert hasattr(decoder_layer, 'norm_self_attention'), (
            "Decoder layer should have LayerNorm modules for post-norm"
        )

    def test_no_encoder(self, model):
        """Simplified model has no compact token encoder."""
        assert not hasattr(model.head, 'transformer_encoder'), (
            "Simplified head should not have a compact token encoder"
        )

    def test_query_scoring_mlp_exists(self, model):
        """Head should have a query_scoring_mlp for mask dot-product scoring."""
        assert hasattr(model.head, 'query_scoring_mlp'), (
            "TauTrackFinderHead should have query_scoring_mlp"
        )

    def test_confidence_head_input_dim(self, model):
        """Confidence head input should be decoder_dim (query only)."""
        first_linear = model.head.confidence_head[0]
        expected_input_dim = model.head.decoder_dim
        assert first_linear.in_features == expected_input_dim, (
            f"Confidence head input dim should be {expected_input_dim} "
            f"(decoder_dim), got {first_linear.in_features}"
        )

    def test_gradient_flow_through_mask_logits(self, model, sample_training_inputs):
        """Total loss should flow gradients through the mask scoring path."""
        points, features, lorentz_vectors, mask, track_labels = sample_training_inputs
        model.train()
        loss_dict = model(points, features, lorentz_vectors, mask, track_labels)
        loss_dict['total_loss'].backward()

        # query_scoring_mlp should receive gradients from mask CE loss
        has_gradient = False
        for param in model.head.query_scoring_mlp.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradient = True
                break
        assert has_gradient, (
            "query_scoring_mlp should receive gradients from "
            "mask cross-entropy loss backpropagation"
        )


# ---- No-Object Weight (eos_coef) Configuration Tests ----

class TestNoObjectWeight:
    """Verify eos_coef (no-object weight) configurability."""

    def test_default_no_object_weight(self):
        """Default no_object_weight should be 0.4."""
        default_model = TauTrackFinder(
            backbone_kwargs=_make_backbone_kwargs(),
            decoder_kwargs=_make_decoder_kwargs(),
        )
        assert default_model.no_object_weight == 0.4

    def test_custom_no_object_weight(self):
        """no_object_weight should be configurable via constructor."""
        custom_model = TauTrackFinder(
            backbone_kwargs=_make_backbone_kwargs(),
            decoder_kwargs=_make_decoder_kwargs(),
            no_object_weight=0.5,
        )
        assert custom_model.no_object_weight == 0.5

    def test_no_object_weight_affects_loss(self):
        """Different eos_coef values should produce different confidence losses
        for the same inputs, because unmatched queries are weighted differently.
        """
        points, features, lorentz_vectors = _make_physical_inputs(
            2, NUM_TRACKS, seed=42,
        )
        mask = torch.ones(2, 1, NUM_TRACKS)
        track_labels = torch.zeros(2, 1, NUM_TRACKS)
        track_labels[0, 0, 10] = 1.0
        track_labels[1, 0, 20] = 1.0

        model_low_eos = TauTrackFinder(
            backbone_kwargs=_make_backbone_kwargs(),
            decoder_kwargs=_make_decoder_kwargs(),
            no_object_weight=0.1,
        )
        model_high_eos = TauTrackFinder(
            backbone_kwargs=_make_backbone_kwargs(),
            decoder_kwargs=_make_decoder_kwargs(),
            no_object_weight=1.0,
        )

        # Copy weights from low to high so only eos_coef differs
        model_high_eos.load_state_dict(model_low_eos.state_dict())

        model_low_eos.train()
        model_high_eos.train()

        loss_low = model_low_eos(
            points, features, lorentz_vectors, mask, track_labels,
        )
        loss_high = model_high_eos(
            points, features, lorentz_vectors, mask, track_labels,
        )

        # With different eos_coef, the confidence BCE loss should differ
        # because unmatched queries (the majority) are weighted differently
        assert loss_low['confidence_loss'].item() != pytest.approx(
            loss_high['confidence_loss'].item(), abs=1e-6,
        ), (
            "Different no_object_weight values should produce different "
            "confidence losses"
        )


# ---- Checkpoint Management Tests ----

class TestCheckpointManager:
    """Verify rolling top-K best checkpoint pruning."""

    def test_keeps_top_k_checkpoints(self):
        """Should delete checkpoints beyond top K by criterion_value."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            manager = CheckpointManager(
                checkpoints_directory=temporary_directory,
                keep_best_k=3,
                criterion_mode='min',
            )

            # Save 5 checkpoints with decreasing quality (higher loss)
            for epoch in range(1, 6):
                checkpoint_data = {'epoch': epoch, 'val_loss': epoch * 0.1}
                manager.save_checkpoint(
                    checkpoint_data,
                    epoch=epoch,
                    criterion_value=epoch * 0.1,
                    is_best=(epoch == 1),
                )

            # Should keep top 3 (epochs 1, 2, 3 with losses 0.1, 0.2, 0.3)
            remaining_files = [
                filename for filename in os.listdir(temporary_directory)
                if filename.startswith('checkpoint_epoch_')
            ]
            assert len(remaining_files) == 3

            # Best 3 should exist
            assert os.path.exists(os.path.join(
                temporary_directory, 'checkpoint_epoch_1.pt',
            ))
            assert os.path.exists(os.path.join(
                temporary_directory, 'checkpoint_epoch_2.pt',
            ))
            assert os.path.exists(os.path.join(
                temporary_directory, 'checkpoint_epoch_3.pt',
            ))

            # Worst 2 should be deleted
            assert not os.path.exists(os.path.join(
                temporary_directory, 'checkpoint_epoch_4.pt',
            ))
            assert not os.path.exists(os.path.join(
                temporary_directory, 'checkpoint_epoch_5.pt',
            ))

    def test_best_model_always_preserved(self):
        """best_model.pt should always exist even after pruning."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            manager = CheckpointManager(
                checkpoints_directory=temporary_directory,
                keep_best_k=2,
                criterion_mode='min',
            )

            # First checkpoint is best
            manager.save_checkpoint(
                {'epoch': 1}, epoch=1, criterion_value=0.5, is_best=True,
            )
            # Second is worse
            manager.save_checkpoint(
                {'epoch': 2}, epoch=2, criterion_value=0.8, is_best=False,
            )
            # Third is better than both — new best
            manager.save_checkpoint(
                {'epoch': 3}, epoch=3, criterion_value=0.3, is_best=True,
            )

            # best_model.pt should exist (never pruned)
            assert os.path.exists(os.path.join(
                temporary_directory, 'best_model.pt',
            ))

    def test_keep_best_k_zero_disables_pruning(self):
        """Setting keep_best_k=0 should keep all checkpoints."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            manager = CheckpointManager(
                checkpoints_directory=temporary_directory,
                keep_best_k=0,
            )

            for epoch in range(1, 8):
                manager.save_checkpoint(
                    {'epoch': epoch},
                    epoch=epoch,
                    criterion_value=epoch * 0.1,
                    is_best=(epoch == 1),
                )

            # All 7 checkpoint files should remain
            remaining_files = [
                filename for filename in os.listdir(temporary_directory)
                if filename.startswith('checkpoint_epoch_')
            ]
            assert len(remaining_files) == 7

    def test_later_better_checkpoint_evicts_earlier_worse(self):
        """A later checkpoint with better loss should evict the worst tracked."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            manager = CheckpointManager(
                checkpoints_directory=temporary_directory,
                keep_best_k=2,
                criterion_mode='min',
            )

            # Save 3 checkpoints: 0.5, 0.8, 0.3
            manager.save_checkpoint(
                {'epoch': 1}, epoch=1, criterion_value=0.5, is_best=True,
            )
            manager.save_checkpoint(
                {'epoch': 2}, epoch=2, criterion_value=0.8, is_best=False,
            )
            manager.save_checkpoint(
                {'epoch': 3}, epoch=3, criterion_value=0.3, is_best=True,
            )

            # Top 2 are: epoch 3 (0.3) and epoch 1 (0.5)
            # Epoch 2 (0.8) should be deleted
            assert os.path.exists(os.path.join(
                temporary_directory, 'checkpoint_epoch_3.pt',
            ))
            assert os.path.exists(os.path.join(
                temporary_directory, 'checkpoint_epoch_1.pt',
            ))
            assert not os.path.exists(os.path.join(
                temporary_directory, 'checkpoint_epoch_2.pt',
            ))
