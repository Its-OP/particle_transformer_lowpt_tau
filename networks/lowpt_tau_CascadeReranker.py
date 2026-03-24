"""Network wrapper for CascadeModel (Stage 1 pre-filter → Stage 2 reranker).

Builds a CascadeModel from:
    - Stage 1: frozen TrackPreFilter loaded from checkpoint
    - Stage 2: pluggable reranker architecture (placeholder until Track A/B/C)

Usage:
    python train_cascade.py --network networks/lowpt_tau_CascadeReranker.py \
        --stage1-checkpoint models/prefilter_best.pt --top-k1 600
"""
import torch
import torch.nn as nn

from weaver.nn.model.CascadeModel import CascadeModel
from weaver.nn.model.TrackPreFilter import TrackPreFilter
from weaver.utils.logger import _logger


class PlaceholderStage2(nn.Module):
    """Minimal Stage 2 reranker: MLP on per-track features + stage1_scores.

    This is a temporary placeholder until CascadeReranker (Track A) is
    implemented. It verifies the cascade pipeline works end-to-end.

    Architecture: cat(features, stage1_score) → MLP → per-track score.
    Trained with ranking loss (same as TrackPreFilter).
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 128,
        ranking_num_samples: int = 50,
        ranking_temperature: float = 1.0,
    ):
        super().__init__()
        self.ranking_num_samples = ranking_num_samples
        self.ranking_temperature = ranking_temperature

        # +1 for stage1_score channel
        self.mlp = nn.Sequential(
            nn.Conv1d(input_dim + 1, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        stage1_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Score each track in the filtered set.

        Args:
            points: (B, 2, K1) coordinates.
            features: (B, input_dim, K1) per-track features.
            lorentz_vectors: (B, 4, K1) 4-vectors.
            mask: (B, 1, K1) validity mask.
            stage1_scores: (B, K1) scores from Stage 1.

        Returns:
            scores: (B, K1) per-track scores.
        """
        valid_mask = mask.squeeze(1).bool()

        # Concatenate stage1 scores as an extra feature channel
        stage1_channel = stage1_scores.unsqueeze(1)  # (B, 1, K1)
        combined = torch.cat([features, stage1_channel], dim=1)

        scores = self.mlp(combined * mask.float()).squeeze(1)
        scores = scores.masked_fill(~valid_mask, float('-inf'))
        return scores

    def compute_loss(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        track_labels: torch.Tensor,
        stage1_scores: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute ranking loss on filtered tracks.

        Same temperature-scaled pairwise ranking loss as TrackPreFilter:
            L = T × softplus((s_neg - s_pos) / T)
        """
        scores = self.forward(
            points, features, lorentz_vectors, mask, stage1_scores,
        )
        valid_mask = mask.squeeze(1).bool()
        labels = track_labels.squeeze(1)[:, :scores.shape[1]] * valid_mask.float()

        batch_size = scores.shape[0]
        temperature = self.ranking_temperature
        event_losses = []

        for event_index in range(batch_size):
            event_scores = scores[event_index]
            event_labels = labels[event_index]
            event_valid = valid_mask[event_index]

            positive_indices = (
                (event_labels == 1.0) & event_valid
            ).nonzero(as_tuple=True)[0]
            negative_indices = (
                (event_labels == 0.0) & event_valid
            ).nonzero(as_tuple=True)[0]

            if len(positive_indices) == 0 or len(negative_indices) == 0:
                continue

            num_samples = min(self.ranking_num_samples, len(negative_indices))
            sample_idx = torch.randint(
                0, len(negative_indices), (num_samples,),
                device=scores.device,
            )
            sampled_negatives = negative_indices[sample_idx]

            positive_scores = event_scores[positive_indices].unsqueeze(1)
            negative_scores = event_scores[sampled_negatives].unsqueeze(0)

            # L = T × log(1 + exp((s_neg - s_pos) / T))
            scaled_margin = (negative_scores - positive_scores) / temperature
            pairwise_loss = temperature * torch.nn.functional.softplus(scaled_margin)
            event_losses.append(pairwise_loss.mean())

        if not event_losses:
            ranking_loss = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
        else:
            ranking_loss = torch.stack(event_losses).mean()

        return {
            'total_loss': ranking_loss,
            'ranking_loss': ranking_loss,
            '_scores': scores,
        }


def get_model(data_config, **kwargs):
    """Build CascadeModel with frozen Stage 1 + Stage 2 reranker.

    Required kwargs (passed via train_cascade.py):
        stage1_checkpoint: Path to trained Stage 1 checkpoint.
        top_k1: Number of tracks to pass from Stage 1 to Stage 2.

    Optional kwargs:
        stage2_hidden_dim: Hidden dimension for Stage 2 MLP (default: 128).
    """
    # Pop cascade-specific args
    stage1_checkpoint = kwargs.pop('stage1_checkpoint', None)
    top_k1 = kwargs.pop('top_k1', 600)
    stage2_hidden_dim = kwargs.pop('stage2_hidden_dim', 128)

    # Pop unused args from other heads
    for unused_arg in [
        'pretrained_backbone_path', 'backbone_mode',
        'mask_ce_loss_weight', 'confidence_loss_weight',
        'no_object_weight', 'num_decoder_layers', 'num_queries',
        'focal_bce_weight', 'potential_loss_weight',
        'beta_loss_weight', 'clustering_dim',
        'per_track_loss_weight', 'refinement_loss_weight',
        'num_enrichment_layers',
    ]:
        kwargs.pop(unused_arg, None)

    input_dim = len(data_config.input_dicts['pf_features'])

    # ---- Stage 1: load frozen pre-filter ----
    if stage1_checkpoint is None:
        raise ValueError(
            'stage1_checkpoint is required. '
            'Pass --stage1-checkpoint to train_cascade.py'
        )

    _logger.info(f'Loading Stage 1 from: {stage1_checkpoint}')
    checkpoint = torch.load(stage1_checkpoint, map_location='cpu', weights_only=False)

    # Infer Stage 1 config from checkpoint
    stage1_state = checkpoint.get('model_state_dict', checkpoint)
    stage1 = TrackPreFilter(mode='mlp', input_dim=input_dim, hidden_dim=192,
                            num_message_rounds=2, num_neighbors=16)
    stage1.load_state_dict(stage1_state)
    _logger.info('Stage 1 loaded successfully')

    stage1_params = sum(p.numel() for p in stage1.parameters())
    _logger.info(f'Stage 1: {stage1_params:,} params (frozen)')

    # ---- Stage 2: placeholder reranker ----
    stage2 = PlaceholderStage2(
        input_dim=input_dim,
        hidden_dim=stage2_hidden_dim,
    )
    stage2_params = sum(p.numel() for p in stage2.parameters())
    _logger.info(f'Stage 2 (placeholder): {stage2_params:,} params (trainable)')

    # ---- Cascade ----
    model = CascadeModel(stage1=stage1, stage2=stage2, top_k1=top_k1)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _logger.info(f'Cascade total: {total_params:,} params | Trainable: {trainable_params:,}')

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {
            key: ((1,) + shape[1:])
            for key, shape in data_config.input_shapes.items()
        },
        'output_names': ['loss'],
        'dynamic_axes': {
            **{
                key: {0: 'N', 2: 'n_' + key.split('_')[0]}
                for key in data_config.input_names
            },
            **{'loss': {0: 'N'}},
        },
    }

    return model, model_info
