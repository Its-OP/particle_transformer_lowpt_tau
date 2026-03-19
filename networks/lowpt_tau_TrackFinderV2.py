"""Network wrapper for V2 tau-origin pion track finder.

Provides get_model() to construct TauTrackFinderV2 with pretrained backbone.
The backbone is frozen by default -- only the scoring and refinement heads
are trained.

V2 replaces the DETR decoder with:
    M1 -- Neighbor-aware per-track scoring (kNN + max-pool + MLP)
    M2 -- Self-attention refinement on top-K candidates
    M5 -- Skip-connected displacement features (dxy_significance, pT)
"""

import torch
from weaver.nn.model.TauTrackFinderV2 import TauTrackFinderV2
from weaver.utils.logger import _logger


def get_model(data_config, **kwargs):
    """Build TauTrackFinderV2 with pretrained backbone.

    Kwargs consumed (popped from kwargs):
        pretrained_backbone_path: Path to pretrained backbone checkpoint.
        num_enrichment_layers: Number of enrichment layers (default: 5).
        per_track_loss_weight: Weight for per-track focal BCE (default: 1.0).
        refinement_loss_weight: Weight for refinement focal BCE (default: 1.0).
    """
    pretrained_backbone_path = kwargs.pop('pretrained_backbone_path', None)
    num_enrichment_layers = kwargs.pop('num_enrichment_layers', 5)

    # Loss weights
    per_track_loss_weight = kwargs.pop('per_track_loss_weight', 1.0)
    refinement_loss_weight = kwargs.pop('refinement_loss_weight', 1.0)

    # Pop DETR-specific args (unused by V2, may be passed by generic
    # training script via CLI args that apply to all head types)
    for detr_arg in [
        'mask_ce_loss_weight', 'confidence_loss_weight',
        'no_object_weight', 'num_decoder_layers', 'num_queries',
    ]:
        kwargs.pop(detr_arg, None)

    # Pop OC-specific args (unused by V2)
    for oc_arg in [
        'focal_bce_weight', 'potential_loss_weight',
        'beta_loss_weight', 'clustering_dim',
    ]:
        kwargs.pop(oc_arg, None)

    # Backbone config: identical to pretraining
    single_layer_params = (32, 256, [(8, 1), (4, 1), (2, 1), (1, 1)], 64)

    backbone_kwargs = dict(
        input_dim=input_dim,
        enrichment_kwargs=dict(
            node_dim=32,
            edge_dim=8,
            num_neighbors=32,
            edge_aggregation='attn8',
            layer_params=[
                single_layer_params
                for _ in range(num_enrichment_layers)
            ],
        ),
        compaction_kwargs=dict(
            stage_output_points=[256, 128],
            stage_output_channels=[256, 256],
            stage_num_neighbors=[16, 16],
        ),
    )

    # V2 model config
    configuration = dict(
        backbone_kwargs=backbone_kwargs,
        # M1: Neighbor-aware scoring
        num_scoring_neighbors=16,
        message_dim=64,
        # M2: Top-K self-attention refinement
        num_refinement_candidates=256,
        refinement_dim=128,
        num_refinement_layers=2,
        refinement_num_heads=4,
        refinement_dropout=0.1,
        # Loss weights
        per_track_loss_weight=per_track_loss_weight,
        refinement_loss_weight=refinement_loss_weight,
        # Focal loss
        focal_alpha=0.75,
        focal_gamma=2.0,
        # M5: dxy_significance index in pf_features
        dxy_significance_feature_index=6,
    )
    configuration.update(**kwargs)
    _logger.info('Model config: %s' % str(configuration))

    model = TauTrackFinderV2(**configuration)

    # Load pretrained backbone weights
    if pretrained_backbone_path:
        _logger.info(
            'Loading pretrained backbone from: %s', pretrained_backbone_path,
        )
        checkpoint = torch.load(
            pretrained_backbone_path, map_location='cpu', weights_only=True,
        )

        state_dict = checkpoint.get('model_state_dict', checkpoint)

        has_backbone_prefix = any(
            key.startswith('backbone.') for key in state_dict
        )

        if has_backbone_prefix:
            backbone_state = {
                key.replace('backbone.', '', 1): value
                for key, value in state_dict.items()
                if key.startswith('backbone.')
            }
        else:
            backbone_state = state_dict

        model.backbone.load_state_dict(backbone_state)
        _logger.info('Pretrained backbone loaded successfully.')

    # Freeze backbone
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    _logger.info('Backbone frozen (no gradients).')

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
