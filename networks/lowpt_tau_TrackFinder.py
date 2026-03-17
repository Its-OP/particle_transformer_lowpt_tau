"""Network wrapper for DETR tau-origin pion track finding.

Provides get_model() to construct TauTrackFinder with pretrained backbone.
The backbone is frozen by default — only the head parameters are trained.
"""

import torch
from weaver.nn.model.TauTrackFinder import TauTrackFinder
from weaver.utils.logger import _logger


def get_model(data_config, **kwargs):
    """Build TauTrackFinder with pretrained backbone.

    Kwargs consumed (popped from kwargs):
        pretrained_backbone_path: Path to pretrained backbone checkpoint.
        num_enrichment_layers: Number of enrichment layers (default: 5).
        num_decoder_layers: Number of query decoder layers (default: 4).
        num_queries: Number of learned queries (default: 30).
        mask_ce_loss_weight: Weight for CE mask loss (default: 2.0).
        confidence_loss_weight: Weight for confidence BCE (default: 2.0).
        no_object_weight: Weight for empty targets (default: 0.4).
    """
    pretrained_backbone_path = kwargs.pop('pretrained_backbone_path', None)
    num_enrichment_layers = kwargs.pop('num_enrichment_layers', 5)
    num_decoder_layers = kwargs.pop('num_decoder_layers', 4)
    num_queries = kwargs.pop('num_queries', 30)

    # Loss weights
    mask_ce_loss_weight = kwargs.pop('mask_ce_loss_weight', 2.0)
    confidence_loss_weight = kwargs.pop('confidence_loss_weight', 2.0)
    per_track_loss_weight = kwargs.pop('per_track_loss_weight', 1.0)
    no_object_weight = kwargs.pop('no_object_weight', 0.4)

    input_dim = len(data_config.input_dicts['pf_features'])

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

    decoder_kwargs = dict(
        num_queries=num_queries,
        max_gt_tracks=6,
        decoder_dim=256,
        mask_dim=128,
        num_heads=8,
        num_decoder_layers=num_decoder_layers,
        dropout=0.1,
    )

    configuration = dict(
        backbone_kwargs=backbone_kwargs,
        decoder_kwargs=decoder_kwargs,
        mask_ce_loss_weight=mask_ce_loss_weight,
        confidence_loss_weight=confidence_loss_weight,
        per_track_loss_weight=per_track_loss_weight,
        no_object_weight=no_object_weight,
    )
    configuration.update(**kwargs)
    _logger.info('Model config: %s' % str(configuration))

    model = TauTrackFinder(**configuration)

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
