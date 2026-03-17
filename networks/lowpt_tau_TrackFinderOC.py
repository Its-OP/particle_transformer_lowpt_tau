"""Network wrapper for object condensation tau-origin pion track finding.

Used by the custom train_trackfinder.py script (not weaver's training loop).
Provides get_model() to construct TauTrackFinderOC with pretrained backbone
and object condensation head.

The backbone is frozen by default — only the OC head parameters are trained.
Pretrained backbone weights are loaded from a checkpoint file via
--pretrained-backbone CLI argument.
"""

import torch
from weaver.nn.model.TauTrackFinderOC import TauTrackFinderOC
from weaver.utils.logger import _logger


def get_model(data_config, **kwargs):
    """Build TauTrackFinderOC with pretrained backbone.

    Kwargs consumed (popped from kwargs):
        pretrained_backbone_path: Path to pretrained backbone checkpoint.
        num_enrichment_layers: Number of enrichment layers (default: 5).
        potential_loss_weight: Weight for attractive + repulsive loss (default: 1.0).
        beta_loss_weight: Weight for beta loss (default: 1.0).
        q_min: Minimum charge offset (default: 0.1).
        suppression_weight: Background beta suppression weight (default: 1.0).
        clustering_dim: Clustering space dimensionality (default: 8).
    """
    pretrained_backbone_path = kwargs.pop('pretrained_backbone_path', None)
    num_enrichment_layers = kwargs.pop('num_enrichment_layers', 5)

    # OC-specific hyperparameters
    focal_bce_weight = kwargs.pop('focal_bce_weight', 1.0)
    potential_loss_weight = kwargs.pop('potential_loss_weight', 0.01)
    beta_loss_weight = kwargs.pop('beta_loss_weight', 0.01)
    q_min = kwargs.pop('q_min', 0.1)
    suppression_weight = kwargs.pop('suppression_weight', 0.01)
    clustering_dim = kwargs.pop('clustering_dim', 8)

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

    head_kwargs = dict(
        clustering_dim=clustering_dim,
    )

    configuration = dict(
        backbone_kwargs=backbone_kwargs,
        head_kwargs=head_kwargs,
        focal_bce_weight=focal_bce_weight,
        potential_loss_weight=potential_loss_weight,
        beta_loss_weight=beta_loss_weight,
        q_min=q_min,
        suppression_weight=suppression_weight,
    )
    configuration.update(**kwargs)
    _logger.info('Model config: %s' % str(configuration))

    model = TauTrackFinderOC(**configuration)

    # Load pretrained backbone weights
    if pretrained_backbone_path:
        _logger.info(
            'Loading pretrained backbone from: %s', pretrained_backbone_path,
        )
        checkpoint = torch.load(
            pretrained_backbone_path, map_location='cpu', weights_only=True,
        )

        # The checkpoint may be either:
        # 1. MaskedTrackPretrainer state_dict (prefixed with 'backbone.')
        # 2. Standalone backbone state_dict (backbone_best.pt)
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
