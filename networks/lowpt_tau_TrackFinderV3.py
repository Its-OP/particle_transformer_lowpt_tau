"""Network wrapper for V3 tau-origin pion track finder.

Provides get_model() to construct TauTrackFinderV3 with pretrained backbone.
The backbone is frozen by default -- only GAPLayers and scoring head are trained.

V3 replaces V2's max-pool kNN + self-attention refinement with:
    - ABCNet-style GAPLayers (attention-weighted edge convolution)
    - Dual kNN: physical (eta, phi) + learned feature space
    - Global context injection (event-level average pool + tile)
    - Multi-scale feature concatenation (all intermediate + raw + global)
"""

import torch
from weaver.nn.model.TauTrackFinderV3 import TauTrackFinderV3
from weaver.utils.logger import _logger


def get_model(data_config, **kwargs):
    """Build TauTrackFinderV3 with pretrained backbone.

    Kwargs consumed (popped from kwargs):
        pretrained_backbone_path: Path to pretrained backbone checkpoint.
        num_enrichment_layers: Number of enrichment layers (default: 5).
    """
    pretrained_backbone_path = kwargs.pop('pretrained_backbone_path', None)
    backbone_mode = kwargs.pop('backbone_mode', 'parallel')
    num_enrichment_layers = kwargs.pop('num_enrichment_layers', 5)

    # Pop args from other head types (unused by V3, may be passed generically)
    for unused_arg in [
        'mask_ce_loss_weight', 'confidence_loss_weight',
        'no_object_weight', 'num_decoder_layers', 'num_queries',
        'focal_bce_weight', 'potential_loss_weight',
        'beta_loss_weight', 'clustering_dim',
        'per_track_loss_weight', 'refinement_loss_weight',
    ]:
        kwargs.pop(unused_arg, None)

    input_dim = len(data_config.input_dicts['pf_features'])

    # Frozen backbone config (used when backbone_mode='frozen')
    single_layer_params = (32, 256, [(8, 1), (4, 1), (2, 1), (1, 1)], 64)
    frozen_backbone_kwargs = dict(
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

    # Parallel backbone config (used when backbone_mode='parallel')
    parallel_backbone_kwargs = dict(
        input_dim=input_dim,
        identity_dim=64,
        context_dim=128,
        num_context_layers=2,
        context_num_neighbors=16,
        context_edge_dim=8,
        context_node_dim=32,
        context_message_dim=64,
    )

    # V3 model config — ABCNet-inspired GAPLayers
    configuration = dict(
        backbone_mode=backbone_mode,
        backbone_kwargs=frozen_backbone_kwargs,
        parallel_backbone_kwargs=parallel_backbone_kwargs,
        # GAPLayer 1: kNN in physical (eta, phi) space
        gap1_encoding_dim=64,
        gap1_num_neighbors=16,
        gap1_num_heads=4,
        # GAPLayer 2: kNN in learned feature space
        gap2_encoding_dim=64,
        gap2_num_neighbors=16,
        gap2_num_heads=4,
        # Intermediate MLPs
        intermediate_dim=128,
        # Global context
        global_context_dim=32,
        # Scoring head
        scoring_dropout=0.4,
        # ASL loss (Ben-Baruch et al., ICCV 2021)
        focal_gamma_positive=1.0,
        focal_gamma_negative=4.0,
        asl_clip=0.05,
        # Ranking loss
        ranking_loss_weight=0.1,
        ranking_num_samples=10,
    )
    configuration.update(**kwargs)
    _logger.info('Backbone mode: %s', backbone_mode)
    _logger.info('Model config: %s' % str(configuration))

    model = TauTrackFinderV3(**configuration)

    # Load pretrained backbone weights (frozen mode only)
    if backbone_mode == 'frozen' and pretrained_backbone_path:
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

    # Freeze backbone (frozen mode only — parallel is always trainable)
    if backbone_mode == 'frozen':
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
        _logger.info('Backbone frozen (no gradients).')
    else:
        _logger.info('Parallel backbone: fully trainable, no pretrained weights.')

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
