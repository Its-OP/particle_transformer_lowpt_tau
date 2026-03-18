"""Network wrapper for TrackPreFilter (Stage 1 of two-stage pipeline).

Default configuration: hybrid mode with hidden_dim=128,
num_message_rounds=2, latent_dim=32, ranking_num_samples=50.
Best discovered combination from hyperparameter sweep + SOTA denoising.
"""

from weaver.nn.model.TrackPreFilter import TrackPreFilter
from weaver.utils.logger import _logger


def get_model(data_config, **kwargs):
    """Build TrackPreFilter with default wide128+2rounds config."""
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

    configuration = dict(
        mode='hybrid',
        input_dim=input_dim,
        hidden_dim=128,
        latent_dim=32,
        num_neighbors=16,
        num_message_rounds=2,
        ranking_num_samples=50,
        use_pairwise_physics=True,
        pairwise_edge_dim=16,
    )
    configuration.update(**kwargs)
    _logger.info('TrackPreFilter config: %s' % str(configuration))

    model = TrackPreFilter(**configuration)

    total_params = sum(p.numel() for p in model.parameters())
    _logger.info(f'TrackPreFilter: {total_params:,} params (all trainable)')

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
