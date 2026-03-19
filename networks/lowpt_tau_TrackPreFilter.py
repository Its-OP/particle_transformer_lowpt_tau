"""Network wrapper for TrackPreFilter (Stage 1 of two-stage pipeline).

Default configuration: hybrid mode with hidden_dim=192,
num_message_rounds=2, latent_dim=48, ranking_num_samples=50.
Widened from 128/32 to 192/48 to accommodate extended 13-feature input
(was 7 features: px, py, pz, eta, phi, charge, dxy_significance;
 now adds dz_significance, pt_error, n_valid_pixel_hits,
 dca_significance, covariance_phi_phi, covariance_lambda_lambda).
"""

from weaver.nn.model.TrackPreFilter import TrackPreFilter
from weaver.utils.logger import _logger


def get_model(data_config, **kwargs):
    """Build TrackPreFilter with default wide192+2rounds config."""
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
        hidden_dim=192,
        latent_dim=48,
        num_neighbors=16,
        num_message_rounds=2,
        ranking_num_samples=50,
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
