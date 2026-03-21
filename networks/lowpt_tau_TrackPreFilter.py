"""Network wrapper for TrackPreFilter (Stage 1 of two-stage pipeline).

Default configuration: MLP mode with hidden_dim=192,
num_message_rounds=2, ranking_num_samples=50.
Autoencoder removed — with 13-dim input and 48-dim latent (3.7× wider),
reconstruction was trivially solvable and added no discriminative value.

Training improvements:
- Temperature-scheduled ranking loss (high T → low T over training)
- Temperature-scheduled denoising sigma (large noise → small noise)
- Deferred Re-Weighting: uniform weights first, then upweight positives
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
        mode='mlp',
        input_dim=input_dim,
        hidden_dim=192,
        num_neighbors=16,
        num_message_rounds=2,
        ranking_num_samples=50,
        # Temperature scheduling (Kukleva et al., ICLR 2023):
        # Ranking temperature: high → low (smooth gradients first, then sharp)
        ranking_temperature_start=2.0,
        ranking_temperature_end=0.5,
        # Denoising sigma: large → small (easy positives first, then hard)
        denoising_sigma_start=1.0,
        denoising_sigma_end=0.1,
        # Deferred Re-Weighting (Cao et al., NeurIPS 2019):
        # Uniform weights for 30% of training, then 2× upweight positives
        drw_warmup_fraction=0.3,
        drw_positive_weight=2.0,
        # SupMin contrastive auxiliary loss (Mildenberger et al., CVPR 2025):
        # Pulls signal embeddings together, spreads noise uniformly.
        supmin_weight=0.5,
        supmin_projection_dim=64,
        supmin_temperature=0.1,
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
