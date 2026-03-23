"""Network wrapper for TrackPreFilter (Stage 1 of two-stage pipeline).

Default configuration: MLP mode with hidden_dim=192,
num_message_rounds=2, ranking_num_samples=50.

Input features (16): px, py, pz, eta, phi, charge, dxy_significance,
log_dz_significance, normalized_chi2, log_pt_error, n_valid_pixel_hits,
dca_significance, log_covariance_phi_phi, log_covariance_lambda_lambda,
log_pt, relative_pt_error.

Key fixes over 13-feature version:
- dz_significance log-transformed (was 99.9% clipped by auto-standardization)
- pt_error, covariance_phi_phi, covariance_lambda_lambda log-transformed
- Added normalized_chi2 (#1 CMS track quality feature)
- Added log(pT) and relative pT error (standard in CMS/ATLAS taggers)
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
        # RS@K surrogate loss (Patel et al., CVPR 2022, arXiv:2108.11179):
        # Anneals from 0 → 0.5 over training via set_temperature_progress.
        # Directly optimizes differentiable Recall@200 alongside ranking loss.
        rs_at_k_weight_start=0.0,
        rs_at_k_weight_end=0.5,
        rs_at_k_target=200,
        rs_at_k_tau1=1.0,
        rs_at_k_tau2=1.0,
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
