"""Network wrapper for ParticleNeXt-based TrackPreFilter (Phase C-1).

Uses ParticleNeXt backbone with multi-scale EdgeConv, attention-weighted
aggregation, and pairwise Lorentz vector edge features. Replaces the
simpler MLP+kNN pre-filter from Phase A/B.

Default config: 3 MultiScaleEdgeConv blocks, k=32, 4 dilation scales,
attention edge aggregation with 8 heads.
"""

from weaver.nn.model.ParticleNeXtPreFilter import ParticleNeXtPreFilter
from weaver.utils.logger import _logger


def get_model(data_config, **kwargs):
    """Build ParticleNeXtPreFilter with default config."""
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
        feature_input_dim=input_dim,
        # ParticleNeXt backbone
        node_dim=32,
        edge_dim=8,
        # 3 MultiScaleEdgeConv blocks: k=32 neighbors, 256-dim output,
        # 4 dilation scales [(8,1),(4,1),(2,1),(1,1)] = 4,8,16,32 neighbors,
        # 64-dim message bottleneck
        layer_params=[
            (32, 256, [(8, 1), (4, 1), (2, 1), (1, 1)], 64),
            (32, 256, [(8, 1), (4, 1), (2, 1), (1, 1)], 64),
            (32, 256, [(8, 1), (4, 1), (2, 1), (1, 1)], 64),
        ],
        fc_params=[(256, 0.1)],
        edge_aggregation='attn8',
        use_rel_lv_fts=True,
        # Training schedule (same as Phase-B)
        ranking_num_samples=50,
        ranking_temperature_start=2.0,
        ranking_temperature_end=0.5,
        denoising_sigma_start=1.0,
        denoising_sigma_end=0.1,
        drw_warmup_fraction=0.3,
        drw_positive_weight=2.0,
    )
    configuration.update(**kwargs)
    _logger.info('ParticleNeXtPreFilter config: %s' % str(configuration))

    model = ParticleNeXtPreFilter(**configuration)

    total_params = sum(p.numel() for p in model.parameters())
    _logger.info(f'ParticleNeXtPreFilter: {total_params:,} params (all trainable)')

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
