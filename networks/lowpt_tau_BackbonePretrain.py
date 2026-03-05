"""Network wrapper for backbone pretraining.

Used by the custom pretrain_backbone.py script (not weaver's training loop).
Provides get_model() to construct the MaskedTrackPretrainer with the
two-stage Enrich-Compact backbone.
"""
from weaver.nn.model.BackbonePretraining import MaskedTrackPretrainer
from weaver.utils.logger import _logger


def get_model(data_config, **kwargs):

    input_dim = len(data_config.input_dicts['pf_features'])

    # Number of enrichment layers is configurable via --num-enrichment-layers.
    # Default: 5 layers (deeper than ParticleNeXt's 3 to compensate for
    # ~10× more particles: ~1130 vs ~30-150).
    num_enrichment_layers = kwargs.pop('num_enrichment_layers', 5)

    # Number of DETR-style decoder layers (self-attn → cross-attn → FFN).
    # Default: 1 layer. More layers give queries more capacity to coordinate
    # and refine predictions.
    num_decoder_layers = kwargs.pop('num_decoder_layers', 1)

    # Each layer: (k, out_dim, reduction_dilation, message_dim)
    single_layer_params = (32, 256, [(8, 1), (4, 1), (2, 1), (1, 1)], 64)

    cfg = dict(
        backbone_kwargs=dict(
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
        ),
        decoder_kwargs=dict(
            decoder_dim=128,
            num_heads=4,
            num_decoder_layers=num_decoder_layers,
            num_output_features=input_dim,
            max_masked_tracks=1600,
            dropout=0.0,
        ),
        mask_ratio=0.4,
    )
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = MaskedTrackPretrainer(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {
            k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()
        },
        'output_names': ['loss'],
        'dynamic_axes': {
            **{
                k: {0: 'N', 2: 'n_' + k.split('_')[0]}
                for k in data_config.input_names
            },
            **{'loss': {0: 'N'}},
        },
    }

    return model, model_info
