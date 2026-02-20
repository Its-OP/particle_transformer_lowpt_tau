"""Network wrapper for backbone pretraining.

Used by the custom pretrain_backbone.py script (not weaver's training loop).
Provides get_model() to construct the MaskedTrackPretrainer with default config.
"""
from weaver.nn.model.BackbonePretraining import MaskedTrackPretrainer
from weaver.utils.logger import _logger


def get_model(data_config, **kwargs):

    cfg = dict(
        backbone_kwargs=dict(
            input_dim=len(data_config.input_dicts['pf_features']),
            embed_dim=64,
            stage_output_points=[512, 256, 128],
            stage_output_channels=[128, 192, 256],
            stage_num_neighbors=[32, 24, 16],
        ),
        decoder_kwargs=dict(
            decoder_dim=128,
            num_heads=4,
            num_output_features=len(data_config.input_dicts['pf_features']),
            max_masked_tracks=1200,
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
