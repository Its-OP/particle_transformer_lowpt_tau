"""Network wrapper for CascadeModel (Stage 1 pre-filter → Stage 2 reranker).

Builds a CascadeModel from:
    - Stage 1: frozen TrackPreFilter loaded from checkpoint
    - Stage 2: CascadeReranker (ParT-style pairwise-bias self-attention)

Usage:
    python train_cascade.py --network networks/lowpt_tau_CascadeReranker.py \
        --stage1-checkpoint models/prefilter_best.pt --top-k1 600
"""
import torch

from weaver.nn.model.CascadeModel import CascadeModel
from weaver.nn.model.CascadeReranker import CascadeReranker
from weaver.nn.model.TrackPreFilter import TrackPreFilter
from weaver.utils.logger import _logger


def get_model(data_config, **kwargs):
    """Build CascadeModel with frozen Stage 1 + CascadeReranker Stage 2.

    Required kwargs (passed via train_cascade.py):
        stage1_checkpoint: Path to trained Stage 1 checkpoint.
        top_k1: Number of tracks to pass from Stage 1 to Stage 2.
    """
    # Pop cascade-specific args
    stage1_checkpoint = kwargs.pop('stage1_checkpoint', None)
    top_k1 = kwargs.pop('top_k1', 600)

    # Stage 2 architecture args (from train_cascade.py --stage2-* flags)
    stage2_embed_dim = kwargs.pop('stage2_embed_dim', 128)
    stage2_num_heads = kwargs.pop('stage2_num_heads', 4)
    stage2_num_layers = kwargs.pop('stage2_num_layers', 3)
    stage2_pair_embed_dims = kwargs.pop('stage2_pair_embed_dims', [64, 64])
    stage2_pair_extra_dim = kwargs.pop('stage2_pair_extra_dim', 6)
    stage2_pair_embed_mode = kwargs.pop('stage2_pair_embed_mode', 'concat')
    stage2_ffn_ratio = kwargs.pop('stage2_ffn_ratio', 4)
    stage2_dropout = kwargs.pop('stage2_dropout', 0.1)
    stage2_loss_mode = kwargs.pop('stage2_loss_mode', 'pairwise')
    stage2_rs_at_k_target = kwargs.pop('stage2_rs_at_k_target', 200)
    # Contrastive denoising — auxiliary regularizer ported from TrackPreFilter
    stage2_use_contrastive_denoising = kwargs.pop(
        'stage2_use_contrastive_denoising', False,
    )
    stage2_denoising_sigma_start = kwargs.pop(
        'stage2_denoising_sigma_start', 0.3,
    )
    stage2_denoising_sigma_end = kwargs.pop(
        'stage2_denoising_sigma_end', 0.05,
    )
    stage2_denoising_loss_weight = kwargs.pop(
        'stage2_denoising_loss_weight', 0.5,
    )

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

    # ---- Stage 1: load frozen pre-filter ----
    if stage1_checkpoint is None:
        raise ValueError(
            'stage1_checkpoint is required. '
            'Pass --stage1-checkpoint to train_cascade.py'
        )

    _logger.info(f'Loading Stage 1 from: {stage1_checkpoint}')
    checkpoint = torch.load(stage1_checkpoint, map_location='cpu', weights_only=False)

    # Infer Stage 1 hidden_dim and input_dim from state dict shapes so the
    # wrapper adapts to checkpoints trained with different widths (e.g., the
    # 2026-04 cutoff run uses hidden_dim=256 while earlier runs used 192).
    # track_mlp.0.weight has shape (hidden_dim, input_dim, 1) in the MLP backbone.
    stage1_state = checkpoint.get('model_state_dict', checkpoint)
    first_layer_key = 'track_mlp.0.weight'
    if first_layer_key not in stage1_state:
        raise ValueError(
            f'Cannot infer Stage 1 dimensions: expected key '
            f'"{first_layer_key}" not found in {stage1_checkpoint}'
        )
    inferred_hidden_dim = stage1_state[first_layer_key].shape[0]
    inferred_input_dim = stage1_state[first_layer_key].shape[1]
    if inferred_input_dim != input_dim:
        raise ValueError(
            f'Stage 1 checkpoint input_dim={inferred_input_dim} does not '
            f'match data config input_dim={input_dim}. '
            f'Retrain Stage 1 on the current feature set.'
        )
    _logger.info(
        f'Stage 1 config from checkpoint: '
        f'hidden_dim={inferred_hidden_dim}, input_dim={inferred_input_dim}'
    )

    stage1 = TrackPreFilter(
        mode='mlp',
        input_dim=inferred_input_dim,
        hidden_dim=inferred_hidden_dim,
        num_message_rounds=2,
        num_neighbors=16,
    )
    stage1.load_state_dict(stage1_state)
    _logger.info('Stage 1 loaded successfully')

    stage1_params = sum(p.numel() for p in stage1.parameters())
    _logger.info(f'Stage 1: {stage1_params:,} params (frozen)')

    # ---- Stage 2: CascadeReranker (ParT-style pairwise-bias attention) ----
    stage2 = CascadeReranker(
        input_dim=input_dim,
        embed_dim=stage2_embed_dim,
        num_heads=stage2_num_heads,
        num_layers=stage2_num_layers,
        pair_input_dim=4,                          # ln kT, ln z, ln ΔR, ln m²
        pair_extra_dim=stage2_pair_extra_dim,      # physics pairwise features
        pair_embed_dims=stage2_pair_embed_dims,
        pair_embed_mode=stage2_pair_embed_mode,    # 'concat' or 'sum'
        ffn_ratio=stage2_ffn_ratio,
        dropout=stage2_dropout,
        ranking_num_samples=50,
        ranking_temperature=1.0,
        loss_mode=stage2_loss_mode,
        rs_at_k_target=stage2_rs_at_k_target,
        use_contrastive_denoising=stage2_use_contrastive_denoising,
        denoising_sigma_start=stage2_denoising_sigma_start,
        denoising_sigma_end=stage2_denoising_sigma_end,
        denoising_loss_weight=stage2_denoising_loss_weight,
    )
    stage2_params = sum(p.numel() for p in stage2.parameters())
    _logger.info(f'Stage 2 (CascadeReranker): {stage2_params:,} params (trainable)')

    # ---- Cascade ----
    model = CascadeModel(stage1=stage1, stage2=stage2, top_k1=top_k1)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _logger.info(f'Cascade total: {total_params:,} params | Trainable: {trainable_params:,}')

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
