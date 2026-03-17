#!/bin/bash
# Train TauTrackFinderV3 (ABCNet-inspired GAPLayer architecture)
#
# Expected to run on GPU server with CUDA.
# For local testing on MPS, use --device mps --batch-size 2 --no-compile

python train_trackfinder.py \
    --data-config data/low-pt/lowpt_tau_trackfinder.yaml \
    --data-dir data/low-pt/ \
    --network networks/lowpt_tau_TrackFinderV3.py \
    --pretrained-backbone models/backbone_best.pt \
    --model-name TrackFinderV3 \
    --epochs 50 \
    --batch-size 96 \
    --lr 1e-4 \
    --scheduler cosine \
    --warmup-fraction 0.05 \
    --weight-decay 0.01 \
    --grad-clip 1.0 \
    --steps-per-epoch 500 \
    --save-every 5 \
    --keep-best-k 5 \
    --device cuda:0 \
    --amp \
    "$@"
