#!/bin/bash
# =============================================================================
# Training launch script for tau-origin pion track finder.
#
# Starts two screen sessions:
#   - "trackfinder_train": Training output (train_trackfinder.py)
#   - "trackfinder_gpu":   GPU monitoring (nvidia-smi, refreshes every second)
#
# Each run creates a new experiment folder:
#   experiments/{model_name}_{timestamp}/
#       ├── training.log
#       ├── loss_history.json
#       ├── loss_curves.png
#       ├── checkpoints/
#       └── tensorboard/
#
# Usage:
#   bash train_trackfinder.sh                  # default settings
#   bash train_trackfinder.sh --epochs 100     # override defaults
#   bash train_trackfinder.sh --resume experiments/TrackFinder_.../checkpoints/checkpoint_epoch_50.pt
#
# To reattach after disconnecting:
#   screen -r trackfinder_train    # training output
#   screen -r trackfinder_gpu      # GPU monitoring
#
# To detach from a screen:  Ctrl+A, then D
# =============================================================================
set -euo pipefail

# ---- Check data split ----
SCRIPT_DIR_CHECK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_PARQUET_COUNT=$(find "${SCRIPT_DIR_CHECK}/data/low-pt/train" -maxdepth 1 -name "*.parquet" 2>/dev/null | wc -l | tr -d ' ')
VAL_PARQUET_COUNT=$(find "${SCRIPT_DIR_CHECK}/data/low-pt/val" -maxdepth 1 -name "*.parquet" 2>/dev/null | wc -l | tr -d ' ')
if [ "$TRAIN_PARQUET_COUNT" -lt 10 ] || [ "$VAL_PARQUET_COUNT" -lt 10 ]; then
    echo "WARNING: Found ${TRAIN_PARQUET_COUNT} train and ${VAL_PARQUET_COUNT} val parquet files."
    echo "  Expected 10+ files each for multi-worker loading."
    echo "  Run: python utils/split_parquet.py data/low-pt/train/train.parquet --num-parts 10"
fi
echo "Found ${TRAIN_PARQUET_COUNT} train parquet files, ${VAL_PARQUET_COUNT} val parquet files."

# ---- Configuration ----
SESSION_TRAIN="trackfinder_train"
SESSION_GPU="trackfinder_gpu"
CONDA_ENV_NAME="part"

# Default training arguments (can be overridden via command-line)
DATA_CONFIG="data/low-pt/lowpt_tau_trackfinder.yaml"
DATA_DIR="data/low-pt/train/"
VAL_DATA_DIR="data/low-pt/val/"
NETWORK="networks/lowpt_tau_TrackFinderV2.py"
MODEL_NAME="TrackFinderV3"
EXPERIMENTS_DIR="experiments"
PRETRAINED_BACKBONE="models/backbone_best.pt"
EPOCHS=50
BATCH_SIZE=128
LEARNING_RATE=1e-4
SCHEDULER="cosine"
DEVICE="cuda:0"
STEPS_PER_EPOCH=350
NUM_WORKERS=10
NO_COMPILE=false
# DETR loss weights (only used with DETR head; ignored by V2 and OC wrappers)
MASK_CE_LOSS_WEIGHT=2.0
CONFIDENCE_LOSS_WEIGHT=2.0
NO_OBJECT_WEIGHT=0.4
# Keep top K best checkpoints by val loss (0 = keep all, for CERN cluster runs)
KEEP_BEST_K=5

# ---- Parse extra arguments ----
# Check if --no-compile was passed via CLI and update the variable
# so the summary display is accurate.
EXTRA_ARGS=""
for arg in "$@"; do
    if [ "$arg" = "--no-compile" ]; then
        NO_COMPILE=true
    else
        EXTRA_ARGS="${EXTRA_ARGS:+${EXTRA_ARGS} }${arg}"
    fi
done

# ---- Resolve conda ----
if command -v conda &>/dev/null; then
    CONDA_BASE=$(conda info --base)
elif [ -d "$HOME/miniconda3" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -d "/opt/miniconda3" ]; then
    CONDA_BASE="/opt/miniconda3"
else
    echo "ERROR: conda not found. Run setup_server.sh first."
    exit 1
fi

CONDA_INIT="source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV_NAME}"

# ---- Resolve script directory ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Build training command ----
TRAIN_CMD="${CONDA_INIT} && cd ${SCRIPT_DIR} && python train_trackfinder.py \
    --data-config ${DATA_CONFIG} \
    --data-dir ${DATA_DIR} \
    --val-data-dir ${VAL_DATA_DIR} \
    --network ${NETWORK} \
    --model-name ${MODEL_NAME} \
    --experiments-dir ${EXPERIMENTS_DIR} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --steps-per-epoch ${STEPS_PER_EPOCH} \
    --lr ${LEARNING_RATE} \
    --scheduler ${SCHEDULER} \
    --device ${DEVICE} \
    --num-workers ${NUM_WORKERS} \
    --mask-ce-loss-weight ${MASK_CE_LOSS_WEIGHT} \
    --confidence-loss-weight ${CONFIDENCE_LOSS_WEIGHT} \
    --no-object-weight ${NO_OBJECT_WEIGHT} \
    --keep-best-k ${KEEP_BEST_K} \
    --amp"

# Add pretrained backbone if specified
if [ -n "$PRETRAINED_BACKBONE" ]; then
    TRAIN_CMD="${TRAIN_CMD} --pretrained-backbone ${PRETRAINED_BACKBONE}"
fi

# Disable torch.compile if requested
if [ "$NO_COMPILE" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --no-compile"
fi

# Append extra arguments
TRAIN_CMD="${TRAIN_CMD} ${EXTRA_ARGS}"

# ---- GPU monitoring command ----
GPU_MONITOR_CMD="watch -n 1 nvidia-smi"

# ---- Clean up existing sessions ----
screen -list 2>/dev/null | grep "\.${SESSION_GPU}" | awk '{print $1}' | while read -r session_id; do
    screen -S "$session_id" -X quit 2>/dev/null || true
done || true

if screen -list | grep -q "${SESSION_TRAIN}"; then
    echo "Screen session '${SESSION_TRAIN}' already exists."
    echo "To reattach:  screen -r ${SESSION_TRAIN}"
    echo "To kill it:   screen -S ${SESSION_TRAIN} -X quit"
    exit 1
fi

# ---- Launch screen sessions ----
echo "============================================"
echo "  Launching track finder training in screen"
echo "============================================"
echo ""
echo "Session:    ${SESSION_TRAIN} (training)"
echo "            ${SESSION_GPU} (GPU monitor)"
echo "Experiments: ${SCRIPT_DIR}/${EXPERIMENTS_DIR}/"
echo "Epochs:     ${EPOCHS}"
echo "Steps/epoch: ${STEPS_PER_EPOCH}"
echo "Batch size: ${BATCH_SIZE}"
echo "LR:         ${LEARNING_RATE}"
echo "Scheduler:  ${SCHEDULER}"
echo "Device:     ${DEVICE}"
echo "Network:    ${NETWORK}"
echo "AMP:        enabled"
if [ "$NO_COMPILE" = true ]; then
    echo "Compile:    disabled"
else
    echo "Compile:    enabled"
fi
if [ -n "$PRETRAINED_BACKBONE" ]; then
    echo "Backbone:   ${PRETRAINED_BACKBONE}"
else
    echo "Backbone:   random init (no pretrained weights)"
fi
if [ -n "$EXTRA_ARGS" ]; then
    echo "Extra args: ${EXTRA_ARGS}"
fi
echo ""

# Launch training in a detached screen session
screen -dmS "$SESSION_TRAIN" bash -c "${TRAIN_CMD}; echo '--- Training finished. Press Enter to close. ---'; read"

# Launch GPU monitoring in a separate detached screen session
screen -dmS "$SESSION_GPU" bash -c "$GPU_MONITOR_CMD"

echo "Screen sessions created."
echo ""
echo "To view training:   screen -r ${SESSION_TRAIN}"
echo "To view GPU usage:  screen -r ${SESSION_GPU}"
echo "To detach:          Ctrl+A, then D"
echo "To list sessions:   screen -ls"
echo ""
echo "TensorBoard (from another terminal):"
echo "  ${CONDA_INIT} && tensorboard --logdir ${SCRIPT_DIR}/${EXPERIMENTS_DIR} --bind_all"
echo ""
