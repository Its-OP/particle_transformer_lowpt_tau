#!/bin/bash
# =============================================================================
# Training launch script for backbone pretraining.
#
# Starts two screen sessions:
#   - "pretrain_train": Training output (pretrain_backbone.py)
#   - "pretrain_gpu":   GPU monitoring (nvidia-smi, refreshes every second)
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
#   bash train_pretrain.sh                  # default settings
#   bash train_pretrain.sh --epochs 200     # override defaults
#   bash train_pretrain.sh --resume experiments/BackbonePretrain_20260219_173000/checkpoints/checkpoint_epoch_50.pt
#
# To reattach after disconnecting:
#   screen -r pretrain_train    # training output
#   screen -r pretrain_gpu      # GPU monitoring
#
# To detach from a screen:  Ctrl+A, then D
# =============================================================================
set -euo pipefail

# ---- Configuration ----
SESSION_TRAIN="pretrain_train"
SESSION_GPU="pretrain_gpu"
CONDA_ENV_NAME="part"

# Default training arguments (can be overridden via command-line)
DATA_CONFIG="data/low-pt/lowpt_tau_pretrain.yaml"
DATA_DIR="data/low-pt/"
NETWORK="networks/lowpt_tau_BackbonePretrain.py"
MODEL_NAME="BackbonePretrain"
EXPERIMENTS_DIR="experiments"
EPOCHS=100
BATCH_SIZE=256
LEARNING_RATE=1e-3
DEVICE="cuda:0"

# ---- Parse extra arguments ----
# All extra arguments are passed directly to pretrain_backbone.py,
# allowing overrides like --epochs 200 --batch-size 64
EXTRA_ARGS="$*"

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
# Ensure we run from the part/ directory regardless of where the script is called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Build training command ----
TRAIN_CMD="${CONDA_INIT} && cd ${SCRIPT_DIR} && python pretrain_backbone.py \
    --data-config ${DATA_CONFIG} \
    --data-dir ${DATA_DIR} \
    --network ${NETWORK} \
    --model-name ${MODEL_NAME} \
    --experiments-dir ${EXPERIMENTS_DIR} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --device ${DEVICE} \
    --amp \
    ${EXTRA_ARGS}"

# ---- GPU monitoring command ----
GPU_MONITOR_CMD="watch -n 1 nvidia-smi"

# ---- Clean up existing sessions ----
# Kill stale GPU monitor session from previous runs (these accumulate otherwise)
if screen -list | grep -q "${SESSION_GPU}"; then
    screen -S "${SESSION_GPU}" -X quit 2>/dev/null || true
fi

if screen -list | grep -q "${SESSION_TRAIN}"; then
    echo "Screen session '${SESSION_TRAIN}' already exists."
    echo "To reattach:  screen -r ${SESSION_TRAIN}"
    echo "To kill it:   screen -S ${SESSION_TRAIN} -X quit"
    exit 1
fi

# ---- Launch screen sessions ----
echo "============================================"
echo "  Launching pretraining in screen"
echo "============================================"
echo ""
echo "Session:    ${SESSION_TRAIN} (training)"
echo "            ${SESSION_GPU} (GPU monitor)"
echo "Experiments: ${SCRIPT_DIR}/${EXPERIMENTS_DIR}/"
echo "Epochs:     ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "LR:         ${LEARNING_RATE}"
echo "Device:     ${DEVICE}"
echo "AMP:        enabled"
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
