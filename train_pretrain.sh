#!/bin/bash
# =============================================================================
# Training launch script for backbone pretraining.
#
# Starts a tmux session with two panes:
#   - Top pane:    Training output (pretrain_backbone.py)
#   - Bottom pane: GPU monitoring (nvidia-smi -l)
#
# Usage:
#   bash train_pretrain.sh                  # default settings
#   bash train_pretrain.sh --epochs 200     # override defaults
#   bash train_pretrain.sh --resume checkpoints/pretrain/checkpoint_epoch_50.pt
#
# The script will:
#   1. Activate the conda environment
#   2. Create a tmux session named "pretrain"
#   3. Run training in the top pane with TensorBoard logging
#   4. Run nvidia-smi monitoring in the bottom pane
#
# To reattach after disconnecting:
#   tmux attach -t pretrain
# =============================================================================
set -euo pipefail

# ---- Configuration ----
SESSION_NAME="pretrain"
CONDA_ENV_NAME="part"

# Default training arguments (can be overridden via command-line)
DATA_CONFIG="data/low-pt/lowpt_tau_pretrain.yaml"
DATA_DIR="data/low-pt/"
NETWORK="networks/lowpt_tau_BackbonePretrain.py"
OUTPUT_DIR="checkpoints/pretrain"
EPOCHS=100
BATCH_SIZE=32
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
    --output-dir ${OUTPUT_DIR} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --device ${DEVICE} \
    --amp \
    ${EXTRA_ARGS}"

# ---- GPU monitoring command ----
GPU_MONITOR_CMD="watch -n 1 nvidia-smi"

# ---- Check for existing session ----
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '${SESSION_NAME}' already exists."
    echo "To reattach:  tmux attach -t ${SESSION_NAME}"
    echo "To kill it:   tmux kill-session -t ${SESSION_NAME}"
    exit 1
fi

# ---- Create output directory ----
mkdir -p "${SCRIPT_DIR}/${OUTPUT_DIR}"

# ---- Launch tmux session ----
echo "============================================"
echo "  Launching pretraining in tmux"
echo "============================================"
echo ""
echo "Session:    ${SESSION_NAME}"
echo "Output dir: ${SCRIPT_DIR}/${OUTPUT_DIR}"
echo "Epochs:     ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "LR:         ${LEARNING_RATE}"
echo "Device:     ${DEVICE}"
echo "AMP:        enabled"
if [ -n "$EXTRA_ARGS" ]; then
    echo "Extra args: ${EXTRA_ARGS}"
fi
echo ""

# Create tmux session with the training command in the first pane
tmux new-session -d -s "$SESSION_NAME" -x 200 -y 50 "$TRAIN_CMD; echo '--- Training finished. Press Enter to close. ---'; read"

# Split horizontally (top/bottom) and run GPU monitoring in the bottom pane
tmux split-window -v -t "$SESSION_NAME" "$GPU_MONITOR_CMD"

# Give training pane (top) 75% of the vertical space
tmux resize-pane -t "${SESSION_NAME}:0.0" -y 75%

# Select the training pane as active
tmux select-pane -t "${SESSION_NAME}:0.0"

echo "tmux session '${SESSION_NAME}' created."
echo ""
echo "To attach:     tmux attach -t ${SESSION_NAME}"
echo "To detach:     Ctrl+B, then D"
echo "To switch pane: Ctrl+B, then arrow keys"
echo ""
echo "TensorBoard (from another terminal):"
echo "  ${CONDA_INIT} && tensorboard --logdir ${SCRIPT_DIR}/${OUTPUT_DIR}/tensorboard --bind_all"
echo ""

# Attach to the session
tmux attach -t "$SESSION_NAME"
