#!/bin/bash
# =============================================================================
# HTCondor execution script for backbone pretraining on lxplus.
#
# This script runs on a GPU-enabled worker node. It:
#   1. Activates the conda environment from EOS
#   2. Creates an experiment placeholder on EOS with job metadata
#   3. Runs pretrain_backbone.py with output in /tmp (fast local I/O)
#   4. Periodically syncs training logs to the EOS placeholder
#   5. Copies full results back to EOS when training completes
#   6. Handles preemption: traps EXIT to salvage partial results
#
# This script controls infrastructure paths (EOS, /tmp, conda, PYTHONPATH).
# All training hyperparameters are passed through from the .sub file
# and forwarded directly to pretrain_backbone.py.
#
# Usage (via condor_submit):
#   condor_submit condor/pretrain.sub
#
# See condor/pretrain.sub for configurable training parameters.
# =============================================================================
set -euo pipefail


# =============================================================================
# EOS paths (CERN EOS filesystem, mounted on all lxplus/batch workers)
# =============================================================================

EOS_HOME="/eos/user/o/oprostak"

# Data directory — parquet files read directly from EOS (no local copy needed).
DATA_DIR="${EOS_HOME}/tau_data/parquet_clean"

# Source repositories on EOS
PART_DIR="${EOS_HOME}/src/part"
WEAVER_DIR="${EOS_HOME}/src/weaver"

# Conda installation on EOS (environment 'part' is pre-created)
CONDA_DIR="${EOS_HOME}/miniconda"
CONDA_ENV_NAME="part"

# EOS experiments directory — final results are copied here
EOS_EXPERIMENTS_DIR="${EOS_HOME}/experiments"

# Training configuration files (inside the part repo on EOS)
DATA_CONFIG="${PART_DIR}/data/low-pt/lowpt_tau_pretrain.yaml"
NETWORK_CONFIG="${PART_DIR}/networks/lowpt_tau_BackbonePretrain.py"

# Condor job identifiers (injected via environment in .sub file)
CLUSTER_ID="${CONDOR_CLUSTER_ID:-$$}"
PROC_ID="${CONDOR_PROC_ID:-0}"


# =============================================================================
# Local paths (fast /tmp on the batch node — lost after job ends)
# =============================================================================

LOCAL_WORK_DIR="/tmp/pretrain_${CLUSTER_ID}_${PROC_ID}"
LOCAL_EXPERIMENTS_DIR="${LOCAL_WORK_DIR}/experiments"

# Background sync interval (seconds) for copying logs to EOS placeholder.
# Set to 0 to disable periodic sync.
LOG_SYNC_INTERVAL=300


# =============================================================================
# Parse --model-name from forwarded arguments
# =============================================================================
# We need the model name in the shell script to locate the experiment
# directory created by pretrain_backbone.py (named {model_name}_{timestamp}).
# All other arguments are forwarded opaquely to the Python script.

TRAINING_ARGS=("$@")
MODEL_NAME="BackbonePretrain"

previous_argument=""
for argument in "${TRAINING_ARGS[@]}"; do
    if [ "${previous_argument}" = "--model-name" ]; then
        MODEL_NAME="${argument}"
    fi
    previous_argument="${argument}"
done


# =============================================================================
# Helper functions
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Find the experiment directory created by pretrain_backbone.py.
# Layout: ${LOCAL_EXPERIMENTS_DIR}/${MODEL_NAME}_{YYYYMMDD_HHMMSS}/
find_local_experiment_dir() {
    find "${LOCAL_EXPERIMENTS_DIR}" -maxdepth 1 -type d \
        -name "${MODEL_NAME}_*" 2>/dev/null | sort | tail -1
}

# Copy full experiment results from /tmp to EOS.
copy_results_to_eos() {
    local experiment_dir
    experiment_dir=$(find_local_experiment_dir)

    if [ -z "${experiment_dir}" ] || [ ! -d "${experiment_dir}" ]; then
        log "WARNING: No experiment directory found in ${LOCAL_EXPERIMENTS_DIR}"
        return 1
    fi

    local experiment_name
    experiment_name=$(basename "${experiment_dir}")
    local eos_target="${EOS_EXPERIMENTS_DIR}/${experiment_name}"

    log "Copying results to EOS..."
    log "  Source: ${experiment_dir}"
    log "  Target: ${eos_target}"

    mkdir -p "${eos_target}"
    # rsync for incremental copy — safe if the job is killed mid-transfer
    rsync -a "${experiment_dir}/" "${eos_target}/"

    # Append completion metadata
    cat >> "${eos_target}/job_info.txt" << COMPLETION_EOF

--- Job Completed ---
End time:    $(date '+%Y-%m-%d %H:%M:%S')
Exit code:   ${TRAINING_EXIT_CODE:-unknown}
COMPLETION_EOF

    log "Results copied to EOS: ${eos_target}"
}

# Background loop: periodically sync training.log and loss_history.json
# to the EOS placeholder so the user can monitor progress without SSH.
sync_logs_periodically() {
    while true; do
        sleep "${LOG_SYNC_INTERVAL}"
        local experiment_dir
        experiment_dir=$(find_local_experiment_dir)
        if [ -n "${experiment_dir}" ] && [ -d "${experiment_dir}" ]; then
            cp -f "${experiment_dir}/training.log"     "${PLACEHOLDER_DIR}/" 2>/dev/null || true
            cp -f "${experiment_dir}/loss_history.json" "${PLACEHOLDER_DIR}/" 2>/dev/null || true
        fi
    done
}

# Cleanup handler — runs on EXIT (normal completion, error, or SIGTERM).
cleanup() {
    local exit_code=$?
    TRAINING_EXIT_CODE=${exit_code}
    log "Exit handler triggered (exit code: ${exit_code})"

    # Stop the background log sync process
    if [ -n "${SYNC_PID:-}" ] && kill -0 "${SYNC_PID}" 2>/dev/null; then
        kill "${SYNC_PID}" 2>/dev/null || true
        wait "${SYNC_PID}" 2>/dev/null || true
    fi

    # Copy results to EOS (partial or complete)
    copy_results_to_eos || log "WARNING: Failed to copy results to EOS"

    # Remove the .running placeholder (results now live in the final dir)
    if [ -d "${PLACEHOLDER_DIR:-}" ]; then
        rm -rf "${PLACEHOLDER_DIR}"
        log "Removed running placeholder"
    fi

    # Clean up local work directory
    log "Cleaning up: ${LOCAL_WORK_DIR}"
    rm -rf "${LOCAL_WORK_DIR}"
    log "Done."
}


# =============================================================================
# Main execution
# =============================================================================

# Register cleanup handler — fires on normal exit, errors, and SIGTERM
# (HTCondor sends SIGTERM before SIGKILL on preemption/eviction).
trap cleanup EXIT

PLACEHOLDER_DIR="${EOS_EXPERIMENTS_DIR}/.running_condor_${CLUSTER_ID}"

log "=========================================="
log "  HTCondor Backbone Pretraining"
log "=========================================="
log ""
log "Condor job:   ${CLUSTER_ID}.${PROC_ID}"
log "Hostname:     $(hostname)"
log "GPU:          $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "Model name:   ${MODEL_NAME}"
log "Training args: ${TRAINING_ARGS[*]:-<defaults>}"
log "Local workdir: ${LOCAL_WORK_DIR}"
log ""


# ---- Set up local work directory ----
mkdir -p "${LOCAL_EXPERIMENTS_DIR}"


# ---- Create EOS placeholder ----
# The placeholder directory tells the user that a job is running.
# It contains job metadata and periodically updated training logs.
# When the job completes, results are copied to the final experiment
# directory and this placeholder is removed.
mkdir -p "${PLACEHOLDER_DIR}"
cat > "${PLACEHOLDER_DIR}/job_info.txt" << JOB_INFO_EOF
HTCondor Pretraining Job
========================
Condor cluster ID: ${CLUSTER_ID}
Condor process ID: ${PROC_ID}
Start time:        $(date '+%Y-%m-%d %H:%M:%S')
Hostname:          $(hostname)
GPU:               $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')

Model name:    ${MODEL_NAME}
Training args: ${TRAINING_ARGS[*]:-<defaults>}

Local workdir: ${LOCAL_WORK_DIR}

To check live progress:
  condor_ssh_to_job ${CLUSTER_ID}.${PROC_ID}
  tail -f ${LOCAL_WORK_DIR}/experiments/${MODEL_NAME}_*/training.log
JOB_INFO_EOF

log "Created EOS placeholder: ${PLACEHOLDER_DIR}"


# ---- Activate conda environment ----
# Activate by full path (more reliable on batch nodes where the env may
# not be registered in conda's envs_dirs).
CONDA_ENV_PATH="${CONDA_DIR}/envs/${CONDA_ENV_NAME}"
log "Activating conda environment: ${CONDA_ENV_PATH}"
# shellcheck source=/dev/null
source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PATH}"

log "Python:  $(which python)"
log "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
log "CUDA:    $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"


# ---- Set PYTHONPATH for weaver imports ----
# pretrain_backbone.py imports from weaver.nn.model and weaver.utils.data
export PYTHONPATH="${WEAVER_DIR}:${PYTHONPATH:-}"


# ---- Start background log sync ----
if [ "${LOG_SYNC_INTERVAL}" -gt 0 ]; then
    sync_logs_periodically &
    SYNC_PID=$!
    log "Background log sync started (PID: ${SYNC_PID}, interval: ${LOG_SYNC_INTERVAL}s)"
fi


# ---- Launch training ----
log ""
log "Launching pretrain_backbone.py..."
log "  Data dir:    ${DATA_DIR}"
log "  Data config: ${DATA_CONFIG}"
log "  Network:     ${NETWORK_CONFIG}"
log "  Output:      ${LOCAL_EXPERIMENTS_DIR}"
log ""

cd "${PART_DIR}"

python pretrain_backbone.py \
    --data-config "${DATA_CONFIG}" \
    --data-dir "${DATA_DIR}" \
    --network "${NETWORK_CONFIG}" \
    --experiments-dir "${LOCAL_EXPERIMENTS_DIR}" \
    --device cuda:0 \
    --amp \
    "${TRAINING_ARGS[@]}"

TRAINING_EXIT_CODE=$?
log "Training completed with exit code: ${TRAINING_EXIT_CODE}"

# The EXIT trap handles copying results and cleanup.
