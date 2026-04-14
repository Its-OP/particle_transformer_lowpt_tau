#!/bin/bash
# =============================================================================
# Sweep launcher: train the per-couple reranker with several values of
# `--top-k2` (the number of cascade-Stage-2 tracks fed into couple
# enumeration). Sequential — single GPU.
#
# Self-detaches into a screen session named `topk2_sweep` and starts a
# second screen `topk2_sweep_gpu` running `watch -n 1 nvidia-smi`. The
# launching shell exits immediately after printing reattach instructions,
# so you can run this from ssh and walk away.
#
# Each subrun gets its own folder under
#     experiments/topk2_sweep_<timestamp>/topk2_<K>/
# and the trainer's normal experiment dir lands inside that. After every
# subrun (and at the end), `diagnostics/aggregate_couple_sweep.py`
# collects each subrun's loss_history.json into:
#     experiments/topk2_sweep_<timestamp>/sweep_summary.json
#     experiments/topk2_sweep_<timestamp>/sweep_summary.md
#
# All training output for one subrun is also tee'd to
#     experiments/topk2_sweep_<timestamp>/topk2_<K>/training.log
# and the sweep-level orchestration log lives at
#     experiments/topk2_sweep_<timestamp>/sweep.log
# so you can grep across runs without opening every per-subrun directory.
#
# Usage:
#   bash sweep_topk2.sh                          # default: 10 K values
#   TOP_K2_VALUES="50 100 200" bash sweep_topk2.sh
#   EPOCHS=30 BATCH_SIZE=64 bash sweep_topk2.sh
#   NO_SCREEN=1 bash sweep_topk2.sh              # run inline (debugging)
#
# Overnight estimate: with the defaults below (10 values × 50 epochs ×
# 100 steps × batch 96), expect ~10-16 hours on a single GPU — total
# step count is 50k (vs ~135k for the old default), but each step is
# heavier because batch is 6x larger. Tune EPOCHS, STEPS_PER_EPOCH,
# or the K-value list if your wallclock budget is tighter; cost grows
# roughly linearly in EPOCHS and roughly as O(top_k2^2) per subrun.
# =============================================================================
set -euo pipefail

# ---- K values to sweep over ----
# Span from the current baseline (50) to the largest sensible pool (200,
# close to top-K1=256). Step is 10 in the lower half (50..100) to give
# fine resolution near the baseline, and 25 in the upper half because the
# cost of each subrun grows roughly as O(K2^2). Override via TOP_K2_VALUES.
TOP_K2_VALUES="${TOP_K2_VALUES:-50 60 70 80 90 100 125 150 175 200}"

# ---- Couple metric K grid for every subrun ----
# 50, 60, 70, ..., 200 — step 10. The trainer's selection criterion is
# C@100_couples, so 100 must be in this list (the trainer will fail
# fast otherwise).
K_VALUES_COUPLES="50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200"
K_VALUES_TRACKS="30 50 75 100 150 200"

# ---- Common training config (overridable via env vars) ----
# Batch size and steps per epoch are coupled: with the 6x batch bump
# from 16 → 96, leaving steps_per_epoch at the old 500 would push the
# per-epoch sample count from 8k to 48k, which causes the model to see
# the dataset many more times per epoch and risks memorization. Cutting
# steps to 100 gives ~9.6k events/epoch — same ballpark as the old
# baseline — while keeping the larger-batch optimization benefits.
EPOCHS="${EPOCHS:-60}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-200}"
BATCH_SIZE="${BATCH_SIZE:-96}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
SCHEDULER="${SCHEDULER:-cosine}"
COSINE_POWER="${COSINE_POWER:-1.0}"
DEVICE="${DEVICE:-cuda:0}"
NUM_WORKERS="${NUM_WORKERS:-10}"
KEEP_BEST_K="${KEEP_BEST_K:-5}"

# ---- CoupleReranker architecture ----
COUPLE_HIDDEN_DIM="${COUPLE_HIDDEN_DIM:-256}"
COUPLE_NUM_RESIDUAL_BLOCKS="${COUPLE_NUM_RESIDUAL_BLOCKS:-4}"
COUPLE_DROPOUT="${COUPLE_DROPOUT:-0.1}"

# ---- Paths (overridable so smoke tests can point at the subset dataset) ----
DATA_CONFIG="${DATA_CONFIG:-data/low-pt/lowpt_tau_trackfinder.yaml}"
DATA_DIR="${DATA_DIR:-data/low-pt/train/}"
VAL_DATA_DIR="${VAL_DATA_DIR:-data/low-pt/val/}"
NETWORK="${NETWORK:-networks/lowpt_tau_CoupleReranker.py}"
CASCADE_CHECKPOINT="${CASCADE_CHECKPOINT:-models/cascade_best.pt}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-part}"

# ---- Screen session names ----
SESSION_SWEEP="topk2_sweep"
SESSION_GPU="topk2_sweep_gpu"

# ---- Resolve script directory ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# Phase 1 — Pre-flight checks (run BEFORE relaunching in screen so the user
# sees errors immediately, on the launching terminal).
# =============================================================================

if [ ! -f "${SCRIPT_DIR}/${CASCADE_CHECKPOINT}" ]; then
    echo "ERROR: Cascade checkpoint not found: ${SCRIPT_DIR}/${CASCADE_CHECKPOINT}"
    echo "  Build the cascade first via train_cascade.sh, or symlink an"
    echo "  existing checkpoint into models/cascade_best.pt."
    exit 1
fi

TRAIN_PARQUET_COUNT=$(find "${SCRIPT_DIR}/${DATA_DIR}" -maxdepth 1 -name "*.parquet" 2>/dev/null | wc -l | tr -d ' ')
VAL_PARQUET_COUNT=$(find "${SCRIPT_DIR}/${VAL_DATA_DIR}" -maxdepth 1 -name "*.parquet" 2>/dev/null | wc -l | tr -d ' ')
if [ "$TRAIN_PARQUET_COUNT" -lt 10 ] || [ "$VAL_PARQUET_COUNT" -lt 10 ]; then
    echo "WARNING: Found ${TRAIN_PARQUET_COUNT} train and ${VAL_PARQUET_COUNT} val parquet files."
fi

# =============================================================================
# Phase 2 — Self-relaunch in a detached screen session.
#
# `STY` is set inside any screen session and unset outside, so we use it
# to detect whether this invocation is the user's interactive launch or
# the relaunched copy running inside screen. The user can opt out via
# NO_SCREEN=1 (useful for testing/debugging — the loop runs inline).
# =============================================================================

if [ -z "${STY:-}" ] && [ "${NO_SCREEN:-0}" != "1" ]; then
    # Refuse to clobber an existing sweep session — the previous one
    # could still be running.
    if screen -list 2>/dev/null | grep -q "\.${SESSION_SWEEP}"; then
        echo "Screen session '${SESSION_SWEEP}' already exists."
        echo "Reattach: screen -r ${SESSION_SWEEP}"
        echo "Kill it:  screen -S ${SESSION_SWEEP} -X quit"
        exit 1
    fi

    # Tear down any stale GPU monitor from a previous sweep.
    screen -list 2>/dev/null | grep "\.${SESSION_GPU}" | awk '{print $1}' \
        | while read -r session_id; do
            screen -S "$session_id" -X quit 2>/dev/null || true
        done || true

    # Create the sweep root NOW so the user-facing banner shows the same
    # path that the inner script will use, and so they can `cd` into it
    # immediately and watch logs accumulate. The relaunched copy reads
    # SWEEP_ROOT_OVERRIDE from the environment instead of generating a
    # new timestamp.
    SWEEP_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    SWEEP_ROOT="${SCRIPT_DIR}/experiments/topk2_sweep_${SWEEP_TIMESTAMP}"
    mkdir -p "${SWEEP_ROOT}"
    export SWEEP_ROOT_OVERRIDE="${SWEEP_ROOT}"

    # Optional GPU monitor session (only if nvidia-smi exists).
    GPU_MONITOR_AVAILABLE=0
    if command -v nvidia-smi &>/dev/null; then
        screen -dmS "${SESSION_GPU}" bash -c "watch -n 1 nvidia-smi"
        GPU_MONITOR_AVAILABLE=1
    fi

    # Relaunch self in a detached screen. The trailing `read` keeps the
    # session alive after the loop finishes so the user can attach and
    # see the final summary.
    screen -dmS "${SESSION_SWEEP}" bash -c "
        bash '${BASH_SOURCE[0]}'
        echo ''
        echo '--- Sweep finished. Press Enter to close this screen session. ---'
        read
    "

    NUM_K_VALUES=$(echo "${TOP_K2_VALUES}" | wc -w | tr -d ' ')
    echo "============================================================"
    echo "  CoupleReranker top_k2 sweep launched in screen"
    echo "============================================================"
    echo ""
    echo "Sweep session:    ${SESSION_SWEEP}"
    if [ "${GPU_MONITOR_AVAILABLE}" -eq 1 ]; then
        echo "GPU monitor:      ${SESSION_GPU}"
    fi
    echo "Sweep root:       ${SWEEP_ROOT}"
    echo ""
    echo "K values:         ${TOP_K2_VALUES} (${NUM_K_VALUES} total)"
    echo "Epochs/run:       ${EPOCHS}"
    echo "Steps/epoch:      ${STEPS_PER_EPOCH}"
    echo "Batch size:       ${BATCH_SIZE}"
    echo "Learning rate:    ${LEARNING_RATE}"
    echo "Device:           ${DEVICE}"
    echo ""
    echo "Reattach sweep:   screen -r ${SESSION_SWEEP}"
    if [ "${GPU_MONITOR_AVAILABLE}" -eq 1 ]; then
        echo "Reattach gpu:     screen -r ${SESSION_GPU}"
    fi
    echo "Detach:           Ctrl+A, then D"
    echo "Kill sweep:       screen -S ${SESSION_SWEEP} -X quit"
    echo ""
    echo "Tail sweep log:   tail -f ${SWEEP_ROOT}/sweep.log"
    echo "Live summary:     watch -n 30 cat ${SWEEP_ROOT}/sweep_summary.md"
    echo ""
    exit 0
fi

# =============================================================================
# Phase 3 — Inside the screen session (or NO_SCREEN=1 inline mode).
# Run the actual sweep loop.
# =============================================================================

# Use the sweep root from the launching shell if it was set, so the
# inner banner matches the outer one. Otherwise (NO_SCREEN=1 path),
# generate a fresh timestamp.
if [ -n "${SWEEP_ROOT_OVERRIDE:-}" ]; then
    SWEEP_ROOT="${SWEEP_ROOT_OVERRIDE}"
else
    SWEEP_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    SWEEP_ROOT="${SCRIPT_DIR}/experiments/topk2_sweep_${SWEEP_TIMESTAMP}"
    mkdir -p "${SWEEP_ROOT}"
fi

SWEEP_LOG="${SWEEP_ROOT}/sweep.log"
exec > >(tee -a "${SWEEP_LOG}") 2>&1

# ---- Conda activation (inside screen) ----
if command -v conda &>/dev/null; then
    CONDA_BASE=$(conda info --base)
elif [ -d "$HOME/miniconda3" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -d "/opt/miniconda3" ]; then
    CONDA_BASE="/opt/miniconda3"
else
    echo "ERROR: conda not found."
    exit 1
fi
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

# ---- Banner ----
NUM_K_VALUES=$(echo "${TOP_K2_VALUES}" | wc -w | tr -d ' ')
echo "================================================================"
echo "  CoupleReranker top_k2 sweep"
echo "================================================================"
echo "Sweep root:            ${SWEEP_ROOT}"
echo "Started:               $(date '+%Y-%m-%d %H:%M:%S')"
echo "K values:              ${TOP_K2_VALUES} (${NUM_K_VALUES} total)"
echo "K_couples (per run):   ${K_VALUES_COUPLES}"
echo "K_tracks (per run):    ${K_VALUES_TRACKS}"
echo "Epochs (per run):      ${EPOCHS}"
echo "Steps/epoch:           ${STEPS_PER_EPOCH}"
echo "Batch size:            ${BATCH_SIZE}"
echo "Learning rate:         ${LEARNING_RATE}"
echo "Device:                ${DEVICE}"
echo "Cascade checkpoint:    ${CASCADE_CHECKPOINT}"
echo "Train parquet files:   ${TRAIN_PARQUET_COUNT}"
echo "Val parquet files:     ${VAL_PARQUET_COUNT}"
echo ""

# ---- Run each top_k2 value ----
NUM_TOTAL=0
NUM_OK=0
NUM_FAILED=0
FAILED_K_VALUES=""

for K in ${TOP_K2_VALUES}; do
    NUM_TOTAL=$((NUM_TOTAL + 1))
    SUBRUN_DIR="${SWEEP_ROOT}/topk2_${K}"
    mkdir -p "${SUBRUN_DIR}"
    SUBRUN_LOG="${SUBRUN_DIR}/training.log"

    echo "----------------------------------------------------------------"
    echo "  [Run ${NUM_TOTAL}/${NUM_K_VALUES}]  top_k2=${K}"
    echo "  Started:  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Subrun:   ${SUBRUN_DIR}"
    echo "----------------------------------------------------------------"

    set +e
    # Adaptive batch size: C(K,2) grows quadratically, reduce batch for
    # large K to prevent OOM (K=200 OOM'd at batch=96 on 95 GB GPU).
    if [ "${K}" -ge 200 ]; then
        EFFECTIVE_BATCH=64
    elif [ "${K}" -ge 175 ]; then
        EFFECTIVE_BATCH=80
    else
        EFFECTIVE_BATCH="${BATCH_SIZE}"
    fi

    python train_couple_reranker.py \
        --data-config "${DATA_CONFIG}" \
        --data-dir "${DATA_DIR}" \
        --val-data-dir "${VAL_DATA_DIR}" \
        --network "${NETWORK}" \
        --cascade-checkpoint "${CASCADE_CHECKPOINT}" \
        --top-k2 "${K}" \
        --k-values-couples ${K_VALUES_COUPLES} \
        --k-values-tracks ${K_VALUES_TRACKS} \
        --couple-hidden-dim "${COUPLE_HIDDEN_DIM}" \
        --couple-num-residual-blocks "${COUPLE_NUM_RESIDUAL_BLOCKS}" \
        --couple-dropout "${COUPLE_DROPOUT}" \
        --model-name "topk2_${K}" \
        --experiments-dir "${SUBRUN_DIR}" \
        --epochs "${EPOCHS}" \
        --batch-size "${EFFECTIVE_BATCH}" \
        --steps-per-epoch "${STEPS_PER_EPOCH}" \
        --lr "${LEARNING_RATE}" \
        --scheduler "${SCHEDULER}" \
        --cosine-power "${COSINE_POWER}" \
        --device "${DEVICE}" \
        --num-workers "${NUM_WORKERS}" \
        --keep-best-k "${KEEP_BEST_K}" \
        2>&1 | tee "${SUBRUN_LOG}"
    RUN_STATUS=${PIPESTATUS[0]}
    set -e

    if [ "${RUN_STATUS}" -eq 0 ]; then
        NUM_OK=$((NUM_OK + 1))
        echo ""
        echo "  ✓ top_k2=${K} finished OK at $(date '+%Y-%m-%d %H:%M:%S')"
    else
        NUM_FAILED=$((NUM_FAILED + 1))
        FAILED_K_VALUES="${FAILED_K_VALUES} ${K}"
        echo ""
        echo "  ✗ top_k2=${K} FAILED with exit code ${RUN_STATUS}"
        echo "  Continuing with the next K value..."
    fi

    # Re-aggregate after every run so partial results are visible even
    # if the sweep is killed mid-flight.
    python diagnostics/aggregate_couple_sweep.py "${SWEEP_ROOT}" || true
    echo ""
done

# ---- Final aggregation ----
echo "================================================================"
echo "  Sweep complete: ${NUM_OK} OK, ${NUM_FAILED} failed, ${NUM_TOTAL} total"
echo "  Finished:       $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
if [ -n "${FAILED_K_VALUES}" ]; then
    echo "Failed K values:${FAILED_K_VALUES}"
fi
echo ""
echo "Sweep root:        ${SWEEP_ROOT}"
echo "Per-run logs:      ${SWEEP_ROOT}/topk2_*/training.log"
echo "Sweep log:         ${SWEEP_LOG}"
echo "Summary (json):    ${SWEEP_ROOT}/sweep_summary.json"
echo "Summary (md):      ${SWEEP_ROOT}/sweep_summary.md"
echo ""
cat "${SWEEP_ROOT}/sweep_summary.md" || true
