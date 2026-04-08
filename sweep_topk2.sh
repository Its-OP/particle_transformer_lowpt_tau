#!/bin/bash
# =============================================================================
# Sweep launcher: train the per-couple reranker with several values of
# `--top-k2` (the number of cascade-Stage-2 tracks fed into couple
# enumeration). Sequential — single GPU.
#
# Each subrun gets its own folder under
#     experiments/topk2_sweep_<timestamp>/topk2_<K>/
# and the trainer's normal experiment dir lands inside that. After all
# subruns finish (or even partway through, if the script is interrupted),
# `diagnostics/aggregate_couple_sweep.py` collects every subrun's
# loss_history.json into:
#     experiments/topk2_sweep_<timestamp>/sweep_summary.json
#     experiments/topk2_sweep_<timestamp>/sweep_summary.md
#
# All training output for one subrun is also tee'd to
#     experiments/topk2_sweep_<timestamp>/topk2_<K>/training.log
# so you can grep across runs without opening every per-subrun directory.
#
# Usage:
#   bash sweep_topk2.sh                          # default: 10 K values
#   TOP_K2_VALUES="50 100 200" bash sweep_topk2.sh
#   EPOCHS=30 BATCH_SIZE=64 bash sweep_topk2.sh
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
EPOCHS="${EPOCHS:-50}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
BATCH_SIZE="${BATCH_SIZE:-96}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
SCHEDULER="${SCHEDULER:-cosine}"
DEVICE="${DEVICE:-cuda:0}"
NUM_WORKERS="${NUM_WORKERS:-10}"
KEEP_BEST_K="${KEEP_BEST_K:-1}"

# ---- CoupleReranker architecture ----
COUPLE_HIDDEN_DIM="${COUPLE_HIDDEN_DIM:-256}"
COUPLE_NUM_RESIDUAL_BLOCKS="${COUPLE_NUM_RESIDUAL_BLOCKS:-4}"
COUPLE_DROPOUT="${COUPLE_DROPOUT:-0.1}"

# ---- Static paths ----
DATA_CONFIG="data/low-pt/lowpt_tau_trackfinder.yaml"
DATA_DIR="data/low-pt/train/"
VAL_DATA_DIR="data/low-pt/val/"
NETWORK="networks/lowpt_tau_CoupleReranker.py"
CASCADE_CHECKPOINT="models/cascade_best.pt"
CONDA_ENV_NAME="part"

# ---- Resolve script directory + sweep root ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_ROOT="${SCRIPT_DIR}/experiments/topk2_sweep_${SWEEP_TIMESTAMP}"
mkdir -p "${SWEEP_ROOT}"

SWEEP_LOG="${SWEEP_ROOT}/sweep.log"
exec > >(tee -a "${SWEEP_LOG}") 2>&1

# ---- Resolve conda ----
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

# ---- Pre-flight checks ----
if [ ! -f "${SCRIPT_DIR}/${CASCADE_CHECKPOINT}" ]; then
    echo "ERROR: Cascade checkpoint not found: ${SCRIPT_DIR}/${CASCADE_CHECKPOINT}"
    exit 1
fi

TRAIN_PARQUET_COUNT=$(find "${SCRIPT_DIR}/${DATA_DIR}" -maxdepth 1 -name "*.parquet" 2>/dev/null | wc -l | tr -d ' ')
VAL_PARQUET_COUNT=$(find "${SCRIPT_DIR}/${VAL_DATA_DIR}" -maxdepth 1 -name "*.parquet" 2>/dev/null | wc -l | tr -d ' ')
if [ "$TRAIN_PARQUET_COUNT" -lt 10 ] || [ "$VAL_PARQUET_COUNT" -lt 10 ]; then
    echo "WARNING: Found ${TRAIN_PARQUET_COUNT} train and ${VAL_PARQUET_COUNT} val parquet files."
fi

# ---- Banner ----
echo "================================================================"
echo "  CoupleReranker top_k2 sweep"
echo "================================================================"
echo "Sweep root:            ${SWEEP_ROOT}"
echo "K values:              ${TOP_K2_VALUES}"
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
    echo "  [Run ${NUM_TOTAL}/$(echo "${TOP_K2_VALUES}" | wc -w | tr -d ' ')]  top_k2=${K}"
    echo "  Started:  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Subrun:   ${SUBRUN_DIR}"
    echo "----------------------------------------------------------------"

    set +e
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
        --batch-size "${BATCH_SIZE}" \
        --steps-per-epoch "${STEPS_PER_EPOCH}" \
        --lr "${LEARNING_RATE}" \
        --scheduler "${SCHEDULER}" \
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
