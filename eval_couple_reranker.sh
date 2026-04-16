#!/bin/bash
# =============================================================================
# Evaluate the trained CoupleReranker on the validation set and export
# results as a parquet file with track indices.
#
# Output: a parquet with per-event (run, event_id, luminosity_block) +
#   couples (list of [idx_i, idx_j] track-index pairs, top-200 by
#   reranker confidence) + remaining_pions (list of track indices from
#   the Stage 1 prefilter top-K1). The consumer joins with the input
#   parquet to look up any track property (pT, eta, charge, etc.).
#
# Usage:
#   bash eval_couple_reranker.sh                    # defaults
#   DATA_DIR=data/low-pt/subset/val DEVICE=mps bash eval_couple_reranker.sh
# =============================================================================
set -euo pipefail

# ---- Checkpoints ----
CASCADE_CHECKPOINT="${CASCADE_CHECKPOINT:-models/cascade_best.pt}"
COUPLE_CHECKPOINT="${COUPLE_CHECKPOINT:-models/couple_reranker_best.pt}"

# ---- Data ----
DATA_CONFIG="${DATA_CONFIG:-data/low-pt/lowpt_tau_trackfinder.yaml}"
DATA_DIR="${DATA_DIR:-data/low-pt/val/}"
OUTPUT="${OUTPUT:-data/low-pt/eval/couple_reranker_val.parquet}"

# ---- Model config ----
TOP_K2="${TOP_K2:-80}"
TOP_K_OUTPUT_COUPLES="${TOP_K_OUTPUT_COUPLES:-200}"
NETWORK="${NETWORK:-networks/lowpt_tau_CoupleReranker.py}"

# ---- Runtime ----
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-10}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-part}"

# ---- Resolve script directory ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Pre-flight checks ----
if [ ! -f "${SCRIPT_DIR}/${CASCADE_CHECKPOINT}" ]; then
    echo "ERROR: Cascade checkpoint not found: ${SCRIPT_DIR}/${CASCADE_CHECKPOINT}"
    exit 1
fi
if [ ! -f "${SCRIPT_DIR}/${COUPLE_CHECKPOINT}" ]; then
    echo "ERROR: Couple reranker checkpoint not found: ${SCRIPT_DIR}/${COUPLE_CHECKPOINT}"
    exit 1
fi

PARQUET_COUNT=$(find "${SCRIPT_DIR}/${DATA_DIR}" -maxdepth 1 -name "*.parquet" 2>/dev/null | wc -l | tr -d ' ')
if [ "$PARQUET_COUNT" -eq 0 ]; then
    echo "ERROR: No .parquet files found in ${SCRIPT_DIR}/${DATA_DIR}"
    exit 1
fi
echo "Found ${PARQUET_COUNT} parquet files in ${DATA_DIR}"

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

# ---- Banner ----
echo "============================================"
echo "  CoupleReranker Evaluation"
echo "============================================"
echo "Cascade checkpoint:    ${CASCADE_CHECKPOINT}"
echo "Couple checkpoint:     ${COUPLE_CHECKPOINT}"
echo "Data directory:        ${DATA_DIR} (${PARQUET_COUNT} files)"
echo "Output:                ${OUTPUT}"
echo "Top-K2:                ${TOP_K2}"
echo "Output couples:        ${TOP_K_OUTPUT_COUPLES}"
echo "Device:                ${DEVICE}"
echo "Batch size:            ${BATCH_SIZE}"
echo ""

# ---- Run ----
cd "${SCRIPT_DIR}"
python diagnostics/eval_couple_reranker.py \
    --cascade-checkpoint "${CASCADE_CHECKPOINT}" \
    --couple-checkpoint "${COUPLE_CHECKPOINT}" \
    --data-config "${DATA_CONFIG}" \
    --data-dir "${DATA_DIR}" \
    --network "${NETWORK}" \
    --output "${OUTPUT}" \
    --top-k2 "${TOP_K2}" \
    --top-k-output-couples "${TOP_K_OUTPUT_COUPLES}" \
    --device "${DEVICE}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}"

echo ""
echo "Done. Output: ${OUTPUT}"
