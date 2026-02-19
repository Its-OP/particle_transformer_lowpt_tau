#!/bin/bash
set -e

CONDA_ENV="part"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

INPUT="${1:-${SCRIPT_DIR}/data/low-pt/merged_noBKstar.root}"
OUTPUT="${2:-${SCRIPT_DIR}/data/low-pt/lowpt_tau_trackorigin.parquet}"

echo "Converting low-pT tau dataset:"
echo "  Input:  ${INPUT}"
echo "  Output: ${OUTPUT}"

conda run --no-capture-output -n "${CONDA_ENV}" \
    python "${SCRIPT_DIR}/utils/convert_lowpt_tau_dataset.py" \
    -i "${INPUT}" \
    -o "${OUTPUT}"
