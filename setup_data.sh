#!/bin/bash
# Download and extract split parquet files from Google Drive.
# Usage: bash setup_data.sh
#
# Set TRAIN_ZIP_ID and VAL_ZIP_ID to the Google Drive file IDs
# before running. File ID is the part after /d/ in the share URL:
# https://drive.google.com/file/d/<FILE_ID>/view

set -euo pipefail

TRAIN_ZIP_ID="${TRAIN_ZIP_ID:-1_TpMxxoBzJOyQlsrgLj5gprz8qmBoK65}"
VAL_ZIP_ID="${VAL_ZIP_ID:-1nSXqH22D9RehRlr9jAYRYoUNNuCuVFW6}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data/low-pt"

# Download and extract train split
TRAIN_FILE_COUNT=$(find "${DATA_DIR}/train" -maxdepth 1 -name "*.parquet" 2>/dev/null | wc -l | tr -d ' ')
if [ ! -d "${DATA_DIR}/train" ] || [ "$TRAIN_FILE_COUNT" -lt 10 ]; then
    echo "Downloading train split..."
    gdown --id "${TRAIN_ZIP_ID}" -O "${DATA_DIR}/train_split.zip" || {
        echo "gdown failed (likely rate-limited). Retrying with curl..."
        curl -L -o "${DATA_DIR}/train_split.zip" \
            "https://drive.google.com/uc?export=download&confirm=t&id=${TRAIN_ZIP_ID}"
    }
    unzip -o "${DATA_DIR}/train_split.zip" -d "${DATA_DIR}/"
    rm "${DATA_DIR}/train_split.zip"
    echo "Train split: $(find "${DATA_DIR}/train" -maxdepth 1 -name "*.parquet" | wc -l | tr -d ' ') files"
fi

# Download and extract val split
VAL_FILE_COUNT=$(find "${DATA_DIR}/val" -maxdepth 1 -name "*.parquet" 2>/dev/null | wc -l | tr -d ' ')
if [ ! -d "${DATA_DIR}/val" ] || [ "$VAL_FILE_COUNT" -lt 10 ]; then
    echo "Downloading val split..."
    gdown --id "${VAL_ZIP_ID}" -O "${DATA_DIR}/val_split.zip" || {
        echo "gdown failed (likely rate-limited). Retrying with curl..."
        curl -L -o "${DATA_DIR}/val_split.zip" \
            "https://drive.google.com/uc?export=download&confirm=t&id=${VAL_ZIP_ID}"
    }
    unzip -o "${DATA_DIR}/val_split.zip" -d "${DATA_DIR}/"
    rm "${DATA_DIR}/val_split.zip"
    echo "Val split: $(find "${DATA_DIR}/val" -maxdepth 1 -name "*.parquet" | wc -l | tr -d ' ') files"
fi

echo "Data ready."
