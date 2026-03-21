#!/bin/bash
# Unzip dataset.zip and remove the archive.
# Usage: bash unzip_dataset.sh
set -euo pipefail
cd "$(dirname "$0")"
unzip -o dataset.zip
rm dataset.zip
echo "Done: $(find train -maxdepth 1 -name '*.parquet' | wc -l | tr -d ' ') train, $(find val -maxdepth 1 -name '*.parquet' | wc -l | tr -d ' ') val files."
