#!/bin/bash
set -e

REPO_DIR="particle_transformer_lowpt_tau"

echo "Starting GPU profiling screen..."
screen -dmS profiling bash -c "nvidia-smi -l 1"

echo "Starting training screen..."
screen -dmS training bash -c "conda activate parT && cd $REPO_DIR && ./train_QuarkGluon.sh ParT kinpid --batch-size 512 --num-workers 0"

echo "Screens started:"
echo "  - profiling: nvidia-smi monitoring"
echo "  - training: ParT training"
echo ""
echo "Attach with: screen -r <name>"
echo "List screens: screen -ls"
echo "Exit screen: Ctrl + A, D"