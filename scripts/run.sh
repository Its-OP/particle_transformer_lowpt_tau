#!/bin/bash
set -e

echo "Starting GPU profiling screen..."
screen -dmS profiling bash -c "nvidia-smi -l 1"

echo "Starting training screen..."
screen -dmS training bash -c "cd particle_transformer_lowpt_tau && conda run --no-capture-output -n parT ./train_QuarkGluon.sh ParT kinpid --batch-size 128 --in-memory --use-amp"
echo "Screens started:"
echo "  - profiling: nvidia-smi monitoring"
echo "  - training: ParT training"
echo ""
echo "Attach with: screen -r <name>"
echo "List screens: screen -ls"
echo "Exit screen: Ctrl + A, D"