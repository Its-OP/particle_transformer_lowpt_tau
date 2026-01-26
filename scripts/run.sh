#!/bin/bash
set -e

cd "$(dirname "$0")/.."

eval "$(conda shell.bash hook)"
conda activate parT

SESSION="parT-training"

screen -dmS "$SESSION"

screen -S "$SESSION" -X screen -t training bash -c './train_QuarkGluon.sh ParT kinpid --batch-size 64 --num-workers 0; exec bash'
screen -S "$SESSION" -X screen -t profiling bash -c 'nvidia-smi -l 1; exec bash'

echo "Started screen session '$SESSION' with windows: training, profiling"
echo "Attach with: screen -r $SESSION"
echo "Detach with: Ctrl+A, D"
echo "Switch windows with: Ctrl+A, n"