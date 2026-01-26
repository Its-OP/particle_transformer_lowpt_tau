#!/bin/bash
set -e

cd "$(dirname "$0")/.."

eval "$(conda shell.bash hook)"

if ! command -v screen &> /dev/null; then
    sudo apt-get update && sudo apt-get install -y screen
fi

conda env create -f environment.yml -n parT
conda activate parT

pip install -r requirements.txt

python ./get_datasets.py QuarkGluon -d datasets

echo "Setup complete. Run 'conda activate parT' to use the environment."