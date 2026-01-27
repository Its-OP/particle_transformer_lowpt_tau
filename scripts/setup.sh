#!/bin/bash
set -e

echo "Cloning repository..."
git clone https://github.com/Its-OP/particle_transformer_lowpt_tau.git
cd particle_transformer_lowpt_tau

echo "Creating conda environment 'parT'..."
conda create -n parT python=3.13 -y

echo "Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate parT

echo "Installing requirements..."
pip install -r requirements.txt

echo "Creating datasets directory..."
mkdir -p ./datasets

echo "Downloading QuarkGluon dataset..."
python ./get_datasets.py QuarkGluon -d datasets

echo "Installing screen..."
sudo apt-get update && sudo apt-get install -y screen

echo "Setup complete. Environment 'parT' is ready."
echo "To activate: conda activate parT"