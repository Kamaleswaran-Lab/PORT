#!/bin/bash
#SBATCH --job-name=lstm_tune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/CHD_MEDS/results/baselines_tuned/slurm_lstm_%j.log
#SBATCH --error=/path/to/CHD_MEDS/results/baselines_tuned/slurm_lstm_%j.err

set -euo pipefail
mkdir -p /path/to/CHD_MEDS/results/baselines_tuned
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ethos
cd .
python -u -m baselines.lstm_tuned --max_epochs 25 --patience 5 --batch_size 64 --seed 42
echo "done"
