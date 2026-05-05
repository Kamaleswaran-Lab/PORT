#!/bin/bash
#SBATCH --job-name=base_tune
#SBATCH --partition=common
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/path/to/CHD_MEDS/results/baselines_tuned/slurm_tune_%j.log
#SBATCH --error=/path/to/CHD_MEDS/results/baselines_tuned/slurm_tune_%j.err

set -euo pipefail
mkdir -p /path/to/CHD_MEDS/results/baselines_tuned
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ethos
cd .
python -u -m baselines.logreg_xgb_tuned --feature_set both --xgb_trials 50 --seed 42
echo "done"
