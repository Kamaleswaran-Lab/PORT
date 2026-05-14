#!/bin/bash
# LR + XGBoost unweighted HP tune (full history, both feature sets).
# CPU-only; submit to common partition.
#
#SBATCH --job-name=lrxgb_uw
#SBATCH --partition=common
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/path/to/CHD_MEDS/results_v4/slurm_lrxgb_uw_%j.log
#SBATCH --error=/path/to/CHD_MEDS/results_v4/slurm_lrxgb_uw_%j.err

set -e
cd /path/to/CHDLLM
export PATH=/path/to/miniforge3/envs/ethos/bin:$PATH
export PYTHONPATH=/path/to/CHDLLM:$PYTHONPATH
export CHD_DATA_ROOT=/path/to/CHD_MEDS

OUT=/path/to/CHD_MEDS/results/baselines_tuned
mkdir -p $OUT

echo "=== LR+XGB unweighted HP tune (manual + MEDS) ==="
python -m baselines.logreg_xgb_tuned \
    --feature_set both \
    --xgb_trials 50 \
    --seed 42 \
    --output_dir $OUT \
    --unweighted \
    --suffix _unweighted_tuned

echo "=== Done ==="
