#!/bin/bash
# LR + XGBoost unweighted, per-window context (7/30/90/365d).
# CPU-only; submit to common partition.
#
# Usage:
#   sbatch --array=0-3 slurm_lr_xgb_unweighted_ctx.sh
#
#SBATCH --job-name=lrxgb_uw_ctx
#SBATCH --partition=common
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/path/to/CHD_MEDS/results_v4/slurm_lrxgb_uw_ctx_%A_%a.log
#SBATCH --error=/path/to/CHD_MEDS/results_v4/slurm_lrxgb_uw_ctx_%A_%a.err
#SBATCH --array=0-3

set -e
cd /path/to/CHDLLM
export PATH=/path/to/miniforge3/envs/ethos/bin:$PATH
export PYTHONPATH=/path/to/CHDLLM:$PYTHONPATH
export CHD_DATA_ROOT=/path/to/CHD_MEDS

WINDOWS=(7 30 90 365)
W=${WINDOWS[$SLURM_ARRAY_TASK_ID]}

OUT=/path/to/CHD_MEDS/results/baselines
mkdir -p $OUT

echo "=== LR+XGB unweighted (window=${W}d, both feature sets) ==="
python -m baselines.logreg_xgb \
    --feature_set both \
    --output_dir $OUT \
    --window_days $W \
    --unweighted \
    --suffix _unweighted_window_${W}d
