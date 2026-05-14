#!/bin/bash
# BiLSTM unweighted, per-window context (7/30/90/365d). Simple model (best
# config will be applied here once HP tune finishes).
#
# Usage:
#   sbatch --array=0-3 slurm_bilstm_unweighted_ctx.sh
#
#SBATCH --job-name=lstm_uw_ctx
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=/path/to/CHD_MEDS/results_v4/slurm_lstm_uw_ctx_%A_%a.log
#SBATCH --error=/path/to/CHD_MEDS/results_v4/slurm_lstm_uw_ctx_%A_%a.err
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

echo "=== BiLSTM unweighted (window=${W}d) ==="
python -m baselines.lstm \
    --output_dir $OUT \
    --seed 42 \
    --epochs 20 --batch_size 64 --lr 1e-3 \
    --window_days $W \
    --unweighted \
    --suffix _unweighted_window_${W}d
