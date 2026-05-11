#!/bin/bash
# Data-efficiency ablation for BiLSTM: subsample training set to
# 1%, 5%, 10%, 25%, 50% of the full corpus (stratified).
#
# Usage:
#   sbatch --array=0-4 slurm_data_eff_bilstm.sh
#
#SBATCH --job-name=data_lstm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --output=/path/to/CHD_MEDS/results_v4/slurm_data_lstm_%A_%a.log
#SBATCH --error=/path/to/CHD_MEDS/results_v4/slurm_data_lstm_%A_%a.err
#SBATCH --array=0-4

set -e
cd /path/to/CHDLLM
export PATH=/path/to/miniforge3/envs/ethos/bin:$PATH
export PYTHONPATH=/path/to/CHDLLM/ethos-ares/src:$PYTHONPATH
export CHD_DATA_ROOT=/path/to/CHD_MEDS

FRACS=(0.01 0.05 0.10 0.25 0.50)
F=${FRACS[$SLURM_ARRAY_TASK_ID]}
TAG=$(echo $F | tr -d '.' | sed 's/^0*//; s/^$/0/')
SEED=123

RESULTS=/path/to/CHD_MEDS/results_v4/baselines
mkdir -p $RESULTS

echo "=== Data efficiency BiLSTM: train_frac=${F}, seed=${SEED} ==="

python baselines/lstm.py \
    --output_dir $RESULTS \
    --seed $SEED \
    --epochs 20 --batch_size 64 --lr 1e-3 \
    --train_frac $F \
    --suffix _frac${TAG}_s${SEED}

echo "=== Done BiLSTM frac=${F} ==="
