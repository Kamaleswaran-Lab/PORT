#!/bin/bash
# BiLSTM unweighted HP tune (8-config grid, full history).
# GPU job; submit AFTER Stage 2 completes.
#
#SBATCH --job-name=lstm_uw_tune
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/CHD_MEDS/results_v4/slurm_lstm_uw_tune_%j.log
#SBATCH --error=/path/to/CHD_MEDS/results_v4/slurm_lstm_uw_tune_%j.err

set -e
cd /path/to/CHDLLM
export PATH=/path/to/miniforge3/envs/ethos/bin:$PATH
export PYTHONPATH=/path/to/CHDLLM:$PYTHONPATH
export CHD_DATA_ROOT=/path/to/CHD_MEDS

OUT=/path/to/CHD_MEDS/results/baselines_tuned
mkdir -p $OUT

echo "=== BiLSTM unweighted HP tune (8 configs) ==="
python -m baselines.lstm_tuned \
    --max_epochs 25 \
    --patience 5 \
    --batch_size 64 \
    --seed 42 \
    --output_dir $OUT \
    --unweighted \
    --suffix _unweighted_tuned

echo "=== Done ==="
