#!/bin/bash
# Data-efficiency ablation for PORT (LoRA r=8): subsample training set to
# 1%, 5%, 10%, 25%, 50% of the full corpus (stratified by IoD label).
#
# Usage:
#   sbatch --array=0-4 slurm_data_eff_port.sh
#
#SBATCH --job-name=data_port
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=/path/to/CHD_MEDS/results_v4/slurm_data_port_%A_%a.log
#SBATCH --error=/path/to/CHD_MEDS/results_v4/slurm_data_port_%A_%a.err
#SBATCH --array=0-4

set -e
cd /path/to/CHDLLM
export PATH=/path/to/miniforge3/envs/ethos/bin:$PATH
export PYTHONPATH=/path/to/CHDLLM/ethos-ares/src:$PYTHONPATH

FRACS=(0.01 0.05 0.10 0.25 0.50)
F=${FRACS[$SLURM_ARRAY_TASK_ID]}
TAG=$(echo $F | tr -d '.' | sed 's/^0*//; s/^$/0/')   # 0.01->1, 0.10->10, 0.50->50
SEED=123

MODEL_FP=/path/to/CHD_MEDS/tokenized_v4/models/chd_v4_layer6_do0.3/best_model.pt
TOK=/path/to/CHD_MEDS/tokenized_v4
RESULTS=/path/to/CHD_MEDS/results_v4/baselines
mkdir -p $RESULTS/ethos/finetune

echo "=== Data efficiency PORT: train_frac=${F}, seed=${SEED} ==="

python ethos/finetune.py \
    --model_fp $MODEL_FP \
    --train_dir $TOK/train --val_dir $TOK/val --test_dir $TOK/test \
    --results_dir $RESULTS \
    --seed $SEED \
    --lr 1e-3 --hidden_dim 256 --head_dropout 0.1 \
    --lora --lora_r 8 --lora_alpha 16 \
    --train_frac $F \
    --suffix _v4_lora_frac${TAG}_s${SEED}

echo "=== Done frac=${F} ==="
