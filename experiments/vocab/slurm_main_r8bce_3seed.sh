#!/bin/bash
# Main (r=8, BCE) at seeds 42 and 456 (s123 already trained in HP grid).
# Used for 3-seed variance + uncertainty (Fig 3) re-evaluation.
#
# Usage:
#   sbatch --array=0-1 slurm_main_r8bce_3seed.sh
#
#SBATCH --job-name=r8bce_3s
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --output=/path/to/CHD_MEDS/results_v4/slurm_r8bce_3s_%A_%a.log
#SBATCH --error=/path/to/CHD_MEDS/results_v4/slurm_r8bce_3s_%A_%a.err
#SBATCH --array=0-1

set -e
cd /path/to/CHDLLM
export PATH=/path/to/miniforge3/envs/ethos/bin:$PATH
export PYTHONPATH=/path/to/CHDLLM/ethos-ares/src:$PYTHONPATH
export CHD_DATA_ROOT=/path/to/CHD_MEDS

SEEDS=(42 456)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

MODEL_FP=/path/to/CHD_MEDS/tokenized_v4/models/chd_v4_layer6_do0.3/best_model.pt
TOK=/path/to/CHD_MEDS/tokenized_v4
RESULTS=/path/to/CHD_MEDS/results_v4/baselines
mkdir -p $RESULTS/ethos/finetune

echo "=== (r=8, bce) seed=${SEED} ==="

python ethos/finetune.py \
    --model_fp $MODEL_FP \
    --train_dir $TOK/train --val_dir $TOK/val --test_dir $TOK/test \
    --results_dir $RESULTS \
    --seed $SEED \
    --epochs 15 --patience 5 \
    --lr 1e-3 --hidden_dim 256 --head_dropout 0.1 \
    --lora --lora_r 8 --lora_alpha 16 \
    --loss_type bce \
    --suffix _v4_lora_r8_bce_s${SEED}
