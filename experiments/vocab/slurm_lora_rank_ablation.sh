#!/bin/bash
# LoRA rank ablation: r ∈ {4, 8, 16, 32, 64}, alpha=2r.
# Submit one job per rank to gpu-hp; each runs a single LoRA fine-tune.
#
# Usage:
#   sbatch --array=0-4 slurm_lora_rank_ablation.sh
#
#SBATCH --job-name=lora_r
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=/path/to/CHD_MEDS/results_v4/slurm_lora_r_%A_%a.log
#SBATCH --error=/path/to/CHD_MEDS/results_v4/slurm_lora_r_%A_%a.err
#SBATCH --array=0-4

set -e
cd /path/to/CHDLLM
export PATH=/path/to/miniforge3/envs/ethos/bin:$PATH
export PYTHONPATH=/path/to/CHDLLM/ethos-ares/src:$PYTHONPATH

RANKS=(4 8 16 32 64)
R=${RANKS[$SLURM_ARRAY_TASK_ID]}
A=$((R * 2))
SEED=123

MODEL_FP=/path/to/CHD_MEDS/tokenized_v4/models/chd_v4_layer6_do0.3/best_model.pt
TOK=/path/to/CHD_MEDS/tokenized_v4
RESULTS=/path/to/CHD_MEDS/results_v4/baselines
mkdir -p $RESULTS/ethos/finetune

echo "=== LoRA rank ablation: r=${R}, alpha=${A}, seed=${SEED} ==="

python ethos/finetune.py \
    --model_fp $MODEL_FP \
    --train_dir $TOK/train --val_dir $TOK/val --test_dir $TOK/test \
    --results_dir $RESULTS \
    --seed $SEED \
    --lr 1e-3 --hidden_dim 256 --head_dropout 0.1 \
    --lora --lora_r $R --lora_alpha $A \
    --suffix _v4_lora_r${R}_s${SEED}

echo "=== Done r=${R} ==="
