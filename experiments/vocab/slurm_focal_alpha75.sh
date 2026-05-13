#!/bin/bash
# Focal loss α-correction sweep: re-run focal_g2 at α=0.75 (positive class
# weighted 0.75) for r ∈ {4, 8}. The original HP grid used Lin et al. RetinaNet
# default α=0.25 which down-weights the positive class — backwards intuition for
# our 1% IoD prevalence task. This 2-cell sweep tests whether properly oriented
# α makes focal competitive with the (r=8, bce) winner.
#
# Usage:
#   sbatch --array=0-1 slurm_focal_alpha75.sh
#
#SBATCH --job-name=focal75
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --output=/path/to/CHD_MEDS/results_v4/slurm_focal75_%A_%a.log
#SBATCH --error=/path/to/CHD_MEDS/results_v4/slurm_focal75_%A_%a.err
#SBATCH --array=0-1

set -e
cd /path/to/CHDLLM
export PATH=/path/to/miniforge3/envs/ethos/bin:$PATH
export PYTHONPATH=/path/to/CHDLLM/ethos-ares/src:$PYTHONPATH
export CHD_DATA_ROOT=/path/to/CHD_MEDS

RANKS=(4 8)
R=${RANKS[$SLURM_ARRAY_TASK_ID]}
A=$((R * 2))
SEED=123

MODEL_FP=/path/to/CHD_MEDS/tokenized_v4/models/chd_v4_layer6_do0.3/best_model.pt
TOK=/path/to/CHD_MEDS/tokenized_v4
RESULTS=/path/to/CHD_MEDS/results_v4/baselines
mkdir -p $RESULTS/ethos/finetune

echo "=== Focal α=0.75 sweep: r=${R}, alpha=${A}, focal γ=2 α=0.75, seed=${SEED} ==="

python ethos/finetune.py \
    --model_fp $MODEL_FP \
    --train_dir $TOK/train --val_dir $TOK/val --test_dir $TOK/test \
    --results_dir $RESULTS \
    --seed $SEED \
    --epochs 15 --patience 5 \
    --lr 1e-3 --hidden_dim 256 --head_dropout 0.1 \
    --lora --lora_r $R --lora_alpha $A \
    --loss_type focal --focal_gamma 2.0 --focal_alpha 0.75 \
    --suffix _v4_lora_r${R}_focal_g2_a75_s${SEED}

echo "=== Done r=${R} ==="
