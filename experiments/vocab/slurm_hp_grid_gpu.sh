#!/bin/bash
# Joint LoRA rank × loss hyperparameter grid (5 × 4 = 20 cells).
#
# Grid axes:
#   rank ∈ {4, 8, 16, 32, 64}        (α = 2r in all cells)
#   loss ∈ {wbce, bce, focal_g2, oversample5}
#
# Array index encoding:  idx = rank_idx * 4 + loss_idx
#   rank_idx ∈ {0..4}, loss_idx ∈ {0..3}
#
# Usage:
#   sbatch --array=0-19 slurm_hp_grid.sh
#
#SBATCH --job-name=hp_grid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --output=/path/to/CHD_MEDS/results_v4/slurm_hp_grid_%A_%a.log
#SBATCH --error=/path/to/CHD_MEDS/results_v4/slurm_hp_grid_%A_%a.err
#SBATCH --array=0-19

set -e
cd /path/to/CHDLLM
export PATH=/path/to/miniforge3/envs/ethos/bin:$PATH
export PYTHONPATH=/path/to/CHDLLM/ethos-ares/src:$PYTHONPATH
export CHD_DATA_ROOT=/path/to/CHD_MEDS

RANKS=(4 8 16 32 64)
LOSS_TAGS=(wbce bce focal_g2 oversample5)
LOSS_ARGS=(
  "--loss_type weighted_bce"
  "--loss_type bce"
  "--loss_type focal --focal_gamma 2.0 --focal_alpha 0.25"
  "--loss_type weighted_bce --sampler oversample_pos --sampler_ratio 5.0"
)

IDX=$SLURM_ARRAY_TASK_ID
RANK_IDX=$(( IDX / 4 ))
LOSS_IDX=$(( IDX % 4 ))

R=${RANKS[$RANK_IDX]}
A=$(( R * 2 ))
LT=${LOSS_TAGS[$LOSS_IDX]}
LA="${LOSS_ARGS[$LOSS_IDX]}"
SEED=123

MODEL_FP=/path/to/CHD_MEDS/tokenized_v4/models/chd_v4_layer6_do0.3/best_model.pt
TOK=/path/to/CHD_MEDS/tokenized_v4
RESULTS=/path/to/CHD_MEDS/results_v4/baselines
mkdir -p $RESULTS/ethos/finetune

echo "=== HP grid cell idx=${IDX}: r=${R}, alpha=${A}, loss=${LT} ==="
echo "    loss args: $LA"

python ethos/finetune.py \
    --model_fp $MODEL_FP \
    --train_dir $TOK/train --val_dir $TOK/val --test_dir $TOK/test \
    --results_dir $RESULTS \
    --seed $SEED \
    --epochs 15 --patience 5 \
    --lr 1e-3 --hidden_dim 256 --head_dropout 0.1 \
    --lora --lora_r $R --lora_alpha $A \
    $LA \
    --suffix _v4_lora_r${R}_${LT}_s${SEED}

echo "=== Done idx=${IDX} (r=${R}, ${LT}) ==="
