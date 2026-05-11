#!/bin/bash
# Loss-function ablation against pos_weight=auto baseline:
#   0: bce          (unweighted BCE)
#   1: focal_g2     (focal loss, gamma=2, alpha=0.25)
#   2: oversample5  (positive oversampled 5x via WeightedRandomSampler, weighted_bce)
#   3: under10      (negatives downsampled to 10:1, weighted_bce on the balanced set)
#
# Usage:
#   sbatch --array=0-3 slurm_loss_ablation.sh
#
#SBATCH --job-name=loss_abl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=/path/to/CHD_MEDS/results_v4/slurm_loss_%A_%a.log
#SBATCH --error=/path/to/CHD_MEDS/results_v4/slurm_loss_%A_%a.err
#SBATCH --array=0-3

set -e
cd /path/to/CHDLLM
export PATH=/path/to/miniforge3/envs/ethos/bin:$PATH
export PYTHONPATH=/path/to/CHDLLM/ethos-ares/src:$PYTHONPATH

SEED=123
MODEL_FP=/path/to/CHD_MEDS/tokenized_v4/models/chd_v4_layer6_do0.3/best_model.pt
TOK=/path/to/CHD_MEDS/tokenized_v4
RESULTS=/path/to/CHD_MEDS/results_v4/baselines
mkdir -p $RESULTS/ethos/finetune

case $SLURM_ARRAY_TASK_ID in
  0)
    NAME="bce"
    EXTRA="--loss_type bce"
    ;;
  1)
    NAME="focal_g2"
    EXTRA="--loss_type focal --focal_gamma 2.0 --focal_alpha 0.25"
    ;;
  2)
    NAME="oversample5"
    EXTRA="--loss_type weighted_bce --sampler oversample_pos --sampler_ratio 5.0"
    ;;
  3)
    NAME="under10"
    EXTRA="--loss_type weighted_bce --sampler undersample_neg --sampler_ratio 10.0"
    ;;
esac

echo "=== Loss ablation: ${NAME}, seed=${SEED} ==="

python ethos/finetune.py \
    --model_fp $MODEL_FP \
    --train_dir $TOK/train --val_dir $TOK/val --test_dir $TOK/test \
    --results_dir $RESULTS \
    --seed $SEED \
    --lr 1e-3 --hidden_dim 256 --head_dropout 0.1 \
    --lora --lora_r 8 --lora_alpha 16 \
    $EXTRA \
    --suffix _v4_lora_loss_${NAME}_s${SEED}

echo "=== Done ${NAME} ==="
