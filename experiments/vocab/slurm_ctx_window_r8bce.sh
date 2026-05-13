#!/bin/bash
# Context-window ablation re-eval at (r=8, BCE), seed 123.
# Windows: 7d, 30d, 90d, 365d. (Full history "All" already covered by
# the HP-grid run _v4_lora_r8_bce_s123.)
#
# Usage:
#   sbatch --array=0-3 slurm_ctx_window_r8bce.sh
#
#SBATCH --job-name=ctx_r8bce
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --output=/path/to/CHD_MEDS/results_v4/slurm_ctx_r8bce_%A_%a.log
#SBATCH --error=/path/to/CHD_MEDS/results_v4/slurm_ctx_r8bce_%A_%a.err
#SBATCH --array=0-3

set -e
cd /path/to/CHDLLM
export PATH=/path/to/miniforge3/envs/ethos/bin:$PATH
export PYTHONPATH=/path/to/CHDLLM/ethos-ares/src:$PYTHONPATH
export CHD_DATA_ROOT=/path/to/CHD_MEDS

WINDOWS=(7 30 90 365)
W=${WINDOWS[$SLURM_ARRAY_TASK_ID]}
SEED=123

MODEL_FP=/path/to/CHD_MEDS/tokenized_v4/models/chd_v4_layer6_do0.3/best_model.pt
TOK=/path/to/CHD_MEDS/tokenized_v4
RESULTS=/path/to/CHD_MEDS/results_v4/baselines
mkdir -p $RESULTS/ethos/finetune

echo "=== (r=8, bce, ctx=${W}d) seed=${SEED} ==="

python ethos/finetune.py \
    --model_fp $MODEL_FP \
    --train_dir $TOK/train --val_dir $TOK/val --test_dir $TOK/test \
    --results_dir $RESULTS \
    --seed $SEED \
    --epochs 15 --patience 5 \
    --lr 1e-3 --hidden_dim 256 --head_dropout 0.1 \
    --lora --lora_r 8 --lora_alpha 16 \
    --loss_type bce \
    --window_days $W \
    --suffix _v4_lora_r8_bce_ctx${W}d_s${SEED}
