#!/bin/bash
#SBATCH --job-name=v4_ctx
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=/path/to/CHD_MEDS/results/slurm_ctx_%x_%j.log
#SBATCH --error=/path/to/CHD_MEDS/results/slurm_ctx_%x_%j.err

# Usage: sbatch --job-name=v4_ctx_7d slurm_context_window.sh 7
#        sbatch --job-name=v4_ctx_90d slurm_context_window.sh 90
#        sbatch --job-name=v4_ctx_365d slurm_context_window.sh 365

set -e
WINDOW_DAYS=${1:?Usage: $0 <window_days>}

cd .
export PATH=${HOME}/miniforge3/envs/ethos/bin:$PATH
export PYTHONPATH=/path/to/ethos-ares/src:$PYTHONPATH

MODEL_FP=/path/to/CHD_MEDS/tokenized/models/chd_v4_layer6_do0.3/best_model.pt
TOK=/path/to/CHD_MEDS/tokenized
RESULTS=/path/to/CHD_MEDS/results
SEED=123

echo "Context window ${WINDOW_DAYS}d, seed $SEED"

CUDA_VISIBLE_DEVICES=0 python ethos/finetune.py \
    --model_fp $MODEL_FP --seed $SEED \
    --train_dir $TOK/train --val_dir $TOK/val --test_dir $TOK/test \
    --results_dir $RESULTS/baselines \
    --lr 1e-3 --hidden_dim 256 --head_dropout 0.1 \
    --lora --lora_r 8 --lora_alpha 16 \
    --window_days $WINDOW_DAYS \
    --suffix _v4_lora_ctx${WINDOW_DAYS}d_s${SEED}

echo "Context window ${WINDOW_DAYS}d done"

# Quick eval
python3 << PYEOF
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
f = "$RESULTS/baselines/ethos_finetune_lora_test_predictions_v4_lora_ctx${WINDOW_DAYS}d_s${SEED}.parquet"
df = pd.read_parquet(f)
auroc = roc_auc_score(df.y_true, df.y_prob)
auprc = average_precision_score(df.y_true, df.y_prob)
print(f"Window={WINDOW_DAYS}d  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")
PYEOF
