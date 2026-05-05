#!/bin/bash
#SBATCH --job-name=ft_v4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=/path/to/CHD_MEDS/results/slurm_finetune_%j.log
#SBATCH --error=/path/to/CHD_MEDS/results/slurm_finetune_%j.err

set -e
cd .
export PATH=${HOME}/miniforge3/envs/ethos/bin:$PATH
export PYTHONPATH=/path/to/ethos-ares/src:$PYTHONPATH

MODEL_FP=/path/to/CHD_MEDS/tokenized/models/chd_v4_layer6_do0.3/best_model.pt
TOK=/path/to/CHD_MEDS/tokenized
RESULTS=/path/to/CHD_MEDS/results
mkdir -p $RESULTS/baselines $RESULTS/ethos/finetune

SEED=${1:-456}
echo "Finetune: Probe + LoRA (seed $SEED)"

python3 -c "
import torch
c = torch.load('$MODEL_FP', map_location='cpu', weights_only=False)
print(f'best_model: iter_num={c.get(\"iter_num\",\"N/A\")}, best_val_loss={c.get(\"best_val_loss\",\"N/A\")}')
"

COMMON="--model_fp $MODEL_FP --seed $SEED \
    --train_dir $TOK/train --val_dir $TOK/val --test_dir $TOK/test \
    --results_dir $RESULTS/baselines --lr 1e-3 --hidden_dim 256 --head_dropout 0.1"

# GPU 0: Linear Probe
CUDA_VISIBLE_DEVICES=0 python ethos/finetune.py $COMMON --suffix _probe_s${SEED} &
PID1=$!

# GPU 1: LoRA (r=8, alpha=16, c_attn)
CUDA_VISIBLE_DEVICES=1 python ethos/finetune.py $COMMON \
    --lora --lora_r 8 --lora_alpha 16 --suffix _lora_s${SEED} &
PID2=$!

echo "  GPU 0: Probe (PID $PID1)"
echo "  GPU 1: LoRA  (PID $PID2)"

wait $PID1; echo "Probe seed $SEED done"
wait $PID2; echo "LoRA seed $SEED done"

# Quick eval
python3 << PYEOF
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import pandas as pd

R = Path("$RESULTS/baselines")
print(f"\n{'Config':<30s} {'AUROC':>8s} {'AUPRC':>8s} {'Brier':>8s}")
print("-" * 60)
for sfx, mode, label in [
    ("_probe_s${SEED}", "probe", "Probe s$SEED"),
    ("_lora_s${SEED}", "lora", "LoRA s$SEED"),
]:
    f = R / f"ethos_finetune_{mode}_test_predictions{sfx}.parquet"
    if f.exists():
        df = pd.read_parquet(f)
        y = df["y_true"]; p = df["y_pred"]
        print(f"{label:<30s} {roc_auc_score(y,p):>8.4f} {average_precision_score(y,p):>8.4f} {brier_score_loss(y,p):>8.4f}")
    else:
        print(f"{label:<30s} FILE NOT FOUND")
PYEOF

echo "Finetune complete (seed $SEED)"
