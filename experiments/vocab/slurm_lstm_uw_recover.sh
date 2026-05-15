#!/bin/bash
# Recover BiLSTM unweighted tuning from existing per-config checkpoints
# (595141 timed out before final eval). Loads each ckpt, evaluates test,
# picks best by test AUPRC, saves predictions.
#
#SBATCH --job-name=lstm_recover
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=/path/to/CHD_MEDS/results_v4/slurm_lstm_recover_%j.log
#SBATCH --error=/path/to/CHD_MEDS/results_v4/slurm_lstm_recover_%j.err

set -e
cd /path/to/CHDLLM
export PATH=/path/to/miniforge3/envs/ethos/bin:$PATH
export PYTHONPATH=/path/to/CHDLLM:$PYTHONPATH
export CHD_DATA_ROOT=/path/to/CHD_MEDS

python3 << 'PYEOF'
import json, pickle, re, glob, logging
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

from baselines.lstm import (
    IoDSequenceDataset, IoDLSTM, build_code_vocab, collate_fn, run_epoch,
    MAX_SEQ_LEN, VOCAB_SIZE,
)
from baselines.features import load_task, load_events, load_splits

OUT = Path("/path/to/CHD_MEDS/results/baselines_tuned")
CKPTS = sorted(OUT.glob("lstm_unweighted_tuned_ckpt_*.pt"))
log.info(f"Found {len(CKPTS)} ckpts:")
for c in CKPTS: log.info(f"  {c.name}")

# Parse config from filename
def parse_cfg(name):
    s = name.replace("lstm_unweighted_tuned_ckpt_", "").replace(".pt", "")
    m = re.match(r"h(\d+)_l(\d+)_do([\d.]+)_lr([\de.\-+]+)_e(\d+)", s)
    return {"name": s, "hidden_dim": int(m.group(1)), "num_layers": int(m.group(2)),
            "dropout": float(m.group(3)), "lr": float(m.group(4)), "embed_dim": int(m.group(5))}

# Reconstruct datasets (same vocab)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
task = load_task(); events = load_events(); splits = load_splits()
task = task.merge(splits[["subject_id", "split"]], on="subject_id", how="left")
task["split"] = task["split"].fillna("train")
train_task = task[task["split"]=="train"].reset_index(drop=True)
val_task   = task[task["split"]=="val"].reset_index(drop=True)
test_task  = task[task["split"]=="test"].reset_index(drop=True)

# Same vocab as during training
vocab_pkl = OUT / "lstm_unweighted_tuned_code_vocab.pkl"
if vocab_pkl.exists():
    code_vocab = pickle.load(open(vocab_pkl, "rb"))
    log.info(f"Loaded vocab from {vocab_pkl} ({len(code_vocab)} codes)")
else:
    log.info("Rebuilding vocab from train")
    code_vocab = build_code_vocab(events, train_task, top_n=VOCAB_SIZE)

val_ds  = IoDSequenceDataset(val_task,  events, code_vocab, MAX_SEQ_LEN)
test_ds = IoDSequenceDataset(test_task, events, code_vocab, MAX_SEQ_LEN)
val_loader  = DataLoader(val_ds,  batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
criterion = nn.BCEWithLogitsLoss()

# Evaluate each ckpt
results = []
for ckpt_fp in CKPTS:
    cfg = parse_cfg(ckpt_fp.name)
    log.info(f"=== {cfg['name']} ===")
    model = IoDLSTM(vocab_size=VOCAB_SIZE+2, embed_dim=cfg["embed_dim"],
                    hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"],
                    dropout=cfg["dropout"]).to(device)
    model.load_state_dict(torch.load(ckpt_fp, map_location=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    _, val_a, val_p, _, _ = run_epoch(model, val_loader, optimizer, criterion, device, train=False)
    _, te_a, te_p, te_probs, te_labels = run_epoch(model, test_loader, optimizer, criterion, device, train=False)
    te_brier = brier_score_loss(te_labels, te_probs)
    log.info(f"  val AUROC={val_a:.4f} AUPRC={val_p:.4f}  |  TEST AUROC={te_a:.4f} AUPRC={te_p:.4f} Brier={te_brier:.4f}")
    results.append({"config": cfg, "val_auroc": val_a, "val_auprc": val_p,
                    "test_auroc": te_a, "test_auprc": te_p, "test_brier": te_brier,
                    "test_probs": te_probs, "test_labels": te_labels})

# Pick best by val AUPRC (matches lstm_tuned.py convention)
best = max(results, key=lambda r: r["val_auprc"])
log.info(f"\nBEST: {best['config']['name']}  val AUPRC={best['val_auprc']:.4f}")
log.info(f"  TEST: AUROC={best['test_auroc']:.4f} AUPRC={best['test_auprc']:.4f} Brier={best['test_brier']:.4f}")

# Save test predictions
test_enc = test_task.reset_index(drop=True)
pred_df = pd.DataFrame({
    "subject_id":    test_enc["subject_id"],
    "encounter_csn": test_enc["encounter_csn"],
    "y_true":        best["test_labels"].astype(int),
    "y_prob":        best["test_probs"],
})
pred_df.to_parquet(OUT / "lstm_unweighted_test_predictions.parquet", index=False)
with open(OUT / "lstm_unweighted_best.json", "w") as f:
    json.dump({"best_config": best["config"], "val_auroc": best["val_auroc"],
               "val_auprc": best["val_auprc"], "test_auroc": best["test_auroc"],
               "test_auprc": best["test_auprc"], "test_brier": best["test_brier"]}, f, indent=2)

# Save sweep summary
summary = pd.DataFrame([{
    "name": r["config"]["name"], "hidden_dim": r["config"]["hidden_dim"],
    "num_layers": r["config"]["num_layers"], "dropout": r["config"]["dropout"],
    "lr": r["config"]["lr"], "embed_dim": r["config"]["embed_dim"],
    "val_auroc": r["val_auroc"], "val_auprc": r["val_auprc"],
    "test_auroc": r["test_auroc"], "test_auprc": r["test_auprc"],
    "test_brier": r["test_brier"],
} for r in results])
summary.to_csv(OUT / "lstm_unweighted_summary.csv", index=False)
log.info(f"Saved predictions, best.json, summary.csv → {OUT}")
PYEOF
echo "Done."
