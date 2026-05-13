#!/bin/bash
# Per-patient occlusion re-eval using the (r=8, BCE, s123) head from the HP grid.
# Mirrors slurm_occlusion_perpatient.sh but points at the new head/LoRA weights.
#
#SBATCH --job-name=occ_r8bce
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --output=/path/to/CHD_MEDS/results_v4/slurm_occ_r8bce_%j.log
#SBATCH --error=/path/to/CHD_MEDS/results_v4/slurm_occ_r8bce_%j.err

set -e
cd /path/to/CHDLLM
export PATH=/path/to/miniforge3/envs/ethos/bin:$PATH
export PYTHONPATH=/path/to/CHDLLM/ethos-ares/src:$PYTHONPATH
export CHD_DATA_ROOT=/path/to/CHD_MEDS

echo "Per-patient occlusion (r=8, BCE, s123)"

python3 << 'PYEOF'
import sys, logging, torch as th, numpy as np, pandas as pd
from pathlib import Path
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_auc_score, average_precision_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, "ethos")
sys.path.insert(0, "ethos/datasets")

MODEL_FP = Path("/path/to/CHD_MEDS/tokenized_v4/models/chd_v4_layer6_do0.3/best_model.pt")
HEAD_FP  = Path("/path/to/CHD_MEDS/results_v4/baselines/ethos/finetune/finetune_lora_head_best_v4_lora_r8_bce_s123.pt")
TEST_DIR = Path("/path/to/CHD_MEDS/tokenized_v4/test")
TASK_PATH = Path("/path/to/CHD_MEDS/outcome/iod_task.parquet")
OUT = Path("/path/to/CHD_MEDS/results_v4/evaluation/per_patient_occlusion_r8bce")
OUT.mkdir(parents=True, exist_ok=True)

from ethos.utils import load_model_checkpoint
model, _ = load_model_checkpoint(MODEL_FP, map_location="cuda:0")
model.to("cuda:0")
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"], lora_dropout=0.0, bias="none")
model = get_peft_model(model, lora_config)
ckpt = th.load(HEAD_FP, map_location="cuda:0", weights_only=False)
if ckpt.get("lora_state_dict"):
    model.load_state_dict(ckpt["lora_state_dict"], strict=False)
cfg = ckpt["head_config"]
head = th.nn.Sequential(
    th.nn.Linear(cfg["n_embd"], cfg["hidden_dim"]),
    th.nn.ReLU(), th.nn.Dropout(0.0),
    th.nn.Linear(cfg["hidden_dim"], 1),
).cuda().eval()
state = {k.replace("net.", ""): v for k, v in ckpt["head_state_dict"].items()}
head.load_state_dict(state)
model.eval()

from iod_dataset import IoDDataset
ds = IoDDataset(TEST_DIR, n_positions=2048)
task = pd.read_parquet(TASK_PATH)
task['sid'] = task['patient_id'].str.lstrip('C').astype(int)
task['pt_us'] = task['prediction_time'].astype('int64') // 1000
label_dict = {(int(r.sid), int(r.pt_us)): int(r.boolean_value) for r in task.itertuples()}

itos = {i: tok for tok, i in ds.vocab.stoi.items()}
Q_TOKEN_IDS = set(tid for tok, tid in ds.vocab.stoi.items() if tok.startswith("Q") and len(tok) <= 3)
or_entry_id = ds.vocab.stoi["ENCOUNTER//AN//OR_ENTRY"]

CATEGORIES = {
    "ICD-10 Codes": ["ICD//CM//"],
    "Medications (ATC)": ["ATC//", "MED//"],
    "Surgical Context (ENCOUNTER)": ["ENCOUNTER//AN//"],
    "Surgical History (PROCEDURE)": ["PROCEDURE//"],
    "Problem List (free-text)": ["PROBLEM//"],
    "Medical History (free-text)": ["DIAGNOSIS//"],
    "Laboratory": ["LAB//"],
    "Vital Signs": ["VITAL//"],
    "Care Trajectory (ADT)": ["ADT//"],
    "Lines/Drains (LDA)": ["LDA//"],
    "Anesthesia Events": ["AN_EVENT//"],
    "Transfusions": ["TRANSFUSION//"],
    "Structured Data (SDE)": ["SDE//"],
    "Insurance": ["INSURANCE//"],
    "Point of Origin": ["POINT_OF_ORIGIN//"],
    "Home County": ["HOME_COUNTY//"],
    "Language": ["LANGUAGE//"],
    "Demographics": ["DEMO//"],
}

cat_token_ids = {}
for cat, prefixes in CATEGORIES.items():
    ids = set()
    for tok, tid in ds.vocab.stoi.items():
        if any(tok.startswith(p) for p in prefixes):
            ids.add(tid)
    # For Surgical Context, exclude OR_ENTRY (cursor artifact)
    if cat == "Surgical Context (ENCOUNTER)":
        ids.discard(or_entry_id)
    cat_token_ids[cat] = ids
    log.info(f"  {cat:<35s} {len(ids):>5d} tokens")

def get_hidden(model, ids):
    base = getattr(model, "base_model", model)
    base = getattr(base, "model", base)
    transformer = base.transformer
    _, t = ids.size()
    x = transformer.drop(transformer.wte(ids) + transformer.wpe(base.pos[:t]))
    for block in transformer.h:
        out = block(x)
        x = out[0] if isinstance(out, tuple) else out
    x = transformer.ln_f(x)
    return x[:, -1, :]

def build_mask_with_adjacent_q(input_ids_1d, category_ids):
    mask = th.zeros_like(input_ids_1d, dtype=th.bool)
    seq_len = input_ids_1d.size(0)
    for i in range(seq_len):
        tid = input_ids_1d[i].item()
        if tid in category_ids:
            mask[i] = True
            if i + 1 < seq_len and input_ids_1d[i + 1].item() in Q_TOKEN_IDS:
                mask[i + 1] = True
    return mask

def evaluate_with_predictions(category=None):
    sids, pts, yt, yp = [], [], [], []
    for i in range(len(ds)):
        x, meta = ds[i]
        pid, pt = meta["patient_id"], meta["prediction_time"]
        pt_us = pt // 1000 if pt > 1e15 else pt
        lab = label_dict.get((pid, pt_us))
        if lab is None: continue
        ids = x.unsqueeze(0).cuda()
        if category is not None:
            mask = build_mask_with_adjacent_q(ids.squeeze(0), cat_token_ids[category])
            ids = ids.masked_fill(mask.unsqueeze(0), 0)
        with th.no_grad():
            prob = th.sigmoid(head(get_hidden(model, ids)).squeeze()).item()
        sids.append(pid); pts.append(pt_us); yt.append(lab); yp.append(prob)
        if (i+1) % 5000 == 0:
            log.info(f"  [{category or 'baseline'}] {i+1}/{len(ds)}")
    return pd.DataFrame({"subject_id": sids, "prediction_time_us": pts,
                         "y_true": yt, "y_prob": yp})

log.info("Baseline (no mask)…")
df_base = evaluate_with_predictions()
df_base.to_parquet(OUT / "predictions_baseline.parquet", index=False)
b_a = roc_auc_score(df_base.y_true, df_base.y_prob)
b_p = average_precision_score(df_base.y_true, df_base.y_prob)
log.info(f"Baseline n={len(df_base):,}  AUROC={b_a:.4f}  AUPRC={b_p:.4f}")

summary = [{"Category": "Baseline", "Tokens": 0, "AUROC": b_a, "AUPRC": b_p,
            "Delta_AUROC": 0.0, "Delta_AUPRC": 0.0}]
for cat in CATEGORIES:
    n_tok = len(cat_token_ids[cat])
    if n_tok == 0:
        log.info(f"  SKIP {cat} (0 tokens)"); continue
    log.info(f"Masking {cat} ({n_tok} tokens)…")
    df_m = evaluate_with_predictions(cat)
    safe = cat.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    df_m.to_parquet(OUT / f"predictions_{safe}.parquet", index=False)
    a = roc_auc_score(df_m.y_true, df_m.y_prob)
    p = average_precision_score(df_m.y_true, df_m.y_prob)
    summary.append({"Category": cat, "Tokens": n_tok, "AUROC": a, "AUPRC": p,
                    "Delta_AUROC": a - b_a, "Delta_AUPRC": p - b_p})
    log.info(f"  {cat}: AUROC={a:.4f} (Δ{a-b_a:+.4f})  AUPRC={p:.4f} (Δ{p-b_p:+.4f})")

pd.DataFrame(summary).sort_values("Delta_AUROC").to_csv(OUT / "summary.csv", index=False)
print("\n", pd.read_csv(OUT / "summary.csv").to_string(index=False))
PYEOF
echo "Done."
