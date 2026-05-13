#!/bin/bash
# Per-patient category attribution + timeline extraction for 4 case examples.
# Uses (r=8, BCE, s123) head from HP grid.
#
#SBATCH --job-name=cases
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=/path/to/CHD_MEDS/results_v4/slurm_cases_%j.log
#SBATCH --error=/path/to/CHD_MEDS/results_v4/slurm_cases_%j.err

set -e
cd /path/to/CHDLLM
export PATH=/path/to/miniforge3/envs/ethos/bin:$PATH
export PYTHONPATH=/path/to/CHDLLM/ethos-ares/src:$PYTHONPATH
export CHD_DATA_ROOT=/path/to/CHD_MEDS

echo "Case examples: per-patient category attribution + timeline"

python3 << 'PYEOF'
import sys, logging, json, torch as th, numpy as np, pandas as pd
from pathlib import Path
from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, "ethos")
sys.path.insert(0, "ethos/datasets")

MODEL_FP = Path("/path/to/CHD_MEDS/tokenized_v4/models/chd_v4_layer6_do0.3/best_model.pt")
HEAD_FP  = Path("/path/to/CHD_MEDS/results_v4/baselines/ethos/finetune/finetune_lora_head_best_v4_lora_r8_bce_s123.pt")
TEST_DIR = Path("/path/to/CHD_MEDS/tokenized_v4/test")
TASK_PATH = Path("/path/to/CHD_MEDS/outcome/iod_task.parquet")
OUT = Path("/path/to/CHD_MEDS/results_v4/evaluation/case_examples_r8bce")
OUT.mkdir(parents=True, exist_ok=True)

# Case patient IDs (subject_id) from test set
CASES = {
    "A_high_conf_pos": 20766307,   # pred 0.748, IoD+
    "B_moderate_pos":    930627,   # pred 0.264, IoD+
    "C_confident_neg":   341986,   # pred 0.001, IoD-
    "D_false_pos":      1327056,   # pred 0.623, IoD-
}

# Load model
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

# Dataset
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
    "Medications (ATC)":            ["ATC//", "MED//"],
    "Care Trajectory (ADT)":        ["ADT//"],
    "Surgical History (PROCEDURE)": ["PROCEDURE//"],
    "ICD-10 Codes":                 ["ICD//CM//"],
    "Surgical Context (ENCOUNTER)": ["ENCOUNTER//AN//"],
    "Laboratory":                   ["LAB//"],
    "Vital Signs":                  ["VITAL//"],
    "Lines/Drains (LDA)":           ["LDA//"],
    "Anesthesia Events":            ["AN_EVENT//"],
    "Problem List":                 ["PROBLEM//"],
    "Medical History":              ["DIAGNOSIS//"],
    "Structured Data (SDE)":        ["SDE//"],
    "Demographics":                 ["DEMO//"],
    "Insurance":                    ["INSURANCE//"],
    "Point of Origin":              ["POINT_OF_ORIGIN//"],
    "Home County":                  ["HOME_COUNTY//"],
    "Language":                     ["LANGUAGE//"],
    "Transfusions":                 ["TRANSFUSION//"],
}

cat_token_ids = {}
for cat, prefixes in CATEGORIES.items():
    ids = set()
    for tok, tid in ds.vocab.stoi.items():
        if any(tok.startswith(p) for p in prefixes):
            ids.add(tid)
    if cat == "Surgical Context (ENCOUNTER)":
        ids.discard(or_entry_id)
    cat_token_ids[cat] = ids

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

# Build sid → ds index map
sid_to_idx = {}
for idx in range(len(ds)):
    _, meta = ds[idx]
    sid_to_idx.setdefault(meta["patient_id"], []).append((idx, meta["prediction_time"]))

results = {}
for case_name, sid in CASES.items():
    if sid not in sid_to_idx:
        log.warning(f"{case_name} (sid={sid}) not in test set")
        continue
    # Pick the encounter matching the labeled prediction_time (most recent if multiple)
    for idx, pt in sorted(sid_to_idx[sid], key=lambda x: -x[1]):
        pt_us = pt // 1000 if pt > 1e15 else pt
        lab = label_dict.get((sid, pt_us))
        if lab is not None:
            break
    log.info(f"\n=== {case_name}  sid={sid}  pt={pt}  label={lab} ===")

    x, meta = ds[idx]
    input_ids = x.cuda().unsqueeze(0)

    # Baseline prediction
    with th.no_grad():
        base_p = th.sigmoid(head(get_hidden(model, input_ids)).squeeze()).item()
    log.info(f"  baseline pred = {base_p:.4f}")

    # Decode the token sequence (last 80 non-pad tokens; reversed→chronological)
    seq = input_ids.squeeze(0).cpu().tolist()
    non_pad = [(i, t) for i, t in enumerate(seq) if t != 0]
    timeline_tokens = [(itos.get(t, f"<{t}>"), i) for i, t in non_pad[-80:]]

    # Per-category attribution
    attribution = {}
    for cat, ids in cat_token_ids.items():
        if not ids: continue
        mask = build_mask_with_adjacent_q(input_ids.squeeze(0), ids)
        masked_ids = input_ids.masked_fill(mask.unsqueeze(0), 0)
        with th.no_grad():
            p = th.sigmoid(head(get_hidden(model, masked_ids)).squeeze()).item()
        attribution[cat] = {
            "n_masked_tokens": int(mask.sum().item()),
            "pred_masked": p,
            "delta_pred": base_p - p,  # positive Δ = category INCREASES risk
        }
        log.info(f"    {cat:<35s}  n={int(mask.sum()):>4d}  masked_pred={p:.4f}  Δ={base_p-p:+.4f}")

    results[case_name] = {
        "subject_id": int(sid),
        "prediction_time_ns": int(pt),
        "label": int(lab),
        "baseline_pred": base_p,
        "timeline_tokens": timeline_tokens,
        "n_total_tokens": len(non_pad),
        "category_attribution": attribution,
    }

import json
out_fp = OUT / "case_examples.json"
with open(out_fp, "w") as f:
    json.dump(results, f, indent=2, default=str)
log.info(f"\nSaved {out_fp}")
PYEOF
echo "Done."
