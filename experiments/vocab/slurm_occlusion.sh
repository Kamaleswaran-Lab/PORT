#!/bin/bash
#SBATCH --job-name=v4_occl
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/CHD_MEDS/results/slurm_occlusion_%j.log
#SBATCH --error=/path/to/CHD_MEDS/results/slurm_occlusion_%j.err

set -e
cd .
export PATH=${HOME}/miniforge3/envs/ethos/bin:$PATH
export PYTHONPATH=/path/to/ethos-ares/src:$PYTHONPATH

echo "Occlusion Analysis (LoRA s123, best AUROC=0.833)"

python3 << 'PYEOF'
import sys, logging, torch as th, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, "ethos")
sys.path.insert(0, "ethos/datasets")

MODEL_FP = Path("/path/to/CHD_MEDS/tokenized/models/chd_v4_layer6_do0.3/best_model.pt")
HEAD_FP = Path("/path/to/CHD_MEDS/results/baselines/ethos/finetune/finetune_lora_head_best_lora_s123.pt")
TEST_DIR = Path("/path/to/CHD_MEDS/tokenized/test")
TASK_PATH = Path("/path/to/CHD_MEDS/outcome/iod_task.parquet")
OUT = Path("/path/to/CHD_MEDS/results/evaluation")
OUT.mkdir(parents=True, exist_ok=True)

# ── Load model ──
from ethos.utils import load_model_checkpoint
model, _ = load_model_checkpoint(MODEL_FP, map_location="cuda:0")
model.to("cuda:0")

lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"], lora_dropout=0.0, bias="none")
model = get_peft_model(model, lora_config)

ckpt = th.load(HEAD_FP, map_location="cuda:0", weights_only=False)
if ckpt.get("lora_state_dict"):
    model.load_state_dict(ckpt["lora_state_dict"], strict=False)
    log.info("Loaded LoRA adapter state")

cfg = ckpt["head_config"]
head = th.nn.Sequential(
    th.nn.Linear(cfg["n_embd"], cfg["hidden_dim"]),
    th.nn.ReLU(), th.nn.Dropout(0.0),
    th.nn.Linear(cfg["hidden_dim"], 1),
).cuda().eval()
state = {k.replace("net.", ""): v for k, v in ckpt["head_state_dict"].items()}
head.load_state_dict(state)
model.eval()

# ── Dataset ──
from iod_dataset import IoDDataset
ds = IoDDataset(TEST_DIR, n_positions=2048)

task = pd.read_parquet(TASK_PATH)
task['sid'] = task['patient_id'].str.lstrip('C').astype(int)
task['pt_us'] = task['prediction_time'].astype('int64') // 1000
label_dict = {(int(r.sid), int(r.pt_us)): int(r.boolean_value) for r in task.itertuples()}

# ── Build vocab lookup ──
itos = {i: tok for tok, i in ds.vocab.stoi.items()}
Q_TOKEN_IDS = set(tid for tok, tid in ds.vocab.stoi.items() if tok.startswith("Q") and len(tok) <= 3)

# ══════════════════════════════════════════════════════════════════════════════
# CATEGORIES — carefully matching actual vocab prefixes
# ══════════════════════════════════════════════════════════════════════════════
CATEGORIES = {
    # ── ICD hierarchical (shared across DIAG/PROBLEM/CARDIOLOGY) ──
    "ICD-10 Codes": ["ICD//CM//"],

    # ── Medications (ATC hierarchical + unmapped) ──
    "Medications (ATC)": ["ATC//", "MED//"],

    # ── Procedures ──
    "Surgical Context (ENCOUNTER)": ["ENCOUNTER//AN//"],
    "Surgical History (PROCEDURE)": ["PROCEDURE//"],

    # ── Free-text problem/diagnosis residuals ──
    "Problem List (free-text)": ["PROBLEM//"],
    "Medical History (free-text)": ["DIAGNOSIS//"],

    # ── Labs / Vitals ──
    "Laboratory": ["LAB//"],
    "Vital Signs": ["VITAL//"],

    # ── Clinical operations ──
    "Care Trajectory (ADT)": ["ADT//"],
    "Lines/Drains (LDA)": ["LDA//"],
    "Anesthesia Events": ["AN_EVENT//"],
    "Transfusions": ["TRANSFUSION//"],

    # ── Structured assessments ──
    "Structured Data (SDE)": ["SDE//"],

    # ── NEW: SES context (socioeconomic context) ──
    "Insurance": ["INSURANCE//"],
    "Point of Origin": ["POINT_OF_ORIGIN//"],
    "Home County": ["HOME_COUNTY//"],
    "Language": ["LANGUAGE//"],

    # ── Demographics (static, in context window) ──
    "Demographics": ["DEMO//"],
}

# Build token ID sets per category
cat_token_ids = {}
for cat, prefixes in CATEGORIES.items():
    ids = set()
    for tok, tid in ds.vocab.stoi.items():
        if any(tok.startswith(p) for p in prefixes):
            ids.add(tid)
    cat_token_ids[cat] = ids
    log.info(f"  {cat:<35s} {len(ids):>5d} tokens")

# ── Forward pass helper ──
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

# ── Masking helper (same as  but works for token pairs) ──
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

# ── Evaluation ──
def evaluate(category=None):
    yt, yp = [], []
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
        yt.append(lab); yp.append(prob)
        if (i+1) % 5000 == 0:
            log.info(f"  [{category or 'baseline'}] {i+1}/{len(ds)}")
    yt, yp = np.array(yt), np.array(yp)
    return roc_auc_score(yt, yp), average_precision_score(yt, yp), len(yt)

# ── Run ──
log.info("Baseline evaluation...")
b_auroc, b_auprc, n = evaluate()
log.info(f"Baseline: AUROC={b_auroc:.4f}  AUPRC={b_auprc:.4f}  n={n:,}")

results = [{"Category": "Baseline (no mask)", "Tokens": "-", "AUROC": b_auroc,
            "Delta_AUROC": 0.0, "AUPRC": b_auprc, "Delta_AUPRC": 0.0}]

for cat in CATEGORIES:
    n_tokens = len(cat_token_ids[cat])
    if n_tokens == 0:
        log.info(f"  SKIP {cat} (0 tokens in vocab)")
        continue
    log.info(f"Evaluating {cat} ({n_tokens} tokens)...")
    auroc, auprc, _ = evaluate(cat)
    d_auroc = auroc - b_auroc
    d_auprc = auprc - b_auprc
    log.info(f"  {cat}: AUROC={auroc:.4f} ({d_auroc:+.4f})  AUPRC={auprc:.4f} ({d_auprc:+.4f})")
    results.append({"Category": cat, "Tokens": n_tokens, "AUROC": auroc,
                     "Delta_AUROC": d_auroc, "AUPRC": auprc, "Delta_AUPRC": d_auprc})

df = pd.DataFrame(results).sort_values("Delta_AUROC")
df.to_csv(OUT / "occlusion_results.csv", index=False)
log.info(f"\nSaved: {OUT/'occlusion_results.csv'}")
log.info(f"\n{df.to_string(index=False)}")
PYEOF

echo "Occlusion complete"
