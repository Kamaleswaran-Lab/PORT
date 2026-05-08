#!/bin/bash
#SBATCH --job-name=fig6_drill
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/CHD_MEDS/results_v4/slurm_fig6_drill_%j.log
#SBATCH --error=/path/to/CHD_MEDS/results_v4/slurm_fig6_drill_%j.err

set -e
cd /path/to/CHDLLM
export PATH=/path/to/miniforge3/envs/ethos/bin:$PATH
export PYTHONPATH=/path/to/CHDLLM/ethos-ares/src:$PYTHONPATH

echo "Drill-down for Fig 6: sub-category decomposition + counterfactual swaps"

python3 << 'PYEOF'
import sys, logging, torch as th, numpy as np, pandas as pd
from pathlib import Path
from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, "ethos")
sys.path.insert(0, "ethos/datasets")

MODEL_FP = Path("/path/to/CHD_MEDS/tokenized_v4/models/chd_v4_layer6_do0.3/best_model.pt")
HEAD_FP  = Path("/path/to/CHD_MEDS/results_v4/baselines/ethos/finetune/finetune_lora_head_best_v4_lora_s123.pt")
TEST_DIR = Path("/path/to/CHD_MEDS/tokenized_v4/test")
TASK_PATH = Path("/path/to/CHD_MEDS/outcome/iod_task.parquet")
OUT = Path("/path/to/CHD_MEDS/results_v4/evaluation/fig6_drilldown")
OUT.mkdir(parents=True, exist_ok=True)

# ── Load model + LoRA + head ──
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

# ── Dataset ──
from iod_dataset import IoDDataset
ds = IoDDataset(TEST_DIR, n_positions=2048)
task = pd.read_parquet(TASK_PATH)
task['sid'] = task['patient_id'].str.lstrip('C').astype(int)
task['pt_us'] = task['prediction_time'].astype('int64') // 1000
label_dict = {(int(r.sid), int(r.pt_us)): int(r.boolean_value) for r in task.itertuples()}

itos = {i: tok for tok, i in ds.vocab.stoi.items()}
Q_TOKEN_IDS = set(tid for tok, tid in ds.vocab.stoi.items() if tok.startswith("Q") and len(tok) <= 3)

# ── Define mask groups ──
def ids_with_prefix(prefix):
    return set(tid for tok, tid in ds.vocab.stoi.items() if tok.startswith(prefix))

# Sub-decomposition of Surgical Context (ENCOUNTER//AN//) — keep OR_ENTRY out of every mask
or_entry_id = ds.vocab.stoi["ENCOUNTER//AN//OR_ENTRY"]
SC_GROUPS = {
    "SC_primary_procedure": ids_with_prefix("ENCOUNTER//AN//PRIMARY_PROCEDURE"),
    "SC_asa_score":         {ds.vocab.stoi["ENCOUNTER//AN//ASA_SCORE"]},
    "SC_admission_type":    ids_with_prefix("ENCOUNTER//AN//ADMISSION_TYPE"),
    "SC_patient_class":     ids_with_prefix("ENCOUNTER//AN//PATIENT_CLASS"),
    "SC_or_markers":        ({ds.vocab.stoi[t] for t in
                              ["ENCOUNTER//AN//AN_START","ENCOUNTER//AN//AN_END",
                               "ENCOUNTER//AN//OR_EXIT","ENCOUNTER//AN//HOSPITAL_ADMISSION",
                               "ENCOUNTER//AN//HOSPITAL_DISCHARGE"]
                              if t in ds.vocab.stoi}),
}
for k in SC_GROUPS:
    SC_GROUPS[k].discard(or_entry_id)

# Sub-decomposition of Medications: non-vasoactive ATC subset
VASO_SUFFIXES = {"A24","A04","E02","A03","A06","A01","B10","A26"}
vaso_ids = set()
for tok, tid in ds.vocab.stoi.items():
    if tok == "ATC//C01":
        vaso_ids.add(tid)
    elif tok.startswith("ATC//SFX//") and tok.replace("ATC//SFX//","") in VASO_SUFFIXES:
        vaso_ids.add(tid)
non_vaso_atc = ids_with_prefix("ATC//") - vaso_ids
MED_GROUPS = {
    "MED_non_vasoactive_atc": non_vaso_atc,
}

ALL_MASK_GROUPS = {**SC_GROUPS, **MED_GROUPS}
for name, ids in ALL_MASK_GROUPS.items():
    log.info(f"  {name:<30s} {len(ids)} tokens")

# Counterfactual swap targets
PROC_SWAP_TARGET = ds.vocab.stoi["ENCOUNTER//AN//PRIMARY_PROCEDURE//MYRINGOTOMY_W/_TUBES"]
ASA_SCORE_ID = ds.vocab.stoi["ENCOUNTER//AN//ASA_SCORE"]
Q1_ID = ds.vocab.stoi["Q1"]
Q9_ID = ds.vocab.stoi["Q9"]
PROC_PREFIX_IDS = ids_with_prefix("ENCOUNTER//AN//PRIMARY_PROCEDURE")
log.info(f"PROC swap target: {PROC_SWAP_TARGET} (MYRINGOTOMY_W/_TUBES)")
log.info(f"ASA swap targets: Q1={Q1_ID}, Q9={Q9_ID}")

# ── Helpers ──
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

def predict(ids_1d):
    ids = ids_1d.unsqueeze(0).cuda()
    with th.no_grad():
        return th.sigmoid(head(get_hidden(model, ids)).squeeze()).item()

def mask_with_adjacent_q(ids_1d, target_set):
    """Set tokens in target_set (and adjacent Q if present) to 0."""
    out = ids_1d.clone()
    seq = out.tolist()
    for i, tid in enumerate(seq):
        if tid in target_set:
            out[i] = 0
            if i + 1 < len(seq) and seq[i + 1] in Q_TOKEN_IDS:
                out[i + 1] = 0
    return out

def swap_procedure(ids_1d, prefix_ids, target_id):
    """Replace any PRIMARY_PROCEDURE token with target_id."""
    out = ids_1d.clone()
    seq = out.tolist()
    swapped = False
    for i, tid in enumerate(seq):
        if tid in prefix_ids:
            out[i] = target_id
            swapped = True
    return out, swapped

def swap_asa_q(ids_1d, asa_id, target_q):
    """Replace the Q value following an ASA_SCORE token with target_q."""
    out = ids_1d.clone()
    seq = out.tolist()
    swapped = False
    for i, tid in enumerate(seq):
        if tid == asa_id and i + 1 < len(seq) and seq[i + 1] in Q_TOKEN_IDS:
            out[i + 1] = target_q
            swapped = True
    return out, swapped

# ── Run all experiments ──
sids, pts, yt = [], [], []
results = {name: [] for name in ALL_MASK_GROUPS}
results["proc_swap_to_myringotomy"] = []
results["asa_swap_to_q1"] = []
results["asa_swap_to_q9"] = []
flags = {"has_proc": [], "has_asa": []}

# Also recompute baseline within this run for parquet alignment
results["baseline"] = []

for i in range(len(ds)):
    x, meta = ds[i]
    pid, pt = meta["patient_id"], meta["prediction_time"]
    pt_us = pt // 1000 if pt > 1e15 else pt
    lab = label_dict.get((pid, pt_us))
    if lab is None: continue

    sids.append(pid); pts.append(pt_us); yt.append(lab)

    # Baseline
    p_base = predict(x)
    results["baseline"].append(p_base)

    # Sub-mask experiments
    for name, target_set in ALL_MASK_GROUPS.items():
        if not target_set:
            results[name].append(p_base)
            continue
        x_m = mask_with_adjacent_q(x, target_set)
        results[name].append(predict(x_m))

    # Procedure swap
    seq_list = x.tolist()
    has_proc = any(t in PROC_PREFIX_IDS for t in seq_list)
    flags["has_proc"].append(has_proc)
    if has_proc:
        x_proc, _ = swap_procedure(x, PROC_PREFIX_IDS, PROC_SWAP_TARGET)
        results["proc_swap_to_myringotomy"].append(predict(x_proc))
    else:
        results["proc_swap_to_myringotomy"].append(p_base)

    # ASA Q swap
    has_asa = ASA_SCORE_ID in seq_list
    flags["has_asa"].append(has_asa)
    if has_asa:
        x_q1, _ = swap_asa_q(x, ASA_SCORE_ID, Q1_ID)
        x_q9, _ = swap_asa_q(x, ASA_SCORE_ID, Q9_ID)
        results["asa_swap_to_q1"].append(predict(x_q1))
        results["asa_swap_to_q9"].append(predict(x_q9))
    else:
        results["asa_swap_to_q1"].append(p_base)
        results["asa_swap_to_q9"].append(p_base)

    if (i + 1) % 5000 == 0:
        log.info(f"  Processed {i+1}/{len(ds)}")

df = pd.DataFrame({
    "subject_id": sids,
    "prediction_time_us": pts,
    "y_true": yt,
    **{name: results[name] for name in results},
    "has_proc": flags["has_proc"],
    "has_asa": flags["has_asa"],
})
df.to_parquet(OUT / "drilldown_predictions.parquet", index=False)
log.info(f"Saved n={len(df):,} per-encounter predictions to {OUT}")
log.info(f"  has_proc: {sum(flags['has_proc']):,};  has_asa: {sum(flags['has_asa']):,}")

# Quick sanity summary
from sklearn.metrics import roc_auc_score, average_precision_score
y = df["y_true"].values
for name in ["baseline"] + list(ALL_MASK_GROUPS.keys()) + ["proc_swap_to_myringotomy", "asa_swap_to_q1", "asa_swap_to_q9"]:
    p = df[name].values
    a = roc_auc_score(y, p); ap = average_precision_score(y, p)
    log.info(f"  {name:<30s}  AUROC={a:.4f}  AUPRC={ap:.4f}")
PYEOF

echo "Drill-down complete"
