#!/bin/bash
#SBATCH --job-name=counterfact
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=/path/to/CHD_MEDS/results_v4/slurm_counterfact_%j.log
#SBATCH --error=/path/to/CHD_MEDS/results_v4/slurm_counterfact_%j.err

set -e
cd /path/to/CHDLLM
export PATH=/path/to/miniforge3/envs/ethos/bin:$PATH
export PYTHONPATH=/path/to/CHDLLM/ethos-ares/src:$PYTHONPATH

echo "Counterfactual analysis (two clinically motivated swaps)"

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
OUT = Path("/path/to/CHD_MEDS/results_v4/evaluation/counterfactual")
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

# ── Define counterfactual swap target sets ──
# Scenario A: "if patient had not been on pre-operative vasoactive support".
# Mask the parent class (ATC//C01 = cardiac stimulants) plus all vasoactive
# leaf-suffix tokens used as IoD3/IoD4 stop tokens.
VASO_SUFFIXES = {"A24", "A04", "E02", "A03", "A06", "A01", "B10", "A26"}
vaso_ids = set()
for tok, tid in ds.vocab.stoi.items():
    if tok == "ATC//C01":
        vaso_ids.add(tid)
    elif tok.startswith("ATC//SFX//"):
        sfx = tok.replace("ATC//SFX//", "")
        if sfx in VASO_SUFFIXES:
            vaso_ids.add(tid)
log.info(f"Scenario A (vasoactive swap-to-absent): {len(vaso_ids)} target tokens")

# Scenario B: "if patient had a lower-acuity ASA score".
# Strategy: locate ENCOUNTER//AN//ASA_SCORE token, then replace the Q value
# at the next position with Q2 (a low-acuity baseline). If Q2 not in vocab,
# fall back to masking the ASA-Q pair.
asa_score_id = ds.vocab.stoi.get("ENCOUNTER//AN//ASA_SCORE", None)
q2_id = ds.vocab.stoi.get("Q2", None)
log.info(f"Scenario B: ASA_SCORE token id={asa_score_id}, Q2 token id={q2_id}")

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

def mask_targets(ids_1d, target_set):
    out = ids_1d.clone()
    seq = out.tolist()
    for i, tid in enumerate(seq):
        if tid in target_set:
            out[i] = 0
            if i + 1 < len(seq) and seq[i+1] in Q_TOKEN_IDS:
                out[i + 1] = 0
    return out

def swap_asa(ids_1d, asa_id, target_q):
    out = ids_1d.clone()
    seq = out.tolist()
    swapped = False
    for i, tid in enumerate(seq):
        if tid == asa_id and i + 1 < len(seq) and seq[i + 1] in Q_TOKEN_IDS:
            out[i + 1] = target_q
            swapped = True
    return out, swapped

# ── Run both scenarios ──
sids, pts, yt = [], [], []
y_base, y_vaso, y_asa = [], [], []
on_vaso, has_asa = [], []

for i in range(len(ds)):
    x, meta = ds[i]
    pid, pt = meta["patient_id"], meta["prediction_time"]
    pt_us = pt // 1000 if pt > 1e15 else pt
    lab = label_dict.get((pid, pt_us))
    if lab is None: continue

    seq_list = x.tolist()
    on_v = any(t in vaso_ids for t in seq_list)
    h_a = (asa_score_id is not None) and (asa_score_id in seq_list)

    p_base = predict(x)

    # Scenario A: vasoactive removal
    if on_v:
        x_v = mask_targets(x, vaso_ids)
        p_v = predict(x_v)
    else:
        p_v = p_base  # no change for patients not on vasoactives

    # Scenario B: ASA -> Q2
    if h_a and q2_id is not None:
        x_a, _ = swap_asa(x, asa_score_id, q2_id)
        p_a = predict(x_a)
    else:
        p_a = p_base

    sids.append(pid); pts.append(pt_us); yt.append(lab)
    y_base.append(p_base); y_vaso.append(p_v); y_asa.append(p_a)
    on_vaso.append(on_v); has_asa.append(h_a)

    if (i + 1) % 5000 == 0:
        log.info(f"  Processed {i+1}/{len(ds)}")

df = pd.DataFrame({
    "subject_id": sids,
    "prediction_time_us": pts,
    "y_true": yt,
    "p_base": y_base,
    "p_no_vasoactives": y_vaso,
    "p_asa_q2": y_asa,
    "on_vasoactive": on_vaso,
    "has_asa": has_asa,
})
df.to_parquet(OUT / "counterfactual_predictions.parquet", index=False)
log.info(f"Saved n={len(df):,} per-encounter counterfactual predictions to {OUT}")
log.info(f"  on_vasoactive: {sum(on_vaso):,} encounters; has_asa: {sum(has_asa):,}")
log.info(f"  mean Δ vaso: {np.mean(np.array(y_vaso) - np.array(y_base)):+.4f}")
log.info(f"  mean Δ asa : {np.mean(np.array(y_asa)  - np.array(y_base)):+.4f}")
PYEOF

echo "Counterfactual complete"
