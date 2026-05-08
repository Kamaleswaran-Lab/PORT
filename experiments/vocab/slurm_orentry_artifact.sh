#!/bin/bash
#SBATCH --job-name=orentry_art
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=/path/to/CHD_MEDS/results_v4/slurm_orentry_art_%j.log
#SBATCH --error=/path/to/CHD_MEDS/results_v4/slurm_orentry_art_%j.err

set -e
cd /path/to/CHDLLM
export PATH=/path/to/miniforge3/envs/ethos/bin:$PATH
export PYTHONPATH=/path/to/CHDLLM/ethos-ares/src:$PYTHONPATH

echo "OR_ENTRY artifact analysis: split SC mask into cursor-only vs content-only"

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
HEAD_FP  = Path("/path/to/CHD_MEDS/results_v4/baselines/ethos/finetune/finetune_lora_head_best_v4_lora_s123.pt")
TEST_DIR = Path("/path/to/CHD_MEDS/tokenized_v4/test")
TASK_PATH = Path("/path/to/CHD_MEDS/outcome/iod_task.parquet")
OUT = Path("/path/to/CHD_MEDS/results_v4/evaluation/orentry_artifact")
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

Q_TOKEN_IDS = set(tid for tok, tid in ds.vocab.stoi.items() if tok.startswith("Q") and len(tok) <= 3)

def ids_with_prefix(prefix):
    return set(tid for tok, tid in ds.vocab.stoi.items() if tok.startswith(prefix))

or_entry_id = ds.vocab.stoi["ENCOUNTER//AN//OR_ENTRY"]

# All ENCOUNTER//AN// tokens (= Fig 5 "Surgical Context")
SC_FULL = ids_with_prefix("ENCOUNTER//AN//")

# Three masks:
MASKS = {
    "sc_full":             SC_FULL,                  # Fig 5 reproduction
    "or_entry_only":       {or_entry_id},            # cursor token only
    "sc_minus_or_entry":   SC_FULL - {or_entry_id},  # content only
}
for name, ids in MASKS.items():
    log.info(f"  {name:<25s} {len(ids):>5d} tokens")

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

def evaluate(category_ids=None):
    sids, pts, yt, yp = [], [], [], []
    for i in range(len(ds)):
        x, meta = ds[i]
        pid, pt = meta["patient_id"], meta["prediction_time"]
        pt_us = pt // 1000 if pt > 1e15 else pt
        lab = label_dict.get((pid, pt_us))
        if lab is None: continue
        ids = x.unsqueeze(0).cuda()
        if category_ids is not None:
            mask = build_mask_with_adjacent_q(ids.squeeze(0), category_ids)
            ids = ids.masked_fill(mask.unsqueeze(0), 0)
        with th.no_grad():
            prob = th.sigmoid(head(get_hidden(model, ids)).squeeze()).item()
        sids.append(pid); pts.append(pt_us); yt.append(lab); yp.append(prob)
        if (i+1) % 5000 == 0:
            log.info(f"  [{('mask' if category_ids else 'baseline')}] {i+1}/{len(ds)}")
    return pd.DataFrame({"subject_id": sids, "prediction_time_us": pts,
                         "y_true": yt, "y_prob": yp})

log.info("=" * 60)
log.info("Baseline (no mask)")
df_base = evaluate(None)
df_base.to_parquet(OUT / "predictions_baseline.parquet", index=False)
b_a = roc_auc_score(df_base.y_true, df_base.y_prob)
b_p = average_precision_score(df_base.y_true, df_base.y_prob)
log.info(f"Baseline n={len(df_base):,}  AUROC={b_a:.4f}  AUPRC={b_p:.4f}")

rows = [{"mask": "baseline", "n_tok": 0, "AUROC": b_a, "AUPRC": b_p,
         "dAUROC": 0.0, "dAUPRC": 0.0}]

for name, ids in MASKS.items():
    log.info("=" * 60)
    log.info(f"Mask: {name} ({len(ids)} tokens)")
    df_m = evaluate(ids)
    df_m.to_parquet(OUT / f"predictions_{name}.parquet", index=False)
    a = roc_auc_score(df_m.y_true, df_m.y_prob)
    p = average_precision_score(df_m.y_true, df_m.y_prob)
    log.info(f"  AUROC={a:.4f} (Δ{a-b_a:+.4f})  AUPRC={p:.4f} (Δ{p-b_p:+.4f})")
    rows.append({"mask": name, "n_tok": len(ids), "AUROC": a, "AUPRC": p,
                 "dAUROC": a-b_a, "dAUPRC": p-b_p})

pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
log.info("=" * 60)
log.info("Summary:")
log.info("\n" + pd.DataFrame(rows).to_string(index=False))
PYEOF

echo "Done."
