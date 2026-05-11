#!/bin/bash
# Quantile-pairing ablation (test-time): mask ALL Q1–Q10 tokens together and
# re-evaluate PORT (LoRA r=8, s123). If AUROC/AUPRC degrade substantially, the
# Q tokens carry essential numeric-magnitude signal beyond code identity.
#
#SBATCH --job-name=q_abl
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=/path/to/CHD_MEDS/results_v4/slurm_q_abl_%j.log
#SBATCH --error=/path/to/CHD_MEDS/results_v4/slurm_q_abl_%j.err

set -e
cd /path/to/CHDLLM
export PATH=/path/to/miniforge3/envs/ethos/bin:$PATH
export PYTHONPATH=/path/to/CHDLLM/ethos-ares/src:$PYTHONPATH
export CHD_DATA_ROOT=/path/to/CHD_MEDS

echo "Quantile-pairing ablation: mask all Q1-Q10 tokens at inference"

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
OUT = Path("/path/to/CHD_MEDS/results_v4/evaluation/quantile_ablation")
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

Q_TOKEN_IDS = set(tid for tok, tid in ds.vocab.stoi.items()
                  if tok.startswith("Q") and len(tok) <= 3)
log.info(f"Q-token IDs to mask: {len(Q_TOKEN_IDS)}  ({sorted(t for t in ds.vocab.stoi if t.startswith('Q') and len(t)<=3)})")

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

def evaluate(mask_q=False):
    sids, pts, yt, yp = [], [], [], []
    for i in range(len(ds)):
        x, meta = ds[i]
        pid, pt = meta["patient_id"], meta["prediction_time"]
        pt_us = pt // 1000 if pt > 1e15 else pt
        lab = label_dict.get((pid, pt_us))
        if lab is None: continue
        ids = x.unsqueeze(0).cuda()
        if mask_q:
            mask = th.zeros_like(ids, dtype=th.bool)
            for qid in Q_TOKEN_IDS:
                mask |= (ids == qid)
            ids = ids.masked_fill(mask, 0)
        with th.no_grad():
            prob = th.sigmoid(head(get_hidden(model, ids)).squeeze()).item()
        sids.append(pid); pts.append(pt_us); yt.append(lab); yp.append(prob)
        if (i+1) % 5000 == 0:
            log.info(f"  [{('mask_q' if mask_q else 'baseline')}] {i+1}/{len(ds)}")
    return pd.DataFrame({"subject_id": sids, "prediction_time_us": pts,
                         "y_true": yt, "y_prob": yp})

log.info("Baseline …")
df_b = evaluate(False)
df_b.to_parquet(OUT / "predictions_baseline.parquet", index=False)
a_b = roc_auc_score(df_b.y_true, df_b.y_prob)
p_b = average_precision_score(df_b.y_true, df_b.y_prob)
log.info(f"Baseline n={len(df_b):,} AUROC={a_b:.4f} AUPRC={p_b:.4f}")

log.info("All-Q mask …")
df_m = evaluate(True)
df_m.to_parquet(OUT / "predictions_mask_all_q.parquet", index=False)
a_m = roc_auc_score(df_m.y_true, df_m.y_prob)
p_m = average_precision_score(df_m.y_true, df_m.y_prob)
log.info(f"Mask-Q  n={len(df_m):,} AUROC={a_m:.4f} (Δ{a_m-a_b:+.4f})  AUPRC={p_m:.4f} (Δ{p_m-p_b:+.4f})")

pd.DataFrame([
    dict(mask="baseline", AUROC=a_b, AUPRC=p_b, dAUROC=0.0, dAUPRC=0.0),
    dict(mask="all_Q_tokens", AUROC=a_m, AUPRC=p_m, dAUROC=a_m-a_b, dAUPRC=p_m-p_b),
]).to_csv(OUT / "summary.csv", index=False)
print("\n", pd.read_csv(OUT / "summary.csv").to_string(index=False))
PYEOF
echo "Done."
