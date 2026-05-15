#!/bin/bash
# Stage 2 launcher — submit all downstream experiments for the (r=8, BCE)
# winner of the HP grid + 3-seed sweep.
#
# Usage:
#   bash submit_stage2.sh BEST_SEED
#
#   where BEST_SEED ∈ {42, 123, 456} chosen by max test AUROC.
#
# Submits:
#   1. Context window 7d/30d/90d/365d at BEST_SEED               (4 jobs)
#   2. Per-patient occlusion using BEST_SEED head                 (1 job)
#   3. Q-token occlusion using BEST_SEED head                     (1 job)
#   4. OR_ENTRY artifact verification using BEST_SEED head        (1 job)
#   5. Data efficiency PORT (r=8, bce) at 1/5/10/25/50%           (5 jobs)
#   6. Data efficiency BiLSTM at 1/5/10/25/50% (independent)      (5 jobs)
#   7. Case examples at BEST_SEED (only if BEST_SEED != 123)      (≤1 job)
#
# Total: up to 18 jobs.

set -e
SEED=${1:?Usage: $0 BEST_SEED}
echo "Stage 2 launch with BEST_SEED=${SEED}"

# Resolved paths — data canonical location is /data/klabFiles/CHD (since 2026-05-15)
DATA=/data/klabFiles/CHD/CHD_MEDS
CODE=/hpc/home/jkim1/workspace/CHDLLM
CONDA=/hpc/home/jkim1/miniforge3
MODEL_FP=$DATA/tokenized_v4/models/chd_v4_layer6_do0.3/best_model.pt
TOK=$DATA/tokenized_v4
RESULTS=$DATA/results_v4/baselines
HEAD_FP=$RESULTS/ethos/finetune/finetune_lora_head_best_v4_lora_r8_bce_s${SEED}.pt

# ── Verify head exists ──
if [ ! -f "$HEAD_FP" ]; then
    echo "ERROR: head not found at $HEAD_FP"
    exit 1
fi
echo "Using head: $HEAD_FP"

mkdir -p /tmp/stage2_runs

# ── 1. Context window (4 jobs) ─────────────────────────────────────────────
cat > /tmp/stage2_runs/ctx_r8bce_s${SEED}.sh << EOF
#!/bin/bash
#SBATCH --job-name=ctx_s${SEED}
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --output=$DATA/results_v4/slurm_ctx_s${SEED}_%A_%a.log
#SBATCH --error=$DATA/results_v4/slurm_ctx_s${SEED}_%A_%a.err
#SBATCH --array=0-3

set -e
cd $CODE
export PATH=$CONDA/envs/ethos/bin:\$PATH
export PYTHONPATH=$CODE/ethos-ares/src:\$PYTHONPATH
export CHD_DATA_ROOT=$DATA

WINDOWS=(7 30 90 365)
W=\${WINDOWS[\$SLURM_ARRAY_TASK_ID]}

echo "Ctx window (r=8 bce s${SEED}, ctx=\${W}d)"
python ethos/finetune.py \\
    --model_fp $MODEL_FP \\
    --train_dir $TOK/train --val_dir $TOK/val --test_dir $TOK/test \\
    --results_dir $RESULTS \\
    --seed ${SEED} \\
    --epochs 15 --patience 5 \\
    --lr 1e-3 --hidden_dim 256 --head_dropout 0.1 \\
    --lora --lora_r 8 --lora_alpha 16 \\
    --loss_type bce \\
    --window_days \$W \\
    --suffix _v4_lora_r8_bce_ctx\${W}d_s${SEED}
EOF
JOB_CTX=$(sbatch /tmp/stage2_runs/ctx_r8bce_s${SEED}.sh | awk '{print $4}')
echo "  [1] Context window submitted: ${JOB_CTX}"

# ── 2. Per-patient occlusion (1 job) ───────────────────────────────────────
cat > /tmp/stage2_runs/occ_perpatient_s${SEED}.sh << EOF
#!/bin/bash
#SBATCH --job-name=occ_s${SEED}
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --output=$DATA/results_v4/slurm_occ_s${SEED}_%j.log
#SBATCH --error=$DATA/results_v4/slurm_occ_s${SEED}_%j.err

set -e
cd $CODE
export PATH=$CONDA/envs/ethos/bin:\$PATH
export PYTHONPATH=$CODE/ethos-ares/src:\$PYTHONPATH
export CHD_DATA_ROOT=$DATA

# Reuse occlusion script logic but point head to best seed
python3 - << 'PYEOF'
import sys, logging, torch as th, numpy as np, pandas as pd
from pathlib import Path
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_auc_score, average_precision_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)
sys.path.insert(0, "ethos"); sys.path.insert(0, "ethos/datasets")

MODEL_FP = Path("$MODEL_FP")
HEAD_FP  = Path("$HEAD_FP")
TEST_DIR = Path("$TOK/test")
TASK_PATH = Path("$DATA/outcome/iod_task.parquet")
OUT = Path("$DATA/results_v4/evaluation/per_patient_occlusion_r8bce_s${SEED}")
OUT.mkdir(parents=True, exist_ok=True)

from ethos.utils import load_model_checkpoint
model, _ = load_model_checkpoint(MODEL_FP, map_location="cuda:0")
model.to("cuda:0")
lc = LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"], lora_dropout=0.0, bias="none")
model = get_peft_model(model, lc)
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
or_entry_id = ds.vocab.stoi["ENCOUNTER//AN//OR_ENTRY"]

CATEGORIES = {
    "ICD-10 Codes": ["ICD//CM//"], "Medications (ATC)": ["ATC//", "MED//"],
    "Surgical Context (ENCOUNTER)": ["ENCOUNTER//AN//"],
    "Surgical History (PROCEDURE)": ["PROCEDURE//"],
    "Problem List (free-text)": ["PROBLEM//"],
    "Medical History (free-text)": ["DIAGNOSIS//"],
    "Laboratory": ["LAB//"], "Vital Signs": ["VITAL//"],
    "Care Trajectory (ADT)": ["ADT//"], "Lines/Drains (LDA)": ["LDA//"],
    "Anesthesia Events": ["AN_EVENT//"], "Transfusions": ["TRANSFUSION//"],
    "Structured Data (SDE)": ["SDE//"], "Insurance": ["INSURANCE//"],
    "Point of Origin": ["POINT_OF_ORIGIN//"], "Home County": ["HOME_COUNTY//"],
    "Language": ["LANGUAGE//"], "Demographics": ["DEMO//"],
}

cat_token_ids = {}
for cat, prefixes in CATEGORIES.items():
    ids = set(tid for tok, tid in ds.vocab.stoi.items() if any(tok.startswith(p) for p in prefixes))
    if cat == "Surgical Context (ENCOUNTER)":
        ids.discard(or_entry_id)
    cat_token_ids[cat] = ids

def get_hidden(model, ids):
    base = getattr(model, "base_model", model); base = getattr(base, "model", base)
    transformer = base.transformer
    _, t = ids.size()
    x = transformer.drop(transformer.wte(ids) + transformer.wpe(base.pos[:t]))
    for block in transformer.h:
        out = block(x); x = out[0] if isinstance(out, tuple) else out
    return transformer.ln_f(x)[:, -1, :]

def build_mask(input_ids_1d, category_ids):
    mask = th.zeros_like(input_ids_1d, dtype=th.bool)
    seq_len = input_ids_1d.size(0)
    for i in range(seq_len):
        tid = input_ids_1d[i].item()
        if tid in category_ids:
            mask[i] = True
            if i + 1 < seq_len and input_ids_1d[i + 1].item() in Q_TOKEN_IDS:
                mask[i + 1] = True
    return mask

def evaluate(category=None):
    sids, pts, yt, yp = [], [], [], []
    for i in range(len(ds)):
        x, meta = ds[i]
        pid, pt = meta["patient_id"], meta["prediction_time"]
        pt_us = pt // 1000 if pt > 1e15 else pt
        lab = label_dict.get((pid, pt_us))
        if lab is None: continue
        ids = x.unsqueeze(0).cuda()
        if category is not None:
            mask = build_mask(ids.squeeze(0), cat_token_ids[category])
            ids = ids.masked_fill(mask.unsqueeze(0), 0)
        with th.no_grad():
            prob = th.sigmoid(head(get_hidden(model, ids)).squeeze()).item()
        sids.append(pid); pts.append(pt_us); yt.append(lab); yp.append(prob)
        if (i+1) % 5000 == 0:
            log.info(f"  [{category or 'baseline'}] {i+1}/{len(ds)}")
    return pd.DataFrame({"subject_id": sids, "prediction_time_us": pts, "y_true": yt, "y_prob": yp})

log.info("Baseline …")
df_base = evaluate(); df_base.to_parquet(OUT / "predictions_baseline.parquet", index=False)
b_a = roc_auc_score(df_base.y_true, df_base.y_prob); b_p = average_precision_score(df_base.y_true, df_base.y_prob)
log.info(f"Baseline n={len(df_base):,} AUROC={b_a:.4f} AUPRC={b_p:.4f}")
summary = [{"Category":"Baseline","Tokens":0,"AUROC":b_a,"AUPRC":b_p,"Delta_AUROC":0.0,"Delta_AUPRC":0.0}]
for cat in CATEGORIES:
    n_tok = len(cat_token_ids[cat])
    if n_tok == 0: continue
    log.info(f"Masking {cat} ({n_tok})…")
    df_m = evaluate(cat); safe = cat.replace(" ","_").replace("/","_").replace("(","").replace(")","")
    df_m.to_parquet(OUT / f"predictions_{safe}.parquet", index=False)
    a = roc_auc_score(df_m.y_true, df_m.y_prob); p = average_precision_score(df_m.y_true, df_m.y_prob)
    summary.append({"Category":cat,"Tokens":n_tok,"AUROC":a,"AUPRC":p,"Delta_AUROC":a-b_a,"Delta_AUPRC":p-b_p})
    log.info(f"  {cat}: AUROC={a:.4f} (Δ{a-b_a:+.4f})  AUPRC={p:.4f} (Δ{p-b_p:+.4f})")
pd.DataFrame(summary).sort_values("Delta_AUROC").to_csv(OUT / "summary.csv", index=False)
PYEOF
EOF
JOB_OCC=$(sbatch /tmp/stage2_runs/occ_perpatient_s${SEED}.sh | awk '{print $4}')
echo "  [2] Per-patient occlusion submitted: ${JOB_OCC}"

# ── 3. Q-token occlusion (1 job) + 4. OR_ENTRY artifact (1 job) ────────────
cat > /tmp/stage2_runs/qtok_orentry_s${SEED}.sh << EOF
#!/bin/bash
#SBATCH --job-name=qor_s${SEED}
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=$DATA/results_v4/slurm_qor_s${SEED}_%j.log
#SBATCH --error=$DATA/results_v4/slurm_qor_s${SEED}_%j.err

set -e
cd $CODE
export PATH=$CONDA/envs/ethos/bin:\$PATH
export PYTHONPATH=$CODE/ethos-ares/src:\$PYTHONPATH
export CHD_DATA_ROOT=$DATA

python3 - << 'PYEOF'
import sys, torch as th, numpy as np, pandas as pd
from pathlib import Path
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, "ethos"); sys.path.insert(0, "ethos/datasets")

MODEL_FP = Path("$MODEL_FP"); HEAD_FP = Path("$HEAD_FP")
TEST_DIR = Path("$TOK/test"); TASK_PATH = Path("$DATA/outcome/iod_task.parquet")
OUT_Q = Path("$DATA/results_v4/evaluation/quantile_ablation_r8bce_s${SEED}"); OUT_Q.mkdir(parents=True, exist_ok=True)
OUT_OR= Path("$DATA/results_v4/evaluation/orentry_artifact_r8bce_s${SEED}"); OUT_OR.mkdir(parents=True, exist_ok=True)

from ethos.utils import load_model_checkpoint
model, _ = load_model_checkpoint(MODEL_FP, map_location="cuda:0"); model.to("cuda:0")
lc = LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"], lora_dropout=0.0, bias="none")
model = get_peft_model(model, lc)
ckpt = th.load(HEAD_FP, map_location="cuda:0", weights_only=False)
if ckpt.get("lora_state_dict"): model.load_state_dict(ckpt["lora_state_dict"], strict=False)
cfg = ckpt["head_config"]
head = th.nn.Sequential(th.nn.Linear(cfg["n_embd"], cfg["hidden_dim"]), th.nn.ReLU(), th.nn.Dropout(0.0), th.nn.Linear(cfg["hidden_dim"], 1)).cuda().eval()
head.load_state_dict({k.replace("net.",""): v for k, v in ckpt["head_state_dict"].items()})
model.eval()

from iod_dataset import IoDDataset
ds = IoDDataset(TEST_DIR, n_positions=2048)
task = pd.read_parquet(TASK_PATH)
task['sid'] = task['patient_id'].str.lstrip('C').astype(int); task['pt_us'] = task['prediction_time'].astype('int64') // 1000
label_dict = {(int(r.sid), int(r.pt_us)): int(r.boolean_value) for r in task.itertuples()}
Q_IDS = set(tid for tok, tid in ds.vocab.stoi.items() if tok.startswith("Q") and len(tok) <= 3)
or_entry_id = ds.vocab.stoi["ENCOUNTER//AN//OR_ENTRY"]
SC_FULL = set(tid for tok, tid in ds.vocab.stoi.items() if tok.startswith("ENCOUNTER//AN//"))

def get_hidden(m, ids):
    base = getattr(m, "base_model", m); base = getattr(base, "model", base); tf = base.transformer
    _, t = ids.size()
    x = tf.drop(tf.wte(ids) + tf.wpe(base.pos[:t]))
    for blk in tf.h: out = blk(x); x = out[0] if isinstance(out, tuple) else out
    return tf.ln_f(x)[:, -1, :]

def build_mask(input_ids_1d, target_set, with_q=True):
    mask = th.zeros_like(input_ids_1d, dtype=th.bool)
    n = input_ids_1d.size(0)
    for i in range(n):
        tid = input_ids_1d[i].item()
        if tid in target_set:
            mask[i] = True
            if with_q and i+1 < n and input_ids_1d[i+1].item() in Q_IDS: mask[i+1] = True
    return mask

def evaluate(target_set=None, q_only=False, adj_q=True):
    sids, pts, yt, yp = [], [], [], []
    for i in range(len(ds)):
        x, meta = ds[i]
        pid, pt = meta["patient_id"], meta["prediction_time"]
        pt_us = pt // 1000 if pt > 1e15 else pt
        lab = label_dict.get((pid, pt_us))
        if lab is None: continue
        ids = x.unsqueeze(0).cuda()
        if q_only:
            m = th.zeros_like(ids, dtype=th.bool)
            for q in Q_IDS: m |= (ids == q)
            ids = ids.masked_fill(m, 0)
        elif target_set is not None:
            m = build_mask(ids.squeeze(0), target_set, with_q=adj_q)
            ids = ids.masked_fill(m.unsqueeze(0), 0)
        with th.no_grad():
            prob = th.sigmoid(head(get_hidden(model, ids)).squeeze()).item()
        sids.append(pid); pts.append(pt_us); yt.append(lab); yp.append(prob)
    return pd.DataFrame({"subject_id":sids,"prediction_time_us":pts,"y_true":yt,"y_prob":yp})

# Baseline (shared)
print("Baseline …")
df_b = evaluate(); df_b.to_parquet(OUT_Q / "predictions_baseline.parquet", index=False)
df_b.to_parquet(OUT_OR / "predictions_baseline.parquet", index=False)
a_b = roc_auc_score(df_b.y_true, df_b.y_prob); p_b = average_precision_score(df_b.y_true, df_b.y_prob)
print(f"Baseline AUROC={a_b:.4f} AUPRC={p_b:.4f}")

# Q-token mask
print("Q mask …")
df_q = evaluate(q_only=True); df_q.to_parquet(OUT_Q / "predictions_mask_all_q.parquet", index=False)
a_q = roc_auc_score(df_q.y_true, df_q.y_prob); p_q = average_precision_score(df_q.y_true, df_q.y_prob)
pd.DataFrame([{"mask":"baseline","AUROC":a_b,"AUPRC":p_b,"dAUROC":0.0,"dAUPRC":0.0},
              {"mask":"all_Q_tokens","AUROC":a_q,"AUPRC":p_q,"dAUROC":a_q-a_b,"dAUPRC":p_q-p_b}
             ]).to_csv(OUT_Q / "summary.csv", index=False)
print(f"Q mask AUROC={a_q:.4f} (Δ{a_q-a_b:+.4f})")

# OR_ENTRY artifact
rows = [{"mask":"baseline","n_tok":0,"AUROC":a_b,"AUPRC":p_b,"dAUROC":0.0,"dAUPRC":0.0}]
for name, ids in [("sc_full", SC_FULL), ("or_entry_only", {or_entry_id}), ("sc_minus_or_entry", SC_FULL - {or_entry_id})]:
    print(f"OR/SC mask: {name} (n={len(ids)})")
    dfm = evaluate(target_set=ids)
    dfm.to_parquet(OUT_OR / f"predictions_{name}.parquet", index=False)
    a = roc_auc_score(dfm.y_true, dfm.y_prob); p = average_precision_score(dfm.y_true, dfm.y_prob)
    rows.append({"mask":name,"n_tok":len(ids),"AUROC":a,"AUPRC":p,"dAUROC":a-a_b,"dAUPRC":p-p_b})
    print(f"  AUROC={a:.4f} (Δ{a-a_b:+.4f})")
pd.DataFrame(rows).to_csv(OUT_OR / "summary.csv", index=False)
PYEOF
EOF
JOB_QOR=$(sbatch /tmp/stage2_runs/qtok_orentry_s${SEED}.sh | awk '{print $4}')
echo "  [3+4] Q-token + OR_ENTRY combined submitted: ${JOB_QOR}"

# ── 5. Data efficiency PORT (5 jobs) ────────────────────────────────────────
cat > /tmp/stage2_runs/data_eff_port_s${SEED}.sh << EOF
#!/bin/bash
#SBATCH --job-name=de_p_s${SEED}
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --output=$DATA/results_v4/slurm_de_p_s${SEED}_%A_%a.log
#SBATCH --error=$DATA/results_v4/slurm_de_p_s${SEED}_%A_%a.err
#SBATCH --array=0-4

set -e
cd $CODE
export PATH=$CONDA/envs/ethos/bin:\$PATH
export PYTHONPATH=$CODE/ethos-ares/src:\$PYTHONPATH
export CHD_DATA_ROOT=$DATA

FRACS=(0.01 0.05 0.10 0.25 0.50)
F=\${FRACS[\$SLURM_ARRAY_TASK_ID]}
TAG=\$(echo \$F | tr -d '.' | sed 's/^0*//; s/^\$/0/')

python ethos/finetune.py \\
    --model_fp $MODEL_FP \\
    --train_dir $TOK/train --val_dir $TOK/val --test_dir $TOK/test \\
    --results_dir $RESULTS \\
    --seed ${SEED} \\
    --epochs 15 --patience 5 \\
    --lr 1e-3 --hidden_dim 256 --head_dropout 0.1 \\
    --lora --lora_r 8 --lora_alpha 16 \\
    --loss_type bce \\
    --train_frac \$F \\
    --suffix _v4_lora_r8_bce_frac\${TAG}_s${SEED}
EOF
JOB_DEP=$(sbatch /tmp/stage2_runs/data_eff_port_s${SEED}.sh | awk '{print $4}')
echo "  [5] Data efficiency PORT submitted: ${JOB_DEP}"

# ── 6. Data efficiency BiLSTM (5 jobs, independent of seed) ─────────────────
cp $CODE/experiments/vocab/slurm_data_eff_bilstm.sh /tmp/stage2_runs/data_eff_bilstm.sh
sed -i "s|/path/to/CHD_MEDS|$DATA|g; s|/path/to/CHDLLM|$CODE|g; s|/path/to/miniforge3|$CONDA|g" /tmp/stage2_runs/data_eff_bilstm.sh
JOB_DEL=$(sbatch /tmp/stage2_runs/data_eff_bilstm.sh | awk '{print $4}')
echo "  [6] Data efficiency BiLSTM submitted: ${JOB_DEL}"

# ── 7. Case examples (only if best != 123) ─────────
if [ "${SEED}" != "123" ]; then
    cp $CODE/experiments/vocab/slurm_case_examples.sh /tmp/stage2_runs/cases_s${SEED}.sh
    sed -i "s|/path/to/CHD_MEDS|$DATA|g; s|/path/to/CHDLLM|$CODE|g; s|/path/to/miniforge3|$CONDA|g" /tmp/stage2_runs/cases_s${SEED}.sh
    sed -i "s|finetune_lora_head_best_v4_lora_r8_bce_s123.pt|finetune_lora_head_best_v4_lora_r8_bce_s${SEED}.pt|g; s|case_examples_r8bce|case_examples_r8bce_s${SEED}|g" /tmp/stage2_runs/cases_s${SEED}.sh
    JOB_CASES=$(sbatch /tmp/stage2_runs/cases_s${SEED}.sh | awk '{print $4}')
    echo "  [7] Case examples re-run for s${SEED} submitted: ${JOB_CASES}"
else
    echo "  [7] Case examples skipped (s123 already done)"
fi

echo ""
echo "Stage 2 launch complete. Submitted jobs:"
squeue -u \$USER -h -o "%i %j %t %P %R" | grep -E "ctx_s|occ_s|qor_s|de_p_s|de_l|data_lstm|cases_s" | head -20
