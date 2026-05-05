"""
full evaluation — all metrics for paper update.
Runs: ECE, confusion matrix, bootstrap, ASA subgroup, 3-seed summary.
Uses seed 123 as primary (best AUROC/AUPRC).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             confusion_matrix, roc_curve)

R = Path("/path/to/CHD_MEDS/results/baselines")
OUT = Path("/path/to/CHD_MEDS/results/evaluation")
OUT.mkdir(parents=True, exist_ok=True)

def load_pred(path):
    df = pd.read_parquet(path)
    return df['y_true'].values, df['y_prob'].values

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if m.sum() == 0: continue
        ece += m.sum() / len(y_true) * abs(y_prob[m].mean() - y_true[m].mean())
    return ece

# ── Load LoRA seed 123 ──
y, p = load_pred(R / "ethos_finetune_lora_test_predictions_lora_s123.parquet")
print(f"n={len(y):,}  pos={int(y.sum())}  prev={100*y.mean():.1f}%")

# ── 1. Main metrics ──
auroc = roc_auc_score(y, p)
auprc = average_precision_score(y, p)
brier = brier_score_loss(y, p)
ece = compute_ece(y, p)
print(f"\nv4 LoRA s123: AUROC={auroc:.4f}  AUPRC={auprc:.4f}  Brier={brier:.4f}  ECE={ece:.4f}")

# Probe
py, pp = load_pred(R / "ethos_finetune_probe_test_predictions_probe_s123.parquet")
print(f"Probe s123: AUROC={roc_auc_score(py,pp):.4f}  AUPRC={average_precision_score(py,pp):.4f}  "
      f"Brier={brier_score_loss(py,pp):.4f}  ECE={compute_ece(py,pp):.4f}")

# ── 2. 3-seed ──
print("\n=== 3-Seed Variance ===")
sa, sp, sb = [], [], []
for s in [42, 123, 456]:
    sy, ss = load_pred(R / f"ethos_finetune_lora_test_predictions_lora_s{s}.parquet")
    a, ap, b = roc_auc_score(sy,ss), average_precision_score(sy,ss), brier_score_loss(sy,ss)
    sa.append(a); sp.append(ap); sb.append(b)
    print(f"  s{s}: AUROC={a:.4f}  AUPRC={ap:.4f}  Brier={b:.4f}")
print(f"  Mean±SD: AUROC={np.mean(sa):.3f}±{np.std(sa):.3f}  AUPRC={np.mean(sp):.3f}±{np.std(sp):.3f}")

# ── 3. Confusion matrix ──
print("\n=== Confusion Matrix ===")
fpr, tpr, th = roc_curve(y, p)
j = tpr - fpr
bi = np.argmax(j)
yt = th[bi]
yp_y = (p >= yt).astype(int)
tn, fp, fn, tp = confusion_matrix(y, yp_y).ravel()
sens = tp/(tp+fn); spec = tn/(tn+fp); ppv = tp/(tp+fp) if (tp+fp)>0 else 0
print(f"Youden threshold={yt:.4f}  Sens={sens:.3f}  Spec={spec:.3f}  PPV={ppv:.3f}  TP={tp} FP={fp} FN={fn} TN={tn}")

# Fixed sens 85%
i85 = np.argmin(np.abs(tpr - 0.85))
t85 = th[i85]
yp85 = (p >= t85).astype(int)
tn2, fp2, fn2, tp2 = confusion_matrix(y, yp85).ravel()
s2 = tp2/(tp2+fn2); sp2 = tn2/(tn2+fp2); ppv2 = tp2/(tp2+fp2) if (tp2+fp2)>0 else 0
print(f"@Sens85% threshold={t85:.4f}  Sens={s2:.3f}  Spec={sp2:.3f}  PPV={ppv2:.3f}  TP={tp2} FP={fp2}")

# ── 4. Bootstrap vs LSTM ──
print("\n=== Bootstrap PORT vs LSTM ===")
lstm_path = Path("/path/to/CHD_MEDS/results/baselines/lstm_test_predictions.parquet")
if lstm_path.exists():
    ld = pd.read_parquet(lstm_path)
    lstm_auroc = roc_auc_score(ld.y_true, ld.y_prob)
    np.random.seed(42)
    deltas = []
    for _ in range(5000):
        idx = np.random.choice(len(y), len(y), replace=True)
        deltas.append(roc_auc_score(y[idx], p[idx]) - lstm_auroc)
    ci = np.percentile(deltas, [2.5, 97.5])
    pv = np.mean(np.array(deltas) <= 0)
    print(f"PORT={auroc:.4f} LSTM={lstm_auroc:.4f} Δ={np.mean(deltas):+.4f} CI=[{ci[0]:+.4f},{ci[1]:+.4f}] p={pv:.4f}")

# ── 5. ASA Subgroup ──
print("\n=== ASA Subgroup ===")
task = pd.read_parquet("/path/to/CHD_MEDS/outcome/iod_task.parquet")
task['sid'] = task['patient_id'].str.lstrip('C').astype(int)
an = pd.read_csv("/path/to/CHOA_RAW_TABLES/CHOA_DATA_Tables_CHD/DR15201_AN_Patients.rpt",
                 sep="|", usecols=["C MRN", "ASA PS Score", "In OR"], dtype=str)
an['sid'] = pd.to_numeric(an['C MRN'].str.lstrip('C'), errors='coerce')
an['asa'] = pd.to_numeric(an['ASA PS Score'], errors='coerce')
an['in_or'] = pd.to_datetime(an['In OR'], errors='coerce')
an = an.dropna(subset=['sid', 'asa', 'in_or'])
an['pt_us'] = an['in_or'].astype('int64') // 1000

lora_df = pd.read_parquet(R / "ethos_finetune_lora_test_predictions_lora_s123.parquet")
if 'subject_id' in lora_df.columns and 'prediction_time_us' in lora_df.columns:
    lora_df = lora_df.merge(an[['sid','pt_us','asa']].drop_duplicates(),
                             left_on=['subject_id','prediction_time_us'], right_on=['sid','pt_us'], how='left')
    for label, mf in [("All", lambda d:d), ("ASA>=3", lambda d:d[d.asa>=3]), ("ASA>=4", lambda d:d[d.asa>=4]),
                       ("ASA I", lambda d:d[d.asa==1]), ("ASA II", lambda d:d[d.asa==2]),
                       ("ASA III", lambda d:d[d.asa==3]), ("ASA IV", lambda d:d[d.asa>=4])]:
        sub = mf(lora_df.dropna(subset=['asa']))
        if len(sub)<10 or sub.y_true.sum()<2: continue
        a = roc_auc_score(sub.y_true, sub.y_prob)
        ap = average_precision_score(sub.y_true, sub.y_prob)
        n_s, n_p = len(sub), int(sub.y_true.sum())
        # PPV at Youden
        fr,tr,thr = roc_curve(sub.y_true, sub.y_prob)
        bi2 = np.argmax(tr-fr)
        yp_s = (sub.y_prob >= thr[bi2]).astype(int)
        tp_s = ((yp_s==1)&(sub.y_true==1)).sum()
        fp_s = ((yp_s==1)&(sub.y_true==0)).sum()
        ppv_s = tp_s/(tp_s+fp_s) if (tp_s+fp_s)>0 else 0
        print(f"  {label:<10s} n={n_s:>6,} IoD+={n_p:>4d} prev={100*n_p/n_s:>5.1f}% AUROC={a:.3f} AUPRC={ap:.3f} PPV={100*ppv_s:.1f}%")
else:
    print("  Cannot match ASA — missing subject_id/prediction_time_us columns")

# ── Save ──
pd.DataFrame([{
    "auroc": auroc, "auprc": auprc, "brier": brier, "ece": ece,
    "probe_auroc": roc_auc_score(py,pp), "probe_auprc": average_precision_score(py,pp),
    "3seed_auroc_mean": np.mean(sa), "3seed_auroc_sd": np.std(sa),
    "3seed_auprc_mean": np.mean(sp), "3seed_auprc_sd": np.std(sp),
    "youden_thresh": yt, "youden_sens": sens, "youden_spec": spec, "youden_ppv": ppv,
    "sens85_thresh": t85, "sens85_sens": s2, "sens85_spec": sp2, "sens85_ppv": ppv2,
}]).to_csv(OUT / "summary.csv", index=False)
print(f"\n Saved {OUT/'summary.csv'}")
