"""
Regenerate ALL paper figures with predictions — FIXED baseline loading.
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R_BASE = Path("/path/to/CHD_MEDS/results/baselines")
R_RESULTS = Path("/path/to/CHD_MEDS/results/baselines")
FIG = Path("paper/overleaf/figures")

# ── Load all models ──
models = {}

# Baselines — manual features
df_m = pd.read_parquet(R_BASE / "test_preds_manual.parquet")
models["LR (manual)"] = (df_m.y_true.values, df_m.prob_lr_manual.values)
models["XGB (manual)"] = (df_m.y_true.values, df_m.prob_xgb_manual.values)

# Baselines — MEDS features
df_d = pd.read_parquet(R_BASE / "test_preds_meds.parquet")
models["LR (MEDS)"] = (df_d.y_true.values, df_d.prob_lr_meds.values)
models["XGB (MEDS)"] = (df_d.y_true.values, df_d.prob_xgb_meds.values)

# ASA
df_a = pd.read_parquet(R_BASE / "asa_test_predictions.parquet")
models["ASA score"] = (df_a.y_true.values, df_a.y_prob.values)

# BiLSTM
df_l = pd.read_parquet(R_BASE / "lstm_test_predictions.parquet")
models["BiLSTM"] = (df_l.y_true.values, df_l.y_prob.values)

# PORT (LoRA fine-tuned)
df_lr = pd.read_parquet(R_RESULTS / "ethos_finetune_lora_test_predictions_lora_s123.parquet")
models["PORT"] = (df_lr.y_true.values, df_lr.y_prob.values)

print(f"Loaded {len(models)} models:")
for name, (y,p) in models.items():
    print(f"  {name}: n={len(y):,}, AUROC={roc_auc_score(y,p):.3f}")

colors = {
    "ASA score": "#d62728", "LR (manual)": "#ff7f0e", "XGB (manual)": "#bcbd22",
    "LR (MEDS)": "#2ca02c", "XGB (MEDS)": "#17becf", "BiLSTM": "#9467bd",
    "PORT": "#1f77b4",
}
order = ["ASA score", "LR (manual)", "XGB (manual)", "LR (MEDS)", "XGB (MEDS)",
         "BiLSTM", "PORT"]

# ── ROC ──
fig, ax = plt.subplots(figsize=(8, 6))
for name in order:
    y, p = models[name]
    fpr, tpr, _ = roc_curve(y, p)
    auroc = roc_auc_score(y, p)
    lw = 2.5 if "PORT" in name else (2.0 if "BiLSTM" in name else 1.2)
    ls = "-" if "PORT" in name or "BiLSTM" in name else "--"
    ax.plot(fpr, tpr, label=f"{name} ({auroc:.3f})", color=colors[name], lw=lw, ls=ls)
ax.plot([0,1],[0,1], 'k--', lw=0.5, alpha=0.3)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — All Models", fontsize=14)
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(FIG/"roc_curves.png", dpi=150)
plt.close()
print(" roc_curves.png")

# ── PR ──
fig, ax = plt.subplots(figsize=(8, 6))
for name in order:
    y, p = models[name]
    prec, rec, _ = precision_recall_curve(y, p)
    auprc = average_precision_score(y, p)
    lw = 2.5 if "PORT" in name else (2.0 if "BiLSTM" in name else 1.2)
    ls = "-" if "PORT" in name or "BiLSTM" in name else "--"
    ax.plot(rec, prec, label=f"{name} ({auprc:.3f})", color=colors[name], lw=lw, ls=ls)
prev = models["PORT"][0].mean()
ax.axhline(prev, color='gray', ls=':', lw=0.8, label=f"Random ({prev:.3f})")
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curves", fontsize=14)
ax.legend(loc="upper right", fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.35)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(FIG/"pr_curves.png", dpi=150)
plt.close()
print(" pr_curves.png")

# ── Calibration ──
fig, ax = plt.subplots(figsize=(8, 6))
for name in ["LR (MEDS)", "XGB (MEDS)", "BiLSTM", "PORT"]:
    y, p = models[name]
    prob_true, prob_pred = calibration_curve(y, p, n_bins=10, strategy='uniform')
    lw = 2.5 if "PORT" in name else 1.5
    ax.plot(prob_pred, prob_true, 'o-', label=name, color=colors[name], lw=lw, markersize=4)
ax.plot([0,1],[0,1], 'k--', lw=0.5)
ax.set_xlabel("Mean Predicted Probability", fontsize=12)
ax.set_ylabel("Fraction of Positives", fontsize=12)
ax.set_title("Calibration Curves", fontsize=14)
ax.legend(fontsize=9)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(FIG/"calibration_curves.png", dpi=150)
plt.close()
print(" calibration_curves.png")

# ── ASA subgroup ──
an = pd.read_csv("/path/to/CHOA_RAW_TABLES/CHOA_DATA_Tables_CHD/DR15201_AN_Patients.rpt",
                 sep="|", usecols=["C MRN","ASA PS Score","In OR"], dtype=str)
an['sid'] = pd.to_numeric(an['C MRN'].str.lstrip('C'), errors='coerce')
an['asa'] = pd.to_numeric(an['ASA PS Score'], errors='coerce')
an['in_or'] = pd.to_datetime(an['In OR'], errors='coerce')
an = an.dropna(subset=['sid','asa','in_or'])
an['pt_us'] = an['in_or'].astype('int64') // 1000

fig, ax = plt.subplots(figsize=(8, 5))
lora_df = pd.read_parquet(R_RESULTS/"ethos_finetune_lora_test_predictions_lora_s123.parquet")
lora_df = lora_df.merge(an[['sid','pt_us','asa']].drop_duplicates(),
                         left_on=['subject_id','prediction_time_us'], right_on=['sid','pt_us'], how='left')
lora_df = lora_df.dropna(subset=['asa'])

asa_labels = ['I','II','III','IV','V+']
aurocs = []
for ac in [1, 2, 3, 4, 5]:
    sub = lora_df[lora_df.asa==ac] if ac < 5 else lora_df[lora_df.asa>=5]
    if len(sub) >= 10 and sub.y_true.sum() >= 2:
        aurocs.append(roc_auc_score(sub.y_true, sub.y_prob))
    else:
        aurocs.append(0)

clrs = ['#3498db','#2ecc71','#e67e22','#e74c3c','#8e44ad']
bars = ax.bar(asa_labels, aurocs, color=clrs, alpha=0.8, edgecolor='black', lw=0.5)
for i, v in enumerate(aurocs):
    if v > 0: ax.text(i, v+0.01, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylim(0.5, 1.0)
ax.set_ylabel('AUROC', fontsize=12)
ax.set_xlabel('ASA Physical Status', fontsize=12)
ax.set_title('PORT AUROC by ASA Class', fontsize=14)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(FIG/'subgroup_asa.png', dpi=150)
plt.close()
print(" subgroup_asa.png")

print(f"\nAll 4 figures saved to {FIG}")
