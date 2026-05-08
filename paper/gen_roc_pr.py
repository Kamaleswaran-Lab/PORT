"""
gen_roc_pr_with_significance.py
-------------------------------
Three-panel figure replacing the previous two-panel ROC/PR:
  (a) ROC curves (PORT + baselines)
  (b) PR curves (PORT + baselines)
  (c) Statistical significance: bootstrap distribution of PORT − tuned-BiLSTM
      ΔAUROC overlaid with PORT 3-seed AUROC points.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score, average_precision_score,
)

R_BASE = Path("/path/to/CHD_MEDS/results/baselines")
R_NEW  = Path("/path/to/CHD_MEDS/results/baselines_tuned")
R_RESULTS   = Path("/path/to/CHD_MEDS/results/baselines")
TASK_PATH = Path("/path/to/CHD_MEDS/outcome/iod_task.parquet")
FIG    = Path("paper/overleaf/figures")

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "legend.fontsize": 11,
    "legend.frameon": False,
    "legend.handlelength": 2.0,
    "legend.handletextpad": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ── Load predictions ──────────────────────────────────────────────────────────
def load_pred(p, ycol="y_true", pcol="y_prob"):
    df = pd.read_parquet(p)
    return df[ycol].astype(int).values, df[pcol].values

models = {}
df_m = pd.read_parquet(R_NEW / "test_preds_manual_tuned.parquet")
models["LR (manual)"]  = (df_m.y_true.astype(int).values, df_m.prob_lr_manual.values)
models["XGB (manual)"] = (df_m.y_true.astype(int).values, df_m.prob_xgb_manual.values)
df_d = pd.read_parquet(R_NEW / "test_preds_meds_tuned.parquet")
models["LR (MEDS)"]    = (df_d.y_true.astype(int).values, df_d.prob_lr_meds.values)
models["XGB (MEDS)"]   = (df_d.y_true.astype(int).values, df_d.prob_xgb_meds.values)
models["ASA score"]    = load_pred(R_BASE / "asa_test_predictions.parquet")
models["BiLSTM"]       = load_pred(R_NEW / "lstm_tuned_test_predictions.parquet")
models["PORT"]         = load_pred(R_RESULTS / "ethos_finetune_lora_test_predictions_lora_s123.parquet")

# ── Visual encoding ───────────────────────────────────────────────────────────
STYLES = {
    "PORT":         dict(color="#1F3A5F", lw=2.4, ls="-"),
    "BiLSTM":       dict(color="#E67E22", lw=1.8, ls="-"),
    "XGB (MEDS)":   dict(color="#2A9199", lw=1.3, ls="--"),
    "LR (MEDS)":    dict(color="#2E864D", lw=1.3, ls="--"),
    "XGB (manual)": dict(color="#B5651D", lw=1.1, ls=":"),
    "LR (manual)":  dict(color="#A06800", lw=1.1, ls=":"),
    "ASA score":    dict(color="#C0392B", lw=1.1, ls="-."),
}

# Compute curves and bootstrap envelopes
def bootstrap_curves(y, p, n_boot=2000, n_grid=200, seed=42):
    """Bootstrap ROC and PR curve envelopes interpolated to a common grid."""
    fpr_grid = np.linspace(0, 1, n_grid)
    rec_grid = np.linspace(0, 1, n_grid)
    tprs = np.full((n_boot, n_grid), np.nan)
    precs = np.full((n_boot, n_grid), np.nan)
    rng = np.random.default_rng(seed)
    n = len(y)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yi = y[idx]
        if yi.sum() == 0 or yi.sum() == n:
            continue
        pi = p[idx]
        fpr_i, tpr_i, _ = roc_curve(yi, pi)
        tprs[i] = np.interp(fpr_grid, fpr_i, tpr_i)
        prec_i, rec_i, _ = precision_recall_curve(yi, pi)
        # PR curve recall is monotonically decreasing; reverse for interp
        order = np.argsort(rec_i)
        precs[i] = np.interp(rec_grid, rec_i[order], prec_i[order])
    tpr_lo = np.nanpercentile(tprs, 2.5, axis=0)
    tpr_hi = np.nanpercentile(tprs, 97.5, axis=0)
    prec_lo = np.nanpercentile(precs, 2.5, axis=0)
    prec_hi = np.nanpercentile(precs, 97.5, axis=0)
    return fpr_grid, tpr_lo, tpr_hi, rec_grid, prec_lo, prec_hi

roc_data = {}
pr_data  = {}
roc_envelope = {}
pr_envelope  = {}
print("Computing bootstrap envelopes (n=2000 each curve) …")
for name, (y, p) in models.items():
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    roc_data[name] = (fpr, tpr, roc_auc_score(y, p))
    pr_data[name]  = (rec, prec, average_precision_score(y, p))
    fg, tlo, thi, rg, plo, phi = bootstrap_curves(y, p, n_boot=2000)
    roc_envelope[name] = (fg, tlo, thi)
    pr_envelope[name]  = (rg, plo, phi)
    print(f"  {name:14s} envelopes computed")

roc_order = sorted(models.keys(), key=lambda n: roc_data[n][2], reverse=True)
pr_order  = sorted(models.keys(), key=lambda n: pr_data[n][2],  reverse=True)
prev = models["PORT"][0].mean()

# ── Compute panel (c) data: bootstrap PORT vs BiLSTM + 3-seed AUROC ──────────
print("Computing 3-seed AUROC and bootstrap distribution …")

# 3-seed PORT AUROC values
task = pd.read_parquet(TASK_PATH)
task["subject_id_int"] = task["patient_id"].str.lstrip("C").astype(int)
task["prediction_time_us"] = task["prediction_time"].astype("int64") // 1000

seed_aurocs = {}
seed_auprcs = {}
port_with_csn = None
for seed in [42, 123, 456]:
    df = pd.read_parquet(R_RESULTS / f"ethos_finetune_lora_test_predictions_lora_s{seed}.parquet")
    y = df.y_true.astype(int).values
    p = df.y_prob.values
    seed_aurocs[seed] = roc_auc_score(y, p)
    seed_auprcs[seed] = average_precision_score(y, p)
    if seed == 123:
        # Add encounter_csn for join with LSTM
        merged = df.merge(
            task[["subject_id_int", "prediction_time_us", "encounter_csn"]].rename(
                columns={"subject_id_int": "subject_id"}),
            on=["subject_id", "prediction_time_us"], how="inner",
        )
        port_with_csn = merged[["encounter_csn", "y_true", "y_prob"]].rename(
            columns={"y_prob": "y_port"})

print(f"  3-seed PORT AUROC: {[f'{seed_aurocs[s]:.3f}' for s in [42,123,456]]}")

# Bootstrap PORT (s123) vs BiLSTM
lstm_df = pd.read_parquet(R_NEW / "lstm_tuned_test_predictions.parquet")[["encounter_csn","y_prob"]].rename(columns={"y_prob":"y_lstm"})
joint = port_with_csn.merge(lstm_df, on="encounter_csn", how="inner")
print(f"  Bootstrap intersection: {len(joint):,} encounters, {int(joint.y_true.sum())} IoD+")

y = joint.y_true.astype(int).values
p_port = joint.y_port.values
p_lstm = joint.y_lstm.values

rng = np.random.default_rng(42)
n = len(joint)
deltas = []
for _ in range(5000):
    idx = rng.integers(0, n, n)
    yi = y[idx]
    if yi.sum() == 0 or yi.sum() == n:
        continue
    deltas.append(roc_auc_score(yi, p_port[idx]) - roc_auc_score(yi, p_lstm[idx]))
deltas = np.array(deltas)
ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
p_two_sided = (deltas <= 0).mean() * 2
print(f"  Bootstrap Δ: {deltas.mean():+.4f}  CI95=[{ci_lo:+.4f}, {ci_hi:+.4f}]  p={p_two_sided:.4g}")

# ── Figure: 2 panels (ROC + PR with bootstrap CI bands) ──────────────────────
fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.6))

# (a) ROC with bootstrap envelopes (95% CI shaded)
for name in roc_order:
    fpr, tpr, auc = roc_data[name]
    fg, tlo, thi = roc_envelope[name]
    s = STYLES[name]
    # Shade only headline models to avoid visual clutter; use a higher alpha
    # plus a thin boundary edge so the band is clearly visible at print size.
    if name in ("PORT", "BiLSTM"):
        ax_roc.fill_between(fg, tlo, thi, color=s["color"], alpha=0.18, lw=0, zorder=1)
        ax_roc.plot(fg, tlo, color=s["color"], lw=0.6, alpha=0.6, zorder=2)
        ax_roc.plot(fg, thi, color=s["color"], lw=0.6, alpha=0.6, zorder=2)
    ax_roc.plot(fpr, tpr, label=f"{name}  ({auc:.3f})", **s)
ax_roc.plot([0, 1], [0, 1], color="0.8", lw=0.7, ls="--", zorder=0)
ax_roc.set_xlim(-0.005, 1.0); ax_roc.set_ylim(0.0, 1.005)
ax_roc.set_xticks(np.linspace(0, 1, 6)); ax_roc.set_yticks(np.linspace(0, 1, 6))
ax_roc.set_xlabel("False positive rate"); ax_roc.set_ylabel("True positive rate")
ax_roc.set_title("(a) Discrimination (AUROC)", fontsize=14, pad=8)
ax_roc.legend(loc="lower right")

# (b) PR with bootstrap envelopes
for name in pr_order:
    rec, prec, auprc = pr_data[name]
    rg, plo, phi = pr_envelope[name]
    s = STYLES[name]
    if name in ("PORT", "BiLSTM"):
        ax_pr.fill_between(rg, plo, phi, color=s["color"], alpha=0.18, lw=0, zorder=1)
        ax_pr.plot(rg, plo, color=s["color"], lw=0.6, alpha=0.6, zorder=2)
        ax_pr.plot(rg, phi, color=s["color"], lw=0.6, alpha=0.6, zorder=2)
    ax_pr.plot(rec, prec, label=f"{name}  ({auprc:.3f})", **s)
ax_pr.axhline(prev, color="0.6", lw=0.8, ls=":", zorder=0)
ax_pr.set_xlim(-0.005, 1.0); ax_pr.set_ylim(0.0, 0.36)
ax_pr.set_xticks(np.linspace(0, 1, 6)); ax_pr.set_yticks(np.arange(0, 0.36, 0.05))
ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
ax_pr.set_title("(b) Positive-class precision (AUPRC)", fontsize=14, pad=8)
ax_pr.legend(loc="upper right")

# Common minimal style for both panels
for ax in (ax_roc, ax_pr):
    ax.tick_params(axis="both", which="major", length=3.5, pad=2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#333"); ax.spines[spine].set_linewidth(0.8)
    ax.grid(True, alpha=0.3, ls="--", zorder=0)
    ax.set_axisbelow(True)

fig.tight_layout(w_pad=2.5)
out_png = FIG / "roc_pr_combined.png"
out_pdf = FIG / "roc_pr_combined.pdf"
fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf,            bbox_inches="tight")
print(f"\nWROTE {out_png}")
print(f"WROTE {out_pdf}")

# Also save the bootstrap stats JSON for reproducibility
stats = {
    "seed_aurocs":  {str(s): float(seed_aurocs[s]) for s in [42, 123, 456]},
    "seed_auprcs":  {str(s): float(seed_auprcs[s]) for s in [42, 123, 456]},
    "bootstrap_n_iter": 5000,
    "bootstrap_intersection_n": int(len(joint)),
    "bootstrap_intersection_pos": int(joint.y_true.sum()),
    "delta_mean": float(deltas.mean()),
    "delta_ci95_lo": float(ci_lo),
    "delta_ci95_hi": float(ci_hi),
    "p_two_sided": float(p_two_sided),
}
(FIG / "../../paper/roc_pr_significance_stats.json").parent.mkdir(parents=True, exist_ok=True)
with open(R_NEW / "port_vs_bilstm_significance.json", "w") as f:
    json.dump(stats, f, indent=2)
print(f"WROTE {R_NEW / 'port_vs_bilstm_significance.json'}")
