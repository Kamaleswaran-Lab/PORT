"""Figure 5: occlusion analysis with bootstrap 95% CI bars.

Replaces the prior point-estimate occlusion plot. Uses per-patient predictions
written by experiments/vocab/slurm_occlusion_perpatient.sh:
  /path/to/CHD_MEDS/results/evaluation/per_patient_occlusion/
    predictions_baseline.parquet
    predictions_<Category>.parquet  (one file per masked category)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import roc_auc_score, average_precision_score

PRED_DIR = Path("/path/to/CHD_MEDS/results_v4/evaluation/per_patient_occlusion_r8bce_s456")
FIG_OUT  = Path("paper/overleaf/figures/occlusion_analysis.png")

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def bootstrap_delta_ci(y, p_base, p_masked, n_boot=2000, seed=42):
    """Bootstrap 95% CI on Δ(masked − baseline) AUROC and AUPRC."""
    rng = np.random.default_rng(seed)
    n = len(y)
    d_aurocs, d_auprcs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yi = y[idx]
        if yi.sum() == 0 or yi.sum() == n: continue
        a_b = roc_auc_score(yi, p_base[idx])
        a_m = roc_auc_score(yi, p_masked[idx])
        p_b = average_precision_score(yi, p_base[idx])
        p_m = average_precision_score(yi, p_masked[idx])
        d_aurocs.append(a_m - a_b)
        d_auprcs.append(p_m - p_b)
    return (np.percentile(d_aurocs, [2.5, 97.5]),
            np.percentile(d_auprcs, [2.5, 97.5]))

# ── Load baseline ──
df_base = pd.read_parquet(PRED_DIR / "predictions_baseline.parquet")
key_cols = ["subject_id", "prediction_time_us"]
y = df_base["y_true"].astype(int).values
p_base = df_base["y_prob"].values

# ── Per-category masked predictions ──
def safe(name):
    return name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")

CATEGORIES = [
    "Surgical Context (ENCOUNTER)", "Care Trajectory (ADT)",
    "Surgical History (PROCEDURE)", "Medications (ATC)",
    "ICD-10 Codes", "Laboratory", "Anesthesia Events",
    "Language", "Point of Origin", "Problem List (free-text)",
    "Demographics", "Home County", "Medical History (free-text)",
    "Insurance", "Structured Data (SDE)", "Transfusions",
    "Lines/Drains (LDA)", "Vital Signs",
]

# OR_ENTRY positional cursor confounds the Surgical Context mask: masking the
# full ENCOUNTER//AN// prefix replaces the prediction-cursor token with PAD,
# producing an artifactual hidden-state collapse at position -1 unrelated to
# surgical-context content. We therefore substitute the SC mask with the
# OR_ENTRY-excluded variant from experiments/vocab/slurm_orentry_artifact.sh.
# SC row in this PRED_DIR already excludes OR_ENTRY (handled in slurm_occlusion_r8bce.sh)
ORENTRY_DIR = Path(str(PRED_DIR).replace("per_patient_occlusion_r8bce_s456", "orentry_artifact_r8bce_s456"))
OVERRIDES = {}  # No override needed; SC pred already excludes OR_ENTRY

rows = []
for cat in CATEGORIES:
    fp = OVERRIDES.get(cat, PRED_DIR / f"predictions_{safe(cat)}.parquet")
    if not fp.exists():
        continue
    df_m = pd.read_parquet(fp)
    # Align by (subject_id, prediction_time_us) to baseline
    merged = df_base.merge(df_m, on=key_cols, suffixes=("_base", "_mask"))
    yb = merged["y_true_base"].astype(int).values
    pb = merged["y_prob_base"].values
    pm = merged["y_prob_mask"].values
    a_b = roc_auc_score(yb, pb); a_m = roc_auc_score(yb, pm)
    p_b = average_precision_score(yb, pb); p_m = average_precision_score(yb, pm)
    (d_lo_a, d_hi_a), (d_lo_p, d_hi_p) = bootstrap_delta_ci(yb, pb, pm)
    rows.append({"Category": cat, "n": len(yb),
                 "Delta_AUROC_pct": (a_m - a_b) * 100,
                 "lo_AUROC_pct": d_lo_a * 100, "hi_AUROC_pct": d_hi_a * 100,
                 "Delta_AUPRC_pct": (p_m - p_b) * 100,
                 "lo_AUPRC_pct": d_lo_p * 100, "hi_AUPRC_pct": d_hi_p * 100})

df = pd.DataFrame(rows).sort_values("Delta_AUROC_pct")

# ── Color by impact magnitude ──
def color_for(delta_pct):
    if delta_pct < -1.0: return "#e74c3c"
    if delta_pct < -0.1: return "#e67e22"
    if delta_pct < 0:    return "#3498db"
    return "#95a5a6"

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 11))
plt.subplots_adjust(left=0.28, right=0.97, hspace=0.32)

y_pos = np.arange(len(df))

# Panel a: ΔAUROC
colors1 = [color_for(v) for v in df["Delta_AUROC_pct"]]
err_lo1 = df["Delta_AUROC_pct"] - df["lo_AUROC_pct"]
err_hi1 = df["hi_AUROC_pct"] - df["Delta_AUROC_pct"]
ax1.barh(y_pos, df["Delta_AUROC_pct"], color=colors1, edgecolor="black",
         lw=0.3, height=0.7,
         xerr=[err_lo1, err_hi1],
         error_kw=dict(ecolor="#222", lw=0.9, capsize=3))
ax1.set_yticks(y_pos); ax1.set_yticklabels(df["Category"], fontsize=11)
ax1.set_xlabel("ΔAUROC (%)")
ax1.set_title("(a) AUROC Impact", fontsize=12, pad=8)
ax1.axvline(0, color="black", lw=0.5)
ax1.grid(axis="x", alpha=0.3, ls="--"); ax1.set_axisbelow(True); ax1.invert_yaxis()

# Panel b: ΔAUPRC
colors2 = [color_for(v) for v in df["Delta_AUPRC_pct"]]
err_lo2 = df["Delta_AUPRC_pct"] - df["lo_AUPRC_pct"]
err_hi2 = df["hi_AUPRC_pct"] - df["Delta_AUPRC_pct"]
ax2.barh(y_pos, df["Delta_AUPRC_pct"], color=colors2, edgecolor="black",
         lw=0.3, height=0.7,
         xerr=[err_lo2, err_hi2],
         error_kw=dict(ecolor="#222", lw=0.9, capsize=3))
ax2.set_yticks(y_pos); ax2.set_yticklabels(df["Category"], fontsize=11)
ax2.set_xlabel("ΔAUPRC (%)")
ax2.set_title("(b) AUPRC Impact", fontsize=12, pad=8)
ax2.axvline(0, color="black", lw=0.5)
ax2.grid(axis="x", alpha=0.3, ls="--"); ax2.set_axisbelow(True); ax2.invert_yaxis()

plt.savefig(FIG_OUT, dpi=300, bbox_inches="tight")
plt.savefig(str(FIG_OUT).replace(".png", ".pdf"), bbox_inches="tight")
plt.close()
print(f"Saved {FIG_OUT}")
