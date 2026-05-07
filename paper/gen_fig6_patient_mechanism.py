"""Figure 6: patient-level mechanism analysis.

Four panels:
  (a) Per-patient ΔP(IoD) when Surgical Context tokens are masked.
  (b) Per-patient ΔP(IoD) when Medication tokens are masked.
  (c) Counterfactual: patient currently on pre-operative vasoactive support
      versus the same patient with vasoactive tokens removed.
  (d) Counterfactual: ASA score quantile swapped to a low-acuity baseline (Q2).

(a) and (b) use predictions written by slurm_occlusion_perpatient.sh.
(c) and (d) use predictions written by slurm_counterfactual.sh.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

OCCL_DIR = Path("/path/to/CHD_MEDS/results/evaluation/per_patient_occlusion")
CF_DIR   = Path("/path/to/CHD_MEDS/results/evaluation/counterfactual")
FIG_OUT  = Path("paper/overleaf/figures/patient_mechanism.png")

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

PORT_BLUE = "#1F3A5F"
POS_RED   = "#C0392B"
NEG_GRAY  = "#888888"

# ── Load baseline + per-patient masked predictions for (a) and (b) ──
df_base = pd.read_parquet(OCCL_DIR / "predictions_baseline.parquet")
df_sc   = pd.read_parquet(OCCL_DIR / "predictions_Surgical_Context_ENCOUNTER.parquet")
df_med  = pd.read_parquet(OCCL_DIR / "predictions_Medications_ATC.parquet")

key = ["subject_id", "prediction_time_us"]
mb_sc  = df_base.merge(df_sc,  on=key, suffixes=("", "_sc"))
mb_med = df_base.merge(df_med, on=key, suffixes=("", "_med"))

# Per-patient delta = baseline minus masked (positive = mask reduced risk)
mb_sc["delta"]  = mb_sc["y_prob"]  - mb_sc["y_prob_sc"]
mb_med["delta"] = mb_med["y_prob"] - mb_med["y_prob_med"]

# ── Load counterfactual outputs for (c) and (d) ──
df_cf = pd.read_parquet(CF_DIR / "counterfactual_predictions.parquet")
df_cf["delta_vaso"] = df_cf["p_base"] - df_cf["p_no_vasoactives"]
df_cf["delta_asa"]  = df_cf["p_base"] - df_cf["p_asa_q2"]

cf_vaso_sub = df_cf[df_cf["on_vasoactive"]].copy()
cf_asa_sub  = df_cf[df_cf["has_asa"]].copy()

# ── Figure ──
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

def scatter_panel(ax, df_m, label_col, title):
    pos = df_m[df_m[label_col] == 1]
    neg = df_m[df_m[label_col] == 0]
    ax.scatter(neg["y_prob"], neg["delta"], s=4, c=NEG_GRAY, alpha=0.35,
               rasterized=True, label=f"IoD− (n={len(neg):,})")
    ax.scatter(pos["y_prob"], pos["delta"], s=14, c=POS_RED, alpha=0.85,
               edgecolors="white", linewidths=0.4,
               label=f"IoD+ (n={len(pos):,})")
    ax.axhline(0, color="0.4", lw=0.7, ls=":", zorder=0)
    ax.set_xscale("log"); ax.set_xlim(1e-4, 1)
    ax.set_xlabel(r"Baseline predicted IoD risk $\hat p_\text{base}$")
    ax.set_ylabel(r"$\hat p_\text{base} - \hat p_\text{masked}$")
    ax.set_title(title, fontsize=12, pad=8)
    ax.grid(True, alpha=0.3, ls="--", zorder=0); ax.set_axisbelow(True)
    ax.legend(loc="upper left")

def violin_panel(ax, sub, delta_col, label_col, title):
    pos = sub[sub[label_col] == 1][delta_col].values
    neg = sub[sub[label_col] == 0][delta_col].values
    parts = ax.violinplot([neg, pos], positions=[0, 1], widths=0.7,
                          showmeans=False, showmedians=True, showextrema=False)
    colors = [NEG_GRAY, POS_RED]
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c); pc.set_alpha(0.55); pc.set_edgecolor(c)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(1.2)
    ax.axhline(0, color="0.4", lw=0.7, ls=":", zorder=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"IoD−\n(n={len(neg):,})",
                        f"IoD+\n(n={len(pos):,})"])
    ax.set_ylabel(r"$\hat p_\text{base} - \hat p_\text{counterfactual}$")
    ax.set_title(title, fontsize=12, pad=8)
    ax.grid(True, axis="y", alpha=0.3, ls="--", zorder=0); ax.set_axisbelow(True)

scatter_panel(axes[0, 0], mb_sc,  "y_true", "(a) Per-patient: Surgical Context masked")
scatter_panel(axes[0, 1], mb_med, "y_true", "(b) Per-patient: Medications masked")
violin_panel (axes[1, 0], cf_vaso_sub, "delta_vaso", "y_true",
              "(c) Counterfactual: vasoactives removed (only encounters with vasoactive use)")
violin_panel (axes[1, 1], cf_asa_sub,  "delta_asa",  "y_true",
              "(d) Counterfactual: ASA quantile swapped to Q2")

fig.tight_layout(w_pad=2.5, h_pad=2.5)
fig.savefig(FIG_OUT, dpi=300, bbox_inches="tight")
fig.savefig(str(FIG_OUT).replace(".png", ".pdf"), bbox_inches="tight")
plt.close(fig)
print(f"Saved {FIG_OUT}")
