"""Figure 6: drill-down within the two dominant occlusion categories from
Figure 5 plus two counterfactual swap scenarios.

Four panels:
  (a) Surgical Context sub-decomposition — masking each component
      (Primary Procedure, ASA score, Admission type, Patient class, OR markers)
      individually; bootstrap 95% CIs on ΔAUROC and ΔAUPRC.
  (b) Medications sub-decomposition — masking the vasoactive subset (ATC C01
      + leaf SFX tokens, n=9) and the non-vasoactive ATC subset (n=268)
      separately.
  (c) Counterfactual procedure swap: replacing the recorded primary-procedure
      token with a low-complexity reference (myringotomy with tubes) for every
      encounter that has a primary-procedure token; per-encounter risk
      reduction ($\hat p_\text{base} - \hat p_\text{swap}$) plotted against
      baseline risk, with the median per baseline-risk decile overlaid.
  (d) Counterfactual ASA-tier swap: replacing the quantile token following
      ENCOUNTER//AN//ASA_SCORE with Q1 (low acuity) or Q9 (high acuity);
      paired ΔP distributions show the model's bidirectional response to ASA
      quantile changes.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import roc_auc_score, average_precision_score

DRILL = Path("/path/to/CHD_MEDS/results/evaluation/fig6_drilldown/drilldown_predictions.parquet")
CF    = Path("/path/to/CHD_MEDS/results/evaluation/counterfactual/counterfactual_predictions.parquet")
FIG   = Path("paper/overleaf/figures/patient_mechanism")  # extension added below

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PORT_BLUE = "#1F3A5F"
RED       = "#C0392B"
GRAY      = "#888888"

# ── Load data ──
d = pd.read_parquet(DRILL)
cf = pd.read_parquet(CF)

# Vasoactive predictions come from the earlier counterfactual job and are
# joined on (subject_id, prediction_time_us).
key = ["subject_id", "prediction_time_us"]
d = d.merge(cf[key + ["p_no_vasoactives", "on_vasoactive"]], on=key, how="left")

y = d["y_true"].astype(int).values
p_base = d["baseline"].values
auroc_base = roc_auc_score(y, p_base)
auprc_base = average_precision_score(y, p_base)

# ── Bootstrap helper ──
def bootstrap_delta_ci(y, p_base, p_masked, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    d_a, d_p = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yi = y[idx]
        if yi.sum() == 0 or yi.sum() == n: continue
        a_b = roc_auc_score(yi, p_base[idx]); a_m = roc_auc_score(yi, p_masked[idx])
        p_b = average_precision_score(yi, p_base[idx]); p_m = average_precision_score(yi, p_masked[idx])
        d_a.append(a_m - a_b); d_p.append(p_m - p_b)
    return np.percentile(d_a, [2.5, 97.5]), np.percentile(d_p, [2.5, 97.5])

# ── Figure ──
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
ax_a, ax_b = axes[0]
ax_c, ax_d = axes[1]

# ══════════════════════════════════════════════════════════════════════════════
# (a) Surgical Context sub-decomposition
# ══════════════════════════════════════════════════════════════════════════════
SC_LABELS = {
    "SC_primary_procedure": "Primary procedure",
    "SC_asa_score":         "ASA score",
    "SC_admission_type":    "Admission type",
    "SC_patient_class":     "Patient class",
    "SC_or_markers":        "OR / hospital markers",
}
sc_rows = []
for col, lbl in SC_LABELS.items():
    p_m = d[col].values
    a_m = roc_auc_score(y, p_m); ap_m = average_precision_score(y, p_m)
    (lo_a, hi_a), (lo_p, hi_p) = bootstrap_delta_ci(y, p_base, p_m)
    sc_rows.append({"label": lbl,
                    "d_auroc": (a_m - auroc_base) * 100,
                    "lo_a": lo_a * 100, "hi_a": hi_a * 100,
                    "d_auprc": (ap_m - auprc_base) * 100,
                    "lo_p": lo_p * 100, "hi_p": hi_p * 100})
sc_df = pd.DataFrame(sc_rows).sort_values("d_auroc")

y_pos = np.arange(len(sc_df))
err_lo = sc_df["d_auroc"] - sc_df["lo_a"]
err_hi = sc_df["hi_a"] - sc_df["d_auroc"]
ax_a.barh(y_pos, sc_df["d_auroc"], color="#9c2a2f", edgecolor="black", lw=0.4,
          height=0.6, xerr=[err_lo, err_hi],
          error_kw=dict(ecolor="#222", lw=0.9, capsize=3))
ax_a.set_yticks(y_pos); ax_a.set_yticklabels(sc_df["label"])
ax_a.set_xlabel("ΔAUROC (%)")
ax_a.set_title("(a) Surgical Context sub-decomposition", fontsize=12, pad=8)
ax_a.axvline(0, color="black", lw=0.5)
ax_a.grid(axis="x", alpha=0.3, ls="--"); ax_a.set_axisbelow(True); ax_a.invert_yaxis()

# ══════════════════════════════════════════════════════════════════════════════
# (b) Medications sub-decomposition
# ══════════════════════════════════════════════════════════════════════════════
# Vasoactive: from counterfactual_predictions.parquet (p_no_vasoactives)
# Non-vasoactive: from drilldown
# Use the same baseline (drilldown's baseline column) for both.
med_rows = []
p_vaso  = d["p_no_vasoactives"].values
a_vaso = roc_auc_score(y, p_vaso); ap_vaso = average_precision_score(y, p_vaso)
(lo_a, hi_a), (lo_p, hi_p) = bootstrap_delta_ci(y, p_base, p_vaso)
med_rows.append({"label": "Vasoactive ATC (C01 + SFX)",
                 "d_auroc": (a_vaso - auroc_base) * 100,
                 "lo_a": lo_a * 100, "hi_a": hi_a * 100,
                 "d_auprc": (ap_vaso - auprc_base) * 100,
                 "lo_p": lo_p * 100, "hi_p": hi_p * 100})

p_nv = d["MED_non_vasoactive_atc"].values
a_nv = roc_auc_score(y, p_nv); ap_nv = average_precision_score(y, p_nv)
(lo_a, hi_a), (lo_p, hi_p) = bootstrap_delta_ci(y, p_base, p_nv)
med_rows.append({"label": "Non-vasoactive ATC",
                 "d_auroc": (a_nv - auroc_base) * 100,
                 "lo_a": lo_a * 100, "hi_a": hi_a * 100,
                 "d_auprc": (ap_nv - auprc_base) * 100,
                 "lo_p": lo_p * 100, "hi_p": hi_p * 100})
med_df = pd.DataFrame(med_rows).sort_values("d_auprc")

y_pos2 = np.arange(len(med_df))
# Show ΔAUPRC for medications since that's where Fig 5 highlights the dominance
err_lo2 = med_df["d_auprc"] - med_df["lo_p"]
err_hi2 = med_df["hi_p"] - med_df["d_auprc"]
ax_b.barh(y_pos2, med_df["d_auprc"], color="#e67e22", edgecolor="black", lw=0.4,
          height=0.55, xerr=[err_lo2, err_hi2],
          error_kw=dict(ecolor="#222", lw=0.9, capsize=3))
ax_b.set_yticks(y_pos2); ax_b.set_yticklabels(med_df["label"])
ax_b.set_xlabel("ΔAUPRC (%)")
ax_b.set_title("(b) Medications sub-decomposition", fontsize=12, pad=8)
ax_b.axvline(0, color="black", lw=0.5)
ax_b.grid(axis="x", alpha=0.3, ls="--"); ax_b.set_axisbelow(True); ax_b.invert_yaxis()

# ══════════════════════════════════════════════════════════════════════════════
# (c) Counterfactual: PROCEDURE swap to myringotomy
# ══════════════════════════════════════════════════════════════════════════════
proc = d[d["has_proc"]].copy()
proc["delta"] = proc["baseline"] - proc["proc_swap_to_myringotomy"]
neg = proc[proc.y_true == 0]; pos = proc[proc.y_true == 1]
ax_c.scatter(neg["baseline"], neg["delta"], s=4, c=GRAY, alpha=0.30,
             rasterized=True, label=f"IoD− (n={len(neg):,})")
ax_c.scatter(pos["baseline"], pos["delta"], s=14, c=RED, alpha=0.85,
             edgecolors="white", linewidths=0.4, label=f"IoD+ (n={len(pos):,})")
ax_c.axhline(0, color="0.4", lw=0.7, ls=":", zorder=0)
proc["dec"] = pd.qcut(proc["baseline"], q=10, labels=False, duplicates="drop")
med_dec = proc.groupby("dec").agg(p_med=("baseline","median"), d_med=("delta","median"))
ax_c.plot(med_dec["p_med"], med_dec["d_med"], "-D", color=PORT_BLUE, lw=2.0,
          ms=6, mec="white", mew=0.8, label="Median ΔP per decile")
ax_c.set_xscale("log"); ax_c.set_xlim(1e-4, 1)
ax_c.set_xlabel(r"Baseline PORT risk $\hat p_\text{base}$ (log scale)")
ax_c.set_ylabel(r"$\hat p_\text{base} - \hat p_\text{swap-to-myringotomy}$")
ax_c.set_title("(c) Counterfactual: primary-procedure swap to myringotomy",
               fontsize=12, pad=8)
ax_c.grid(True, alpha=0.3, ls="--", zorder=0); ax_c.set_axisbelow(True)
ax_c.legend(loc="upper left", fontsize=9)

# ══════════════════════════════════════════════════════════════════════════════
# (d) Counterfactual: ASA Q swap, bidirectional
# ══════════════════════════════════════════════════════════════════════════════
asa = d[d["has_asa"]].copy()
asa["d_q1"] = asa["baseline"] - asa["asa_swap_to_q1"]   # >0 = lowering ASA reduces risk
asa["d_q9"] = asa["asa_swap_to_q9"] - asa["baseline"]   # >0 = raising ASA increases risk

# Side-by-side histograms with median markers
def hist_overlay(ax, vals_neg, vals_pos, color_neg=GRAY, color_pos=RED, label_neg="IoD-", label_pos="IoD+"):
    bins = np.linspace(min(vals_neg.min(), vals_pos.min(), -0.05),
                       max(vals_neg.max(), vals_pos.max(), 0.05), 60)
    ax.hist(vals_neg, bins=bins, alpha=0.45, color=color_neg, edgecolor="white",
            lw=0.4, label=label_neg, density=True)
    ax.hist(vals_pos, bins=bins, alpha=0.65, color=color_pos, edgecolor="white",
            lw=0.4, label=label_pos, density=True)
    ax.axvline(0, color="0.4", lw=0.7, ls=":", zorder=0)
    ax.axvline(np.median(vals_neg), color=color_neg, lw=1.5, ls="--", zorder=4)
    ax.axvline(np.median(vals_pos), color=color_pos, lw=1.5, ls="--", zorder=4)

# Compose two-row sub panel using a 1x2 grid INSIDE ax_d
ax_d.axis("off")
gs_in = ax_d.get_subplotspec().subgridspec(1, 2, wspace=0.32)
ax_d1 = fig.add_subplot(gs_in[0, 0])
ax_d2 = fig.add_subplot(gs_in[0, 1])

neg_q1 = asa[asa.y_true == 0]["d_q1"].values
pos_q1 = asa[asa.y_true == 1]["d_q1"].values
hist_overlay(ax_d1, neg_q1, pos_q1)
ax_d1.set_xlabel(r"$\hat p_\text{base} - \hat p_\text{ASA \to Q1}$")
ax_d1.set_ylabel("Density")
ax_d1.set_title("(d) ASA -> Q1 (lower acuity)", fontsize=11, pad=6)
ax_d1.grid(True, alpha=0.3, ls="--"); ax_d1.set_axisbelow(True)
ax_d1.legend(loc="upper right", fontsize=9)

neg_q9 = asa[asa.y_true == 0]["d_q9"].values
pos_q9 = asa[asa.y_true == 1]["d_q9"].values
hist_overlay(ax_d2, neg_q9, pos_q9)
ax_d2.set_xlabel(r"$\hat p_\text{ASA \to Q9} - \hat p_\text{base}$")
ax_d2.set_title("ASA -> Q9 (higher acuity)", fontsize=11, pad=6)
ax_d2.grid(True, alpha=0.3, ls="--"); ax_d2.set_axisbelow(True)

# ── Save ──
fig.tight_layout()
fig.savefig(str(FIG) + ".png", dpi=300, bbox_inches="tight")
fig.savefig(str(FIG) + ".pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved {FIG}.png/.pdf")

# Numerical summaries
print("\nSurgical Context sub-decomposition:")
print(sc_df[["label","d_auroc","lo_a","hi_a","d_auprc","lo_p","hi_p"]].to_string(index=False))
print("\nMedications sub-decomposition:")
print(med_df[["label","d_auroc","lo_a","hi_a","d_auprc","lo_p","hi_p"]].to_string(index=False))
print(f"\nProcedure swap (n with proc = {len(proc):,}):")
print(f"  median ΔP overall: {proc['delta'].median():+.4f}")
print(f"  IoD+ median ΔP: {pos['delta'].median():+.4f}")
print(f"  IoD- median ΔP: {neg['delta'].median():+.4f}")
print(f"\nASA->Q1 swap (n with ASA = {len(asa):,}):  median d_q1 overall: {asa['d_q1'].median():+.4f}")
print(f"ASA->Q9 swap:                                  median d_q9 overall: {asa['d_q9'].median():+.4f}")
