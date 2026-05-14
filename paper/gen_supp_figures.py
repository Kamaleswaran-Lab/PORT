"""
Publication-quality supplementary figures for the JAMIA paper.

Re-renders four supplementary figures with the same visual style as
the main-text ROC/PR and context-window figures:
  - figures/history_length_distribution.png  (3-panel)
  - figures/calibration_curves.png           (single panel)
  - figures/subgroup_asa.png                 (single panel bar)
  - figures/decision_curve.png               (single panel)

Shared style:
  Helvetica sans-serif, font.size 10
  Full box (4-side spines), inward mirror ticks (major + minor)
  Light major (0.85) + minor (0.92) gridlines
  Framed legend (white, thin gray border, no metric title)
  Bold panel title on top; bold (a/b/c) panel labels for multi-panel
  PORT in deep navy (#1F3A5F), BiLSTM purple (#E67E22), conventional
  baselines in muted teal/green/orange
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

R_BASE = Path("/path/to/CHD_MEDS/results/baselines")
R_RESULTS   = Path("/path/to/CHD_MEDS/results/baselines")
FIG    = Path("paper/overleaf/figures")
TASK   = Path("/path/to/CHD_MEDS/outcome/iod_task.parquet")
ANP    = Path("/path/to/CHOA_RAW_TABLES/CHOA_DATA_Tables_CHD/DR15201_AN_Patients.rpt")

# ── matplotlib defaults (matches main-text figures) ──
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

STYLE_PORT  = dict(color="#1F3A5F", lw=2.4)
STYLE_LSTM  = dict(color="#E67E22", lw=1.8)
STYLE_XGB   = dict(color="#2A9199", lw=1.3, ls="--")
STYLE_LR    = dict(color="#2E864D", lw=1.3, ls="--")
STYLE_ASA   = dict(color="#C0392B", lw=1.1, ls="-.")


def style_box(ax):
    """Minimal style: hide top/right spines + dashed major grid (Fig 3 convention)."""
    ax.tick_params(axis="both", which="major", length=3.5, pad=2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#333")
        ax.spines[spine].set_linewidth(0.8)
    ax.grid(True, alpha=0.3, ls="--", zorder=0)
    ax.set_axisbelow(True)


def panel_letter(ax, letter, dx=-0.13, dy=1.04):
    """No-op kept for backward compat: panel letters are now inline in titles."""
    pass


# ════════════════════════════════════════════════════════════════════
# Load shared predictions
# ════════════════════════════════════════════════════════════════════
df_d  = pd.read_parquet(R_BASE / "test_preds_meds.parquet")
df_l  = pd.read_parquet(R_BASE / "lstm_test_predictions.parquet")
df_a  = pd.read_parquet(R_BASE / "asa_test_predictions.parquet")
df_p  = pd.read_parquet(R_RESULTS / "ethos_finetune_lora_test_predictions_v4_lora_r8_bce_s456.parquet")

models = {
    "PORT":       (df_p.y_true.astype(int).values, df_p.y_prob.values),
    "BiLSTM":     (df_l.y_true.astype(int).values, df_l.y_prob.values),
    "XGB (MEDS)": (df_d.y_true.astype(int).values, df_d.prob_xgb_meds.values),
    "LR (MEDS)":  (df_d.y_true.astype(int).values, df_d.prob_lr_meds.values),
    "ASA score":  (df_a.y_true.astype(int).values, df_a.y_prob.values),
}
STYLES = {
    "PORT":       STYLE_PORT,
    "BiLSTM":     STYLE_LSTM,
    "XGB (MEDS)": STYLE_XGB,
    "LR (MEDS)":  STYLE_LR,
    "ASA score":  STYLE_ASA,
}


# ════════════════════════════════════════════════════════════════════
# (1) Calibration curves
# ════════════════════════════════════════════════════════════════════
def fig_calibration():
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    # quantile-binning is more informative than uniform for rare events
    for name in ["PORT", "BiLSTM", "XGB (MEDS)", "LR (MEDS)"]:
        y, p = models[name]
        prob_true, prob_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        # Drop bins with zero observed positives — log(0) = -inf creates an
        # artifactual vertical drop on log-scale axes.
        mask = prob_true > 0
        s = STYLES[name]
        ax.plot(prob_pred[mask], prob_true[mask], marker="o", ms=5, mec="white", mew=0.6,
                label=name, **s)
    ax.plot([0, 1], [0, 1], color="0.7", lw=0.8, ls="--", zorder=0,
            label="Perfect")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-4, 1.0)
    ax.set_ylim(1e-4, 1.0)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title("Reliability by predicted-risk decile",
                 fontsize=12, pad=8)
    ax.legend(loc="upper left")
    style_box(ax)
    fig.tight_layout()
    fig.savefig(FIG / "calibration_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "calibration_curves.pdf",            bbox_inches="tight")
    plt.close(fig)
    print("WROTE", FIG / "calibration_curves.png")


# ════════════════════════════════════════════════════════════════════
# (2) Subgroup ASA — PORT AUROC by ASA class
# ════════════════════════════════════════════════════════════════════
def fig_subgroup_asa():
    # Match ASA score: AN_Patients (sid, csn) -> task -> PORT (subject_id, prediction_time)
    an = pd.read_csv(ANP, sep="|", usecols=["C MRN","ASA PS Score","Encounter CSN"], dtype=str)
    an["sid"] = pd.to_numeric(an["C MRN"].str.lstrip("C"), errors="coerce")
    an["asa"] = pd.to_numeric(an["ASA PS Score"], errors="coerce")
    an["enc_csn"] = pd.to_numeric(an["Encounter CSN"], errors="coerce")
    an = an.dropna(subset=["sid", "asa", "enc_csn"]).drop_duplicates(["sid", "enc_csn"])

    task = pd.read_parquet("/path/to/CHD_MEDS/outcome/iod_task.parquet")
    task["sid"] = task.patient_id.str.lstrip("C").astype(int)
    task = task.merge(an[["sid", "enc_csn", "asa"]],
                      left_on=["sid", "encounter_csn"],
                      right_on=["sid", "enc_csn"], how="left")

    df = pd.read_parquet(R_RESULTS / "ethos_finetune_lora_test_predictions_v4_lora_r8_bce_s456.parquet")
    df["prediction_time"] = pd.to_datetime(df.prediction_time_us, unit="us")
    df = df.merge(task[["sid", "prediction_time", "asa"]],
                  left_on=["subject_id", "prediction_time"],
                  right_on=["sid", "prediction_time"],
                  how="left").dropna(subset=["asa"])

    classes  = ["I", "II", "III", "IV", "V+"]
    aurocs, ns, ci_los, ci_his = [], [], [], []
    for ac in [1, 2, 3, 4, 5]:
        sub = df[df.asa == ac] if ac < 5 else df[df.asa >= 5]
        if len(sub) >= 10 and sub.y_true.sum() >= 2:
            ys = sub.y_true.astype(int).values
            ps = sub.y_prob.values
            aurocs.append(roc_auc_score(ys, ps))
            # Bootstrap CI for AUROC at this subgroup
            rng = np.random.default_rng(42)
            n = len(ys)
            boots = []
            for _ in range(2000):
                idx = rng.integers(0, n, n)
                yi = ys[idx]
                if yi.sum() == 0 or yi.sum() == n: continue
                boots.append(roc_auc_score(yi, ps[idx]))
            ci_los.append(np.percentile(boots, 2.5))
            ci_his.append(np.percentile(boots, 97.5))
        else:
            aurocs.append(np.nan); ci_los.append(np.nan); ci_his.append(np.nan)
        ns.append(len(sub))

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    bar_colors = ["#3D5A8A", "#3D5A8A", "#1F3A5F", "#9C2A2F", "#9C2A2F"]
    err_lo = [v - l if not np.isnan(v) else 0 for v, l in zip(aurocs, ci_los)]
    err_hi = [h - v if not np.isnan(v) else 0 for v, h in zip(aurocs, ci_his)]
    bars = ax.bar(classes, aurocs, color=bar_colors, edgecolor="black", lw=0.6, width=0.65,
                  yerr=[err_lo, err_hi], error_kw=dict(ecolor="#222", lw=1.0, capsize=4))
    for b, v, n, ch in zip(bars, aurocs, ns, ci_his):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width()/2, ch + 0.008,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
            ax.text(b.get_x() + b.get_width()/2, 0.515,
                    f"n={n:,}", ha="center", va="bottom",
                    fontsize=10, color="white")
    ax.axhline(0.5, color="0.6", lw=0.7, ls="--", zorder=0)
    ax.set_ylim(0.5, 0.92)
    ax.set_xlabel("ASA physical status class")
    ax.set_ylabel("AUROC")
    ax.set_title("PORT discrimination by patient acuity (ASA)",
                 fontsize=12, pad=8)
    style_box(ax)
    fig.tight_layout()
    fig.savefig(FIG / "subgroup_asa.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "subgroup_asa.pdf",            bbox_inches="tight")
    plt.close(fig)
    print("WROTE", FIG / "subgroup_asa.png")


# ════════════════════════════════════════════════════════════════════
# (3) Decision-curve analysis
# ════════════════════════════════════════════════════════════════════
def fig_decision_curve():
    thresholds = np.linspace(0.005, 0.20, 80)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))

    def net_benefit(y, p, t):
        flag = p >= t
        N = len(y)
        return (flag & (y == 1)).sum() / N - (flag & (y == 0)).sum() / N * (t / (1 - t))

    # Order to plot: weakest first so PORT lands on top.
    # Bootstrap envelope (95% CI shaded) for headline models only (PORT, BiLSTM)
    # to avoid visual clutter; other curves drawn as point estimates.
    for name in ["ASA score", "LR (MEDS)", "XGB (MEDS)", "BiLSTM", "PORT"]:
        y, p = models[name]
        nb = np.array([net_benefit(y, p, t) for t in thresholds])
        s = STYLES[name]
        if name in ("PORT", "BiLSTM"):
            # Bootstrap CI
            rng = np.random.default_rng(42)
            n = len(y)
            boot_nb = np.full((2000, len(thresholds)), np.nan)
            for i in range(2000):
                idx = rng.integers(0, n, n)
                yi = y[idx]
                if yi.sum() == 0: continue
                pi = p[idx]
                boot_nb[i] = [net_benefit(yi, pi, t) for t in thresholds]
            lo = np.nanpercentile(boot_nb, 2.5, axis=0)
            hi = np.nanpercentile(boot_nb, 97.5, axis=0)
            ax.fill_between(thresholds, lo, hi, color=s["color"], alpha=0.15, lw=0, zorder=1)
        ax.plot(thresholds, nb, label=name, **s)

    # Reference strategies
    y_port = models["PORT"][0]
    prev = y_port.mean()
    nb_all = [prev - (1 - prev) * (t / (1 - t)) for t in thresholds]
    ax.plot(thresholds, nb_all, color="0.45", lw=0.8, ls=":",
            label="Treat all")
    ax.axhline(0, color="0.45", lw=0.8, ls="--", label="Treat none")

    ax.set_xlim(thresholds[0], thresholds[-1])
    ax.set_ylim(-0.005, 0.012)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision-curve analysis (clinical net benefit)",
                 fontsize=12, pad=8)
    ax.legend(loc="upper right")
    style_box(ax)
    fig.tight_layout()
    fig.savefig(FIG / "decision_curve.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "decision_curve.pdf",            bbox_inches="tight")
    plt.close(fig)
    print("WROTE", FIG / "decision_curve.png")


# ════════════════════════════════════════════════════════════════════
# (4) Pre-operative history length distribution
# ════════════════════════════════════════════════════════════════════
def fig_history_length():
    # Build per-encounter pre-operative history length from iod_task and merged events.
    # Restrict to TEST SET (n=37,948) so the IoD+ count matches the paper's headline
    # test-set numbers (393 IoD+, 1.04%); the full-cohort version drops ~110 IoD+
    # encounters whose patient's earliest event in events.parquet predates their
    # task prediction_time, producing inconsistent counts.
    task = pd.read_parquet(TASK)
    SPLITS = Path("/path/to/CHD_MEDS/splits/splits.parquet")
    if SPLITS.exists():
        splits = pd.read_parquet(SPLITS)
        test_pids = set(splits[splits["split"] == "test"]["subject_id"])
        task["sid"] = task["patient_id"].str.lstrip("C").astype(int)
        task = task[task["sid"].isin(test_pids)].copy()
        print(f"  restricted to test set: {len(task):,} encounters")
    EVENTS = Path("/path/to/CHD_MEDS/merged/events.parquet")
    if not EVENTS.exists():
        print("WARN: events file missing; skipping history figure")
        return
    # Stream events row-group at a time to avoid loading 26.5M rows in memory
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(EVENTS))
    earliest = {}
    for rg in range(pf.num_row_groups):
        chunk = pf.read_row_group(rg, columns=["patient_id", "time"]).to_pandas()
        chunk = chunk.dropna(subset=["time"])
        g = chunk.groupby("patient_id")["time"].min()
        for pid, t in g.items():
            if pid not in earliest or t < earliest[pid]:
                earliest[pid] = t
    ear = pd.Series(earliest, name="first_event")
    task["prediction_time"] = pd.to_datetime(task["prediction_time"])
    task = task.merge(ear, left_on="patient_id", right_index=True, how="left")
    task["hist_days"] = (task["prediction_time"] - task["first_event"]).dt.days
    task = task.dropna(subset=["hist_days"])
    task = task[task["hist_days"] >= 0]

    pos = task[task.boolean_value == True]["hist_days"].values
    neg = task[task.boolean_value == False]["hist_days"].values

    # Vertical 3×1 layout so each panel gets full figure width and large readable text.
    fig, axes = plt.subplots(3, 1, figsize=(9, 12))

    # ── (a) Window-bucket bar chart: encounters per cutoff bin ──
    # Buckets: ≤7d, 8–30d, 31–90d, 91–365d, >365d. Each bar shows
    # exact encounter count + % of total → 7d / 30d are now individual
    # bars, no longer collapsed into a single histogram bin.
    ax = axes[0]
    edges = [0, 7, 30, 90, 365, np.inf]
    labels = ["≤ 7 d", "8–30 d", "31–90 d", "91–365 d", "> 365 d"]
    counts_neg = [((neg >= lo) & (neg < hi)).sum() for lo, hi in zip(edges[:-1], edges[1:])]
    counts_pos = [((pos >= lo) & (pos < hi)).sum() for lo, hi in zip(edges[:-1], edges[1:])]
    counts_tot = np.array(counts_neg) + np.array(counts_pos)
    pct = 100 * counts_tot / counts_tot.sum()
    x = np.arange(len(labels))
    ax.bar(x, counts_neg, color="#888", edgecolor="black", lw=0.5, label=f"IoD− (n={len(neg):,})")
    ax.bar(x, counts_pos, bottom=counts_neg, color="#C0392B", edgecolor="black", lw=0.5,
           label=f"IoD+ (n={len(pos):,})")
    for xi, ct, p in zip(x, counts_tot, pct):
        ax.text(xi, ct + counts_tot.max()*0.018,
                f"{ct:,}\n({p:.1f}%)", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, counts_tot.max() * 1.32)
    ax.set_xlabel("Pre-operative history window")
    ax.set_ylabel("Number of encounters")
    ax.set_title("(a) Encounters by history-length bin",
                 fontsize=12, pad=8)
    ax.legend(loc="upper left", fontsize=11)
    style_box(ax); panel_letter(ax, "a")

    # ── (b) Cumulative distribution (log-x so cut-offs are evenly spaced) ──
    ax = axes[1]
    h_sorted = np.sort(task.hist_days.values)
    cdf = np.arange(1, len(h_sorted) + 1) / len(h_sorted)
    # Replace 0-day values with 0.5 to keep them visible on log axis
    h_plot = np.where(h_sorted < 0.5, 0.5, h_sorted)
    ax.plot(h_plot, cdf, color="#1F3A5F", lw=2.0)
    for cut, lab in [(7, "7 d"), (30, "30 d"), (90, "90 d"), (365, "365 d")]:
        frac = (task.hist_days <= cut).mean()
        ax.axvline(cut, color="0.55", lw=0.8, ls="--", zorder=0)
        ax.text(cut, 1.04, lab, ha="center", va="bottom", fontsize=11, color="0.35")
        ax.plot([cut], [frac], "o", color="#1F3A5F", ms=6, mec="white", mew=0.7, zorder=5)
        ax.annotate(f"{frac*100:.1f}%", (cut, frac),
                    xytext=(8, -3), textcoords="offset points",
                    fontsize=11, color="#1F3A5F", fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(0.5, h_sorted[-1] * 1.05)
    ax.set_ylim(0, 1.10)
    ax.set_xlabel("Pre-operative history (days, log scale)")
    ax.set_ylabel("Cumulative fraction of encounters")
    ax.set_title("(b) Cumulative distribution",
                 fontsize=12, pad=8)
    style_box(ax); panel_letter(ax, "b")

    # ── (c) Density comparison IoD+ vs IoD− on log-x ──
    # Log-spaced bins so the 0-365d region is well resolved.
    ax = axes[2]
    bins = np.logspace(0, np.log10(max(neg.max(), pos.max())+1), 40)
    # Replace 0-day values with 0.5 day to plot on log axis
    neg_p = np.where(neg < 0.5, 0.5, neg)
    pos_p = np.where(pos < 0.5, 0.5, pos)
    ax.hist(neg_p, bins=bins, density=True, color="#888", alpha=0.65,
            edgecolor="white", lw=0.4, label=f"IoD− (n={len(neg):,}, median={np.median(neg):.0f} d)")
    ax.hist(pos_p, bins=bins, density=True, color="#C0392B", alpha=0.75,
            edgecolor="white", lw=0.4, label=f"IoD+ (n={len(pos):,}, median={np.median(pos):.0f} d)")
    ax.axvline(np.median(neg), color="#444", ls=":", lw=1.4, zorder=4)
    ax.axvline(np.median(pos), color="#7B1F19", ls=":", lw=1.4, zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("Pre-operative history (days, log scale)")
    ax.set_ylabel("Density")
    ax.set_title("(c) Distribution by IoD label",
                 fontsize=12, pad=8)
    ax.legend(loc="upper left", fontsize=10.5)
    style_box(ax); panel_letter(ax, "c")

    fig.tight_layout(h_pad=2.5)
    fig.savefig(FIG / "history_length_distribution.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "history_length_distribution.pdf",            bbox_inches="tight")
    plt.close(fig)
    print("WROTE", FIG / "history_length_distribution.png")


if __name__ == "__main__":
    fig_calibration()
    fig_subgroup_asa()
    fig_decision_curve()
    fig_history_length()
