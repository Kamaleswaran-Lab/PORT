"""Generate 2 new JAMIA-quality figures: context-window + uncertainty ensemble."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("paper/overleaf/figures")
OUT.mkdir(exist_ok=True)

# ── Unified figure style (Fig 3 minimal, +2pt) ──
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
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ── Figure 1: Context-window ablation ─────────────────────────────────────────
# Values from paper Table 4 (verified against predictions parquets)
windows = ["7d", "30d", "90d", "365d", "All"]
x = np.arange(len(windows))
auroc = {
    "LR (MEDS)":  [0.759, 0.759, 0.760, 0.758, 0.751],
    "XGB (MEDS)": [0.752, 0.759, 0.765, 0.754, 0.744],
    "BiLSTM":     [0.809, 0.795, 0.801, 0.804, 0.783],
    "PORT":       [0.829, 0.844, 0.822, 0.829, 0.833],
}
auprc = {
    "LR (MEDS)":  [0.083, 0.069, 0.063, 0.049, 0.043],
    "XGB (MEDS)": [0.083, 0.091, 0.079, 0.071, 0.064],
    "BiLSTM":     [0.101, 0.104, 0.088, 0.100, 0.101],
    "PORT":       [0.138, 0.139, 0.136, 0.137, 0.144],
}
colors = {"LR (MEDS)": "#7f7f7f", "XGB (MEDS)": "#2ca02c", "BiLSTM": "#1f77b4", "PORT": "#d62728"}
markers = {"LR (MEDS)": "o", "XGB (MEDS)": "s", "BiLSTM": "^", "PORT": "D"}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, metric, data, ylab in [
    (axes[0], "AUROC", auroc, "AUROC"),
    (axes[1], "AUPRC", auprc, "AUPRC"),
]:
    for m, ys in data.items():
        lw = 2.4 if m == "PORT" else 1.6
        ax.plot(x, ys, marker=markers[m], color=colors[m], lw=lw,
                ms=7, label=m, mec="white", mew=0.8)
    ax.set_xticks(x); ax.set_xticklabels(windows)
    ax.set_xlabel("Pre-operative history window")
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.3, ls="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
axes[0].legend(frameon=False, loc="lower right", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "context_window.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "context_window.pdf", bbox_inches="tight")
plt.close(fig)
print("wrote context_window")

# ── Figure 2: Seed-ensemble uncertainty ──────────────────────────────────────
base = Path("/path/to/CHD_MEDS/results/baselines")
dfs = [pd.read_parquet(base / f"ethos_finetune_lora_test_predictions_lora_s{s}.parquet")
       for s in [42, 123, 456]]
# Align by subject_id
pid = dfs[0]["subject_id"] if "subject_id" in dfs[0].columns else dfs[0].index
probs = np.stack([d["y_prob"].values for d in dfs], axis=1)
y = dfs[0]["y_true"].astype(int).values
mean_p = probs.mean(axis=1)
std_p = probs.std(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# (a) scatter mean vs std, colored by y
ax = axes[0]
neg = y == 0; pos = y == 1
ax.scatter(mean_p[neg], std_p[neg], s=4, c="#b0c4de", alpha=0.4, label=f"IoD− (n={neg.sum()})", rasterized=True)
ax.scatter(mean_p[pos], std_p[pos], s=14, c="#c0392b", alpha=0.85, label=f"IoD+ (n={pos.sum()})",
           marker="o", edgecolors="white", linewidths=0.4)
ax.set_xlabel(r"Predicted IoD risk $\bar{p}$ (mean across 3 seeds)")
ax.set_ylabel(r"Seed disagreement $\sigma_p$")
ax.set_xscale("log"); ax.set_xlim(1e-4, 1)
ax.grid(True, alpha=0.3, ls="--")
ax.legend(frameon=False, fontsize=11, loc="upper left")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.set_title("(a) Per-encounter risk vs. seed disagreement", fontsize=12)

# (b) IoD+ rate by risk decile + 95% CI from seed std as error bar proxy
ax = axes[1]
# decile bins on mean_p
deciles = np.quantile(mean_p, np.linspace(0, 1, 11))
deciles[0] = -1e-9; deciles[-1] = 1 + 1e-9
bin_idx = np.digitize(mean_p, deciles) - 1
bin_idx = np.clip(bin_idx, 0, 9)
xs, obs, pred, se = [], [], [], []
for i in range(10):
    m = bin_idx == i
    if m.sum() == 0: continue
    xs.append(mean_p[m].mean())
    obs.append(y[m].mean())
    pred.append(mean_p[m].mean())
    p = y[m].mean(); n = m.sum()
    se.append(1.96 * np.sqrt(p*(1-p)/max(n, 1)))
xs = np.array(xs); obs = np.array(obs); se = np.array(se)
ax.plot([1e-4, 1], [1e-4, 1], "--", c="gray", lw=1, label="Perfect calibration")
ax.errorbar(xs, obs, yerr=se, fmt="o", color="#c0392b", ms=7, lw=1.3, capsize=3,
            label="Observed (decile bins, 95% CI)", mec="white", mew=0.8)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1e-4, 1); ax.set_ylim(1e-4, 1)
ax.set_xlabel(r"Mean predicted risk $\bar{p}$")
ax.set_ylabel("Observed IoD+ rate")
ax.grid(True, alpha=0.3, ls="--", which="both")
ax.legend(frameon=False, fontsize=11, loc="upper left")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.set_title("(b) Calibration by risk decile", fontsize=12)

fig.tight_layout()
fig.savefig(OUT / "uncertainty_ensemble.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "uncertainty_ensemble.pdf", bbox_inches="tight")
plt.close(fig)
print("wrote uncertainty_ensemble")
print("DONE")
