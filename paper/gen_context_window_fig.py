"""
Publication-quality context-window line plot.

Two panels: (a) AUROC, (b) AUPRC across pre-operative history windows.
- Same visual conventions as the ROC/PR figure: colour-coded curves
  with PORT in deep navy (headline), BiLSTM purple, conventional
  baselines muted; full box, inward mirror ticks, light gridlines,
  framed legend, panel labels.
- Legends are sorted by performance descending and placed in the
  lower-left of the right (AUPRC) panel, where no curve passes,
  to avoid overlapping the data.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

OUT = Path("paper/overleaf/figures")
OUT.mkdir(exist_ok=True)

# ── matplotlib defaults (matches ROC/PR figure) ──
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

windows = ["7d", "30d", "90d", "365d", "All"]
x = np.arange(len(windows))

auroc = {
    "PORT":       [0.829, 0.844, 0.822, 0.829, 0.833],
    "BiLSTM":     [0.809, 0.795, 0.801, 0.804, 0.783],
    "XGB (MEDS)": [0.752, 0.759, 0.765, 0.754, 0.744],
    "LR (MEDS)":  [0.759, 0.759, 0.760, 0.758, 0.751],
}
auprc = {
    "PORT":       [0.138, 0.139, 0.136, 0.137, 0.144],
    "BiLSTM":     [0.101, 0.104, 0.088, 0.100, 0.101],
    "XGB (MEDS)": [0.083, 0.091, 0.079, 0.071, 0.064],
    "LR (MEDS)":  [0.083, 0.069, 0.063, 0.049, 0.043],
}

# Colour scheme aligned with the ROC/PR figure
STYLES = {
    "PORT":       dict(color="#1F3A5F", lw=2.4, ls="-",  marker="D", ms=7),
    "BiLSTM":     dict(color="#9467BD", lw=1.8, ls="-",  marker="^", ms=7),
    "XGB (MEDS)": dict(color="#2A9199", lw=1.3, ls="--", marker="s", ms=6),
    "LR (MEDS)":  dict(color="#2E864D", lw=1.3, ls="--", marker="o", ms=6),
}

fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.4))

# Order each panel's iteration by its own metric at the headline column ('All')
# so the curve drawn last (= on top) is the strongest.  Legend entries also
# sort by mean metric across windows, descending.
def panel(ax, data, ylab, panel_letter, title, legend_loc, ylim):
    means = {m: np.mean(v) for m, v in data.items()}
    order = sorted(data.keys(), key=lambda m: means[m], reverse=True)
    # Plot weakest first so PORT lands on top
    for m in reversed(order):
        s = STYLES[m]
        ax.plot(x, data[m], label=m, mec="white", mew=0.8, **s)
    ax.set_xticks(x); ax.set_xticklabels(windows)
    ax.set_xlabel("Pre-operative history window")
    ax.set_ylabel(ylab)
    ax.set_ylim(*ylim)
    ax.set_title(f"({panel_letter}) {title}", fontsize=12, pad=8)
    if legend_loc is not None:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend([by_label[m] for m in order], order, loc=legend_loc)
    ax.tick_params(axis="both", which="major", length=3.5, pad=2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#333"); ax.spines[spine].set_linewidth(0.8)
    ax.grid(True, alpha=0.3, ls="--", zorder=0)
    ax.set_axisbelow(True)

# Left panel: AUROC, no legend (legend lives on the right panel only).
panel(ax_roc, auroc, "AUROC", "a",
      title="Discrimination (AUROC)",
      legend_loc=None,
      ylim=(0.730, 0.860))
# Right panel: AUPRC, legend at lower-left where curves ascend
# rightward and leave the low-recall region clear.
panel(ax_pr, auprc, "AUPRC", "b",
      title="Positive-class precision (AUPRC)",
      legend_loc="lower left",
      ylim=(0.035, 0.155))

fig.tight_layout(w_pad=3.0)
fig.savefig(OUT / "context_window.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "context_window.pdf",            bbox_inches="tight")
print("WROTE", OUT / "context_window.png")
print("WROTE", OUT / "context_window.pdf")
