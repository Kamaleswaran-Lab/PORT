"""Supplementary HP-grid heatmap: 5×4 (rank × loss) for AUROC, AUPRC, Brier.

Reads test predictions from the 20-cell sweep and renders a 3-panel heatmap.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

R    = Path("/path/to/CHD_MEDS/results_v4/baselines")
FIG  = Path("paper/overleaf/figures/hp_grid_heatmap")

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.linewidth": 0.7,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

RANKS  = [4, 8, 16, 32, 64]
LOSSES = ["wbce", "bce", "focal_g2", "oversample5"]
LOSS_LABELS = {
    "wbce":         "weighted BCE\n(pos_w=101)",
    "bce":          "BCE\n(unweighted)",
    "focal_g2":     r"Focal $\gamma{=}2,\alpha{=}0.25$",
    "oversample5":  r"Oversample $5\times$",
}

def load_metric(r, loss):
    fp = R / f"ethos_finetune_lora_test_predictions_v4_lora_r{r}_{loss}_s123.parquet"
    df = pd.read_parquet(fp)
    y = df["y_true"].astype(int).values
    p = df["y_prob"].values
    return (roc_auc_score(y, p),
            average_precision_score(y, p),
            brier_score_loss(y, p))

AUROC = np.full((len(RANKS), len(LOSSES)), np.nan)
AUPRC = np.full((len(RANKS), len(LOSSES)), np.nan)
BRIER = np.full((len(RANKS), len(LOSSES)), np.nan)
for i, r in enumerate(RANKS):
    for j, loss in enumerate(LOSSES):
        try:
            a, ap, b = load_metric(r, loss)
            AUROC[i, j] = a
            AUPRC[i, j] = ap
            BRIER[i, j] = b
        except FileNotFoundError:
            pass

# Selected cell coordinates: (r=8, bce) → row 1, col 1
SEL = (1, 1)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

def plot_panel(ax, data, title, cmap, fmt, lo=None, hi=None, invert=False):
    if lo is None: lo = np.nanmin(data)
    if hi is None: hi = np.nanmax(data)
    im = ax.imshow(data, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                # White text on dark cells, black on light
                norm = (v - lo) / (hi - lo) if hi > lo else 0.5
                if invert:
                    color = "white" if norm < 0.35 else "black"
                else:
                    color = "white" if norm > 0.65 else "black"
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=10, color=color)
            if (i, j) == SEL:
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                           edgecolor="red", lw=2.0, zorder=5))
    ax.set_xticks(range(len(LOSSES)))
    ax.set_xticklabels([LOSS_LABELS[l] for l in LOSSES], fontsize=9)
    ax.set_yticks(range(len(RANKS)))
    ax.set_yticklabels([f"r={r}" for r in RANKS])
    ax.set_title(title, fontsize=12, pad=8)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)

plot_panel(axes[0], AUROC, "(a) Test AUROC",        cmap="viridis", fmt="{:.3f}")
plot_panel(axes[1], AUPRC, "(b) Test AUPRC",        cmap="viridis", fmt="{:.3f}")
plot_panel(axes[2], BRIER, "(c) Test Brier (lower=better)",
           cmap="viridis_r", fmt="{:.4f}", invert=True)

fig.suptitle("LoRA hyperparameter sweep (5 ranks × 4 losses; seed 123); "
             "red box = selected config (rank=8, unweighted BCE).",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(str(FIG) + ".png", dpi=300, bbox_inches="tight")
fig.savefig(str(FIG) + ".pdf",            bbox_inches="tight")
plt.close(fig)
print(f"Saved {FIG}.png/.pdf")
