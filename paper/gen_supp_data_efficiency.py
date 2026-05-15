"""Supplementary Data Efficiency figure: PORT vs BiLSTM under decreasing
training-set fractions (1, 5, 10, 25, 50, 100%). Both models trained with
unweighted BCE for fair comparison.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import roc_auc_score, average_precision_score

R = Path("/path/to/CHD_MEDS/results_v4/baselines")
FIG = Path("paper/overleaf/figures/data_efficiency")

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
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
LSTM_ORANGE = "#E67E22"

fracs_pct = [1, 5, 10, 25, 50, 100]

def load(model, tag):
    if model == "PORT":
        if tag == 100:
            fp = R / "ethos_finetune_lora_test_predictions_v4_lora_r8_bce_s456.parquet"
        else:
            fp = R / f"ethos_finetune_lora_test_predictions_v4_lora_r8_bce_frac{tag}_s456.parquet"
    else:
        if tag == 100:
            fp = R / "lstm_unweighted_test_predictions.parquet"
            if not fp.exists():
                fp = Path("/path/to/CHD_MEDS/results/baselines_tuned/lstm_tuned_test_predictions.parquet")
        else:
            fp = R / f"lstm_test_predictions_unweighted_frac{tag}_s123.parquet"
    if not fp.exists():
        return float("nan"), float("nan")
    df = pd.read_parquet(fp)
    y = df.y_true.astype(int).values; p = df.y_prob.values
    return roc_auc_score(y, p), average_precision_score(y, p)

port_auroc, port_auprc = [], []
lstm_auroc, lstm_auprc = [], []
for tag in fracs_pct:
    pa, pp = load("PORT", tag)
    la, lp = load("LSTM", tag)
    port_auroc.append(pa); port_auprc.append(pp)
    lstm_auroc.append(la); lstm_auprc.append(lp)

print("PORT AUROC:", [f"{x:.4f}" for x in port_auroc])
print("BiLSTM AUROC:", [f"{x:.4f}" for x in lstm_auroc])
print("PORT AUPRC:", [f"{x:.4f}" for x in port_auprc])
print("BiLSTM AUPRC:", [f"{x:.4f}" for x in lstm_auprc])

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.4))

x = np.array(fracs_pct)
for ax, port, lstm, ylab, title in [
    (ax_a, port_auroc, lstm_auroc, "Test AUROC", "(a) Discrimination (AUROC)"),
    (ax_b, port_auprc, lstm_auprc, "Test AUPRC", "(b) Positive-class precision (AUPRC)"),
]:
    ax.plot(x, lstm, color=LSTM_ORANGE, lw=2.0, marker="^", ms=8, mec="white",
            mew=0.8, label="BiLSTM (unweighted)")
    ax.plot(x, port, color=PORT_BLUE, lw=2.4, marker="D", ms=8, mec="white",
            mew=0.8, label="PORT (LoRA r=8, BCE)")
    ax.set_xscale("log")
    ax.set_xticks(fracs_pct)
    ax.set_xticklabels([f"{f}%" for f in fracs_pct])
    ax.set_xlabel("Training set fraction")
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=12, pad=8)
    ax.grid(True, alpha=0.3, ls="--", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")

fig.tight_layout(w_pad=2.5)
fig.savefig(str(FIG) + ".png", dpi=300, bbox_inches="tight")
fig.savefig(str(FIG) + ".pdf",            bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {FIG}.png/.pdf")
