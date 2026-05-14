"""Supplementary case-examples figure.

Renders a 2x2 grid of per-patient category attribution bars for four selected
cases from the held-out test set, illustrating how PORT decomposes risk
across token categories at the individual level.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

CASE_JSON = Path("/path/to/CHD_MEDS/results_v4/evaluation/case_examples_r8bce_s456/case_examples.json")
FIG       = Path("paper/overleaf/figures/case_examples")

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Color: red increases risk (Δp > 0 when masked → mask removes risk-elevating signal)
# Blue decreases risk (Δp < 0 when masked → mask removes risk-lowering signal)
COLOR_POS = "#C0392B"   # red
COLOR_NEG = "#2A5DA0"   # blue

CASE_TITLES = {
    "A_high_conf_pos": ("Case A: High-confidence true positive",
                        "Confirmed IoD+ at high predicted risk"),
    "B_moderate_pos":  ("Case B: Moderate-risk true positive",
                        "Confirmed IoD+ at deployment-relevant threshold"),
    "C_confident_neg": ("Case C: Confident true negative",
                        "Routine encounter — model confidently low"),
    "D_false_pos":     ("Case D: False positive (transparency)",
                        "No IoD event despite abnormal preoperative markers"),
}

data = json.load(open(CASE_JSON))

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
axes = axes.flatten()

TOP_K = 6

for ax, (case_key, info) in zip(axes, data.items()):
    title, subtitle = CASE_TITLES.get(case_key, (case_key, ""))
    base_p = info["baseline_pred"]
    label  = info["label"]
    n_tot  = info["n_total_tokens"]

    # Top-K categories by |Δp|
    attrib = sorted(info["category_attribution"].items(),
                    key=lambda x: -abs(x[1]["delta_pred"]))[:TOP_K]
    labels = [c for c, _ in attrib]
    deltas = [m["delta_pred"] for _, m in attrib]
    ntok   = [m["n_masked_tokens"] for _, m in attrib]

    # Plot from largest |Δ| at top
    y = np.arange(len(labels))[::-1]
    colors = [COLOR_POS if d > 0 else COLOR_NEG for d in deltas]
    bars = ax.barh(y, deltas, color=colors, edgecolor="black", lw=0.4, height=0.65)

    # Annotate each bar with token count
    xlim_pad = max(abs(min(deltas)), abs(max(deltas))) * 0.18 + 0.01
    for yi, d, n in zip(y, deltas, ntok):
        x_text = d + (xlim_pad * 0.15 if d >= 0 else -xlim_pad * 0.15)
        ha = "left" if d >= 0 else "right"
        ax.text(x_text, yi, f"n={n:,}", ha=ha, va="center",
                fontsize=8.5, color="#333")

    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(r"$\Delta\hat p$ when category masked"
                  " (positive = removes risk-elevating signal)")
    ax.set_title(f"{title}\n"
                 f"baseline PORT $\\hat p$ = {base_p:.3f}; "
                 f"actual = {'IoD$+$' if label==1 else 'IoD$-$'}; "
                 f"timeline = {n_tot:,} tokens",
                 fontsize=11, pad=10)
    ax.grid(axis="x", alpha=0.3, ls="--", zorder=0)
    ax.set_axisbelow(True)

    # Symmetric-ish x-limits with padding
    lo = min(deltas) - xlim_pad
    hi = max(deltas) + xlim_pad
    ax.set_xlim(min(lo, -0.02), max(hi, 0.02))

fig.suptitle("Per-encounter token-category attribution for four representative test cases",
             fontsize=13, y=1.00)
fig.tight_layout()
fig.savefig(str(FIG) + ".png", dpi=300, bbox_inches="tight")
fig.savefig(str(FIG) + ".pdf",            bbox_inches="tight")
plt.close(fig)
print(f"Saved {FIG}.png/.pdf")
