"""Figure 6 (single-panel): risk-stratified IoD rate with fine-grained
sub-strata above 10% predicted risk to expose the monotonic risk gradient.

Replaces the 4-panel clinical figure with a focused 9-stratum bar chart
(<1, 1-2, 2-5, 5-10, 10-15, 15-20, 20-30, 30-50, >=50 percent).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

PORT = Path("/path/to/CHD_MEDS/results_v4/baselines/ethos_finetune_lora_test_predictions_v4_lora_s123.parquet")
FIG  = Path("paper/overleaf/figures/patient_mechanism")

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PORT_BLUE = "#1F3A5F"

df = pd.read_parquet(PORT)
y = df.y_true.astype(int).values
p = df.y_prob.values

EDGES  = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.01]
LABELS = ["<1%", "1–2%", "2–5%", "5–10%",
          "10–15%", "15–20%", "20–30%", "30–50%", "≥50%"]

df = pd.DataFrame({"p": p, "y": y})
df["bin"] = pd.cut(df.p, EDGES, labels=LABELS, include_lowest=True, right=False)
g = df.groupby("bin", observed=True).agg(n=("y", "size"), pos=("y", "sum"))
g["rate"] = g.pos / g.n
g = g.reindex(LABELS)
print(g.to_string())

fig, ax = plt.subplots(1, 1, figsize=(10, 5.2))

x = np.arange(len(LABELS))
bars = ax.bar(x, g.rate.values * 100, color=PORT_BLUE, edgecolor="black",
              lw=0.4, width=0.72, zorder=3)
for i, (n, pos, rate) in enumerate(zip(g.n, g.pos, g.rate)):
    ax.text(i, rate * 100 + 0.6, f"n={int(n):,}\n{int(pos)} IoD+",
            ha="center", va="bottom", fontsize=9, color="#222")

ax.axhline(1.0, color="0.55", lw=0.9, ls=":", zorder=1)
ax.text(len(LABELS) - 0.5, 1.3, "Population prevalence 1.0%",
        ha="right", va="bottom", fontsize=10, color="0.4", style="italic")

ax.set_xticks(x)
ax.set_xticklabels(LABELS)
ax.set_xlabel("PORT predicted risk")
ax.set_ylabel("Observed IoD rate (%)")
ax.set_ylim(0, max(g.rate.values * 100) * 1.22)
ax.grid(axis="y", alpha=0.3, ls="--", zorder=0)
ax.set_axisbelow(True)
ax.set_title("Observed IoD rate by predicted-risk stratum",
             fontsize=13, pad=10)

fig.tight_layout()
fig.savefig(str(FIG) + ".png", dpi=300, bbox_inches="tight")
fig.savefig(str(FIG) + ".pdf",            bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {FIG}.png/.pdf")

# Useful stats for caption / narrative
top10 = (df.p >= 0.10).sum(); top10_pos = ((df.p >= 0.10) & (df.y == 1)).sum()
top50 = (df.p >= 0.50).sum(); top50_pos = ((df.p >= 0.50) & (df.y == 1)).sum()
total_pos = (df.y == 1).sum()
print(f"\nTop ≥10% stratum: {top10:,} encounters ({top10/len(df)*100:.1f}%), "
      f"{top10_pos} IoD+ ({top10_pos/total_pos*100:.1f}% of all events)")
print(f"Top ≥50% stratum: {top50:,} encounters ({top50/len(df)*100:.2f}%), "
      f"{top50_pos} IoD+ ({top50_pos/total_pos*100:.1f}% of all events)")
