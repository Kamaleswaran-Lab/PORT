"""Generate occlusion analysis figure from CSV results."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

CSV = "/path/to/CHD_MEDS/results/evaluation/occlusion_results.csv"
FIG = "paper/overleaf/figures/occlusion_analysis.png"
FIG_PDF = FIG.replace(".png", ".pdf")

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
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "legend.fontsize": 11,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

df = pd.read_csv(CSV)
# Remove baseline row
df = df[df.Category != "Baseline (no mask)"].copy()

# Sort by Delta_AUROC (most negative first)
df = df.sort_values("Delta_AUROC", ascending=True)

# Vertical 2x1 layout: each panel gets full width so long category labels
# (Surgical Context, Care Trajectory, etc.) never collide with adjacent panel.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 11))
plt.subplots_adjust(left=0.28, right=0.97, hspace=0.32)

colors = []
for _, row in df.iterrows():
    if row['Delta_AUROC'] < -0.01:
        colors.append('#e74c3c')  # red for big impact
    elif row['Delta_AUROC'] < -0.001:
        colors.append('#e67e22')  # orange for moderate
    elif row['Delta_AUROC'] < 0:
        colors.append('#3498db')  # blue for small
    else:
        colors.append('#95a5a6')  # gray for no/positive impact

# AUROC drop
y_pos = range(len(df))
ax1.barh(y_pos, df['Delta_AUROC'].values * 100, color=colors, edgecolor='black', lw=0.3, height=0.7)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(df['Category'].values, fontsize=11)
ax1.set_xlabel('ΔAUROC (%)')
ax1.set_title('(a) AUROC Impact', fontsize=12, pad=8)
ax1.axvline(0, color='black', lw=0.5)
ax1.grid(axis='x', alpha=0.3, ls="--")
ax1.set_axisbelow(True)
ax1.invert_yaxis()

# Add value labels — place to the right of the bar (positive x) so they
# never overlap the y-axis labels even when the bar is large negative.
xmin1, xmax1 = ax1.get_xlim()
pad1 = (xmax1 - xmin1) * 0.01
for i, (_, row) in enumerate(df.iterrows()):
    v = row['Delta_AUROC'] * 100
    if abs(v) > 0.05:
        ax1.text(pad1, i, f'{v:+.1f} %',
                va='center', ha='left', fontsize=10)

# AUPRC drop
colors2 = []
for _, row in df.iterrows():
    if row['Delta_AUPRC'] < -0.01:
        colors2.append('#e74c3c')
    elif row['Delta_AUPRC'] < -0.001:
        colors2.append('#e67e22')
    elif row['Delta_AUPRC'] < 0:
        colors2.append('#3498db')
    else:
        colors2.append('#95a5a6')

ax2.barh(y_pos, df['Delta_AUPRC'].values * 100, color=colors2, edgecolor='black', lw=0.3, height=0.7)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(df['Category'].values, fontsize=11)
ax2.set_xlabel('ΔAUPRC (%)')
ax2.set_title('(b) AUPRC Impact', fontsize=12, pad=8)
ax2.axvline(0, color='black', lw=0.5)
ax2.grid(axis='x', alpha=0.3, ls="--")
ax2.set_axisbelow(True)
ax2.invert_yaxis()

xmin2, xmax2 = ax2.get_xlim()
pad2 = (xmax2 - xmin2) * 0.01
for i, (_, row) in enumerate(df.iterrows()):
    v = row['Delta_AUPRC'] * 100
    if abs(v) > 0.05:
        ax2.text(pad2, i, f'{v:+.1f} %',
                va='center', ha='left', fontsize=10)

plt.savefig(FIG, dpi=300, bbox_inches='tight')
plt.savefig(FIG_PDF, bbox_inches='tight')
plt.close()
print(f"Saved: {FIG}")
print(f"Saved: {FIG_PDF}")
