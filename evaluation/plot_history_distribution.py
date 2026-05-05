"""Plot history length distribution for paper Figure."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/path/to/CHD_MEDS/results/evaluation/history_length_distribution.png"

task = pd.read_parquet("/path/to/CHD_MEDS/outcome/iod_task.parquet")
events = pd.read_parquet("/path/to/CHD_MEDS/merged/events.parquet")
splits = pd.read_parquet("/path/to/CHD_MEDS/splits/splits.parquet")

test_pids = set(splits[splits["split"] == "test"]["subject_id"])
task["pid_int"] = task["patient_id"].str.replace("C", "").astype(int)
test_task = task[task["pid_int"].isin(test_pids)].copy()

events["pid_int"] = events["patient_id"].str.replace("C", "").astype(int)
first_ev = events.groupby("pid_int")["time"].min().reset_index().rename(columns={"time": "t0"})
test_task = test_task.merge(first_ev, on="pid_int", how="left")
test_task["hdays"] = (
    pd.to_datetime(test_task["prediction_time"]) - pd.to_datetime(test_task["t0"])
).dt.total_seconds() / 86400

v = test_task.dropna(subset=["hdays"])
iod_p = v[v["boolean_value"] == True]["hdays"]
iod_n = v[v["boolean_value"] == False]["hdays"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel A: histogram
ax = axes[0]
ax.hist(v["hdays"], bins=100, range=(0, 5000), color="#4e79a7", alpha=0.8,
        edgecolor="white", linewidth=0.3)
for w, c in [(7, "#e15759"), (30, "#f28e2b"), (90, "#59a14f"), (365, "#b07aa1")]:
    ax.axvline(w, color=c, ls="--", lw=1.5, label=f"{w}d")
ax.set_xlabel("Days of history before OR entry")
ax.set_ylabel("Number of encounters")
ax.set_title("(A) Distribution of Pre-operative History Length\n(Test set, n=37,948)",
             fontweight="bold")
ax.legend(fontsize=9, title="Window cutoffs")
ax.set_xlim(0, 5000)

# Panel B: CDF
ax2 = axes[1]
dr = np.arange(0, 3650, 1)
cdf = [(v["hdays"] >= d).mean() * 100 for d in dr]
ax2.plot(dr, cdf, color="#4e79a7", lw=2)
for w, c in [(7, "#e15759"), (30, "#f28e2b"), (90, "#59a14f"), (365, "#b07aa1")]:
    p = (v["hdays"] >= w).mean() * 100
    ax2.axvline(w, color=c, ls="--", lw=1.5)
    ax2.plot(w, p, "o", color=c, ms=8, zorder=5)
    ax2.annotate(f"{p:.1f}%", (w + 30, p + 1), fontsize=9, color=c, fontweight="bold")
ax2.set_xlabel("Minimum history length (days)")
ax2.set_ylabel("% encounters with >= X days")
ax2.set_title("(B) Cumulative Distribution", fontweight="bold")
ax2.set_xlim(0, 2000)
ax2.set_ylim(0, 105)
ax2.grid(alpha=0.3)

# Panel C: IoD+ vs IoD-
ax3 = axes[2]
bins = np.linspace(0, 4000, 60)
ax3.hist(iod_n, bins=bins, density=True, alpha=0.6, color="#4e79a7",
         label=f"IoD- (n={len(iod_n):,})", edgecolor="white", linewidth=0.3)
ax3.hist(iod_p, bins=bins, density=True, alpha=0.7, color="#e15759",
         label=f"IoD+ (n={len(iod_p):,})", edgecolor="white", linewidth=0.3)
ax3.axvline(iod_n.median(), color="#4e79a7", ls=":", lw=2,
            label=f"IoD- median={iod_n.median():.0f}d")
ax3.axvline(iod_p.median(), color="#e15759", ls=":", lw=2,
            label=f"IoD+ median={iod_p.median():.0f}d")
ax3.set_xlabel("Days of history before OR entry")
ax3.set_ylabel("Density")
ax3.set_title("(C) History Length: IoD+ vs IoD-", fontweight="bold")
ax3.legend(fontsize=9)
ax3.set_xlim(0, 4000)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}")
