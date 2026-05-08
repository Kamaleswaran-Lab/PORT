"""Figure 6: Clinical decision support analysis.

Four panels translating PORT predictions into actionable clinical guidance:

(a) Risk-stratified IoD rate per PORT decile, with recommended clinical
    action tier for each stratum.
(b) PORT-vs-ASA discordance: 2x2 contingency table (PORT high/low x ASA
    low/high) showing IoD rate per cell. Identifies the patient subset
    where PORT adds value beyond the ASA score alone.
(c) Modifiable risk component for the patients with documented pre-operative
    vasoactive use (n=10,708): baseline predicted IoD risk vs. risk reduction
    after simulated vasoactive removal. High-baseline-risk patients benefit
    most from medical optimization.
(d) Decision-curve analysis with threshold-keyed clinical actions annotated.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Rectangle

R_V4   = Path("/path/to/CHD_MEDS/results/baselines")
R_LSTM = Path("/path/to/CHD_MEDS/results/baselines_tuned")
CF     = Path("/path/to/CHD_MEDS/results/evaluation/counterfactual/counterfactual_predictions.parquet")
TASK   = Path("/path/to/CHD_MEDS/outcome/iod_task.parquet")
ANP    = Path("/path/to/CHOA_RAW_TABLES/CHOA_DATA_Tables_CHD/DR15201_AN_Patients.rpt")
FIG    = Path("paper/overleaf/figures/patient_mechanism")  # extension added below

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
    "legend.fontsize": 10,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PORT_BLUE = "#1F3A5F"
LSTM_PURPLE = "#9467BD"
RED       = "#C0392B"
GRAY      = "#888888"
GREEN     = "#2E864D"

# ── Load data ──
pp = pd.read_parquet(R_V4 / "ethos_finetune_lora_test_predictions_v4_lora_s123.parquet")
lstm = pd.read_parquet(R_LSTM / "lstm_tuned_test_predictions.parquet")
cf = pd.read_parquet(CF)

an = pd.read_csv(ANP, sep="|", usecols=["C MRN","ASA PS Score","Encounter CSN"], dtype=str)
an["sid"] = pd.to_numeric(an["C MRN"].str.lstrip("C"), errors="coerce")
an["asa"] = pd.to_numeric(an["ASA PS Score"], errors="coerce")
an["enc_csn"] = pd.to_numeric(an["Encounter CSN"], errors="coerce")
an = an.dropna(subset=["sid","asa","enc_csn"]).drop_duplicates(["sid","enc_csn"])

task = pd.read_parquet(TASK)
task["sid"] = task.patient_id.str.lstrip("C").astype(int)
task = task.merge(an[["sid","enc_csn","asa"]], left_on=["sid","encounter_csn"],
                  right_on=["sid","enc_csn"], how="left")

pp["prediction_time"] = pd.to_datetime(pp.prediction_time_us, unit="us")
pp = pp.merge(task[["sid","prediction_time","asa"]], left_on=["subject_id","prediction_time"],
              right_on=["sid","prediction_time"], how="left")
pp_with_asa = pp.dropna(subset=["asa"]).copy()

# ── Figure ──
fig = plt.figure(figsize=(14, 12))
gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.32)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

# ══════════════════════════════════════════════════════════════════════════════
# (a) Risk-stratified IoD rate + action tier
# ══════════════════════════════════════════════════════════════════════════════
strata = [(0.000, 0.005, "P < 0.5%",      "#bdc3c7"),
          (0.005, 0.020, "0.5 – 2 %",     "#7fb069"),
          (0.020, 0.050, "2 – 5 %",       "#f4a261"),
          (0.050, 0.100, "5 – 10 %",      "#e76f51"),
          (0.100, 1.001, "> 10 %",        "#9c2a2f")]

s_labels, s_iod_rate, s_n, s_pos, s_color = [], [], [], [], []
for lo, hi, lbl, color in strata:
    sub = pp[(pp.y_prob >= lo) & (pp.y_prob < hi)]
    s_labels.append(lbl); s_color.append(color)
    s_n.append(len(sub)); s_pos.append(int(sub.y_true.sum()))
    s_iod_rate.append(100 * sub.y_true.mean() if len(sub) > 0 else 0.0)

x = np.arange(len(strata))
bars = ax_a.bar(x, s_iod_rate, color=s_color, edgecolor="black", lw=0.6, width=0.7,
                label="PORT")
for xi, rate, n, pos in zip(x, s_iod_rate, s_n, s_pos):
    ax_a.text(xi, rate + 0.15, f"{rate:.1f}%", ha="center", va="bottom",
              fontsize=11, fontweight="bold")
    ax_a.text(xi, rate + 0.85, f"n={n:,}\n({pos} IoD+)", ha="center", va="bottom",
              fontsize=8, color="0.35")

# BiLSTM IoD rate at the SAME PORT-defined strata boundaries (same patients
# binned by their LSTM predicted probability) for direct comparison.
y_lstm = lstm.y_true.astype(int).values
p_lstm = lstm.y_prob.values
lstm_rates = []
for lo, hi, *_ in strata:
    sub = (p_lstm >= lo) & (p_lstm < hi)
    lstm_rates.append(100 * y_lstm[sub].mean() if sub.sum() > 0 else 0.0)
ax_a.plot(x, lstm_rates, "-^", color=LSTM_PURPLE, lw=1.6, ms=8, mec="white",
          mew=0.8, label="BiLSTM (same thresholds)", zorder=5)

ax_a.set_xticks(x); ax_a.set_xticklabels(s_labels, fontsize=10)
ax_a.set_ylim(0, max(s_iod_rate) * 1.35)
ax_a.set_ylabel("Observed IoD+ rate (%)")
ax_a.set_xlabel("Predicted-risk stratum")
ax_a.set_title("(a) Risk concentration across strata", fontsize=12, pad=8)
ax_a.grid(axis="y", alpha=0.3, ls="--", zorder=0); ax_a.set_axisbelow(True)
ax_a.legend(loc="upper left", fontsize=10)

# ══════════════════════════════════════════════════════════════════════════════
# (b) PORT vs ASA discordance (2x2 contingency)
# ══════════════════════════════════════════════════════════════════════════════
PORT_HIGH = 0.05  # 5% threshold (matches "cardiology consult" boundary)
df = pp_with_asa.copy()
df["asa_high"] = df["asa"] >= 3
df["port_high"] = df.y_prob >= PORT_HIGH

cells = []
for ah_lbl, ah_val in [("ASA I-II (low)", False), ("ASA III+ (high)", True)]:
    for ph_lbl, ph_val in [("PORT < 5%", False), ("PORT ≥ 5%", True)]:
        sub = df[(df.asa_high == ah_val) & (df.port_high == ph_val)]
        cells.append({
            "asa_high": ah_val, "port_high": ph_val,
            "asa_lbl": ah_lbl, "port_lbl": ph_lbl,
            "n": len(sub), "pos": int(sub.y_true.sum()),
            "rate": 100 * sub.y_true.mean() if len(sub) > 0 else 0.0,
        })

# 2x2 grid: rows = ASA high/low, cols = PORT high/low
heat = np.array([[cells[1]["rate"], cells[0]["rate"]],   # ASA low: PORT high, PORT low
                 [cells[3]["rate"], cells[2]["rate"]]])  # ASA high: PORT high, PORT low
n_grid = np.array([[cells[1]["n"], cells[0]["n"]],
                   [cells[3]["n"], cells[2]["n"]]])
pos_grid = np.array([[cells[1]["pos"], cells[0]["pos"]],
                     [cells[3]["pos"], cells[2]["pos"]]])

im = ax_b.imshow(heat, cmap="OrRd", vmin=0, vmax=max(heat.max() * 1.05, 5))
for i in range(2):
    for j in range(2):
        v = heat[i, j]; n = n_grid[i, j]; p = pos_grid[i, j]
        col = "white" if v > heat.max() * 0.55 else "black"
        ax_b.text(j, i - 0.05, f"{v:.1f}%", ha="center", va="center",
                  fontsize=18, fontweight="bold", color=col)
        ax_b.text(j, i + 0.22, f"n={n:,} ({p} IoD+)", ha="center", va="center",
                  fontsize=10, color=col)

ax_b.set_xticks([0, 1]); ax_b.set_yticks([0, 1])
ax_b.set_xticklabels(["PORT ≥ 5 %", "PORT < 5 %"])
ax_b.set_yticklabels(["ASA I-II", "ASA III-V"])
ax_b.set_xlabel("PORT prediction"); ax_b.set_ylabel("ASA classification")
ax_b.set_title("(b) PORT vs ASA: discordance reveals added value", fontsize=12, pad=8)
# Box the "PORT high, ASA low" cell — where PORT adds value
rect = Rectangle((-0.5, -0.5), 1.0, 1.0, linewidth=2.5, edgecolor=RED,
                 facecolor="none", zorder=5)
ax_b.add_patch(rect)
# "PORT-only flagged" small label inside the top-left (PORT≥5%, ASA I-II) cell
ax_b.text(0.0, -0.43, "PORT-only flagged\n(ASA missed)", ha="center", va="top",
          fontsize=9, color=RED, fontweight="bold")
ax_b.spines["top"].set_visible(False); ax_b.spines["right"].set_visible(False)
ax_b.spines["left"].set_visible(False); ax_b.spines["bottom"].set_visible(False)

# ══════════════════════════════════════════════════════════════════════════════
# (c) Modifiable risk among vasoactive-using patients
# ══════════════════════════════════════════════════════════════════════════════
sub_v = cf[cf.on_vasoactive].copy()
sub_v["delta"] = sub_v.p_base - sub_v.p_no_vasoactives
neg = sub_v[sub_v.y_true == 0]; pos = sub_v[sub_v.y_true == 1]
ax_c.scatter(neg.p_base, neg.delta, s=4, c=GRAY, alpha=0.35, rasterized=True,
             label=f"IoD− (n={len(neg):,})")
ax_c.scatter(pos.p_base, pos.delta, s=14, c=RED, alpha=0.85,
             edgecolors="white", linewidths=0.4, label=f"IoD+ (n={len(pos):,})")
ax_c.axhline(0, color="0.4", lw=0.7, ls=":", zorder=0)
# Median ΔP per baseline-risk decile (shows trend among vasoactive patients)
sub_v["dec"] = pd.qcut(sub_v.p_base, q=10, labels=False, duplicates="drop")
med_by_dec = sub_v.groupby("dec").agg(p_med=("p_base","median"),
                                      d_med=("delta","median"))
ax_c.plot(med_by_dec.p_med, med_by_dec.d_med, "-D", color=PORT_BLUE,
          lw=2.0, ms=6, mec="white", mew=0.8, label="Median ΔP per decile")
ax_c.set_xscale("log"); ax_c.set_xlim(1e-4, 1)
ax_c.set_xlabel(r"Baseline PORT risk $\hat p_\text{base}$ (log scale)")
ax_c.set_ylabel(r"Modifiable component  $\hat p_\text{base} - \hat p_\text{no-vaso}$")
ax_c.set_title("(c) Modifiable risk: vasoactive removal in n=10,708 affected encounters",
               fontsize=12, pad=8)
ax_c.grid(True, alpha=0.3, ls="--", zorder=0); ax_c.set_axisbelow(True)
ax_c.legend(loc="upper left", fontsize=9)

# ══════════════════════════════════════════════════════════════════════════════
# (d) Cumulative IoD capture by PORT-ranked screening
# ══════════════════════════════════════════════════════════════════════════════
y_full = pp.y_true.astype(int).values
p_full = pp.y_prob.values
N = len(y_full); P = int(y_full.sum())

# Sort by PORT score descending
order = np.argsort(-p_full)
y_sorted = y_full[order]
cum_cap = np.cumsum(y_sorted) / P            # cumulative fraction of IoD+ captured
cum_pop = (np.arange(N) + 1) / N             # cumulative fraction of population screened

# BiLSTM cumulative capture, computed on its own test split.
order_l = np.argsort(-p_lstm)
y_l_sorted = y_lstm[order_l]
P_l = int(y_lstm.sum()); N_l = len(y_lstm)
cum_cap_l = np.cumsum(y_l_sorted) / P_l
cum_pop_l = (np.arange(N_l) + 1) / N_l

ax_d.plot(cum_pop * 100, cum_cap * 100, color=PORT_BLUE, lw=2.4, label="PORT")
ax_d.plot(cum_pop_l * 100, cum_cap_l * 100, color=LSTM_PURPLE, lw=1.8, ls="-",
          label="BiLSTM")
ax_d.plot([0, 100], [0, 100], color="0.55", lw=1.0, ls="--", label="Random ranking")

# Annotate selected screening fractions
def at_population_fraction(frac):
    idx = int(np.ceil(frac * N)) - 1
    return cum_pop[idx] * 100, cum_cap[idx] * 100

annot = [
    (0.088, f"Top 8.8 %: {at_population_fraction(0.088)[1]:.0f}%\nof IoD events"),
    (0.20,  f"Top 20 %: {at_population_fraction(0.20)[1]:.0f}%"),
    (0.50,  f"Top 50 %: {at_population_fraction(0.50)[1]:.0f}%"),
]
for frac, txt in annot:
    xp, yp = at_population_fraction(frac)
    ax_d.scatter([xp], [yp], color=PORT_BLUE, s=42, edgecolor="white", lw=1.0, zorder=5)
    ax_d.annotate(txt, xy=(xp, yp), xytext=(xp + 8, yp - 12),
                  fontsize=9, color=PORT_BLUE, fontweight="bold",
                  arrowprops=dict(arrowstyle="-", color="0.6", lw=0.6))

ax_d.set_xlim(0, 100); ax_d.set_ylim(0, 100)
ax_d.set_xlabel("Patients screened, ordered by PORT score (%)")
ax_d.set_ylabel("IoD+ events captured (%)")
ax_d.set_title("(d) Cumulative IoD capture by PORT-ranked screening", fontsize=12, pad=8)
ax_d.legend(loc="lower right", fontsize=10)
ax_d.grid(True, alpha=0.3, ls="--", zorder=0); ax_d.set_axisbelow(True)

# ── Save ──
fig.savefig(str(FIG) + ".png", dpi=300, bbox_inches="tight")
fig.savefig(str(FIG) + ".pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved {FIG}.png/.pdf")
print(f"\nStratum summary (panel a):")
for lbl, n, pos, rate in zip(s_labels, s_n, s_pos, s_iod_rate):
    print(f"  {lbl:<14s}  n={n:>7,d}  IoD+={pos:>4d}  rate={rate:5.2f}%")
print(f"\nDiscordance summary (panel b):")
for c in cells:
    print(f"  {c['asa_lbl']} & {c['port_lbl']:<11s}  n={c['n']:>7,d}  IoD+={c['pos']:>4d}  rate={c['rate']:5.2f}%")
