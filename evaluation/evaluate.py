"""
evaluate.py
-----------
Unified evaluation and visualization for all IoD prediction models.

Loads prediction parquets from each model and produces:
  1. Summary table  — AUROC / AUPRC / Brier score per model
  2. ROC curves     — all models on one plot
  3. PR curves      — all models on one plot
  4. Calibration curves — reliability diagrams
  5. Subgroup analysis  — by ASA class, procedure type

Input parquets (y_true, y_prob columns):
  /path/to/CHD_MEDS/results/baselines/logreg_manual_test_predictions.parquet
  /path/to/CHD_MEDS/results/baselines/logreg_meds_test_predictions.parquet
  /path/to/CHD_MEDS/results/baselines/xgb_manual_test_predictions.parquet
  /path/to/CHD_MEDS/results/baselines/xgb_meds_test_predictions.parquet
  /path/to/CHD_MEDS/results/baselines/lstm_test_predictions.parquet
  /path/to/CHD_MEDS/results/ethos/iod/<latest>.parquet  (after ETHOS inference)

Usage:
    conda activate tccc
    python evaluation/evaluate.py [--output_dir DIR]
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

RESULTS_ROOT   = Path("/path/to/CHD_MEDS/results")
BASELINES_DIR  = RESULTS_ROOT / "baselines"
ETHOS_DIR      = RESULTS_ROOT / "ethos" / "iod"
TASK_PATH      = Path("/path/to/CHD_MEDS/outcome/iod_task.parquet")
DEFAULT_OUTPUT = RESULTS_ROOT / "evaluation"

# Model display names — models stored as (path, y_prob_col) tuples
# New combined format: test_preds_{feature_set}.parquet with multiple prob columns
MODELS = {
    "ASA (clinical)":  (BASELINES_DIR / "asa_test_predictions.parquet",  "y_prob"),
    "LR (manual)":     (BASELINES_DIR / "test_preds_manual.parquet",      "prob_lr_manual"),
    "LR (MEDS)":       (BASELINES_DIR / "test_preds_meds.parquet",        "prob_lr_meds"),
    "XGB (manual)":    (BASELINES_DIR / "test_preds_manual.parquet",      "prob_xgb_manual"),
    "XGB (MEDS)":      (BASELINES_DIR / "test_preds_meds.parquet",        "prob_xgb_meds"),
    "LSTM":            (BASELINES_DIR / "lstm_test_predictions.parquet",  "y_prob"),
    "ETHOS (zero-shot)":   (BASELINES_DIR / "ethos_test_predictions.parquet",          "y_prob"),
    "ETHOS (fine-tuned)":  (BASELINES_DIR / "ethos_finetune_test_predictions.parquet",  "y_prob"),
}

COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
]


# ── loaders ───────────────────────────────────────────────────────────────────

def load_predictions(path: Path, y_prob_col: str = "y_prob") -> pd.DataFrame | None:
    if not path.exists():
        log.warning(f"  Missing: {path}")
        return None
    raw = pd.read_parquet(path, engine="pyarrow")
    assert "y_true" in raw.columns, f"Missing y_true in {path}"
    assert y_prob_col in raw.columns, f"Missing {y_prob_col} in {path}"
    # Normalise to standard (y_true, y_prob) + optional id columns
    df = raw[["y_true", y_prob_col]].rename(columns={y_prob_col: "y_prob"})
    for col in ("subject_id", "encounter_csn"):
        if col in raw.columns:
            df[col] = raw[col]
    return df


def load_all_predictions() -> dict[str, pd.DataFrame]:
    preds = {}
    for name, (path, y_prob_col) in MODELS.items():
        df = load_predictions(path, y_prob_col)
        if df is not None:
            preds[name] = df
            log.info(f"  Loaded {name}: {path.name} (col={y_prob_col})")

    # ETHOS: prefer the iod_predictions.parquet from analyze_infer if present
    ethos_pred_path = BASELINES_DIR / "ethos_test_predictions.parquet"
    if not ethos_pred_path.exists() and ETHOS_DIR.exists():
        parquets = sorted(ETHOS_DIR.glob("*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
        if parquets:
            df = load_predictions(parquets[0])
            if df is not None:
                preds["ETHOS (zero-shot)"] = df
                log.info(f"  Loaded ETHOS predictions: {parquets[0].name}")

    return preds


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) via equal-width bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece / len(y_true))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "auroc":   roc_auc_score(y_true, y_prob),
        "auprc":   average_precision_score(y_true, y_prob),
        "brier":   brier_score_loss(y_true, y_prob),
        "ece":     compute_ece(y_true, y_prob),
        "n_total": len(y_true),
        "n_pos":   int(y_true.sum()),
        "prev_pct": float(y_true.mean() * 100),
    }


def build_summary_table(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in preds.items():
        m = compute_metrics(df["y_true"].values, df["y_prob"].values)
        rows.append({"Model": name, **m})
    summary = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    # Reorder columns for readability
    col_order = ["Model", "auroc", "auprc", "brier", "ece", "n_total", "n_pos", "prev_pct"]
    summary = summary[[c for c in col_order if c in summary.columns]]
    return summary


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_roc(preds: dict, output_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")

    for (name, df), color in zip(preds.items(), COLORS):
        fpr, tpr, _ = roc_curve(df["y_true"], df["y_prob"])
        auroc = roc_auc_score(df["y_true"], df["y_prob"])
        ax.plot(fpr, tpr, color=color, lw=1.8, label=f"{name} (AUROC={auroc:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — IoD Prediction", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(f"  ROC curve saved → {output_path}")


def plot_pr(preds: dict, output_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))

    # Baseline: random classifier AUPRC ≈ prevalence
    prev = next(iter(preds.values()))["y_true"].mean()
    ax.axhline(prev, color="k", linestyle="--", lw=0.8, label=f"Random (prev={prev*100:.1f}%)")

    for (name, df), color in zip(preds.items(), COLORS):
        prec, rec, _ = precision_recall_curve(df["y_true"], df["y_prob"])
        auprc = average_precision_score(df["y_true"], df["y_prob"])
        ax.plot(rec, prec, color=color, lw=1.8, label=f"{name} (AUPRC={auprc:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — IoD Prediction", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(f"  PR curve saved → {output_path}")


def plot_calibration(preds: dict, output_path: Path, n_bins: int = 10):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")

    for (name, df), color in zip(preds.items(), COLORS):
        fraction_of_positives, mean_predicted_value = calibration_curve(
            df["y_true"], df["y_prob"], n_bins=n_bins, strategy="quantile"
        )
        ax.plot(mean_predicted_value, fraction_of_positives,
                "o-", color=color, lw=1.8, ms=5, label=name)

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Curves — IoD Prediction", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(f"  Calibration curve saved → {output_path}")


def plot_score_distribution(preds: dict, output_path: Path):
    """Histogram of predicted probabilities by true label for each model."""
    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (name, df) in zip(axes, preds.items()):
        neg = df[df["y_true"] == 0]["y_prob"]
        pos = df[df["y_true"] == 1]["y_prob"]
        ax.hist(neg, bins=30, alpha=0.6, color="steelblue", label="IoD−", density=True)
        ax.hist(pos, bins=30, alpha=0.6, color="firebrick", label="IoD+", density=True)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted Probability")
        ax.legend(fontsize=8)

    fig.suptitle("Score Distributions by True Label", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(f"  Score distributions saved → {output_path}")


# ── subgroup analysis ─────────────────────────────────────────────────────────

def subgroup_analysis(preds: dict, task: pd.DataFrame, output_path: Path):
    """
    AUROC broken down by ASA class (1–5) for each model.
    Requires preds to have subject_id + encounter_csn columns.
    """
    task_sub = task[["subject_id", "encounter_csn", "boolean_value"]].copy()

    # Load ASA scores from events — use first model's pred file as base roster
    events_path = Path("/path/to/CHD_MEDS/merged/events.parquet")
    if not events_path.exists():
        log.warning("  events.parquet not found; skipping subgroup analysis")
        return

    events = pd.read_parquet(events_path, engine="pyarrow",
                             filters=[("code", "=", "ENCOUNTER//AN//ASA_SCORE")])
    asa = (
        events[events["code"] == "ENCOUNTER//AN//ASA_SCORE"]
        [["subject_id", "numeric_value"]]
        .dropna(subset=["numeric_value"])
        .rename(columns={"numeric_value": "asa_score"})
    )
    # subject_id in events is int; task may have string — align types
    asa["subject_id"] = asa["subject_id"].astype("int64")
    task_sub["subject_id"] = task_sub["subject_id"].astype("int64")

    # Group ASA into integer class
    asa["asa_class"] = asa["asa_score"].round().clip(1, 5).astype(int)
    asa = asa.drop_duplicates("subject_id")

    rows = []
    for name, df in preds.items():
        if "subject_id" not in df.columns:
            continue
        df2 = df.merge(asa[["subject_id", "asa_class"]], on="subject_id", how="left")
        for asa_cls in sorted(df2["asa_class"].dropna().unique()):
            sub = df2[df2["asa_class"] == asa_cls]
            if sub["y_true"].sum() < 5:
                continue  # skip if too few positives
            auroc = roc_auc_score(sub["y_true"], sub["y_prob"])
            rows.append({
                "Model": name, "ASA_class": int(asa_cls),
                "n": len(sub), "n_pos": int(sub["y_true"].sum()),
                "auroc": auroc,
            })

    if not rows:
        log.warning("  No subgroup data; skipping plot")
        return

    sg = pd.DataFrame(rows)
    sg.to_csv(output_path.with_suffix(".csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for (name, grp), color in zip(sg.groupby("Model"), COLORS):
        ax.plot(grp["ASA_class"], grp["auroc"], "o-", color=color, lw=1.8, ms=6, label=name)

    ax.set_xlabel("ASA Class", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("Subgroup AUROC by ASA Class", fontsize=13)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.5, color="k", linestyle="--", lw=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(f"  Subgroup plot saved → {output_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== IoD Prediction Evaluation ===")
    log.info("Loading predictions …")
    preds = load_all_predictions()

    if not preds:
        log.error("No prediction files found. Run baselines first.")
        return

    log.info(f"  Models loaded: {list(preds.keys())}")

    # Summary table
    summary = build_summary_table(preds)
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info("\n" + summary.to_string(index=False))
    log.info(f"\n  Summary saved → {summary_path}")

    # Plots
    log.info("\nGenerating plots …")
    plot_roc(preds,         output_dir / "roc_curves.png")
    plot_pr(preds,          output_dir / "pr_curves.png")
    plot_calibration(preds, output_dir / "calibration_curves.png")
    plot_score_distribution(preds, output_dir / "score_distributions.png")

    # Subgroup analysis (ASA class)
    task = pd.read_parquet(TASK_PATH, engine="pyarrow")
    task["subject_id"] = pd.to_numeric(
        task["patient_id"].str.lstrip("C").str.strip(), errors="coerce"
    ).astype("int64")
    subgroup_analysis(preds, task, output_dir / "subgroup_asa.png")

    log.info(f"\n=== Done. All outputs in {output_dir} ===")


if __name__ == "__main__":
    main()
