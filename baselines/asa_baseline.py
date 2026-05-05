"""
asa_baseline.py
---------------
Clinical ASA-score-only baseline for IoD prediction.

Uses a single feature (ASA physical status score, 1-5) to train a logistic
regression model as the "current clinical standard" Layer 1 baseline.

ASA=6 (brain-dead donor) is excluded (not a surgical candidate).

Results:
  /path/to/CHD_MEDS/results/baselines/asa_test_predictions.parquet
  /path/to/CHD_MEDS/results/baselines/results_summary.csv  (appended)

Usage:
    conda activate ethos
    python -m baselines.asa_baseline
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from baselines.features import load_task, load_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

EVENTS_PATH    = Path("/path/to/CHD_MEDS/merged/events.parquet")
OUTPUT_DIR     = Path("/path/to/CHD_MEDS/results/baselines")
SUMMARY_PATH   = OUTPUT_DIR / "results_summary.csv"
ASA_CODE       = "ENCOUNTER//AN//ASA_SCORE"


def load_asa_scores(task: pd.DataFrame) -> pd.DataFrame:
    """
    Extract ASA score per (subject_id, encounter_csn) from merged events.
    Keeps the single ASA score recorded at prediction_time (In OR) or before.
    """
    log.info("Loading ASA scores from merged events …")
    events = pd.read_parquet(
        EVENTS_PATH,
        engine="pyarrow",
        columns=["patient_id", "time", "code", "numeric_value"],
    )
    events["time"] = pd.to_datetime(events["time"])
    events["subject_id"] = (
        pd.to_numeric(events["patient_id"].str.lstrip("C").str.strip(), errors="coerce")
        .astype("Int64")
    )

    asa = events[events["code"] == ASA_CODE].copy()
    log.info(f"  Raw ASA rows: {len(asa):,}")

    # Join with task to get prediction_time per encounter
    pt = task[["subject_id", "encounter_csn", "prediction_time"]].copy()
    asa = asa.merge(pt, on="subject_id", how="inner")

    # Keep only rows at or before In OR
    asa = asa[asa["time"].isna() | (asa["time"] <= asa["prediction_time"])]

    # Take the last (most recent) ASA score before prediction_time
    asa = (
        asa.sort_values("time")
        .groupby(["subject_id", "encounter_csn"], as_index=False)["numeric_value"]
        .last()
        .rename(columns={"numeric_value": "asa_score"})
    )

    # Exclude ASA=6 (brain-dead donors)
    asa = asa[asa["asa_score"] != 6]
    log.info(f"  Encounters with valid ASA score: {len(asa):,}")
    return asa


def build_dataset(task: pd.DataFrame, splits: pd.DataFrame, asa: pd.DataFrame) -> pd.DataFrame:
    """Merge task labels + splits + ASA scores."""
    task["boolean_value"] = task["boolean_value"].astype(int)
    df = task[["subject_id", "encounter_csn", "boolean_value"]].merge(
        splits[["subject_id", "split"]], on="subject_id", how="inner"
    )
    df = df.merge(asa, on=["subject_id", "encounter_csn"], how="inner")
    log.info(f"  Final dataset size: {len(df):,} encounters, {df['boolean_value'].sum()} IoD+")
    return df


def evaluate(y_true, y_prob, split, model_name):
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    n_pos = int(y_true.sum())
    n_tot = len(y_true)
    log.info(
        f"  [{split}] {model_name}: AUROC={auroc:.4f}  AUPRC={auprc:.4f}  Brier={brier:.4f}"
        f"  (n={n_tot:,}, pos={n_pos:,}, {n_pos/n_tot*100:.1f}%)"
    )
    return {"model": model_name, "split": split, "auroc": auroc, "auprc": auprc,
            "brier": brier, "n_total": n_tot, "n_positive": n_pos}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    task   = load_task()
    splits = load_splits()
    asa    = load_asa_scores(task)
    df     = build_dataset(task, splits, asa)

    train = df[df["split"] == "train"]
    val   = df[df["split"] == "val"]
    test  = df[df["split"] == "test"]

    log.info(f"Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

    X_train = train[["asa_score"]].values
    y_train = train["boolean_value"].values
    X_val   = val[["asa_score"]].values
    y_val   = val["boolean_value"].values
    X_test  = test[["asa_score"]].values
    y_test  = test["boolean_value"].values

    # Logistic regression — simple calibrated probability from ASA score
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    log.info(f"  ASA coef={model.coef_[0][0]:.4f}, intercept={model.intercept_[0]:.4f}")

    results = []
    for X, y, split_name in [(X_train, y_train, "train"), (X_val, y_val, "val"), (X_test, y_test, "test")]:
        y_prob = model.predict_proba(X)[:, 1]
        results.append(evaluate(y, y_prob, split_name, "ASA (clinical)"))

    # Save test predictions
    y_prob_test = model.predict_proba(X_test)[:, 1]
    pred_df = test[["subject_id", "encounter_csn", "boolean_value"]].copy()
    pred_df = pred_df.rename(columns={"boolean_value": "y_true"})
    pred_df["y_prob"] = y_prob_test
    out_path = OUTPUT_DIR / "asa_test_predictions.parquet"
    pred_df.to_parquet(out_path, index=False)
    log.info(f"  Saved test predictions → {out_path}")

    # Append to results_summary.csv
    new_rows = pd.DataFrame(results)
    if SUMMARY_PATH.exists():
        summary = pd.read_csv(SUMMARY_PATH)
        # Remove existing ASA rows if any
        summary = summary[summary["model"] != "ASA (clinical)"]
        summary = pd.concat([summary, new_rows], ignore_index=True)
    else:
        summary = new_rows
    summary.to_csv(SUMMARY_PATH, index=False)
    log.info(f"  Updated results summary → {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
