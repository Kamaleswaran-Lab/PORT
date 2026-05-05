"""
aggregate_zeroshot.py
------------------------
Re-aggregate zero-shot trajectory inference outputs with the correct
IoD stop-token criterion. The previous analyze_infer.py used a previous
hardcoded token list which did not recognize ATC tokens (e.g.,
ATC//SFX//A24 = epinephrine), producing essentially random predictions.

For: per-encounter IoD probability =
  fraction of trajectories whose stop_reason == 'token_of_interest'
(token_of_interest fires only when the trajectory terminates at a stop
token, which in vocab = CPR + vasoactive ATC SFX + ATC//C01 +
arterial-line LDA).

Output:
  /path/to/CHD_MEDS/results/baselines/
    ethos_zeroshot_test_predictions_v4_fixed.parquet

Then computes AUROC, AUPRC, Brier, ECE on the encounter test split.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

INFER_DIR  = Path("/path/to/CHD_MEDS/results/ethos/iod")
TASK_PATH  = Path("/path/to/CHD_MEDS/outcome/iod_task.parquet")
OUT_PATH   = Path("/path/to/CHD_MEDS/results/baselines/"
                  "ethos_zeroshot_test_predictions_v4_fixed.parquet")


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bi = np.clip(np.digitize(y_prob, bin_edges) - 1, 0, n_bins - 1)
    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        m = bi == b
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(y_prob[m].mean() - y_true[m].mean())
    return float(ece)


def main():
    files = sorted(INFER_DIR.glob("samples_*.parquet"))
    log.info(f"Reading {len(files)} trajectory shards …")

    cols = ["data_idx", "patient_id", "prediction_time", "stop_reason"]
    chunks = []
    for i, fp in enumerate(files):
        chunks.append(pd.read_parquet(fp, columns=cols))
        if (i + 1) % 500 == 0:
            log.info(f"  loaded {i+1}/{len(files)} shards")
    df = pd.concat(chunks, ignore_index=True)
    log.info(f"  total rows: {len(df):,}")

    # Per-encounter aggregation: n_traj, n_iod (token_of_interest stops)
    log.info("Aggregating per encounter (data_idx) …")
    agg = df.groupby("data_idx").agg(
        n_traj=("stop_reason", "size"),
        n_iod=("stop_reason", lambda x: (x == "token_of_interest").sum()),
        patient_id=("patient_id", "first"),
        prediction_time=("prediction_time", "first"),
    ).reset_index()
    agg["y_prob"] = agg["n_iod"] / agg["n_traj"]
    log.info(f"  encounters: {len(agg):,}")
    log.info(f"  trajectories per encounter: median={int(agg['n_traj'].median())}  "
             f"min={int(agg['n_traj'].min())}  max={int(agg['n_traj'].max())}")

    # Merge ground truth from task parquet
    log.info("Merging ground truth from iod_task.parquet …")
    task = pd.read_parquet(TASK_PATH)
    task["subject_id"] = task["patient_id"].str.lstrip("C").astype(int)
    task["prediction_time_us"] = (task["prediction_time"].astype("int64") // 1000)

    # The trajectory data has patient_id (int) and prediction_time (datetime ns)
    # Convert agg's prediction_time to us for matching with task
    agg["prediction_time_us"] = (agg["prediction_time"].astype("int64") // 1000)
    log.info(f"  agg prediction_time dtype: {agg['prediction_time'].dtype}")
    log.info(f"  task prediction_time_us range: [{task['prediction_time_us'].min()}, {task['prediction_time_us'].max()}]")
    log.info(f"  agg prediction_time_us range: [{agg['prediction_time_us'].min()}, {agg['prediction_time_us'].max()}]")

    # patient_id in trajectory is int; in task it's "Cnnnnn"
    merged = agg.merge(
        task[["subject_id", "prediction_time_us", "boolean_value", "encounter_csn"]].rename(
            columns={"subject_id": "patient_id"}
        ),
        on=["patient_id", "prediction_time_us"],
        how="inner",
    )
    log.info(f"  merged rows: {len(merged):,}  (lost {len(agg) - len(merged):,})")

    if len(merged) == 0:
        log.error("No matches between trajectory data and task labels. Check id/time conventions.")
        log.info(f"  sample agg pid+ptus: {agg[['patient_id','prediction_time_us']].head(3).values.tolist()}")
        log.info(f"  sample task pid+ptus: {task[['subject_id','prediction_time_us']].head(3).values.tolist()}")
        raise SystemExit(1)

    y_true = merged["boolean_value"].astype(int).values
    y_prob = merged["y_prob"].values

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ece   = expected_calibration_error(y_true, y_prob)

    log.info("")
    log.info(f"=== Zero-shot (fixed aggregation) on n={len(merged):,} encounters ===")
    log.info(f"  IoD+: {y_true.sum():,} ({y_true.mean()*100:.2f}%)")
    log.info(f"  AUROC = {auroc:.4f}")
    log.info(f"  AUPRC = {auprc:.4f}  (random baseline: {y_true.mean():.4f})")
    log.info(f"  Brier = {brier:.4f}")
    log.info(f"  ECE   = {ece:.4f}")
    log.info(f"  y_prob distribution: median={np.median(y_prob):.3f}  "
             f"mean={y_prob.mean():.3f}  std={y_prob.std():.3f}")

    out = pd.DataFrame({
        "subject_id":    merged["patient_id"].values,
        "encounter_csn": merged["encounter_csn"].values,
        "y_true":        y_true,
        "y_prob":        y_prob,
        "n_traj":        merged["n_traj"].values,
    })
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    log.info(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
