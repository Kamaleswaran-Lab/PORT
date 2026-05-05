"""
analyze_infer.py
----------------
Post-process raw ETHOS inference parquet files → final predictions with
uncertainty quantification and token-level risk attribution.

For each encounter (N trajectories), computes:
  y_prob          : fraction of trajectories that generated an IoD token
                    (= "X% of simulated futures had IoD")
  uncertainty     : empirical std of per-trajectory IoD indicators
                    (low = model is confident; high = genuinely uncertain)
  token_attribution: JSON dict {iod_token: count} across IoD+ trajectories
                    (which IoD mechanism dominates — CPR vs vasoactives vs A-line)
  top_iod_token   : most frequent IoD token (primary risk mechanism)

Output:
  <output_dir>/iod_predictions.parquet
    columns: patient_id, encounter_csn, prediction_time,
             y_true, y_prob, uncertainty, n_traj, n_iod_traj,
             top_iod_token, token_attribution

Also:
  - Appends ETHOS row to results/baselines/results_summary.csv
  - Re-runs evaluation/evaluate.py to regenerate all plots

Usage:
    conda activate ethos
    # Auto-selects latest subdir in ethos/iod/:
    python ethos/analyze_infer.py

    # Or specify a particular inference run:
    python ethos/analyze_infer.py \\
        --infer_dir /path/to/CHD_MEDS/results/ethos/iod/iod_rep100_2026-XX-XX_XX-XX-XX
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

TASK_PATH     = Path("/path/to/CHD_MEDS/outcome/iod_task.parquet")
SPLITS_PATH   = Path("/path/to/CHD_MEDS/splits/splits.parquet")
ETHOS_DIR     = Path("/path/to/CHD_MEDS/results/ethos/iod")
SUMMARY_PATH  = Path("/path/to/CHD_MEDS/results/baselines/results_summary.csv")

IOD_STOP_STOKENS = [
    # IoD1 — cardiac arrest
    "AN_EVENT//CPR",
    # IoD3/4 — epinephrine (infusion + bolus)
    "MED//EPINEPHRINE_DRIP",
    "MED//EPINEPHRINE_DRIP_16_MCG/ML_30_ML_SYRINGE",
    "MED//EPINEPHRINE_01_MG/ML_INJECTION_SYRINGE",
    "MED//EPINEPHRINE_1_MG/ML_1_ML_INJECTION_SOLUTION",
    "MED//EPINEPHRINE_HCL_PF_1_MG/ML_1_ML_INJECTION_SOLUTION",
    "MED//EPINEPHRINE_1_MG/ML_INJECTION_SOLUTION",
    # IoD4 — dopamine infusion
    "MED//DOPAMINE_DRIP",
    "MED//ANE_DOPAMINE_INFUSION",
    # IoD4 — milrinone infusion
    "MED//MILRINONE_DRIP",
    "MED//MILRINONE_DRIP_LOADING_DOSE",
    # IoD4 — norepinephrine infusion
    "MED//NOREPINEPHRINE_DRIP",
    # IoD3/4 — phenylephrine
    "MED//PHENYLEPHRINE_DRIP",
    # IoD3/4 — vasopressin
    "MED//VASOPRESSIN_DRIP_FOR_HYPOTENSION",
    # IoD3 — adenosine bolus
    "MED//ADENOSINE_3_MG/ML_INTRAVENOUS_SOLUTION",
    # IoD3 — atropine IV
    "MED//ATROPINE_04_MG/ML_INJECTION_SOLUTION",
    # IoD3 — ephedrine IV
    "MED//EPHEDRINE_SULFATE_50_MG/ML_INJECTION_SOLUTION",
    # IoD5 — arterial line
    "LDA//ARTERIAL_LIN",
]

IOD_TOKEN_LABELS = {
    "AN_EVENT//CPR":                                    "CPR (IoD1)",
    "MED//EPINEPHRINE_DRIP":                            "Epinephrine drip (IoD4)",
    "MED//EPINEPHRINE_DRIP_16_MCG/ML_30_ML_SYRINGE":   "Epinephrine drip syringe (IoD4)",
    "MED//EPINEPHRINE_01_MG/ML_INJECTION_SYRINGE":     "Epinephrine bolus (IoD3)",
    "MED//EPINEPHRINE_1_MG/ML_1_ML_INJECTION_SOLUTION":"Epinephrine bolus 1mg (IoD3)",
    "MED//EPINEPHRINE_HCL_PF_1_MG/ML_1_ML_INJECTION_SOLUTION": "Epinephrine HCl PF bolus (IoD3)",
    "MED//EPINEPHRINE_1_MG/ML_INJECTION_SOLUTION":     "Epinephrine 1mg/mL (IoD3)",
    "MED//DOPAMINE_DRIP":                               "Dopamine drip (IoD4)",
    "MED//ANE_DOPAMINE_INFUSION":                       "Dopamine infusion OR (IoD4)",
    "MED//MILRINONE_DRIP":                              "Milrinone drip (IoD4)",
    "MED//MILRINONE_DRIP_LOADING_DOSE":                 "Milrinone loading dose (IoD4)",
    "MED//NOREPINEPHRINE_DRIP":                         "Norepinephrine drip (IoD4)",
    "MED//PHENYLEPHRINE_DRIP":                          "Phenylephrine drip (IoD3/4)",
    "MED//VASOPRESSIN_DRIP_FOR_HYPOTENSION":            "Vasopressin drip (IoD3/4)",
    "MED//ADENOSINE_3_MG/ML_INTRAVENOUS_SOLUTION":     "Adenosine IV (IoD3)",
    "MED//ATROPINE_04_MG/ML_INJECTION_SOLUTION":       "Atropine IV (IoD3)",
    "MED//EPHEDRINE_SULFATE_50_MG/ML_INJECTION_SOLUTION": "Ephedrine IV (IoD3)",
    "LDA//ARTERIAL_LIN":                                "Arterial Line (IoD5)",
}


def find_latest_infer_dir(ethos_dir: Path) -> Path | None:
    """Find most recently modified subdirectory (or parquet file) in ethos_dir."""
    # Prefer subdirectory (named iod_repNNN_date)
    subdirs = [p for p in ethos_dir.iterdir() if p.is_dir()] if ethos_dir.exists() else []
    if subdirs:
        return max(subdirs, key=lambda p: p.stat().st_mtime)
    # Fallback: parquet files directly in ethos_dir
    parquets = list(ethos_dir.glob("*.parquet")) if ethos_dir.exists() else []
    return ethos_dir if parquets else None


def load_raw_inference(infer_dir: Path) -> pd.DataFrame:
    """Read all samples_*.parquet files from an inference run directory."""
    parquet_files = sorted(infer_dir.glob("samples*.parquet"))
    if not parquet_files:
        # Try direct parquet files
        parquet_files = sorted(infer_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {infer_dir}")

    log.info(f"  Loading {len(parquet_files)} parquet file(s) from {infer_dir}")
    dfs = [pd.read_parquet(f, engine="pyarrow") for f in parquet_files]
    df  = pd.concat(dfs, ignore_index=True)
    log.info(f"  Total rows (trajectories): {len(df):,}")
    return df


def aggregate_encounters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate N trajectories per encounter → risk_score, uncertainty, attribution.

    Returns one row per (patient_id, prediction_time) with:
      y_prob           : fraction of IoD trajectories
      uncertainty      : empirical std across trajectories
      token_attribution: JSON {token: count} for IoD+ trajectories
      top_iod_token    : most frequent IoD token
    """
    df = df.copy()

    # Normalize prediction_time to microseconds int for grouping.
    # ETHOS safetensors store times as int64 nanoseconds; when pandas reads them
    # back as datetime64[us] the raw int64 value is in ns, so divide by 1000
    # to get the correct microsecond offset before comparing with iod_task.
    if pd.api.types.is_datetime64_any_dtype(df["prediction_time"]):
        df["prediction_time_us"] = df["prediction_time"].astype("int64") // 1000
    else:
        df["prediction_time_us"] = df["prediction_time"].astype("int64") // 1000

    df["is_iod"] = df["actual"].isin(IOD_STOP_STOKENS).astype(int)

    group_keys = ["patient_id", "prediction_time_us"]

    # Basic stats
    stats = (
        df.groupby(group_keys)
        .agg(
            y_true       =("iod_label",  "first"),
            y_prob       =("is_iod",     "mean"),
            uncertainty  =("is_iod",     "std"),
            n_traj       =("is_iod",     "count"),
            n_iod_traj   =("is_iod",     "sum"),
        )
        .reset_index()
    )
    stats["uncertainty"] = stats["uncertainty"].fillna(0.0)

    # Token attribution per encounter (from IoD+ trajectories only)
    iod_trajs = df[df["is_iod"] == 1].copy()

    def _attr_json(series):
        counts = series.value_counts()
        return json.dumps(counts.to_dict())

    token_attr = (
        iod_trajs.groupby(group_keys)["actual"]
        .apply(_attr_json)
        .reset_index()
        .rename(columns={"actual": "token_attribution"})
    )
    stats = stats.merge(token_attr, on=group_keys, how="left")
    stats["token_attribution"] = stats["token_attribution"].fillna("{}")

    # Top IoD token (primary risk mechanism)
    def _top_token(s):
        d = json.loads(s)
        return max(d, key=d.get) if d else None

    stats["top_iod_token"] = stats["token_attribution"].apply(_top_token)
    stats["top_iod_label"] = stats["top_iod_token"].map(IOD_TOKEN_LABELS)

    log.info(f"  Encounters after aggregation: {len(stats):,}")
    return stats


def match_to_task(stats: pd.DataFrame) -> pd.DataFrame:
    """
    Merge aggregated encounters with iod_task.parquet to add encounter_csn
    and verify y_true against ground truth.
    """
    task   = pd.read_parquet(TASK_PATH, engine="pyarrow")
    splits = pd.read_parquet(SPLITS_PATH, engine="pyarrow")

    # Convert task patient_id → int (strip "C")
    task["patient_id_int"] = (
        pd.to_numeric(task["patient_id"].str.lstrip("C"), errors="coerce").astype("Int64")
    )

    # Convert task prediction_time → microseconds
    pt = pd.to_datetime(task["prediction_time"])
    if str(pt.dtype) == "datetime64[ns]":
        task["prediction_time_us"] = pt.astype("int64") // 1000
    else:
        task["prediction_time_us"] = pt.astype("int64")

    # Add subject_id to splits
    splits["subject_id"] = splits["subject_id"].astype("int64")

    merged = stats.merge(
        task[["patient_id_int", "prediction_time_us", "encounter_csn", "boolean_value"]],
        left_on=["patient_id", "prediction_time_us"],
        right_on=["patient_id_int", "prediction_time_us"],
        how="left",
    )
    merged = merged.merge(
        splits[["subject_id", "split"]].rename(columns={"subject_id": "patient_id"}),
        on="patient_id",
        how="left",
    )

    n_matched = merged["encounter_csn"].notna().sum()
    log.info(f"  Matched to iod_task: {n_matched:,} / {len(merged):,} encounters")

    # Always prefer iod_task ground truth over the tokenized-timeline iod_label.
    # iod_label is based on token presence after OR_ENTRY (e.g. LDA//ARTERIAL_LIN
    # fires for routine pre-incision arterial lines, inflating positives 1.8%→6.3%).
    # boolean_value from iod_task uses the proper per-encounter IoD definition.
    if merged["boolean_value"].notna().any():
        n_updated = merged["boolean_value"].notna().sum()
        log.info(f"  Overriding y_true with iod_task boolean_value for {n_updated:,} encounters")
        merged.loc[merged["boolean_value"].notna(), "y_true"] = (
            merged.loc[merged["boolean_value"].notna(), "boolean_value"]
            .astype(int)
        )
    else:
        log.warning("  boolean_value not available — falling back to tokenized iod_label (may be inflated)")

    return merged


def evaluate_and_save(merged: pd.DataFrame, output_dir: Path):
    """Compute metrics, save predictions parquet, update results_summary.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)

    test = merged[merged["split"] == "test"].dropna(subset=["y_prob"])
    if len(test) == 0:
        log.warning("No test-split encounters found. Cannot compute metrics.")
        return

    y_true = test["y_true"].astype(int).values
    y_prob = test["y_prob"].values

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    n_pos = int(y_true.sum())
    n_tot = len(y_true)

    log.info(
        f"\n  [test] ETHOS (zero-shot): AUROC={auroc:.4f}  AUPRC={auprc:.4f}  Brier={brier:.4f}"
        f"  (n={n_tot:,}, pos={n_pos:,}, {n_pos/n_tot*100:.1f}%)"
    )
    log.info(
        f"  Uncertainty: mean={test['uncertainty'].mean():.4f}  "
        f"  High-risk (>0.5) encounters: {(test['y_prob'] > 0.5).sum():,}"
    )

    # Save full predictions parquet (test split only)
    out_cols = [
        "patient_id", "encounter_csn", "prediction_time_us",
        "y_true", "y_prob", "uncertainty", "n_traj", "n_iod_traj",
        "top_iod_token", "top_iod_label", "token_attribution",
    ]
    save_cols = [c for c in out_cols if c in test.columns]
    out_path = output_dir / "iod_predictions.parquet"
    test[save_cols].to_parquet(out_path, index=False)
    log.info(f"  Saved predictions → {out_path}")

    # Also save evaluate.py-compatible file (y_true, y_prob, subject_id)
    compat = test[["y_true", "y_prob"]].copy()
    compat["subject_id"] = test["patient_id"]
    compat_path = output_dir.parent.parent / "baselines" / "ethos_test_predictions.parquet"
    compat_path.parent.mkdir(parents=True, exist_ok=True)
    compat.to_parquet(compat_path, index=False)
    log.info(f"  Saved evaluate.py-compatible predictions → {compat_path}")

    # Update results_summary.csv
    new_row = pd.DataFrame([{
        "model":      "ETHOS (zero-shot)",
        "split":      "test",
        "auroc":      auroc,
        "auprc":      auprc,
        "brier":      brier,
        "n_total":    n_tot,
        "n_positive": n_pos,
    }])
    if SUMMARY_PATH.exists():
        summary = pd.read_csv(SUMMARY_PATH)
        summary = summary[summary["model"] != "ETHOS (zero-shot)"]
        summary = pd.concat([summary, new_row], ignore_index=True)
    else:
        summary = new_row
    summary.to_csv(SUMMARY_PATH, index=False)
    log.info(f"  Updated results_summary.csv → {SUMMARY_PATH}")

    # Trigger evaluate.py for updated plots
    log.info("  Re-running evaluation/evaluate.py to update plots …")
    result = subprocess.run(
        [sys.executable, "-m", "evaluation.evaluate"],
        cwd=str(Path(__file__).parent.parent),
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        log.info("  evaluate.py completed successfully.")
    else:
        log.warning(f"  evaluate.py failed:\n{result.stderr}")


def print_uncertainty_summary(merged: pd.DataFrame, output_dir: Path | None = None):
    """
    Print a clinical summary of uncertainty-stratified predictions.
    Adds Fisher's exact test between quadrants and exports CSV.

    2x2 triage grid:
      ┌─────────────┬──────────────────────┬──────────────────────┐
      │             │ Low uncertainty      │ High uncertainty     │
      │             │ (std < 0.3)          │ (std ≥ 0.3)          │
      ├─────────────┼──────────────────────┼──────────────────────┤
      │ High risk   │ Confident High Risk  │ Uncertain High Risk  │
      │ (prob > 0.3)│ → prioritize         │ → investigate further│
      ├─────────────┼──────────────────────┼──────────────────────┤
      │ Low risk    │ Confident Low Risk   │ Uncertain Low Risk   │
      │ (prob ≤ 0.3)│ → routine care       │ → monitor            │
      └─────────────┴──────────────────────┴──────────────────────┘
    """
    test = merged[merged["split"] == "test"].dropna(subset=["y_prob"])
    if len(test) == 0:
        return

    log.info("\n=== Uncertainty-Stratified Risk Summary (test set) ===")

    risk_thresh = 0.3
    unc_thresh  = 0.3

    quadrants = {
        "high_risk_confident": test[(test["y_prob"] >  risk_thresh) & (test["uncertainty"] <  unc_thresh)],
        "high_risk_uncertain": test[(test["y_prob"] >  risk_thresh) & (test["uncertainty"] >= unc_thresh)],
        "low_risk_confident":  test[(test["y_prob"] <= risk_thresh) & (test["uncertainty"] <  unc_thresh)],
        "low_risk_uncertain":  test[(test["y_prob"] <= risk_thresh) & (test["uncertainty"] >= unc_thresh)],
    }

    labels = {
        "high_risk_confident": f"High risk, confident  (prob>{risk_thresh}, std<{unc_thresh})",
        "high_risk_uncertain": f"High risk, uncertain  (prob>{risk_thresh}, std≥{unc_thresh})",
        "low_risk_confident":  f"Low risk,  confident  (prob≤{risk_thresh}, std<{unc_thresh})",
        "low_risk_uncertain":  f"Low risk,  uncertain  (prob≤{risk_thresh}, std≥{unc_thresh})",
    }

    rows = []
    for key, grp in quadrants.items():
        if len(grp) == 0:
            continue
        iod_rate = grp["y_true"].mean()
        n_pos    = int(grp["y_true"].sum())
        log.info(f"  {labels[key]}: n={len(grp):,}, IoD+ {n_pos} ({iod_rate*100:.2f}%)")
        rows.append({
            "quadrant": key,
            "label":    labels[key],
            "n":        len(grp),
            "n_pos":    n_pos,
            "iod_rate": iod_rate,
        })

    # Fisher's exact test: high-risk-confident vs high-risk-uncertain
    hrc = quadrants["high_risk_confident"]
    hru = quadrants["high_risk_uncertain"]
    if len(hrc) > 0 and len(hru) > 0:
        ct = np.array([
            [int(hrc["y_true"].sum()), len(hrc) - int(hrc["y_true"].sum())],
            [int(hru["y_true"].sum()), len(hru) - int(hru["y_true"].sum())],
        ])
        _, p = fisher_exact(ct)
        log.info(
            f"\n  Fisher's exact (high-risk-conf vs high-risk-uncert): p={p:.4f}"
            f"  {'*' if p < 0.05 else 'ns'}"
        )

    # Fisher's exact test: high-risk-confident vs low-risk-confident
    lrc = quadrants["low_risk_confident"]
    if len(hrc) > 0 and len(lrc) > 0:
        ct2 = np.array([
            [int(hrc["y_true"].sum()), len(hrc) - int(hrc["y_true"].sum())],
            [int(lrc["y_true"].sum()), len(lrc) - int(lrc["y_true"].sum())],
        ])
        _, p2 = fisher_exact(ct2)
        log.info(
            f"  Fisher's exact (high-risk-conf vs low-risk-conf): p={p2:.4e}"
            f"  {'***' if p2 < 0.001 else '*' if p2 < 0.05 else 'ns'}"
        )

    # Export CSV
    if output_dir is not None and rows:
        csv_path = output_dir / "uncertainty_triage_summary.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        log.info(f"  Uncertainty triage table → {csv_path}")

    # Token attribution summary
    all_iod = test[test["top_iod_token"].notna()]
    if len(all_iod) > 0:
        log.info("\n  Top predicted IoD mechanisms (among high-risk encounters):")
        counts = all_iod["top_iod_token"].value_counts()
        for tok, cnt in counts.head(5).items():
            label = IOD_TOKEN_LABELS.get(tok, tok)
            log.info(f"    {label}: {cnt:,} encounters ({cnt/len(all_iod)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="ETHOS inference post-processing")
    parser.add_argument(
        "--infer_dir", type=Path, default=None,
        help="Directory with raw inference parquets (auto-selects latest if omitted)"
    )
    parser.add_argument(
        "--output_dir", type=Path,
        default=Path("/path/to/CHD_MEDS/results/ethos/iod"),
        help="Where to save iod_predictions.parquet"
    )
    args = parser.parse_args()

    infer_dir = args.infer_dir
    if infer_dir is None:
        infer_dir = find_latest_infer_dir(ETHOS_DIR)
        if infer_dir is None:
            log.error(f"No inference results found in {ETHOS_DIR}. Run bash ethos/infer.sh first.")
            sys.exit(1)
        log.info(f"Auto-selected inference dir: {infer_dir}")

    log.info("=== ETHOS Inference Post-Processing ===")
    df     = load_raw_inference(infer_dir)
    stats  = aggregate_encounters(df)
    merged = match_to_task(stats)
    evaluate_and_save(merged, args.output_dir)
    print_uncertainty_summary(merged, output_dir=args.output_dir)
    log.info("\n=== Done ===")


if __name__ == "__main__":
    main()
