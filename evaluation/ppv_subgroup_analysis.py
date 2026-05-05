"""
ppv_subgroup_analysis.py
------------------------
Compute PPV and other metrics by ASA subgroup to assess clinical utility
at higher base rates.

Usage:
    python -m evaluation.ppv_subgroup_analysis
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

BASELINES = Path("/path/to/CHD_MEDS/results/baselines")
OUTPUT_DIR = Path("/path/to/CHD_MEDS/results/evaluation")

def compute_metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "sensitivity": sens, "specificity": spec, "PPV": ppv, "NPV": npv}

def youden_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return thresholds[np.argmax(j)]

def analyze_subgroup(y_true, y_prob, label):
    n = len(y_true)
    n_pos = int(y_true.sum())
    prev = n_pos / n if n > 0 else 0

    if n_pos < 5 or n_pos == n:
        return {"subgroup": label, "n": n, "n_pos": n_pos, "prevalence": prev,
                "AUROC": np.nan, "AUPRC": np.nan, "threshold": np.nan,
                "sensitivity": np.nan, "specificity": np.nan,
                "PPV": np.nan, "NPV": np.nan, "note": "too few positives"}

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    thresh = youden_threshold(y_true, y_prob)
    metrics = compute_metrics_at_threshold(y_true, y_prob, thresh)

    return {"subgroup": label, "n": n, "n_pos": n_pos, "prevalence": round(prev, 4),
            "AUROC": round(auroc, 4), "AUPRC": round(auprc, 4),
            "threshold": round(thresh, 6), **{k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)},
            **{k: v for k, v in metrics.items() if isinstance(v, (int, np.integer))}}

def main():
    # Load predictions
    models = {
        "ETHOS FT": (BASELINES / "ethos_finetune_test_predictions.parquet", "y_prob"),
        "LSTM": (BASELINES / "lstm_test_predictions.parquet", "y_prob"),
        "ASA": (BASELINES / "asa_test_predictions.parquet", "y_prob"),
    }

    # Load ASA data from AN_Patients
    asa_pred = pd.read_parquet(BASELINES / "asa_test_predictions.parquet")
    # ASA predictions have subject_id and ASA info
    print(f"ASA pred columns: {asa_pred.columns.tolist()}")
    print(f"ASA pred shape: {asa_pred.shape}")

    # Load features to get ASA score
    # ASA scores are in the manual features or ASA predictions
    # Check if ASA score is directly in the prediction file
    if "asa_score" in asa_pred.columns:
        asa_map = asa_pred.set_index("subject_id")["asa_score"]
    else:
        # Load from features
        manual_preds = pd.read_parquet(BASELINES / "test_preds_manual.parquet")
        print(f"Manual pred columns: {manual_preds.columns.tolist()}")
        if "asa_score" in manual_preds.columns:
            asa_map = manual_preds.set_index("subject_id")["asa_score"] if "subject_id" in manual_preds.columns else None
        else:
            asa_map = None

    # Try to get ASA from the features file
    if asa_map is None:
        features_path = BASELINES / "test_features_manual.parquet"
        if features_path.exists():
            feats = pd.read_parquet(features_path)
            print(f"Features columns: {feats.columns.tolist()[:20]}...")
            if "asa_score" in feats.columns:
                asa_map = feats.set_index("subject_id")["asa_score"] if "subject_id" in feats.columns else None

    results = []

    for model_name, (fpath, prob_col) in models.items():
        if not fpath.exists():
            print(f"  {model_name}: file not found")
            continue
        df = pd.read_parquet(fpath)
        yt = df["y_true"].values.astype(float)
        yp = df[prob_col].values.astype(float)
        valid = ~(np.isnan(yt) | np.isnan(yp))
        yt, yp = yt[valid], yp[valid]

        # Overall
        results.append({**analyze_subgroup(yt, yp, "All"), "model": model_name})

        # Try to match ASA scores
        if "subject_id" in df.columns and asa_map is not None:
            df_valid = df[valid].copy()
            df_valid["asa"] = df_valid["subject_id"].map(asa_map)

            for asa_group, asa_label in [
                (df_valid["asa"] >= 3, "ASA >= 3"),
                (df_valid["asa"] >= 4, "ASA >= 4"),
                (df_valid["asa"] == 1, "ASA 1"),
                (df_valid["asa"] == 2, "ASA 2"),
                (df_valid["asa"] == 3, "ASA 3"),
                (df_valid["asa"] == 4, "ASA 4"),
                (df_valid["asa"] == 5, "ASA 5"),
            ]:
                mask = asa_group.values
                if mask.sum() > 0:
                    sub_yt = df_valid.loc[mask, "y_true"].values.astype(float)
                    sub_yp = df_valid.loc[mask, prob_col].values.astype(float)
                    results.append({**analyze_subgroup(sub_yt, sub_yp, asa_label), "model": model_name})
        elif "asa_score" in df.columns:
            for asa_group_val, asa_label in [
                (">=3", "ASA >= 3"), (">=4", "ASA >= 4"),
                (1, "ASA 1"), (2, "ASA 2"), (3, "ASA 3"), (4, "ASA 4"), (5, "ASA 5"),
            ]:
                if isinstance(asa_group_val, str):
                    if asa_group_val == ">=3":
                        mask = df["asa_score"].values >= 3
                    elif asa_group_val == ">=4":
                        mask = df["asa_score"].values >= 4
                else:
                    mask = df["asa_score"].values == asa_group_val
                mask_valid = mask[valid] if len(mask) == len(df) else mask
                if mask_valid.sum() > 0:
                    results.append({**analyze_subgroup(yt[mask_valid], yp[mask_valid], asa_label), "model": model_name})

    # If no ASA mapping found, try encounter-level approach
    # Load iod_task + an_patients for ASA
    if not any(r["subgroup"] != "All" for r in results):
        print("\nNo ASA mapping found in predictions. Loading from source data...")
        task = pd.read_parquet("/path/to/CHD_MEDS/outcome/iod_task.parquet")

        # Load AN_Patients for ASA scores
        import pyarrow.parquet as pq
        an_patients = pd.read_parquet("/path/to/CHD_MEDS/data/an_patients.parquet")
        print(f"an_patients columns: {an_patients.columns.tolist()[:15]}")

        # ASA is likely encoded as ENCOUNTER//AN//ASA_SCORE with numeric_value
        asa_events = an_patients[an_patients["code"].str.contains("ASA", na=False)]
        print(f"ASA events: {len(asa_events)}")
        print(asa_events.head())

        if len(asa_events) > 0:
            # Map patient_id + time to ASA score
            asa_events = asa_events.dropna(subset=["numeric_value"])
            # Join with task on patient_id and closest time
            task_asa = task.merge(
                asa_events[["patient_id", "time", "numeric_value"]].rename(
                    columns={"numeric_value": "asa_score", "time": "asa_time"}
                ),
                on="patient_id",
                how="left"
            )
            # Keep the ASA closest to prediction_time
            task_asa["time_diff"] = abs((task_asa["prediction_time"] - task_asa["asa_time"]).dt.total_seconds())
            task_asa = task_asa.sort_values("time_diff").drop_duplicates(subset=["patient_id", "prediction_time"], keep="first")

            print(f"\nASA distribution in task:")
            print(task_asa["asa_score"].value_counts().sort_index())

            # Now join with each model's predictions
            for model_name, (fpath, prob_col) in models.items():
                if not fpath.exists():
                    continue
                df = pd.read_parquet(fpath)

                # Join on subject_id
                if "subject_id" in df.columns:
                    task_asa["sid"] = task_asa["patient_id"].str.replace("C", "").astype(int)
                    merged = df.merge(task_asa[["sid", "asa_score"]].drop_duplicates("sid"),
                                      left_on="subject_id", right_on="sid", how="left")

                    yt = merged["y_true"].values.astype(float)
                    yp = merged[prob_col].values.astype(float)
                    asa = merged["asa_score"].values
                    valid = ~(np.isnan(yt) | np.isnan(yp))

                    for asa_label, mask_fn in [
                        ("ASA >= 3", lambda a: a >= 3),
                        ("ASA >= 4", lambda a: a >= 4),
                        ("ASA 1", lambda a: a == 1),
                        ("ASA 2", lambda a: a == 2),
                        ("ASA 3", lambda a: a == 3),
                        ("ASA 4", lambda a: a == 4),
                        ("ASA 5", lambda a: a == 5),
                    ]:
                        mask = valid & (~np.isnan(asa)) & mask_fn(asa)
                        if mask.sum() > 0:
                            results.append({**analyze_subgroup(yt[mask], yp[mask], asa_label), "model": model_name})

    # Print results
    result_df = pd.DataFrame(results)
    cols = ["model", "subgroup", "n", "n_pos", "prevalence", "AUROC", "AUPRC", "PPV", "sensitivity", "specificity", "NPV"]
    cols = [c for c in cols if c in result_df.columns]
    result_df = result_df[cols]

    print("\n" + "="*100)
    print("PPV Subgroup Analysis Results")
    print("="*100)

    for model_name in ["ETHOS FT", "LSTM", "ASA"]:
        sub = result_df[result_df["model"] == model_name]
        if len(sub) > 0:
            print(f"\n--- {model_name} ---")
            print(sub.to_string(index=False))

    # Save
    out_path = OUTPUT_DIR / "ppv_subgroup_analysis.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
