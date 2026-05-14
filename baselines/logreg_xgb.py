"""
logreg_xgb.py
-------------
Train and evaluate Logistic Regression and XGBoost baselines for IoD prediction.

Models trained on two feature sets:
  - manual : demographics, vitals, ASA score, procedure (Layer 2)
  - meds   : manual + labs + medications + diagnoses + ADT + LDAs + transfusions (Layer 3)

Results saved to /path/to/CHD_MEDS/results/baselines/

Usage:
    conda activate tccc
    python baselines/logreg_xgb.py [--feature_set manual|meds|both] [--output_dir DIR]
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    brier_score_loss, classification_report,
)
import xgboost as xgb

from baselines.features import load_task, load_events, load_splits, build_manual_features, build_meds_features

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("/path/to/CHD_MEDS/results/baselines")


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_prob: np.ndarray, split: str, model_name: str) -> dict:
    auroc  = roc_auc_score(y_true, y_prob)
    auprc  = average_precision_score(y_true, y_prob)
    brier  = brier_score_loss(y_true, y_prob)
    n_pos  = int(y_true.sum())
    n_tot  = len(y_true)

    log.info(f"  [{split}] {model_name}: AUROC={auroc:.4f}  AUPRC={auprc:.4f}  Brier={brier:.4f}"
             f"  (n={n_tot:,}, pos={n_pos:,}, {n_pos/n_tot*100:.1f}%)")
    return {
        "model":  model_name,
        "split":  split,
        "auroc":  auroc,
        "auprc":  auprc,
        "brier":  brier,
        "n_total": n_tot,
        "n_positive": n_pos,
    }


# ── train / eval pipeline ────────────────────────────────────────────────────

def run_feature_set(
    feature_set: str,
    task: pd.DataFrame,
    events: pd.DataFrame,
    splits: pd.DataFrame,
    output_dir: Path,
    window_days: int | None = None,
    suffix: str = "",
    unweighted: bool = False,
) -> list[dict]:

    log.info(f"\n{'='*60}")
    log.info(f"Feature set: {feature_set.upper()}  window_days={window_days}  suffix={suffix}")
    log.info(f"{'='*60}")

    if feature_set == "manual":
        X, y, groups = build_manual_features(task, events, window_days=window_days)
    else:
        X, y, groups = build_meds_features(task, events, window_days=window_days)

    # Attach split labels via subject_id
    split_map = splits.set_index("subject_id")["split"]
    split_labels = groups.map(split_map)

    train_mask = split_labels == "train"
    val_mask   = split_labels == "val"
    test_mask  = split_labels == "test"

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    log.info(f"  Train: {train_mask.sum():,} | Val: {val_mask.sum():,} | Test: {test_mask.sum():,}")

    # Convert bool columns to int (XGBoost/sklearn compatibility)
    bool_cols = X_train.select_dtypes("bool").columns.tolist()
    X_train = X_train.copy()
    X_val   = X_val.copy()
    X_test  = X_test.copy()
    for df in [X_train, X_val, X_test]:
        df[bool_cols] = df[bool_cols].astype(int)

    results = []

    # ── Logistic Regression ──────────────────────────────────────────────────
    log.info("Training Logistic Regression …")
    lr_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(
            max_iter=1000,
            class_weight=None if unweighted else "balanced",
            solver="lbfgs",
            C=0.1,
        )),
    ])
    lr_pipe.fit(X_train, y_train)

    model_name = f"LogReg_{feature_set}"
    for split, X_s, y_s in [("val", X_val, y_val), ("test", X_test, y_test)]:
        prob = lr_pipe.predict_proba(X_s)[:, 1]
        results.append(evaluate(y_s.values, prob, split, model_name))

    # Save LR feature importances (coefficients)
    # After imputation, all-NaN columns are dropped — use get_feature_names_out() to match
    imputer_step = lr_pipe.named_steps["imputer"]
    kept_mask = ~np.isnan(imputer_step.statistics_)  # True for columns imputer kept
    kept_cols  = X_train.columns[kept_mask]
    lr_coef = pd.Series(
        lr_pipe.named_steps["clf"].coef_[0],
        index=kept_cols,
    ).sort_values(key=abs, ascending=False)
    lr_coef.to_csv(output_dir / f"lr_{feature_set}_coef.csv")

    # ── XGBoost ─────────────────────────────────────────────────────────────
    log.info("Training XGBoost …")
    scale_pos_weight = 1.0 if unweighted else (y_train == 0).sum() / (y_train == 1).sum()

    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,   # handles class imbalance
        use_label_encoder=False,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )

    # Impute before XGBoost (XGBoost handles NaN natively but imputing helps LR reuse)
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp   = imputer.transform(X_val)
    X_test_imp  = imputer.transform(X_test)

    xgb_model.fit(
        X_train_imp, y_train,
        eval_set=[(X_val_imp, y_val)],
        verbose=50,
    )

    model_name = f"XGBoost_{feature_set}"
    for split, X_s, y_s in [("val", X_val_imp, y_val), ("test", X_test_imp, y_test)]:
        prob = xgb_model.predict_proba(X_s)[:, 1]
        results.append(evaluate(y_s.values, prob, split, model_name))

    # Save XGBoost feature importances
    # X_train_imp was imputed: use the same kept_mask logic as LR
    imputer_xgb = SimpleImputer(strategy="median").fit(X_train)
    kept_mask_xgb = ~np.isnan(imputer_xgb.statistics_)
    kept_cols_xgb = X_train.columns[kept_mask_xgb]
    fi = pd.Series(
        xgb_model.feature_importances_,
        index=kept_cols_xgb,
    ).sort_values(ascending=False)
    fi.to_csv(output_dir / f"xgb_{feature_set}_importance.csv")

    # Save model predictions on test set for calibration plots
    test_preds = pd.DataFrame({
        "subject_id":    groups[test_mask].values,
        "encounter_csn": task.loc[test_mask, "encounter_csn"].values if "encounter_csn" in task.columns else np.nan,
        "y_true":        y_test.values,
        f"prob_lr_{feature_set}":  lr_pipe.predict_proba(X_test)[:, 1],
        f"prob_xgb_{feature_set}": xgb_model.predict_proba(X_test_imp)[:, 1],
    })
    test_preds.to_parquet(output_dir / f"test_preds_{feature_set}{suffix}.parquet", index=False)

    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main(feature_sets: list[str], output_dir: Path, window_days: int | None = None,
         suffix: str = "", unweighted: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading data …")
    task   = load_task()
    events = load_events()
    splits = load_splits()

    all_results = []
    for fs in feature_sets:
        res = run_feature_set(fs, task, events, splits, output_dir,
                              window_days=window_days, suffix=suffix, unweighted=unweighted)
        all_results.extend(res)

    # Summary table
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(["split", "auroc"], ascending=[True, False])
    log.info("\n" + results_df.to_string(index=False))

    summary_name = f"results_summary{suffix}.csv"
    results_df.to_csv(output_dir / summary_name, index=False)
    log.info(f"\nResults saved → {output_dir / summary_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LR + XGBoost baselines for IoD prediction")
    parser.add_argument(
        "--feature_set", choices=["manual", "meds", "both"], default="both",
        help="Which feature set to use (default: both)",
    )
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--window_days", type=int, default=None,
                        help="Context window in days before OR entry (None=all history)")
    parser.add_argument("--suffix", default="",
                        help="Suffix for output filenames (e.g., _window_30d)")
    parser.add_argument("--unweighted", action="store_true",
                        help="Disable class re-weighting (class_weight=None for LR; scale_pos_weight=1 for XGB)")
    args = parser.parse_args()

    feature_sets = ["manual", "meds"] if args.feature_set == "both" else [args.feature_set]
    main(feature_sets, Path(args.output_dir),
         window_days=args.window_days, suffix=args.suffix, unweighted=args.unweighted)
