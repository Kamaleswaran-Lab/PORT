"""
logreg_xgb_tuned.py
-------------------
Hyperparameter-tuned LR + XGBoost baselines for IoD prediction.

Tuning protocol:
  - Fit on train split.
  - For each candidate hyperparameter config, evaluate on val split (AUPRC).
  - Pick best config by val AUPRC.
  - Final eval on test split (no refit on train+val to avoid val→test leakage
    via tuning signal).

LR search:
  C in {1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0}, with class_weight='balanced'.

XGB random search (50 trials):
  n_estimators in {200, 500, 1000}
  max_depth   in {3, 4, 5, 6, 8, 10}
  learning_rate in {0.01, 0.03, 0.05, 0.1}
  subsample   in {0.6, 0.8, 1.0}
  colsample_bytree in {0.6, 0.8, 1.0}
  min_child_weight in {1, 5, 10}
  reg_lambda  in {0.5, 1.0, 5.0}

Usage:
    conda activate ethos
    python baselines/logreg_xgb_tuned.py [--feature_set manual|meds|both]
                                         [--xgb_trials 50] [--seed 42]
                                         [--output_dir DIR]
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
    roc_auc_score, average_precision_score, brier_score_loss,
)
import xgboost as xgb

from baselines.features import (
    load_task, load_events, load_splits,
    build_manual_features, build_meds_features,
)

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("/path/to/CHD_MEDS/results/baselines_tuned")


def evaluate(y_true, y_prob, split, model_name):
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    n_pos = int(y_true.sum())
    n_tot = len(y_true)
    log.info(f"  [{split}] {model_name}: AUROC={auroc:.4f}  AUPRC={auprc:.4f}  Brier={brier:.4f}"
             f"  (n={n_tot:,}, pos={n_pos:,}, {n_pos/n_tot*100:.1f}%)")
    return {
        "model": model_name, "split": split,
        "auroc": auroc, "auprc": auprc, "brier": brier,
        "n_total": n_tot, "n_positive": n_pos,
    }


# ── LR tuning ─────────────────────────────────────────────────────────────────

def tune_lr(X_train, y_train, X_val, y_val, feature_set):
    log.info("Tuning Logistic Regression …")
    C_grid = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    best_auprc = -1.0
    best_C = None
    best_pipe = None

    for C in C_grid:
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("clf",     LogisticRegression(
                max_iter=2000, class_weight="balanced",
                solver="lbfgs", C=C,
            )),
        ])
        pipe.fit(X_train, y_train)
        prob_val = pipe.predict_proba(X_val)[:, 1]
        auprc_val = average_precision_score(y_val, prob_val)
        log.info(f"    LR C={C:>6.3g}  val AUPRC={auprc_val:.4f}")
        if auprc_val > best_auprc:
            best_auprc = auprc_val
            best_C = C
            best_pipe = pipe

    log.info(f"  → Best LR ({feature_set}): C={best_C}  val AUPRC={best_auprc:.4f}")
    return best_pipe, {"C": best_C, "val_auprc": best_auprc}


# ── XGB random search ─────────────────────────────────────────────────────────

def sample_xgb_params(rng):
    return {
        "n_estimators":     int(rng.choice([200, 500, 1000])),
        "max_depth":        int(rng.choice([3, 4, 5, 6, 8, 10])),
        "learning_rate":    float(rng.choice([0.01, 0.03, 0.05, 0.1])),
        "subsample":        float(rng.choice([0.6, 0.8, 1.0])),
        "colsample_bytree": float(rng.choice([0.6, 0.8, 1.0])),
        "min_child_weight": int(rng.choice([1, 5, 10])),
        "reg_lambda":       float(rng.choice([0.5, 1.0, 5.0])),
    }


def tune_xgb(X_train, y_train, X_val, y_val, feature_set, n_trials, seed):
    log.info(f"Tuning XGBoost (random search, {n_trials} trials) …")
    rng = np.random.default_rng(seed)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Pre-impute (XGB handles NaN natively but be consistent)
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)

    best_auprc = -1.0
    best_params = None
    best_model = None
    seen = set()

    for t in range(n_trials):
        # Re-sample if duplicate
        for _ in range(10):
            params = sample_xgb_params(rng)
            key = tuple(sorted(params.items()))
            if key not in seen:
                seen.add(key)
                break

        model = xgb.XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            early_stopping_rounds=30,
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(
            X_train_imp, y_train,
            eval_set=[(X_val_imp, y_val)],
            verbose=False,
        )
        prob_val = model.predict_proba(X_val_imp)[:, 1]
        auprc_val = average_precision_score(y_val, prob_val)
        if auprc_val > best_auprc:
            best_auprc = auprc_val
            best_params = params
            best_model = model
            log.info(f"    [trial {t+1:>3d}/{n_trials}] new best: val AUPRC={auprc_val:.4f}  params={params}")
        else:
            if (t + 1) % 10 == 0:
                log.info(f"    [trial {t+1:>3d}/{n_trials}] val AUPRC={auprc_val:.4f}  (best so far: {best_auprc:.4f})")

    log.info(f"  → Best XGB ({feature_set}): val AUPRC={best_auprc:.4f}  params={best_params}")
    return best_model, imputer, {"params": best_params, "val_auprc": best_auprc}


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_feature_set(
    feature_set: str,
    task: pd.DataFrame,
    events: pd.DataFrame,
    splits: pd.DataFrame,
    output_dir: Path,
    n_trials: int,
    seed: int,
):
    log.info("\n" + "=" * 70)
    log.info(f"Feature set: {feature_set.upper()}")
    log.info("=" * 70)

    if feature_set == "manual":
        X, y, groups = build_manual_features(task, events)
    else:
        X, y, groups = build_meds_features(task, events)

    split_map = splits.set_index("subject_id")["split"]
    split_labels = groups.map(split_map)

    train_mask = split_labels == "train"
    val_mask   = split_labels == "val"
    test_mask  = split_labels == "test"

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    log.info(f"  Train: {train_mask.sum():,} | Val: {val_mask.sum():,} | Test: {test_mask.sum():,}")

    # bool → int
    bool_cols = X_train.select_dtypes("bool").columns.tolist()
    X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()
    for df in [X_train, X_val, X_test]:
        df[bool_cols] = df[bool_cols].astype(int)

    results = []

    # ── LR ─────────────────────────────────────────────────────────────────────
    lr_best, lr_meta = tune_lr(X_train, y_train, X_val, y_val, feature_set)
    for split_name, X_s, y_s in [("val", X_val, y_val), ("test", X_test, y_test)]:
        prob = lr_best.predict_proba(X_s)[:, 1]
        results.append(evaluate(y_s.values, prob, split_name, f"LogReg_{feature_set}"))

    # ── XGB ────────────────────────────────────────────────────────────────────
    xgb_best, imputer, xgb_meta = tune_xgb(X_train, y_train, X_val, y_val, feature_set,
                                            n_trials=n_trials, seed=seed)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)
    for split_name, X_s, y_s in [("val", X_val_imp, y_val), ("test", X_test_imp, y_test)]:
        prob = xgb_best.predict_proba(X_s)[:, 1]
        results.append(evaluate(y_s.values, prob, split_name, f"XGBoost_{feature_set}"))

    # ── Save predictions on test set ───────────────────────────────────────────
    test_preds = pd.DataFrame({
        "subject_id":    groups[test_mask].values,
        "encounter_csn": (task.loc[test_mask, "encounter_csn"].values
                          if "encounter_csn" in task.columns else np.nan),
        "y_true":        y_test.values,
        f"prob_lr_{feature_set}":  lr_best.predict_proba(X_test)[:, 1],
        f"prob_xgb_{feature_set}": xgb_best.predict_proba(X_test_imp)[:, 1],
    })
    test_preds.to_parquet(output_dir / f"test_preds_{feature_set}_tuned.parquet", index=False)

    # ── Save best hyperparameters ──────────────────────────────────────────────
    meta = {"feature_set": feature_set, "lr": lr_meta, "xgb": xgb_meta, "seed": seed}
    with open(output_dir / f"best_hparams_{feature_set}.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return results


def main(feature_sets, output_dir, n_trials, seed):
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading data …")
    task = load_task()
    events = load_events()
    splits = load_splits()

    all_results = []
    for fs in feature_sets:
        all_results.extend(run_feature_set(fs, task, events, splits, output_dir, n_trials, seed))

    df = pd.DataFrame(all_results)
    df = df.sort_values(["split", "auroc"], ascending=[True, False])
    log.info("\n" + df.to_string(index=False))
    df.to_csv(output_dir / "results_summary_tuned.csv", index=False)
    log.info(f"\nResults saved → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_set", choices=["manual", "meds", "both"], default="both")
    parser.add_argument("--xgb_trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    feature_sets = ["manual", "meds"] if args.feature_set == "both" else [args.feature_set]
    main(feature_sets, Path(args.output_dir), args.xgb_trials, args.seed)
