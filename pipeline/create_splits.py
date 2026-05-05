"""
create_splits.py
----------------
Create patient-level train / val / test splits for the ETHOS pipeline.

Split is done at the PATIENT level (not encounter level) to prevent data leakage:
a patient's multiple surgical encounters must all land in the same split.

Split ratios: train 70% / val 10% / test 20%  (configurable via CLI)

Outputs (saved to /path/to/CHD_MEDS/splits/):
  splits.parquet   — MEDS SubjectSplitSchema:
                       subject_id (int64), split (str: "train" | "val" | "test")
  splits.csv       — same content, human-readable

The split is seeded for reproducibility and stratified by IoD label:
  - Compute per-patient IoD positivity (1 if any encounter is IoD+, else 0)
  - Stratify on this flag so IoD+ patients are distributed proportionally

Usage:
    conda activate tccc
    python pipeline/create_splits.py [--train 0.7] [--val 0.1] [--test 0.2] [--seed 42]
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

MERGED_EVENTS = Path("/path/to/CHD_MEDS/merged/events.parquet")
IOD_TASK = Path("/path/to/CHD_MEDS/outcome/iod_task.parquet")
DEFAULT_OUTPUT_DIR = Path("/path/to/CHD_MEDS/splits")


def patient_id_to_subject_id(patient_id: pd.Series) -> pd.Series:
    return pd.to_numeric(patient_id.str.lstrip("C").str.strip(), errors="coerce").astype("Int64")


def main(train_ratio: float, val_ratio: float, test_ratio: float, seed: int, output_dir: Path):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Get all unique subject_ids from the merged events ─────────────────────
    log.info(f"Loading subject_ids from {MERGED_EVENTS} …")
    events = pd.read_parquet(MERGED_EVENTS, columns=["subject_id"], engine="pyarrow")
    all_subjects = events["subject_id"].dropna().unique()
    log.info(f"  Total unique patients: {len(all_subjects):,}")

    # ── Load IoD task to get per-patient positivity for stratification ────────
    log.info(f"Loading IoD task from {IOD_TASK} …")
    iod = pd.read_parquet(IOD_TASK, columns=["patient_id", "boolean_value"], engine="pyarrow")
    iod["subject_id"] = patient_id_to_subject_id(iod["patient_id"]).astype("int64")

    # A patient is IoD+ if ANY of their encounters is positive
    iod_positive = iod.groupby("subject_id")["boolean_value"].any().reset_index()
    iod_positive.columns = ["subject_id", "iod_positive"]

    # Build subject-level DataFrame
    subjects_df = pd.DataFrame({"subject_id": all_subjects})
    subjects_df = subjects_df.merge(iod_positive, on="subject_id", how="left")
    subjects_df["iod_positive"] = subjects_df["iod_positive"].fillna(False)

    n_iod_pos = subjects_df["iod_positive"].sum()
    log.info(f"  IoD+ patients: {n_iod_pos:,} / {len(subjects_df):,} "
             f"({n_iod_pos / len(subjects_df) * 100:.1f}%)")

    # ── Stratified split ──────────────────────────────────────────────────────
    # Step 1: Split off test set
    train_val, test = train_test_split(
        subjects_df,
        test_size=test_ratio,
        stratify=subjects_df["iod_positive"],
        random_state=seed,
    )

    # Step 2: Split train_val into train / val
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio_adjusted,
        stratify=train_val["iod_positive"],
        random_state=seed,
    )

    # ── Assemble splits DataFrame ─────────────────────────────────────────────
    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"

    splits = pd.concat([train, val, test], ignore_index=True)[["subject_id", "split"]]
    splits = splits.sort_values("subject_id").reset_index(drop=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    parquet_path = output_dir / "splits.parquet"
    csv_path = output_dir / "splits.csv"
    splits.to_parquet(parquet_path, index=False, engine="pyarrow")
    splits.to_csv(csv_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=" * 60)
    for split_name in ["train", "val", "test"]:
        mask = splits["split"] == split_name
        n = mask.sum()
        n_pos = subjects_df.loc[subjects_df["subject_id"].isin(
            splits.loc[mask, "subject_id"]), "iod_positive"].sum()
        log.info(f"  {split_name:<6}: {n:>7,} patients  IoD+ {n_pos:>5,} ({n_pos/n*100:.1f}%)")
    log.info(f"  Seed: {seed}")
    log.info(f"  Saved → {parquet_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create patient-level train/val/test splits")
    parser.add_argument("--train", type=float, default=0.7, help="Train ratio (default 0.7)")
    parser.add_argument("--val",   type=float, default=0.1, help="Val ratio (default 0.1)")
    parser.add_argument("--test",  type=float, default=0.2, help="Test ratio (default 0.2)")
    parser.add_argument("--seed",  type=int,   default=42,  help="Random seed (default 42)")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    main(args.train, args.val, args.test, args.seed, Path(args.output_dir))
