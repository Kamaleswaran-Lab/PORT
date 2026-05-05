"""
prepare_ethos_data.py
---------------------
Split merged_events.parquet into train/val/test directories expected by ethos_tokenize.

ethos_tokenize expects:
  input_dir/
    *.parquet   — each file contains a subset of patients (partitioned by subject_id)

This script:
  1. Loads merged events and patient splits
  2. For each split (train/val/test), writes N_SHARDS parquet files
     (each shard ~equal number of patients, sorted by subject_id within each file)
  3. Also converts NaT static events: ETHOS uses time=0 (epoch) for static events
     when sorting, so we replace NaT → datetime(1970, 1, 1) as the MEDS convention

Output structure:
  /path/to/CHD_MEDS/ethos_input/
    train/
      shard_000.parquet ... shard_NNN.parquet
    val/
      shard_000.parquet
    test/
      shard_000.parquet ... shard_NNN.parquet

Usage:
    conda activate ethos
    python pipeline/prepare_ethos_data.py [--n_shards 8]
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

EVENTS_PATH = Path("/path/to/CHD_MEDS/merged/events.parquet")
SPLITS_PATH = Path("/path/to/CHD_MEDS/splits/splits.parquet")
DEFAULT_OUTPUT_DIR = Path("/path/to/CHD_MEDS/ethos_input")

# Columns expected by ethos (subject_id must be int64)
ETHOS_COLS = ["subject_id", "time", "code", "numeric_value", "text_value"]


def write_split(df: pd.DataFrame, out_dir: Path, n_shards: int, split_name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects = df["subject_id"].unique()
    shards = np.array_split(subjects, n_shards)
    for i, shard_subjects in enumerate(shards):
        shard_df = df[df["subject_id"].isin(shard_subjects)]
        out_path = out_dir / f"{i:03d}.parquet"
        shard_df.to_parquet(out_path, index=False, engine="pyarrow")
    log.info(f"  {split_name}: {len(subjects):,} patients → {n_shards} shards in {out_dir}")


def main(output_dir: Path, n_shards_train: int, n_shards_val: int, n_shards_test: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading events …")
    events = pd.read_parquet(EVENTS_PATH, engine="pyarrow")

    log.info("Loading splits …")
    splits = pd.read_parquet(SPLITS_PATH, engine="pyarrow")

    # Keep only ETHOS columns + drop text_value (not used in tokenization)
    # subject_id already int64 from merge_meds.py
    events = events[ETHOS_COLS].copy()

    # ETHOS requires subject_id as int64 — confirm
    events["subject_id"] = events["subject_id"].astype("int64")

    # Sort: subject_id first, then time (NaT sorts to front — MEDS static convention)
    events = events.sort_values(["subject_id", "time"], na_position="first").reset_index(drop=True)

    log.info(f"  Total events: {len(events):,}  patients: {events['subject_id'].nunique():,}")

    for split_name, n_shards in [("train", n_shards_train), ("val", n_shards_val), ("test", n_shards_test)]:
        split_subjects = splits.loc[splits["split"] == split_name, "subject_id"].values
        split_df = events[events["subject_id"].isin(split_subjects)]
        log.info(f"  {split_name}: {split_df['subject_id'].nunique():,} patients, "
                 f"{len(split_df):,} events")
        write_split(split_df, output_dir / split_name, n_shards, split_name)

    log.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MEDS data for ethos_tokenize")
    parser.add_argument("--output_dir",      default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n_shards_train",  type=int, default=8,
                        help="Number of parquet shards for train split (default 8)")
    parser.add_argument("--n_shards_val",    type=int, default=2,
                        help="Number of parquet shards for val split (default 2)")
    parser.add_argument("--n_shards_test",   type=int, default=4,
                        help="Number of parquet shards for test split (default 4)")
    args = parser.parse_args()
    main(
        Path(args.output_dir),
        args.n_shards_train,
        args.n_shards_val,
        args.n_shards_test,
    )
