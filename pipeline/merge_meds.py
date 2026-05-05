"""
merge_meds.py
-------------
Merge all 13 MEDS parquet files into a single unified events dataset.

- NO labels included — this is pure event data, reused across all prediction tasks.
- cardiology_notes.parquet is excluded (free-text, not ETHOS-tokenizable).
- Static events (time=NaT) are kept; they sort to the front of each patient timeline.
- subject_id (int64) is derived from patient_id by stripping the "C" prefix.

Output:
  /path/to/CHD_MEDS/merged/events.parquet
    columns: subject_id (int64), patient_id (str), time (datetime64[us]),
             code (str), numeric_value (float32), text_value (str)

Usage:
    conda activate tccc
    python pipeline/merge_meds.py [--output_dir DIR]
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

MEDS_DIR = Path("/path/to/CHD_MEDS/data")
DEFAULT_OUTPUT_DIR = Path("/path/to/CHD_MEDS/merged")

# All parquets to merge — cardiology_notes excluded (free-text only)
PARQUET_FILES = [
    "an_patients.parquet",
    "an_events.parquet",
    "labs.parquet",
    "medications.parquet",
    "ldas.parquet",
    "adt.parquet",
    "transfusions.parquet",
    "problem_list.parquet",
    "medical_history.parquet",
    "surgical_history.parquet",
    "sde.parquet",
    "cardiology_encounters.parquet",
]

MEDS_COLS = ["patient_id", "time", "code", "numeric_value", "text_value"]


def patient_id_to_subject_id(patient_id: pd.Series) -> pd.Series:
    """Convert 'C800881' → 800881 (int64). Non-numeric remainder → NaN → dropped."""
    numeric = patient_id.str.lstrip("C").str.strip()
    return pd.to_numeric(numeric, errors="coerce").astype("Int64")


def load_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow", columns=MEDS_COLS)
    # Ensure correct dtypes
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=False)
    df["numeric_value"] = pd.to_numeric(df["numeric_value"], errors="coerce").astype("float32")
    df["text_value"] = df["text_value"].astype(str).where(df["text_value"].notna(), None)
    df["code"] = df["code"].astype(str)
    df["patient_id"] = df["patient_id"].astype(str)
    return df


def main(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    total_rows = 0

    for fname in PARQUET_FILES:
        path = MEDS_DIR / fname
        if not path.exists():
            log.warning(f"  MISSING: {fname} — skipping")
            continue

        df = load_parquet(path)
        n = len(df)
        total_rows += n
        log.info(f"  {fname:<45} {n:>10,} rows  {df['patient_id'].nunique():>7,} patients")
        frames.append(df)

    log.info(f"Concatenating {len(frames)} tables ({total_rows:,} total rows) …")
    merged = pd.concat(frames, ignore_index=True)

    # Derive subject_id (int64) from patient_id
    log.info("Deriving subject_id …")
    merged["subject_id"] = patient_id_to_subject_id(merged["patient_id"])

    # Drop rows where subject_id could not be parsed (malformed patient_id)
    bad = merged["subject_id"].isna()
    if bad.any():
        log.warning(f"  Dropping {bad.sum():,} rows with unparseable patient_id")
        merged = merged[~bad]

    merged["subject_id"] = merged["subject_id"].astype("int64")

    # Deduplication: exact duplicate rows across tables
    before = len(merged)
    merged = merged.drop_duplicates()
    log.info(f"  Deduplication: {before:,} → {len(merged):,} rows ({before - len(merged):,} removed)")

    # Sort: by patient, then NaT (static events) first, then chronological
    log.info("Sorting by subject_id + time …")
    merged = merged.sort_values(
        ["subject_id", "time"],
        na_position="first",   # NaT static events sort to front of each patient timeline
        kind="stable",
    ).reset_index(drop=True)

    # Reorder columns: subject_id first for ETHOS compatibility
    merged = merged[["subject_id", "patient_id", "time", "code", "numeric_value", "text_value"]]

    out_path = output_dir / "events.parquet"
    log.info(f"Saving → {out_path}")
    merged.to_parquet(out_path, index=False, engine="pyarrow")

    # Summary
    log.info("=" * 60)
    log.info(f"  Total events    : {len(merged):>12,}")
    log.info(f"  Unique patients : {merged['subject_id'].nunique():>12,}")
    log.info(f"  Unique codes    : {merged['code'].nunique():>12,}")
    log.info(f"  Static events   : {merged['time'].isna().sum():>12,}  (NaT)")
    log.info(f"  Date range      : {merged['time'].min()} – {merged['time'].max()}")
    log.info(f"  Output          : {out_path}  ({out_path.stat().st_size / 1e9:.2f} GB)")
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge all MEDS parquets into a single events file")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    main(Path(args.output_dir))
