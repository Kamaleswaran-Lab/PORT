"""
lda_to_meds.py
--------------
Convert DR15201_LDAs.rpt → MEDS-format Parquet.

Two MEDS events are emitted per LDA record:
    PLACED  event at LDA Placed  → "LDA//<LDA_TYPE_NORMALIZED>"
    REMOVED event at LDA Removed → "LDA//<LDA_TYPE_NORMALIZED>//REMOVED"

MEDS code examples:
    LDA//ARTERIAL_LIN
    LDA//ARTERIAL_LIN//REMOVED
    LDA//ETT
    LDA//CVL
    LDA//ECMO//REMOVED

Usage:
    python lda_to_meds.py [--input PATH] [--output_dir DIR]
"""

import re
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DEFAULT_INPUT = (
    "/path/to/CHOA_RAW_TABLES/"
    "CHOA_DATA_Tables_CHD/DR15201_LDAs.rpt"
)
DEFAULT_OUTPUT_DIR = Path("/path/to/CHD_MEDS/data")

_MULTI_SPACE  = re.compile(r"\s+")
_UNSAFE_CHARS = re.compile(r"[^A-Z0-9_/]")

def normalise_lda(name: str) -> str:
    s = str(name).upper().strip()
    s = _MULTI_SPACE.sub("_", s)
    s = _UNSAFE_CHARS.sub("", s)
    return re.sub(r"_+", "_", s).strip("_")


def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["C MRN"].astype(str).str.startswith("C")].copy()

    placed  = pd.to_datetime(df["LDA Placed"],  errors="coerce")
    removed = pd.to_datetime(df["LDA Removed"], errors="coerce")

    # Clamp erroneous dates to NaT
    valid_range = lambda t: (t >= "1950-01-01") & (t <= "2030-01-01")
    placed[~valid_range(placed)]   = pd.NaT
    removed[~valid_range(removed)] = pd.NaT
    lda_norm = df["LDA Type"].fillna("UNKNOWN").map(normalise_lda)
    lda_raw  = df["LDA Type"].values

    # PLACED events
    valid_p = placed.notna()
    placed_events = pd.DataFrame({
        "patient_id":    df.loc[valid_p, "C MRN"].values,
        "time":          placed[valid_p].values,
        "code":          ("LDA//" + lda_norm[valid_p]).values,
        "numeric_value": np.full(valid_p.sum(), np.nan, dtype="float32"),
        "text_value":    lda_raw[valid_p.values],
    })

    # REMOVED events
    valid_r = removed.notna()
    removed_events = pd.DataFrame({
        "patient_id":    df.loc[valid_r, "C MRN"].values,
        "time":          removed[valid_r].values,
        "code":          ("LDA//" + lda_norm[valid_r] + "//REMOVED").values,
        "numeric_value": np.full(valid_r.sum(), np.nan, dtype="float32"),
        "text_value":    lda_raw[valid_r.values],
    })

    return pd.concat([placed_events, removed_events], ignore_index=True)


def main(input_path: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "ldas.parquet"

    log.info(f"Input : {input_path}")
    log.info(f"Output: {out_path}")

    df = pd.read_csv(
        input_path,
        delimiter="|",
        encoding="utf-8-sig",
        encoding_errors="replace",
        on_bad_lines="skip",
        low_memory=False,
        usecols=["C MRN", "LDA Type", "LDA Placed", "LDA Removed"],
    )
    df = df.iloc[:-2]
    log.info(f"Loaded {len(df):,} raw rows")

    result = transform(df)

    before = len(result)
    result.drop_duplicates(inplace=True)
    log.info(f"Dropped {before - len(result):,} fully duplicate rows ({before:,} → {len(result):,})")

    log.info("Sorting by patient_id, time …")
    result.sort_values(["patient_id", "time"], inplace=True, ignore_index=True)

    log.info(f"Writing parquet → {out_path}")
    result.to_parquet(out_path, index=False, engine="pyarrow")

    n_placed  = (~result["code"].str.endswith("//REMOVED")).sum()
    n_removed = result["code"].str.endswith("//REMOVED").sum()

    log.info("=" * 55)
    log.info(f"  MEDS events total  : {len(result):>10,}")
    log.info(f"    PLACED           : {n_placed:>10,}")
    log.info(f"    REMOVED          : {n_removed:>10,}")
    log.info(f"  Unique patients    : {result['patient_id'].nunique():>10,}")
    log.info(f"  Unique LDA codes   : {result['code'].nunique() // 2:>10,}")
    log.info(f"  Time range         : {result['time'].min()}  →  {result['time'].max()}")
    log.info("=" * 55)
    log.info(f"Done. Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LDAs .rpt to MEDS parquet")
    parser.add_argument("--input",      default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    main(args.input, Path(args.output_dir))
