"""
an_events_to_meds.py
--------------------
Convert DR15201_AN_Events.rpt → MEDS-format Parquet.

MEDS event schema (per row):
    patient_id     : str         — C MRN
    time           : datetime64  — Recorded Time
    code           : str         — "AN_EVENT//<EVENT_NORMALIZED>"
    numeric_value  : float32     — NaN (events carry no numeric value)
    text_value     : str         — Event Comment (short free-text, e.g. Quick Note)
                                   NaN when no comment

Event Comment note:
    Unlike full cardiology progress notes, AN_Events comments are short
    (1–3 sentences). Quick Note (94.9% have comments) are stored directly
    in text_value and are already used for IoD2 labeling.

MEDS code examples:
    AN_EVENT//QUICK_NOTE
    AN_EVENT//CPR
    AN_EVENT//ANESTHESIA_STOP
    AN_EVENT//LABS_TAKEN
    AN_EVENT//MARK_NOW

Usage:
    python an_events_to_meds.py [--input PATH] [--output_dir DIR]
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
    "CHOA_DATA_Tables_CHD/DR15201_AN_Events.rpt"
)
DEFAULT_OUTPUT_DIR = Path("/path/to/CHD_MEDS/data")

_MULTI_SPACE  = re.compile(r"\s+")
_UNSAFE_CHARS = re.compile(r"[^A-Z0-9_/]")

def normalise_event(name: str) -> str:
    s = str(name).upper().strip()
    s = _MULTI_SPACE.sub("_", s)
    s = _UNSAFE_CHARS.sub("", s)
    return re.sub(r"_+", "_", s).strip("_")


def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["C MRN"].astype(str).str.startswith("C")].copy()

    time = pd.to_datetime(df["Recorded Time"], errors="coerce")
    valid = time.notna()
    df   = df[valid]
    time = time[valid]

    code = "AN_EVENT//" + df["Event"].fillna("UNKNOWN").map(normalise_event)

    return pd.DataFrame({
        "patient_id":    df["C MRN"].values,
        "time":          time.values,
        "code":          code.values,
        "numeric_value": np.full(len(df), np.nan, dtype="float32"),
        "text_value":    df["Event Comment"].values,
    })


def main(input_path: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "an_events.parquet"

    log.info(f"Input : {input_path}")
    log.info(f"Output: {out_path}")

    df = pd.read_csv(
        input_path,
        delimiter="|",
        encoding="utf-8-sig",
        encoding_errors="replace",
        on_bad_lines="skip",
        low_memory=False,
        usecols=["C MRN", "Recorded Time", "Event", "Event Comment"],
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

    n_patients   = result["patient_id"].nunique()
    n_codes      = result["code"].nunique()
    has_comment  = result["text_value"].notna().sum()

    log.info("=" * 55)
    log.info(f"  MEDS events total  : {len(result):>10,}")
    log.info(f"  With text comment  : {has_comment:>10,}  ({has_comment/len(result)*100:.1f}%)")
    log.info(f"  Unique patients    : {n_patients:>10,}")
    log.info(f"  Unique event codes : {n_codes:>10,}")
    log.info("  Top events:")
    for code, cnt in result["code"].value_counts().head(8).items():
        log.info(f"    {code:<45}: {cnt:>8,}")
    log.info(f"  Time range         : {result['time'].min()}  →  {result['time'].max()}")
    log.info("=" * 55)
    log.info(f"Done. Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert AN_Events .rpt to MEDS parquet")
    parser.add_argument("--input",      default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    main(args.input, Path(args.output_dir))
