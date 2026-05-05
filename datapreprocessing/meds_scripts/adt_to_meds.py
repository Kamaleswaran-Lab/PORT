"""
adt_to_meds.py
--------------
Convert DR15201_ADT_Patient_Location.rpt → MEDS-format Parquet.

Two MEDS events are emitted per unique ADT event (EVENT_ID):
    IN  event at IN_DTTM  → "ADT//<DEPT_NORMALIZED>"
    OUT event at OUT_DTTM → "ADT//<DEPT_NORMALIZED>//OUT"

This lets temporal models track when a patient enters and leaves each
department (OR, ICU, PACU, floor, ED, etc.).

Deduplication:
    The raw file has duplicate rows per EVENT_ID (mean 1.2×, max 17×).
    We deduplicate on EVENT_ID before emitting events.

MEDS code examples:
    ADT//EG_OR
    ADT//EG_OR//OUT
    ADT//EG_CARDIAC_ICU
    ADT//EG_NEONATAL_ICU//OUT
    ADT//SR_PEDIATRIC_ICU

Usage:
    python adt_to_meds.py [--input PATH] [--output_dir DIR] [--chunk_size N]
"""

import re
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── defaults ───────────────────────────────────────────────────────────────────
DEFAULT_INPUT = (
    "/path/to/CHOA_RAW_TABLES/"
    "CHOA_DATA_Tables_CHD/DR15201_ADT_Patient_Location.rpt"
)
DEFAULT_OUTPUT_DIR = Path("/path/to/CHD_MEDS/data")
CHUNK_SIZE = 300_000

# ── normalisation ──────────────────────────────────────────────────────────────
_MULTI_SPACE  = re.compile(r"\s+")
_UNSAFE_CHARS = re.compile(r"[^A-Z0-9_/]")

def normalise_dept(name: str) -> str:
    """'EG CARDIAC ICU' → 'EG_CARDIAC_ICU'"""
    s = str(name).upper().strip()
    s = _MULTI_SPACE.sub("_", s)
    s = _UNSAFE_CHARS.sub("", s)
    return re.sub(r"_+", "_", s).strip("_")


# ── per-chunk transform ────────────────────────────────────────────────────────
def transform_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    # Valid MRNs
    chunk = chunk[chunk["C MRN"].astype(str).str.startswith("C")].copy()
    if chunk.empty:
        return pd.DataFrame(columns=["patient_id", "time", "code", "numeric_value", "text_value"])

    # Deduplicate on EVENT_ID (keep first)
    chunk = chunk.drop_duplicates(subset=["EVENT_ID"])

    in_time  = pd.to_datetime(chunk["IN_DTTM"],  errors="coerce")
    out_time = pd.to_datetime(chunk["OUT_DTTM"], errors="coerce")

    dept_norm = chunk["ADT_DEPARTMENT_NAME"].fillna("UNKNOWN").map(normalise_dept)
    dept_raw  = chunk["ADT_DEPARTMENT_NAME"].values

    # ── IN events ────────────────────────────────────────────────────────────
    valid_in = in_time.notna()
    in_events = pd.DataFrame({
        "patient_id":    chunk.loc[valid_in, "C MRN"].values,
        "time":          in_time[valid_in].values,
        "code":          ("ADT//" + dept_norm[valid_in]).values,
        "numeric_value": np.full(valid_in.sum(), np.nan, dtype="float32"),
        "text_value":    dept_raw[valid_in.values],
    })

    # ── OUT events ────────────────────────────────────────────────────────────
    valid_out = out_time.notna()
    out_events = pd.DataFrame({
        "patient_id":    chunk.loc[valid_out, "C MRN"].values,
        "time":          out_time[valid_out].values,
        "code":          ("ADT//" + dept_norm[valid_out] + "//OUT").values,
        "numeric_value": np.full(valid_out.sum(), np.nan, dtype="float32"),
        "text_value":    dept_raw[valid_out.values],
    })

    return pd.concat([in_events, out_events], ignore_index=True)


# ── main ───────────────────────────────────────────────────────────────────────
def main(input_path: str, output_dir: Path, chunk_size: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "adt.parquet"

    log.info(f"Input : {input_path}")
    log.info(f"Output: {out_path}")

    reader = pd.read_csv(
        input_path,
        delimiter="|",
        encoding="utf-8-sig",
        encoding_errors="replace",
        on_bad_lines="skip",
        low_memory=False,
        chunksize=chunk_size,
        usecols=["C MRN", "EVENT_ID", "EVENT_TYPE",
                 "IN_DTTM", "OUT_DTTM", "ADT_DEPARTMENT_NAME"],
    )

    parts = []
    total_in = 0
    total_out = 0

    for i, chunk in enumerate(reader, 1):
        # Strip footer rows
        chunk = chunk[chunk["C MRN"].astype(str).str.startswith("C")]
        total_in += len(chunk)
        transformed = transform_chunk(chunk)
        total_out += len(transformed)
        parts.append(transformed)
        if i % 3 == 0:
            log.info(f"  chunk {i} — {total_in:>9,} in → {total_out:>9,} out")

    log.info("Concatenating …")
    df = pd.concat(parts, ignore_index=True)

    before = len(df)
    df.drop_duplicates(inplace=True)
    log.info(f"Dropped {before - len(df):,} fully duplicate rows ({before:,} → {len(df):,})")

    log.info("Sorting by patient_id, time …")
    df.sort_values(["patient_id", "time"], inplace=True, ignore_index=True)

    log.info(f"Writing parquet → {out_path}")
    df.to_parquet(out_path, index=False, engine="pyarrow")

    # ── summary ───────────────────────────────────────────────────────────────
    n_patients = df["patient_id"].nunique()
    n_codes    = df["code"].nunique()
    in_events  = (~df["code"].str.endswith("//OUT")).sum()
    out_events = df["code"].str.endswith("//OUT").sum()

    log.info("=" * 55)
    log.info(f"  Raw rows read      : {total_in:>10,}")
    log.info(f"  MEDS events total  : {total_out:>10,}")
    log.info(f"    IN  events       : {in_events:>10,}")
    log.info(f"    OUT events       : {out_events:>10,}")
    log.info(f"  Unique patients    : {n_patients:>10,}")
    log.info(f"  Unique dept codes  : {n_codes // 2:>10,}")
    log.info(f"  Time range         : {df['time'].min()}  →  {df['time'].max()}")
    log.info("=" * 55)
    log.info(f"Done. Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ADT Patient Location .rpt to MEDS parquet")
    parser.add_argument("--input",      default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--chunk_size", default=CHUNK_SIZE, type=int)
    args = parser.parse_args()
    main(args.input, Path(args.output_dir), args.chunk_size)
