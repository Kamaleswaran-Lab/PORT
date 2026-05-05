"""
medications_to_meds.py
----------------------
Convert DR15201_Medication_Administration.rpt → MEDS-format Parquet.

MEDS event schema (per row):
    patient_id     : str         — C MRN
    time           : datetime64  — MAR Time (medication administration record time)
    code           : str         — "MED//<NORMALIZED_MEDICATION_NAME>"
    numeric_value  : float32     — Dose amount (NaN if non-numeric or missing)
    text_value     : str         — original Medication name

Source is pipe-delimited .rpt (17.3M rows) — no CSV column-shift issue.
The truncated CSV version (DR15201_Medication_Administrations_V.csv) contains
only 1,048,575 rows due to Excel export row limit; use this .rpt instead.

Usage:
    python medications_to_meds.py [--input PATH] [--output_dir DIR] [--chunk_size N]
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
    "/path/to/CHOA_RAW/DR15201_Medication_Administration/"
    "DR15201_Medication_Administration/DR15201_Medication_Administration.rpt"
)
DEFAULT_OUTPUT_DIR = Path("/path/to/CHD_MEDS/data")
CHUNK_SIZE = 500_000

# ── normalisation ──────────────────────────────────────────────────────────────
_MULTI_SPACE  = re.compile(r"\s+")
_UNSAFE_CHARS = re.compile(r"[^A-Z0-9_/]")

def normalise_med(name: str) -> str:
    """'MORPHINE 2 MG/ML INTRAVENOUS CARTRIDGE' → 'MORPHINE_2_MGML_INTRAVENOUS_CARTRIDGE'"""
    s = str(name).upper().strip()
    s = s.replace("%", "PCT")
    s = _MULTI_SPACE.sub("_", s)
    s = _UNSAFE_CHARS.sub("", s)
    return re.sub(r"_+", "_", s).strip("_")


# ── per-chunk transform ────────────────────────────────────────────────────────
def transform_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk[chunk["C MRN"].astype(str).str.startswith("C")].copy()
    if chunk.empty:
        return pd.DataFrame(columns=["patient_id","time","code","numeric_value","text_value"])

    time = pd.to_datetime(chunk["MAR Time"], errors="coerce")

    # Drop rows with no usable timestamp
    valid = time.notna()
    chunk = chunk[valid]
    time  = time[valid]

    code    = "MED//" + chunk["Medication"].fillna("UNKNOWN").map(normalise_med)
    numeric = pd.to_numeric(chunk["Dose"], errors="coerce").astype("float32")

    return pd.DataFrame({
        "patient_id":    chunk["C MRN"].values,
        "time":          time.values,
        "code":          code.values,
        "numeric_value": numeric.values,
        "text_value":    chunk["Medication"].values,
    })


# ── main ───────────────────────────────────────────────────────────────────────
def main(input_path: str, output_dir: Path, chunk_size: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "medications.parquet"

    log.info(f"Input : {input_path}")
    log.info(f"Output: {out_path}")
    log.info(f"Chunk size: {chunk_size:,}")

    reader = pd.read_csv(
        input_path,
        delimiter="|",
        encoding="utf-8-sig",
        encoding_errors="replace",
        on_bad_lines="skip",
        low_memory=False,
        chunksize=chunk_size,
        usecols=["C MRN", "MAR Time", "Medication", "Dose"],
    )

    parts    = []
    total_in = 0
    total_out = 0

    for i, chunk in enumerate(reader, 1):
        # Strip footer rows
        chunk = chunk[chunk["C MRN"].astype(str).str.startswith("C")]
        total_in += len(chunk)
        transformed = transform_chunk(chunk)
        total_out += len(transformed)
        parts.append(transformed)
        if i % 5 == 0:
            log.info(f"  chunk {i:3d} — {total_in:>10,} in → {total_out:>10,} out")

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
    n_patients  = df["patient_id"].nunique()
    n_codes     = df["code"].nunique()
    pct_numeric = df["numeric_value"].notna().mean() * 100

    log.info("=" * 55)
    log.info(f"  Input rows         : {total_in:>10,}")
    log.info(f"  MEDS events        : {total_out:>10,}")
    log.info(f"  Numeric dose       : {pct_numeric:>9.1f}%")
    log.info(f"  Unique patients    : {n_patients:>10,}")
    log.info(f"  Unique med codes   : {n_codes:>10,}")
    log.info(f"  Time range         : {df['time'].min()}  →  {df['time'].max()}")
    log.info("=" * 55)
    log.info(f"Done. Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Medication Administration .rpt to MEDS parquet")
    parser.add_argument("--input",      default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--chunk_size", default=CHUNK_SIZE, type=int)
    args = parser.parse_args()
    main(args.input, Path(args.output_dir), args.chunk_size)
