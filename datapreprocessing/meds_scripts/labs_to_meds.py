"""
labs_to_meds.py
---------------
Convert DR15201_LABS.rpt → MEDS-format Parquet.

MEDS event schema (per row):
    patient_id     : str   — C MRN as-is (e.g. "C800881")
    time           : datetime64[us]  — Specimen_Taken_Time, fallback Result Date
    code           : str   — "LAB//<COMPONENT_NORMALIZED>"
    numeric_value  : float — parsed Result (NaN if non-numeric)
    text_value     : str   — original Result string when non-numeric, else None

Special cases:
    - Results like "<0.2", "<=0.5"  → numeric_value = parsed float,
                                       code gets suffix "//LT" (less-than flag)
    - Results like ">100"           → numeric_value = parsed float,
                                       code gets suffix "//GT"
    - Purely text results           → numeric_value = NaN, text_value = Result

Usage:
    python labs_to_meds.py [--input PATH] [--output_dir DIR] [--chunk_size N]
"""

import re
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_INPUT = (
    "/path/to/CHOA_RAW/DR15201_LABS/DR15201_LABS/DR15201_LABS.rpt"
)
DEFAULT_OUTPUT_DIR = Path("/path/to/CHD_MEDS/data")
CHUNK_SIZE = 500_000          # rows per chunk; tune to available RAM

# ── code normalisation ─────────────────────────────────────────────────────────
_MULTI_SPACE = re.compile(r"\s+")
_UNSAFE_CHARS = re.compile(r"[^A-Z0-9_/]")

def normalise_component(name: str) -> str:
    """'Hemoglobin A1c (%)' → 'HEMOGLOBIN_A1C_PCT'"""
    s = name.upper().strip()
    s = s.replace("%", "PCT").replace("#", "NUM")
    s = _MULTI_SPACE.sub("_", s)
    s = _UNSAFE_CHARS.sub("", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# ── result parsing ─────────────────────────────────────────────────────────────
_LT_RE = re.compile(r"^[<＜]\s*=?\s*([0-9]*\.?[0-9]+)")
_GT_RE = re.compile(r"^[>＞]\s*=?\s*([0-9]*\.?[0-9]+)")

def parse_result(series: pd.Series):
    """
    Returns (numeric_values, text_values, code_suffixes).
    Each is a pd.Series aligned to the input index.
    """
    s = series.astype(str).str.strip()

    # Straight numeric
    numeric = pd.to_numeric(s, errors="coerce")
    is_numeric = numeric.notna()

    # Less-than  (<0.2, <=0.5)
    lt_match = s.str.extract(_LT_RE, expand=False).rename("lt")
    is_lt = lt_match.notna() & ~is_numeric

    # Greater-than (>100)
    gt_match = s.str.extract(_GT_RE, expand=False).rename("gt")
    is_gt = gt_match.notna() & ~is_numeric & ~is_lt

    # Build outputs
    num_vals = numeric.copy()
    num_vals[is_lt] = pd.to_numeric(lt_match[is_lt], errors="coerce")
    num_vals[is_gt] = pd.to_numeric(gt_match[is_gt], errors="coerce")

    code_suffix = pd.Series("", index=s.index, dtype=str)
    code_suffix[is_lt] = "//LT"
    code_suffix[is_gt] = "//GT"

    # text_value: keep original string for purely-text results
    is_text = ~is_numeric & ~is_lt & ~is_gt
    text_vals = pd.Series(pd.NA, index=s.index, dtype=object)
    text_vals[is_text] = series[is_text]
    # store original for lt/gt too so nothing is lost
    text_vals[is_lt] = series[is_lt]
    text_vals[is_gt] = series[is_gt]

    return num_vals, text_vals, code_suffix


# ── per-chunk transform ────────────────────────────────────────────────────────
def transform_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    # ── time: Specimen_Taken_Time, fallback Result Date ──
    time = pd.to_datetime(chunk["Specimen_Taken_Time"], errors="coerce")
    fallback = pd.to_datetime(chunk["Result Date"], errors="coerce")
    time = time.fillna(fallback)

    # Drop rows with no usable time
    valid = time.notna()
    chunk = chunk[valid].copy()
    time = time[valid]

    # ── result parsing ───────────────────────────────────
    num_vals, text_vals, code_suffix = parse_result(chunk["Result"])

    # ── code ─────────────────────────────────────────────
    comp_norm = chunk["Component"].fillna("UNKNOWN").map(normalise_component)
    code = "LAB//" + comp_norm + code_suffix

    # ── assemble ─────────────────────────────────────────
    out = pd.DataFrame(
        {
            "patient_id":    chunk["C MRN"].values,
            "time":          time.values,
            "code":          code.values,
            "numeric_value": num_vals.values.astype("float32"),
            "text_value":    text_vals.values,
        }
    )
    return out


# ── main ───────────────────────────────────────────────────────────────────────
def main(input_path: str, output_dir: Path, chunk_size: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "labs.parquet"

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
        usecols=[
            "C MRN", "Specimen_Taken_Time", "Result Date",
            "Component", "Result",
        ],
    )

    parts = []
    total_rows_in = 0
    total_rows_out = 0
    chunk_idx = 0

    for chunk in reader:
        chunk_idx += 1

        # strip footer rows (contain "UTC DateTime" etc.)
        chunk = chunk[chunk["C MRN"].astype(str).str.startswith("C")].copy()

        total_rows_in += len(chunk)
        transformed = transform_chunk(chunk)
        total_rows_out += len(transformed)
        parts.append(transformed)

        if chunk_idx % 5 == 0:
            log.info(
                f"  Processed {total_rows_in:>9,} input rows → "
                f"{total_rows_out:>9,} MEDS events"
            )

    log.info(f"Finished reading. Concatenating {len(parts)} chunks …")
    df = pd.concat(parts, ignore_index=True)

    before = len(df)
    df.drop_duplicates(inplace=True)
    log.info(f"Dropped {before - len(df):,} fully duplicate rows ({before:,} → {len(df):,})")

    # Sort by patient then time (required by MEDS spec)
    log.info("Sorting by patient_id, time …")
    df.sort_values(["patient_id", "time"], inplace=True, ignore_index=True)

    log.info(f"Writing parquet → {out_path}")
    df.to_parquet(out_path, index=False, engine="pyarrow")

    # ── summary ──────────────────────────────────────────
    n_patients = df["patient_id"].nunique()
    n_codes    = df["code"].nunique()
    pct_numeric = df["numeric_value"].notna().mean() * 100

    log.info("=" * 55)
    log.info(f"  Input rows       : {total_rows_in:>10,}")
    log.info(f"  MEDS events      : {total_rows_out:>10,}")
    log.info(f"  Unique patients  : {n_patients:>10,}")
    log.info(f"  Unique codes     : {n_codes:>10,}")
    log.info(f"  Numeric results  : {pct_numeric:>9.1f}%")
    log.info(f"  Time range       : {df['time'].min()}  →  {df['time'].max()}")
    log.info("=" * 55)
    log.info(f"Done. Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LABS .rpt to MEDS parquet")
    parser.add_argument("--input",       default=DEFAULT_INPUT,      help="Path to DR15201_LABS.rpt")
    parser.add_argument("--output_dir",  default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--chunk_size",  default=CHUNK_SIZE, type=int, help="Rows per chunk")
    args = parser.parse_args()

    main(args.input, Path(args.output_dir), args.chunk_size)
