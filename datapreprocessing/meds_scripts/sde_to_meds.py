"""
sde_to_meds.py
--------------
Convert DR15201_SDEs_V.csv → MEDS-format Parquet.

MEDS event schema (per row):
    patient_id     : str   — C MRN as-is (e.g. "C800881")
    time           : datetime64[us]  — Date of Service (full datetime)
    code           : str   — "SDE//<NOTE_TYPE>//<NORMALIZED_ELEMENT_NAME>"
                             NOTE_TYPE: PRE | POST | PROC
    numeric_value  : float — parsed SDE Response if numeric, else NaN
    text_value     : str   — full reconstructed SDE Response string

Overflow reconstruction:
    When SDE Response contains '|' characters, the CSV parser splits the
    value across Column1, _1 … _20.  These fragments are rejoined with '|'
    to recover the original response.

Multi-line responses:
    A Note ID + Element Name combination can span multiple rows (Line 1, 2, …).
    Each row is kept as a separate MEDS event (one row = one event).

Note Type abbreviations:
    Anesthesia Preprocedure Evaluation  → PRE
    Anesthesia Postprocedure Evaluation → POST
    Anesthesia Procedure Notes          → PROC
    (anything else)                     → OTHER

Usage:
    python sde_to_meds.py [--input PATH] [--output_dir DIR] [--chunk_size N]
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
DEFAULT_INPUT      = "/path/to/CHOA_RAW/DR15201_SDEs_V.csv"
DEFAULT_OUTPUT_DIR = Path("/path/to/CHD_MEDS/data")
CHUNK_SIZE         = 200_000

# ── constants ─────────────────────────────────────────────────────────────────
NOTE_TYPE_MAP = {
    "anesthesia preprocedure evaluation":  "PRE",
    "anesthesia postprocedure evaluation": "POST",
    "anesthesia procedure notes":          "PROC",
}

OVERFLOW_COLS = ["Column1"] + [f"_{i}" for i in range(1, 21)]

# ── normalisation ──────────────────────────────────────────────────────────────
_MULTI_SPACE  = re.compile(r"\s+")
_SEP          = re.compile(r"\s*-\s*")        # " - " separators in element names
_UNSAFE_CHARS = re.compile(r"[^A-Z0-9_/]")

def normalise_element(name: str) -> str:
    """
    'FINDINGS - PHYSICAL EXAM - CARDIOVASCULAR - MURMUR'
    → 'FINDINGS_PHYSICAL_EXAM_CARDIOVASCULAR_MURMUR'
    """
    s = str(name).upper().strip()
    s = s.replace("%", "PCT").replace("#", "NUM")
    s = _SEP.sub("_", s)
    s = _MULTI_SPACE.sub("_", s)
    s = _UNSAFE_CHARS.sub("", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def note_type_short(nt_series: pd.Series) -> pd.Series:
    return nt_series.str.lower().str.strip().map(NOTE_TYPE_MAP).fillna("OTHER")


# ── overflow reconstruction ────────────────────────────────────────────────────
def reconstruct_response(chunk: pd.DataFrame) -> pd.Series:
    """
    Rejoin SDE Response fragments split by '|' across Column1 and _1.._20.
    Only rows where overflow cols are non-null get the extra join overhead.
    """
    base = chunk["SDE Response"].astype(str).str.strip().replace("nan", "")

    # Which rows actually have overflow?
    present = [c for c in OVERFLOW_COLS if c in chunk.columns]
    if not present:
        return base

    has_overflow = chunk[present].notna().any(axis=1)
    if not has_overflow.any():
        return base

    # Build concatenated string only for overflow rows
    parts = [base]
    for col in present:
        if col in chunk.columns:
            parts.append(
                chunk[col].where(chunk[col].notna(), "").astype(str).replace("nan", "")
            )

    joined = parts[0].copy()
    for extra in parts[1:]:
        mask = extra != ""
        joined[mask] = joined[mask] + "|" + extra[mask]

    # Strip leading/trailing pipes introduced by joining
    joined[has_overflow] = joined[has_overflow].str.strip("|")

    return joined


# ── per-chunk transform ────────────────────────────────────────────────────────
def transform_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    # Valid MRNs only
    chunk = chunk[chunk["C MRN"].astype(str).str.startswith("C")].copy()
    if chunk.empty:
        return pd.DataFrame(columns=["patient_id","time","code","numeric_value","text_value"])

    # Time
    time = pd.to_datetime(chunk["Date of Service"], errors="coerce")

    # Code
    nt_short = note_type_short(chunk["Note Type"])
    elem_norm = chunk["Element Name"].fillna("UNKNOWN").map(normalise_element)
    code = "SDE//" + nt_short + "//" + elem_norm

    # Response — reconstruct overflow then parse numeric
    response = reconstruct_response(chunk)
    # Replace empty / "nan" strings with NA
    response = response.replace({"nan": pd.NA, "": pd.NA})

    numeric = pd.to_numeric(response, errors="coerce").astype("float32")

    out = pd.DataFrame({
        "patient_id":    chunk["C MRN"].values,
        "time":          time.values,
        "code":          code.values,
        "numeric_value": numeric.values,
        "text_value":    response.values,
    })

    # Drop rows with no usable time
    out = out[pd.to_datetime(out["time"], errors="coerce").notna()]
    return out


# ── main ───────────────────────────────────────────────────────────────────────
def main(input_path: str, output_dir: Path, chunk_size: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "sde.parquet"

    log.info(f"Input : {input_path}")
    log.info(f"Output: {out_path}")
    log.info(f"Chunk size: {chunk_size:,}")

    # Columns to load
    usecols = [
        "C MRN", "Date of Service", "Note Type",
        "Element Name", "SDE Response",
    ] + [c for c in OVERFLOW_COLS]

    reader = pd.read_csv(
        input_path,
        encoding="utf-8-sig",
        encoding_errors="replace",
        on_bad_lines="skip",
        low_memory=False,
        chunksize=chunk_size,
        usecols=lambda c: c in usecols,
    )

    parts = []
    total_in = 0
    total_out = 0

    for i, chunk in enumerate(reader, 1):
        # Strip last 2 footer rows only in final chunk — filter by MRN prefix instead
        total_in += len(chunk)
        transformed = transform_chunk(chunk)
        total_out += len(transformed)
        parts.append(transformed)

        if i % 5 == 0:
            log.info(f"  chunk {i:3d} — {total_in:>9,} in → {total_out:>9,} out")

    log.info(f"Finished reading. Concatenating {len(parts)} chunks …")
    df = pd.concat(parts, ignore_index=True)

    log.info("Sorting by patient_id, time …")
    df.sort_values(["patient_id", "time"], inplace=True, ignore_index=True)

    log.info(f"Writing parquet → {out_path}")
    df.to_parquet(out_path, index=False, engine="pyarrow")

    # ── summary ───────────────────────────────────────────────────────────────
    n_patients  = df["patient_id"].nunique()
    n_codes     = df["code"].nunique()
    pct_numeric = df["numeric_value"].notna().mean() * 100

    note_counts = df["code"].str.extract(r"SDE//(PRE|POST|PROC|OTHER)//")[0].value_counts()

    log.info("=" * 55)
    log.info(f"  Input rows         : {total_in:>10,}")
    log.info(f"  MEDS events        : {total_out:>10,}")
    for nt, cnt in note_counts.items():
        log.info(f"    {nt:<6}           : {cnt:>10,}  ({cnt/total_out*100:.1f}%)")
    log.info(f"  Numeric responses  : {pct_numeric:>9.1f}%")
    log.info(f"  Unique patients    : {n_patients:>10,}")
    log.info(f"  Unique codes       : {n_codes:>10,}")
    log.info(f"  Time range         : {df['time'].min()}  →  {df['time'].max()}")
    log.info("=" * 55)
    log.info(f"Done. Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SDEs CSV to MEDS parquet")
    parser.add_argument("--input",      default=DEFAULT_INPUT,           help="Path to DR15201_SDEs_V.csv")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--chunk_size", default=CHUNK_SIZE, type=int,    help="Rows per chunk")
    args = parser.parse_args()

    main(args.input, Path(args.output_dir), args.chunk_size)
