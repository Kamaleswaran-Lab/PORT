"""
surgical_history_to_meds.py
---------------------------
Convert DR15201_Surgical_History_V.csv → MEDS-format Parquet.

MEDS event schema (per row):
    patient_id     : str   — C MRN as-is (e.g. "C800881")
    time           : datetime64[us]
                       Priority: Surg Hx Date Text → Surg Hx Start Date → NaT
                       "Surg Hx Start Date" is often "00:00.0" (time-only artifact) — skipped
    code           : str   — "PROCEDURE//SURG_HX//<NORMALIZED_PROC_NAME>"
    numeric_value  : float — NaN (procedures carry no numeric value)
    text_value     : str   — original Proc Name (human-readable)

No End-date events: Surg Hx Start Date and End Date are both "00:00.0" artifacts
and carry no usable information beyond what Date Text already provides.

Time resolution priority:
    1. Surg Hx Date Text   (M/D/YYYY, 92.1% of rows)
    2. Surg Hx Start Date  (fallback; filtered to rows that parse as a real date,
                            not the "00:00.0" time-only artifact)
    3. NaT                 → null time (MEDS static event, sorted to front)

Usage:
    python surgical_history_to_meds.py [--input PATH] [--output_dir DIR]
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
DEFAULT_INPUT     = "/path/to/CHOA_RAW/DR15201_Surgical_History_V.csv"
DEFAULT_OUTPUT_DIR = Path("/path/to/CHD_MEDS/data")

# ── code normalisation ─────────────────────────────────────────────────────────
_MULTI_SPACE  = re.compile(r"\s+")
_UNSAFE_CHARS = re.compile(r"[^A-Z0-9_/]")

def normalise_name(name: str) -> str:
    """'Repair Strabismus' → 'REPAIR_STRABISMUS'"""
    s = str(name).upper().strip()
    s = s.replace("%", "PCT").replace("#", "NUM")
    s = _MULTI_SPACE.sub("_", s)
    s = _UNSAFE_CHARS.sub("", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# ── time parsing ───────────────────────────────────────────────────────────────
# Values that look like time-only artifacts or non-dates
_TIME_ONLY_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?(\.\d+)?$")  # "00:00.0", "0:00"
_NON_DATE_KEYWORDS = {"null", "n/a", "na", "none", "unknown", "today", "now"}

def is_garbage(s: str) -> bool:
    """Return True for time-only strings and known non-date keywords."""
    s = s.strip().lower()
    return bool(_TIME_ONLY_RE.match(s)) or s in _NON_DATE_KEYWORDS


def parse_date_col(series: pd.Series) -> pd.Series:
    """
    Parse a date column, masking garbage values (time-only artifacts, keywords).
    Handles M/D/YYYY, YYYY-MM-DD, year-only (YYYY → YYYY-01-01), M/YYYY, etc.
    """
    s = series.astype(str).str.strip()

    # Mask garbage
    garbage_mask = s.map(is_garbage) | s.isin(["nan", "NaT", ""])
    s_clean = s.copy()
    s_clean[garbage_mask] = ""

    # Year-only → YYYY-01-01 (prevent pandas from misinterpreting)
    year_only = s_clean.str.fullmatch(r"\d{4}")
    s_clean[year_only] = s_clean[year_only] + "-01-01"

    # Month/year → MM/YYYY (pandas handles this with format inference)
    month_year = s_clean.str.fullmatch(r"\d{1,2}/\d{4}")
    s_clean[month_year] = pd.to_datetime(
        s_clean[month_year], format="%m/%Y", errors="coerce"
    ).dt.strftime("%Y-%m-%d").fillna("")

    parsed = pd.to_datetime(s_clean, errors="coerce", dayfirst=False)
    return parsed


def resolve_time(df: pd.DataFrame) -> pd.Series:
    """
    Priority: Surg Hx Date Text → Surg Hx Start Date → NaT
    """
    date_text = parse_date_col(df["Surg Hx Date Text"])
    start     = parse_date_col(df["Surg Hx Start Date"])

    time = date_text.fillna(start)
    return time


# ── main transform ─────────────────────────────────────────────────────────────
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Filter to valid MRNs
    df = df[df["C MRN"].astype(str).str.startswith("C")].copy()

    time = resolve_time(df)

    # Clamp obviously erroneous dates to NaT
    valid_range = (time >= "1950-01-01") & (time <= "2030-01-01")
    time[~valid_range] = pd.NaT

    code = "PROCEDURE//SURG_HX//" + df["Proc Name"].fillna("UNKNOWN").map(normalise_name)

    out = pd.DataFrame({
        "patient_id":    df["C MRN"].values,
        "time":          time.values,
        "code":          code.values,
        "numeric_value": np.full(len(df), np.nan, dtype="float32"),
        "text_value":    df["Proc Name"].values,
    })
    return out


# ── main ───────────────────────────────────────────────────────────────────────
def main(input_path: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "surgical_history.parquet"

    log.info(f"Input : {input_path}")
    log.info(f"Output: {out_path}")

    df = pd.read_csv(
        input_path,
        encoding="utf-8-sig",
        encoding_errors="replace",
        on_bad_lines="skip",
        low_memory=False,
        usecols=["C MRN", "Date of Birth", "Proc Name",
                 "Surg Hx Date Text", "Surg Hx Start Date", "Surg Hx End Date"],
    )
    df = df.iloc[:-2]
    log.info(f"Loaded {len(df):,} raw rows")

    result = transform(df)

    before = len(result)
    result.drop_duplicates(inplace=True)
    log.info(f"Dropped {before - len(result):,} fully duplicate rows ({before:,} → {len(result):,})")

    log.info("Sorting by patient_id, time …")
    result.sort_values(["patient_id", "time"], inplace=True,
                       na_position="first", ignore_index=True)

    log.info(f"Writing parquet → {out_path}")
    result.to_parquet(out_path, index=False, engine="pyarrow")

    # ── summary ───────────────────────────────────────────────────────────────
    n_patients   = result["patient_id"].nunique()
    n_codes      = result["code"].nunique()
    n_null_time  = result["time"].isna().sum()
    timed        = result["time"].dropna()

    log.info("=" * 55)
    log.info(f"  MEDS events total  : {len(result):>10,}")
    log.info(f"  Null time (static) : {n_null_time:>10,}  ({n_null_time/len(result)*100:.1f}%)")
    log.info(f"  Unique patients    : {n_patients:>10,}")
    log.info(f"  Unique codes       : {n_codes:>10,}")
    if len(timed):
        log.info(f"  Time range         : {timed.min()}  →  {timed.max()}")
    log.info("=" * 55)
    log.info(f"Done. Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Surgical History CSV to MEDS parquet")
    parser.add_argument("--input",      default=DEFAULT_INPUT,           help="Path to DR15201_Surgical_History_V.csv")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    main(args.input, Path(args.output_dir))
