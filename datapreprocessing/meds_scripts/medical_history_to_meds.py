"""
medical_history_to_meds.py
--------------------------
Convert DR15201_Medical_History.rpt → MEDS-format Parquet.

MEDS event schema (per row):
    patient_id     : str   — C MRN as-is (e.g. "C800881")
    time           : datetime64[us]  — Med Hx Start Date, fallback Med Hx Date Text parse
                                       "At Birth" → Date of Birth
                                       unparseable / missing → NaT (null = static event)
    code           : str   — "DIAGNOSIS//ICD10//<ICD10_CODE>"
                             fallback: "DIAGNOSIS//MEDHX//<NORMALIZED_DIAGNOSIS_NAME>"
    numeric_value  : float — NaN (diagnoses carry no numeric value)
    text_value     : str   — Diagnosis Name (human-readable label)

End-date events:
    When Med Hx End Date is present, a second event is emitted with
    code suffix "//END" at that timestamp (MEDS convention for date ranges).

Time resolution priority:
    1. Med Hx Start Date   (full datetime, 23.5% of rows)
    2. Med Hx Date Text    (free-text: "2/3/2016", "2019", "1/2016", "3-1-17", …)
       "At Birth"          → use Date of Birth column
    3. NaT                 → null time (MEDS static event, sorted to front)

Usage:
    python medical_history_to_meds.py [--input PATH] [--output_dir DIR]
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
    "CHOA_DATA_Tables_CHD/DR15201_Medical_History.rpt"
)
DEFAULT_OUTPUT_DIR = Path("/path/to/CHD_MEDS/data")

# ── code normalisation ─────────────────────────────────────────────────────────
_MULTI_SPACE = re.compile(r"\s+")
_UNSAFE_CHARS = re.compile(r"[^A-Z0-9_/]")

def normalise_name(name: str) -> str:
    """'Reactive airway disease' → 'REACTIVE_AIRWAY_DISEASE'"""
    s = str(name).upper().strip()
    s = s.replace("%", "PCT").replace("#", "NUM")
    s = _MULTI_SPACE.sub("_", s)
    s = _UNSAFE_CHARS.sub("", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# ── time parsing ───────────────────────────────────────────────────────────────
def parse_date_text(date_text: pd.Series, date_of_birth: pd.Series) -> pd.Series:
    """
    Parse Med Hx Date Text free-text column into datetime.
    Special cases:
        "At Birth"  → use Date of Birth
        year-only   → YYYY-01-01
        month/year  → YYYY-MM-01
        garbage     → NaT
    """
    s = date_text.astype(str).str.strip()

    # Known non-date keywords → NaT before any parsing
    NON_DATE_KEYWORDS = {"at birth", "today", "now", "unknown", "n/a", "na", "none", "null"}
    is_at_birth = s.str.lower() == "at birth"
    is_non_date = s.str.lower().isin(NON_DATE_KEYWORDS - {"at birth"})

    # Try pandas flexible parser (handles M/D/YYYY, YYYY-MM-DD, M-D-YY, etc.)
    s_masked = s.copy()
    s_masked[is_non_date] = pd.NaT  # prevent pandas from interpreting "today" etc.
    parsed = pd.to_datetime(s_masked, errors="coerce", dayfirst=False)

    # Year-only strings ("2016") — pandas may misparse as today's date; detect & override
    year_only = s.str.fullmatch(r"\d{4}")
    parsed[year_only] = pd.to_datetime(s[year_only] + "-01-01", errors="coerce")

    # Month/year strings ("1/2016", "01/2016")
    month_year = s.str.fullmatch(r"\d{1,2}/\d{4}")
    parsed[month_year] = pd.to_datetime(s[month_year], format="%m/%Y", errors="coerce")

    # "At Birth" override
    dob_parsed = pd.to_datetime(date_of_birth, errors="coerce")
    parsed[is_at_birth] = dob_parsed[is_at_birth]

    return parsed


def resolve_time(df: pd.DataFrame) -> pd.Series:
    """
    Priority: Med Hx Start Date → Med Hx Date Text → NaT
    """
    start = pd.to_datetime(df["Med Hx Start Date "], errors="coerce")
    text  = parse_date_text(df["Med Hx Date Text"], df["Date of Birth"])

    # Fill: use text time where start is missing
    time = start.fillna(text)
    return time


# ── code building ──────────────────────────────────────────────────────────────
def build_code(icd10: pd.Series, diag_name: pd.Series) -> pd.Series:
    """
    ICD10 present  → "DIAGNOSIS//ICD10//<code>"
    ICD10 missing  → "DIAGNOSIS//MEDHX//<NORMALIZED_NAME>"
    """
    has_icd = icd10.notna() & (icd10.astype(str).str.strip() != "NULL")

    code = pd.Series("", index=icd10.index, dtype=str)
    code[has_icd]  = "DIAGNOSIS//ICD10//" + icd10[has_icd].astype(str).str.strip()
    code[~has_icd] = "DIAGNOSIS//MEDHX//" + diag_name[~has_icd].fillna("UNKNOWN").map(normalise_name)
    return code


# ── main transform ─────────────────────────────────────────────────────────────
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Filter to valid MRNs
    df = df[df["C MRN"].astype(str).str.startswith("C")].copy()

    time  = resolve_time(df)
    # Clamp obviously erroneous dates (EHR data entry errors) to NaT
    valid_range = (time >= "1950-01-01") & (time <= "2030-01-01")
    time[~valid_range] = pd.NaT

    code  = build_code(df["ICD10"], df["Diagnosis Name"])
    end_t = pd.to_datetime(df["Med Hx End Date "], errors="coerce")
    end_t[~((end_t >= "1950-01-01") & (end_t <= "2030-01-01"))] = pd.NaT

    # ── onset events ──────────────────────────────────────────────────────────
    onset = pd.DataFrame({
        "patient_id":    df["C MRN"].values,
        "time":          time.values,
        "code":          code.values,
        "numeric_value": np.full(len(df), np.nan, dtype="float32"),
        "text_value":    df["Diagnosis Name"].values,
    })

    # ── end events (only where End Date present) ───────────────────────────────
    has_end = end_t.notna()
    end_rows = pd.DataFrame({
        "patient_id":    df.loc[has_end, "C MRN"].values,
        "time":          end_t[has_end].values,
        "code":          (code[has_end] + "//END").values,
        "numeric_value": np.full(has_end.sum(), np.nan, dtype="float32"),
        "text_value":    df.loc[has_end, "Diagnosis Name"].values,
    })

    out = pd.concat([onset, end_rows], ignore_index=True)
    return out


# ── main ───────────────────────────────────────────────────────────────────────
def main(input_path: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "medical_history.parquet"

    log.info(f"Input : {input_path}")
    log.info(f"Output: {out_path}")

    df = pd.read_csv(
        input_path,
        delimiter="|",
        encoding="utf-8-sig",
        encoding_errors="replace",
        on_bad_lines="skip",
        low_memory=False,
        usecols=[
            "C MRN", "Date of Birth", "Diagnosis Name",
            "ICD10", "Med Hx Date Text", "Med Hx Start Date ", "Med Hx End Date ",
        ],
    )
    # Strip footer rows
    df = df.iloc[:-2]
    log.info(f"Loaded {len(df):,} raw rows")

    result = transform(df)

    before = len(result)
    result.drop_duplicates(inplace=True)
    log.info(f"Dropped {before - len(result):,} fully duplicate rows ({before:,} → {len(result):,})")

    # Sort by patient then time (NaT sorts to front — MEDS static event convention)
    log.info("Sorting by patient_id, time …")
    result.sort_values(["patient_id", "time"], inplace=True, na_position="first", ignore_index=True)

    log.info(f"Writing parquet → {out_path}")
    result.to_parquet(out_path, index=False, engine="pyarrow")

    # ── summary ───────────────────────────────────────────────────────────────
    n_patients  = result["patient_id"].nunique()
    n_codes     = result["code"].nunique()
    n_null_time = result["time"].isna().sum()
    n_end_events = result["code"].str.endswith("//END").sum()
    icd_events   = result["code"].str.startswith("DIAGNOSIS//ICD10//").sum()
    medhx_events = result["code"].str.startswith("DIAGNOSIS//MEDHX//").sum()

    log.info("=" * 55)
    log.info(f"  MEDS events total  : {len(result):>10,}")
    log.info(f"    ICD10 coded      : {icd_events:>10,}  ({icd_events/len(result)*100:.1f}%)")
    log.info(f"    Name fallback    : {medhx_events:>10,}  ({medhx_events/len(result)*100:.1f}%)")
    log.info(f"    End events       : {n_end_events:>10,}")
    log.info(f"  Null time (static) : {n_null_time:>10,}  ({n_null_time/len(result)*100:.1f}%)")
    log.info(f"  Unique patients    : {n_patients:>10,}")
    log.info(f"  Unique codes       : {n_codes:>10,}")
    timed = result["time"].dropna()
    if len(timed):
        log.info(f"  Time range         : {timed.min()}  →  {timed.max()}")
    log.info("=" * 55)
    log.info(f"Done. Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Medical History .rpt to MEDS parquet")
    parser.add_argument("--input",      default=DEFAULT_INPUT,           help="Path to DR15201_Medical_History.rpt")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    main(args.input, Path(args.output_dir))
