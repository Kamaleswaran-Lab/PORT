"""
problem_list_to_meds.py
-----------------------
Convert DR15201_Problem_List_V.csv → MEDS-format Parquet.

MEDS event schema (per row):
    patient_id     : str         — C MRN
    time           : datetime64  — Noted Date (when problem was added to list)
                                   NaT → MEDS static event (sorted to front)
    code           : str         — "PROBLEM//ICD10//<ICD10_CODE>"
                                   fallback: "PROBLEM//NAME//<NORMALIZED_DIAGNOSIS>"
    numeric_value  : float32     — NaN (diagnoses carry no numeric value)
    text_value     : str         — Diagnosis name (original)

Resolved-date events:
    When Resolved Date is present, a second event is emitted with
    code suffix "//END" at that timestamp.

Column-shift handling:
    Diagnosis field (col 4) may contain commas (e.g. "Prematurity, 1,000-1,249 grams"),
    causing downstream columns to shift right. Shift is detected by finding the first
    yyyy-mm-dd datetime starting from the Noted Date column position.

    Detected shifts (out of 842,538 rows):
        shift=0 : majority of rows
        shift=1 : ~50,491 rows (single comma in Diagnosis or two ICD10 codes)
        shift=2+: remaining shifted rows

Usage:
    python problem_list_to_meds.py [--input PATH] [--output_dir DIR]
"""

import re
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DEFAULT_INPUT     = "/path/to/CHOA_RAW/DR15201_Problem_List_V.csv"
DEFAULT_OUTPUT_DIR = Path("/path/to/CHD_MEDS/data")

_MULTI_SPACE  = re.compile(r"\s+")
_UNSAFE_CHARS = re.compile(r"[^A-Z0-9_/]")
_ICD10_RE     = re.compile(r"^[A-Z]\d{2}")
_DT_RE        = re.compile(r"^\d{4}-\d{2}-\d{2}")

def norm(s: str) -> str:
    s = str(s).upper().strip()
    s = s.replace("%", "PCT").replace("#", "NUM")
    s = _MULTI_SPACE.sub("_", s)
    s = _UNSAFE_CHARS.sub("", s)
    return re.sub(r"_+", "_", s).strip("_")


def extract_fields(df: pd.DataFrame):
    """
    Detect per-row column shift and return corrected arrays for
    icd10, noted_date, resolved_date, diagnosis_name.

    Pool columns (from ICD10 position onward):
      pool[0]=ICD10, pool[1]=Noted Date, pool[2]=Resolved Date,
      pool[3]=Problem Comment, pool[4]=KEY_ID, pool[5]=Column1,
      pool[6]=_1 … pool[9]=_4

    shift=N means the Noted Date landed at pool[N+1].
    """
    pool = ["ICD10", "Noted Date", "Resolved Date", "Problem Comment",
            "KEY_ID", "Column1", "_1", "_2", "_3", "_4"]

    result_icd10    = pd.Series("",     index=df.index, dtype=object)
    result_noted    = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    result_resolved = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    result_diag     = df["Diagnosis"].astype(str).copy()

    remaining = pd.Series(True, index=df.index)

    for shift in range(len(pool) - 2):
        noted_col = pool[shift + 1]
        icd_col   = pool[shift]
        res_col   = pool[shift + 2]

        if noted_col not in df.columns:
            break

        mask = remaining & df[noted_col].astype(str).str.match(_DT_RE, na=False)
        if not mask.any():
            remaining &= ~mask
            continue

        # ICD10 at icd_col
        if icd_col in df.columns:
            result_icd10[mask] = df.loc[mask, icd_col].astype(str).str.strip()

        # Timestamps
        result_noted[mask] = pd.to_datetime(
            df.loc[mask, noted_col], errors="coerce")
        if res_col in df.columns:
            result_resolved[mask] = pd.to_datetime(
                df.loc[mask, res_col], errors="coerce")

        # Reconstruct Diagnosis for shifted rows
        if shift > 0:
            combined = df.loc[mask, "Diagnosis"].astype(str).str.strip()
            for ep in pool[:shift]:
                if ep in df.columns:
                    extra = df.loc[mask, ep].astype(str).str.strip()
                    # Only append non-garbage parts
                    extra_clean = extra.where(
                        ~extra.str.lower().isin(["nan", "nat", "null", "none", ""]),
                        ""
                    )
                    combined = combined + extra_clean.apply(
                        lambda v: (", " + v) if v else "")
            result_diag[mask] = combined

        remaining &= ~mask

    # Undetected rows (no datetime in any position) — use raw columns as-is
    if remaining.any():
        result_icd10[remaining] = df.loc[remaining, "ICD10"].astype(str).str.strip()
        # Noted / Resolved stay NaT

    return result_icd10, result_noted, result_resolved, result_diag


def build_code(icd10: pd.Series, diag: pd.Series) -> pd.Series:
    """
    Valid ICD10 (starts with letter + 2 digits) → "PROBLEM//ICD10//<code>"
    Otherwise → "PROBLEM//NAME//<NORMALIZED_DIAGNOSIS>"
    """
    is_valid = icd10.str.match(_ICD10_RE, na=False)
    code = pd.Series("", index=icd10.index, dtype=object)
    code[is_valid]  = "PROBLEM//ICD10//" + icd10[is_valid].str.strip()
    code[~is_valid] = "PROBLEM//NAME//"  + diag[~is_valid].fillna("UNKNOWN").map(norm)
    return code


def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["C MRN"].astype(str).str.startswith("C")].copy().reset_index(drop=True)

    icd10, noted, resolved, diag = extract_fields(df)

    # Clamp erroneous dates
    noted    = noted.where((noted    >= "1950-01-01") & (noted    <= "2030-01-01"), other=pd.NaT)
    resolved = resolved.where((resolved >= "1950-01-01") & (resolved <= "2030-01-01"), other=pd.NaT)

    code = build_code(icd10, diag)

    mrn = df["C MRN"]
    nan = np.nan

    # Onset events
    onset = pd.DataFrame({
        "patient_id":    mrn.values,
        "time":          noted.values,
        "code":          code.values,
        "numeric_value": np.full(len(df), nan, dtype="float32"),
        "text_value":    diag.values,
    })

    # Resolved/end events
    has_end = resolved.notna()
    end_rows = pd.DataFrame({
        "patient_id":    mrn[has_end].values,
        "time":          resolved[has_end].values,
        "code":          (code[has_end] + "//END").values,
        "numeric_value": np.full(has_end.sum(), nan, dtype="float32"),
        "text_value":    diag[has_end].values,
    })

    return pd.concat([onset, end_rows], ignore_index=True)


def main(input_path: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "problem_list.parquet"

    log.info(f"Input : {input_path}")
    log.info(f"Output: {out_path}")

    df = pd.read_csv(
        input_path,
        encoding="utf-8-sig",
        encoding_errors="replace",
        on_bad_lines="skip",
        low_memory=False,
        dtype=str,
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

    n_patients  = result["patient_id"].nunique()
    n_codes     = result["code"].nunique()
    n_null_time = result["time"].isna().sum()
    n_end       = result["code"].str.endswith("//END").sum()
    n_icd10     = result["code"].str.startswith("PROBLEM//ICD10//").sum()
    n_name      = result["code"].str.startswith("PROBLEM//NAME//").sum()
    timed       = result["time"].dropna()

    log.info("=" * 55)
    log.info(f"  MEDS events total  : {len(result):>10,}")
    log.info(f"    ICD10 coded      : {n_icd10:>10,}  ({n_icd10/len(result)*100:.1f}%)")
    log.info(f"    Name fallback    : {n_name:>10,}  ({n_name/len(result)*100:.1f}%)")
    log.info(f"    End events       : {n_end:>10,}")
    log.info(f"  Null time (static) : {n_null_time:>10,}  ({n_null_time/len(result)*100:.1f}%)")
    log.info(f"  Unique patients    : {n_patients:>10,}")
    log.info(f"  Unique codes       : {n_codes:>10,}")
    if len(timed):
        log.info(f"  Time range         : {timed.min()}  →  {timed.max()}")
    log.info("=" * 55)
    log.info(f"Done. Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Problem List CSV to MEDS parquet")
    parser.add_argument("--input",      default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    main(args.input, Path(args.output_dir))
