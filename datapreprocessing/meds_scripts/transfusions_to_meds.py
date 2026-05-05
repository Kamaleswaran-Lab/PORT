"""
transfusions_to_meds.py
-----------------------
Convert DR15201_Transfusions_V.csv → MEDS-format Parquet.

MEDS event schema (per row):
    patient_id     : str         — C MRN
    time           : datetime64  — Blood Admin Start
    code           : str         — "TRANSFUSION//<PRODUCT>"
                                   e.g. TRANSFUSION//RBC, TRANSFUSION//PLATELETS,
                                        TRANSFUSION//FFP, TRANSFUSION//CRYO,
                                        TRANSFUSION//EXCHANGE_RBC
    numeric_value  : float32     — NaN (volume/unit not consistently captured)
    text_value     : str         — original Order Name

End events:
    When Blood Admin End is valid, a second event is emitted with
    code suffix "//END" at that timestamp.

Filtering:
    - Only rows where Blood Admin Start parses to a valid date (not 1/1/1900, not NaT)
    - Canceled orders are excluded (keep Completed + any non-Canceled with valid timestamps)

Product normalization (from Order Name):
    Transfuse RBC (units/mL)         → RBC
    Transfuse Platelets (units/mL)   → PLATELETS
    Transfuse Frozen Plasma (FFP)    → FFP
    Transfuse Cryoprecipitate        → CRYO
    Exchange RBC                     → EXCHANGE_RBC
    Other                            → normalized name

Usage:
    python transfusions_to_meds.py [--input PATH] [--output_dir DIR]
"""

import re
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DEFAULT_INPUT      = "/path/to/CHOA_RAW/DR15201_Transfusions_V.csv"
DEFAULT_OUTPUT_DIR = Path("/path/to/CHD_MEDS/data")

_MULTI_SPACE  = re.compile(r"\s+")
_UNSAFE_CHARS = re.compile(r"[^A-Z0-9_/]")

def norm(s: str) -> str:
    s = str(s).upper().strip()
    s = s.replace("%", "PCT").replace("#", "NUM")
    s = _MULTI_SPACE.sub("_", s)
    s = _UNSAFE_CHARS.sub("", s)
    return re.sub(r"_+", "_", s).strip("_")


def normalize_product(name: str) -> str:
    """
    Map Order Name to a clean blood product code.
    Strips any 'Transfusion rate: ...' suffix first (column-shift artifact).
    """
    s = str(name).split("Transfusion rate")[0].strip().upper()
    if "EXCHANGE" in s:
        return "EXCHANGE_RBC"
    if "CRYOPRECIPITATE" in s or "CRYO" in s:
        return "CRYO"
    if "FROZEN PLASMA" in s or "FFP" in s:
        return "FFP"
    if "PLATELET" in s:
        return "PLATELETS"
    if "RBC" in s or "RED BLOOD" in s:
        return "RBC"
    # fallback: normalize the whole name
    return norm(s)


def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["C MRN"].astype(str).str.startswith("C")].copy()

    # Parse timestamps
    admin_start = pd.to_datetime(df["Blood Admin Start"], errors="coerce")
    admin_end   = pd.to_datetime(df["Blood Admin End"],   errors="coerce")

    # Drop 1/1/1900 sentinel dates
    sentinel = pd.Timestamp("1900-01-01")
    admin_start[admin_start.dt.date == sentinel.date()] = pd.NaT
    admin_end[admin_end.dt.date   == sentinel.date()] = pd.NaT

    # Clamp erroneous dates
    admin_start = admin_start.where(
        (admin_start >= "1950-01-01") & (admin_start <= "2030-01-01"), pd.NaT)
    admin_end = admin_end.where(
        (admin_end >= "1950-01-01") & (admin_end <= "2030-01-01"), pd.NaT)

    # Keep only rows with a valid start timestamp and not Canceled
    valid = admin_start.notna() & (df["Order Status"].astype(str).str.strip() != "Canceled")
    df         = df[valid].copy()
    admin_start = admin_start[valid]
    admin_end   = admin_end[valid]

    product = df["Order Name"].fillna("UNKNOWN").map(normalize_product)
    code    = "TRANSFUSION//" + product
    mrn     = df["C MRN"]
    nan     = np.nan

    # Start events
    start_events = pd.DataFrame({
        "patient_id":    mrn.values,
        "time":          admin_start.values,
        "code":          code.values,
        "numeric_value": np.full(len(df), nan, dtype="float32"),
        "text_value":    df["Order Name"].values,
    })

    # End events
    has_end = admin_end.notna()
    end_events = pd.DataFrame({
        "patient_id":    mrn[has_end].values,
        "time":          admin_end[has_end].values,
        "code":          (code[has_end] + "//END").values,
        "numeric_value": np.full(has_end.sum(), nan, dtype="float32"),
        "text_value":    df.loc[has_end, "Order Name"].values,
    })

    return pd.concat([start_events, end_events], ignore_index=True)


def main(input_path: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "transfusions.parquet"

    log.info(f"Input : {input_path}")
    log.info(f"Output: {out_path}")

    df = pd.read_csv(
        input_path,
        encoding="utf-8-sig",
        encoding_errors="replace",
        on_bad_lines="skip",
        low_memory=False,
        usecols=["C MRN", "Order Name", "Order Status",
                 "Blood Admin Start", "Blood Admin End"],
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

    n_patients = result["patient_id"].nunique()
    n_start    = (~result["code"].str.endswith("//END")).sum()
    n_end      = result["code"].str.endswith("//END").sum()

    log.info("=" * 55)
    log.info(f"  MEDS events total  : {len(result):>10,}")
    log.info(f"    START events     : {n_start:>10,}")
    log.info(f"    END events       : {n_end:>10,}")
    log.info(f"  Unique patients    : {n_patients:>10,}")
    log.info(f"  Product breakdown (START events):")
    for code, cnt in result[~result["code"].str.endswith("//END")]["code"].value_counts().items():
        log.info(f"    {code:<40}: {cnt:>8,}")
    timed = result["time"].dropna()
    log.info(f"  Time range         : {timed.min()}  →  {timed.max()}")
    log.info("=" * 55)
    log.info(f"Done. Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Transfusions CSV to MEDS parquet")
    parser.add_argument("--input",      default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    main(args.input, Path(args.output_dir))
