"""
an_patients_to_meds.py
----------------------
Convert DR15201_AN_Patients.rpt → MEDS-format Parquet.

The raw file has 552,208 rows but only 210,272 unique encounters (Encounter CSN).
All events are emitted from the deduplicated encounter-level view.

Events emitted per encounter:
─────────────────────────────────────────────────────────────────────────────
  STATIC (time = null — sorted to front of patient timeline by MEDS spec)
    DEMO//SEX//<Female|Male>
    DEMO//RACE//<race>
    DEMO//ETHNICITY//<ethnicity>
    DEMO//LANGUAGE//<language>
    DEMO//GESTATIONAL_AGE          numeric_value = weeks

  ENCOUNTER TIMING
    ENCOUNTER//AN//HOSPITAL_ADMISSION   at Hospital Admission Date
    ENCOUNTER//AN//HOSPITAL_DISCHARGE   at Hospital Discharge Date
    ENCOUNTER//AN//OR_ENTRY             at In OR
    ENCOUNTER//AN//OR_EXIT              at Out OR
    ENCOUNTER//AN//AN_START             at AN Start
    ENCOUNTER//AN//AN_END              at AN End

  ENCOUNTER CONTEXT  (at AN Start; categorical → null numeric)
    ENCOUNTER//AN//PRIMARY_PROCEDURE//<proc_normalized>
    ENCOUNTER//AN//PATIENT_CLASS//<class_normalized>
    ENCOUNTER//AN//ADMISSION_TYPE//<type_normalized>
    ENCOUNTER//AN//ASA_SCORE            numeric_value = ASA score (1–5)

  PREPROCEDURE VITALS  (at AN Start)
    VITAL//PREPROCEDURE//WEIGHT_KG      numeric_value = kg
    VITAL//PREPROCEDURE//HEIGHT_CM      numeric_value = cm
    VITAL//PREPROCEDURE//HEART_RATE     numeric_value = bpm
    VITAL//PREPROCEDURE//RR             numeric_value = breaths/min
    VITAL//PREPROCEDURE//O2_SAT         numeric_value = %
    VITAL//PREPROCEDURE//TEMP           numeric_value = °F
    VITAL//PREPROCEDURE//SBP            numeric_value = mmHg  (parsed from "116/66")
    VITAL//PREPROCEDURE//DBP            numeric_value = mmHg

  OUTCOME
    OUTCOME//DEATH                      at DEATH_DATE  (only if non-null)

Usage:
    python an_patients_to_meds.py [--input PATH] [--output_dir DIR]
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
    "CHOA_DATA_Tables_CHD/DR15201_AN_Patients.rpt"
)
DEFAULT_OUTPUT_DIR = Path("/path/to/CHD_MEDS/data")

# ── helpers ────────────────────────────────────────────────────────────────────
_MULTI_SPACE  = re.compile(r"\s+")
_UNSAFE_CHARS = re.compile(r"[^A-Z0-9_/]")

def norm(s: str) -> str:
    s = str(s).upper().strip()
    s = _MULTI_SPACE.sub("_", s)
    s = _UNSAFE_CHARS.sub("", s)
    return re.sub(r"_+", "_", s).strip("_")

def parse_time(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")

def rows(patient_ids, times, codes, numerics, texts):
    """Helper to build a partial DataFrame."""
    return pd.DataFrame({
        "patient_id":    patient_ids,
        "time":          times,
        "code":          codes,
        "numeric_value": np.array(numerics, dtype="float32"),
        "text_value":    texts,
    })

# ── transform ──────────────────────────────────────────────────────────────────
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Deduplicate on Encounter CSN (keep first occurrence)
    df = (df[df["C MRN"].astype(str).str.startswith("C")]
          .drop_duplicates(subset=["Encounter CSN"])
          .copy()
          .reset_index(drop=True))

    mrn = df["C MRN"]
    n   = len(df)
    nat = pd.NaT
    nan = np.nan

    parts = []

    # ── static demographics (time = NaT) ──────────────────────────────────────
    for col, prefix in [("Legal Sex ", "DEMO//SEX"),
                        ("Race",       "DEMO//RACE"),
                        ("Ethnicity",  "DEMO//ETHNICITY"),
                        ("Language ",  "DEMO//LANGUAGE")]:
        valid = df[col].notna()
        parts.append(rows(
            mrn[valid].values,
            [nat] * valid.sum(),
            ("DEMO//" + df.loc[valid, col].map(norm)).values,
            [nan] * valid.sum(),
            df.loc[valid, col].values,
        ))

    # Gestational age (static numeric)
    ga = pd.to_numeric(df["Gestational age at birth"], errors="coerce")
    valid = ga.notna()
    parts.append(rows(
        mrn[valid].values,
        [nat] * valid.sum(),
        ["DEMO//GESTATIONAL_AGE"] * valid.sum(),
        ga[valid].values.astype("float32"),
        [None] * valid.sum(),
    ))

    # ── encounter timing events ────────────────────────────────────────────────
    timing_cols = [
        ("Hospital Admission Date",  "ENCOUNTER//AN//HOSPITAL_ADMISSION"),
        ("Hospital Discharge Date",  "ENCOUNTER//AN//HOSPITAL_DISCHARGE"),
        ("In OR",                    "ENCOUNTER//AN//OR_ENTRY"),
        ("Out OR",                   "ENCOUNTER//AN//OR_EXIT"),
        ("AN Start",                 "ENCOUNTER//AN//AN_START"),
        ("AN End",                   "ENCOUNTER//AN//AN_END"),
    ]
    for col, code in timing_cols:
        t = parse_time(df[col])
        valid = t.notna()
        parts.append(rows(
            mrn[valid].values,
            t[valid].values,
            [code] * valid.sum(),
            [nan] * valid.sum(),
            [None] * valid.sum(),
        ))

    # ── encounter context (at AN Start) ───────────────────────────────────────
    an_start = parse_time(df["AN Start"])

    # Primary Procedure
    valid = df["Primary Procedure"].notna() & an_start.notna()
    parts.append(rows(
        mrn[valid].values,
        an_start[valid].values,
        ("ENCOUNTER//AN//PRIMARY_PROCEDURE//" +
         df.loc[valid, "Primary Procedure"].map(norm)).values,
        [nan] * valid.sum(),
        df.loc[valid, "Primary Procedure"].values,
    ))

    # Patient Class
    valid = df["Patient Class"].notna() & an_start.notna()
    parts.append(rows(
        mrn[valid].values,
        an_start[valid].values,
        ("ENCOUNTER//AN//PATIENT_CLASS//" +
         df.loc[valid, "Patient Class"].map(norm)).values,
        [nan] * valid.sum(),
        df.loc[valid, "Patient Class"].values,
    ))

    # Admission Type
    valid = df["Hospital Admission Type"].notna() & an_start.notna()
    parts.append(rows(
        mrn[valid].values,
        an_start[valid].values,
        ("ENCOUNTER//AN//ADMISSION_TYPE//" +
         df.loc[valid, "Hospital Admission Type"].map(norm)).values,
        [nan] * valid.sum(),
        df.loc[valid, "Hospital Admission Type"].values,
    ))

    # ASA Score
    asa = pd.to_numeric(df["ASA PS Score"], errors="coerce")
    valid = asa.notna() & an_start.notna()
    parts.append(rows(
        mrn[valid].values,
        an_start[valid].values,
        ["ENCOUNTER//AN//ASA_SCORE"] * valid.sum(),
        asa[valid].values.astype("float32"),
        [None] * valid.sum(),
    ))

    # ── preprocedure vitals (at AN Start) ─────────────────────────────────────
    vital_cols = [
        ("Weight in kg",             "VITAL//PREPROCEDURE//WEIGHT_KG"),
        ("Height in cm",             "VITAL//PREPROCEDURE//HEIGHT_CM"),
        ("Preprocedure HR",          "VITAL//PREPROCEDURE//HEART_RATE"),
        ("Preprocedure RR",          "VITAL//PREPROCEDURE//RR"),
        ("Preprocedure O2 Saturation","VITAL//PREPROCEDURE//O2_SAT"),
        ("Preprocedure Temperature", "VITAL//PREPROCEDURE//TEMP"),
    ]
    for col, code in vital_cols:
        val = pd.to_numeric(df[col], errors="coerce")
        valid = val.notna() & an_start.notna()
        parts.append(rows(
            mrn[valid].values,
            an_start[valid].values,
            [code] * valid.sum(),
            val[valid].values.astype("float32"),
            [None] * valid.sum(),
        ))

    # Blood pressure — parse "SBP/DBP"
    bp_str = df["Preprocedure Blood Pressure"].astype(str).str.strip()
    bp_parsed = bp_str.str.extract(r"^(\d+)/(\d+)$")
    sbp = pd.to_numeric(bp_parsed[0], errors="coerce")
    dbp = pd.to_numeric(bp_parsed[1], errors="coerce")
    for val, code in [(sbp, "VITAL//PREPROCEDURE//SBP"),
                      (dbp, "VITAL//PREPROCEDURE//DBP")]:
        valid = val.notna() & an_start.notna()
        parts.append(rows(
            mrn[valid].values,
            an_start[valid].values,
            [code] * valid.sum(),
            val[valid].values.astype("float32"),
            [None] * valid.sum(),
        ))

    # ── death outcome ──────────────────────────────────────────────────────────
    death_t = parse_time(df["DEATH_DATE"])
    valid = death_t.notna()
    parts.append(rows(
        mrn[valid].values,
        death_t[valid].values,
        ["OUTCOME//DEATH"] * valid.sum(),
        [nan] * valid.sum(),
        [None] * valid.sum(),
    ))

    result = pd.concat(parts, ignore_index=True)

    # Deduplicate: same patient + same time + same code across multiple encounters
    result.drop_duplicates(inplace=True)

    return result


# ── main ───────────────────────────────────────────────────────────────────────
def main(input_path: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "an_patients.parquet"

    log.info(f"Input : {input_path}")
    log.info(f"Output: {out_path}")

    df = pd.read_csv(
        input_path,
        delimiter="|",
        encoding="utf-8-sig",
        encoding_errors="replace",
        on_bad_lines="skip",
        low_memory=False,
    )
    df = df.iloc[:-2]
    log.info(f"Loaded {len(df):,} raw rows → {df['Encounter CSN'].nunique():,} unique encounters")

    result = transform(df)

    log.info("Sorting by patient_id, time …")
    result.sort_values(["patient_id", "time"], inplace=True,
                       na_position="first", ignore_index=True)

    log.info(f"Writing parquet → {out_path}")
    result.to_parquet(out_path, index=False, engine="pyarrow")

    # ── summary ───────────────────────────────────────────────────────────────
    n_patients  = result["patient_id"].nunique()
    n_codes     = result["code"].nunique()
    n_null_time = result["time"].isna().sum()
    pct_numeric = result["numeric_value"].notna().mean() * 100

    # Event type breakdown
    prefixes = ["DEMO//", "ENCOUNTER//AN//HOSPITAL", "ENCOUNTER//AN//OR",
                "ENCOUNTER//AN//AN_", "ENCOUNTER//AN//PRIMARY",
                "ENCOUNTER//AN//PATIENT_CLASS", "ENCOUNTER//AN//ADMISSION_TYPE",
                "ENCOUNTER//AN//ASA", "VITAL//", "OUTCOME//"]
    log.info("=" * 55)
    log.info(f"  MEDS events total  : {len(result):>10,}")
    log.info(f"  Null time (static) : {n_null_time:>10,}  ({n_null_time/len(result)*100:.1f}%)")
    log.info(f"  Numeric values     : {pct_numeric:>9.1f}%")
    log.info(f"  Unique patients    : {n_patients:>10,}")
    log.info(f"  Unique codes       : {n_codes:>10,}")
    for pfx in prefixes:
        cnt = result["code"].str.startswith(pfx).sum()
        if cnt:
            log.info(f"    {pfx:<35}: {cnt:>8,}")
    timed = result["time"].dropna()
    log.info(f"  Time range         : {timed.min()}  →  {timed.max()}")
    log.info("=" * 55)
    log.info(f"Done. Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert AN_Patients .rpt to MEDS parquet")
    parser.add_argument("--input",      default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    main(args.input, Path(args.output_dir))
