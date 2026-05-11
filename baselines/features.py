"""
features.py
-----------
Shared feature extraction for all baseline models (LR, XGBoost, LSTM).

Two feature sets are provided:

  1. Manual features (Layer 2 baselines)
     - Demographics: age at surgery, sex, weight, height, BMI percentile
     - Clinical: ASA score, admission type, patient class
     - Pre-op vitals: SBP, DBP, HR, RR, O2 sat, temp
     - Procedure: primary procedure code (one-hot top-N)

  2. MEDS aggregate features (Layer 3 baselines)
     All of the above PLUS, from events with time < prediction_time:
     - Labs: most recent value per code, flag for abnormal (outside 5th/95th pct)
     - Medications: count of unique drug types administered
     - Diagnoses (medical_history + problem_list): presence flags per ICD10 chapter
     - ADT: total prior hospital stays, ICU flag
     - LDAs: presence of prior arterial line, ETT, CVL, ECMO
     - Transfusions: prior transfusion count

Usage:
    from baselines.features import build_manual_features, build_meds_features
    X_manual = build_manual_features(task_df, events_df)
    X_meds   = build_meds_features(task_df, events_df)

Both functions return:
    X : pd.DataFrame  — one row per encounter, columns = feature names
    y : pd.Series     — boolean_value (IoD label)
    groups : pd.Series — subject_id (for GroupKFold / split-aware eval)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

import os
_CHD_DATA_ROOT = Path(os.environ.get("CHD_DATA_ROOT", "/path/to/CHD_MEDS"))
SPLITS_PATH = _CHD_DATA_ROOT / "splits"  / "splits.parquet"
EVENTS_PATH = _CHD_DATA_ROOT / "merged"  / "events.parquet"
TASK_PATH   = _CHD_DATA_ROOT / "outcome" / "iod_task.parquet"


# ── helpers ────────────────────────────────────────────────────────────────────

def _patient_id_to_subject_id(patient_id: pd.Series) -> pd.Series:
    return pd.to_numeric(patient_id.str.lstrip("C").str.strip(), errors="coerce").astype("Int64")


def load_task(task_path: Path = TASK_PATH) -> pd.DataFrame:
    """Load iod_task.parquet and add subject_id column."""
    task = pd.read_parquet(task_path, engine="pyarrow")
    task["subject_id"] = _patient_id_to_subject_id(task["patient_id"]).astype("int64")
    task["prediction_time"] = pd.to_datetime(task["prediction_time"])
    return task


def load_splits(splits_path: Path = SPLITS_PATH) -> pd.DataFrame:
    return pd.read_parquet(splits_path, engine="pyarrow")


def load_events(events_path: Path = EVENTS_PATH) -> pd.DataFrame:
    df = pd.read_parquet(events_path, engine="pyarrow")
    df["time"] = pd.to_datetime(df["time"])
    return df


def _preop_events(events: pd.DataFrame, task: pd.DataFrame, window_days: int | None = None) -> pd.DataFrame:
    """
    Filter events to only those occurring BEFORE prediction_time (In OR) for each patient.
    Merges on subject_id and keeps rows where time < prediction_time.
    Static events (time=NaT) are always included.

    If window_days is set, only events within (prediction_time - window_days) to
    prediction_time are included (context window ablation).
    """
    # One prediction_time per patient: use the LATEST In OR across all encounters.
    # (For feature building we want all pre-op history up to any of their surgeries;
    #  per-encounter filtering happens inside aggregate functions.)
    pt = task[["subject_id", "prediction_time", "encounter_csn"]].copy()

    merged = events.merge(pt, on="subject_id", how="inner")
    # Use <= so that events timestamped exactly at In OR (e.g. ASA score, pre-op vitals)
    # are included — these are administrative/pre-op data recorded at encounter start.
    mask = merged["time"].isna() | (merged["time"] <= merged["prediction_time"])

    if window_days is not None:
        window_start = merged["prediction_time"] - pd.Timedelta(days=window_days)
        mask = mask & (merged["time"].isna() | (merged["time"] >= window_start))

    return merged[mask]


# ── AN_Patients-derived features (demographics, vitals, procedure) ────────────

_AN_DEMO_CODES = {
    "DEMO//SEX":              "sex",
    "DEMO//RACE":             "race",
    "DEMO//ETHNICITY":        "ethnicity",
}

_AN_VITAL_CODES = {
    "VITAL//PREPROCEDURE//SBP":        "vitals_sbp",
    "VITAL//PREPROCEDURE//DBP":        "vitals_dbp",
    "VITAL//PREPROCEDURE//HEART_RATE": "vitals_hr",
    "VITAL//PREPROCEDURE//RR":         "vitals_rr",
    "VITAL//PREPROCEDURE//O2_SAT":     "vitals_o2",
    "VITAL//PREPROCEDURE//TEMP":       "vitals_temp",
    "VITAL//PREPROCEDURE//WEIGHT_KG":  "vitals_wt",
    "VITAL//PREPROCEDURE//HEIGHT_CM":  "vitals_ht",
}

_AN_ASA_CODE    = "ENCOUNTER//AN//ASA_SCORE"
_AN_PROC_PREFIX = "ENCOUNTER//AN//PRIMARY_PROCEDURE//"
# Sex: DEMO//MALE or DEMO//FEMALE as presence-flag codes (no single SEX code)
_AN_SEX_MALE_CODE = "DEMO//MALE"


def _extract_an_features(preop: pd.DataFrame) -> pd.DataFrame:
    """
    Extract structured AN_Patients-derived features per (subject_id, encounter_csn).
    Fully vectorized — no Python loops over groups.
    """
    key = ["subject_id", "encounter_csn"]

    # Numeric codes: pivot → one column per code
    num_codes = {
        _AN_ASA_CODE: "asa_score",
        **_AN_VITAL_CODES,
    }
    num_df = preop[preop["code"].isin(num_codes) & preop["numeric_value"].notna()].copy()
    num_df["feat_name"] = num_df["code"].map(num_codes)
    num_pivot = (
        num_df.groupby(key + ["feat_name"])["numeric_value"]
        .first()
        .unstack("feat_name")
        .reset_index()
    )
    # Ensure all expected columns exist
    for col in num_codes.values():
        if col not in num_pivot.columns:
            num_pivot[col] = np.nan

    # Sex: DEMO//MALE presence → 1, DEMO//FEMALE → 0, missing → NaN
    sex_df = preop[preop["code"].isin([_AN_SEX_MALE_CODE, "DEMO//FEMALE"])].copy()
    sex_df = sex_df.groupby(key)["code"].apply(
        lambda codes: 1 if _AN_SEX_MALE_CODE in codes.values else 0
    ).reset_index()
    sex_df.columns = key + ["sex_male"]

    # Primary procedure: first matching code, strip prefix
    proc_df = preop[preop["code"].str.startswith(_AN_PROC_PREFIX, na=False)].copy()
    proc_df["primary_procedure"] = proc_df["code"].str.replace(_AN_PROC_PREFIX, "", regex=False)
    proc_df = proc_df.groupby(key)["primary_procedure"].first().reset_index()

    # All unique (subject_id, encounter_csn) pairs
    base = preop[key].drop_duplicates()

    result = (
        base
        .merge(num_pivot, on=key, how="left")
        .merge(sex_df,    on=key, how="left")
        .merge(proc_df,   on=key, how="left")
    )
    result["primary_procedure"] = result["primary_procedure"].fillna("UNKNOWN")
    return result


# ── MEDS aggregate features ────────────────────────────────────────────────────

def _extract_lab_features(preop: pd.DataFrame, top_n_labs: int = 50) -> pd.DataFrame:
    """Most recent numeric lab value per patient-encounter, top_n_labs by frequency."""
    labs = preop[preop["code"].str.startswith("LAB//", na=False) & preop["numeric_value"].notna()].copy()
    if labs.empty:
        return pd.DataFrame()

    # Top N lab codes by frequency
    top_codes = labs["code"].value_counts().head(top_n_labs).index.tolist()
    labs = labs[labs["code"].isin(top_codes)]

    # Most recent value per (subject_id, encounter_csn, code)
    labs = labs.sort_values("time").groupby(["subject_id", "encounter_csn", "code"]).last().reset_index()
    pivoted = labs.pivot_table(
        index=["subject_id", "encounter_csn"],
        columns="code",
        values="numeric_value",
        aggfunc="last",
    )
    pivoted.columns = [f"lab__{c.replace('LAB//', '')}" for c in pivoted.columns]
    return pivoted.reset_index()


def _extract_med_features(preop: pd.DataFrame) -> pd.DataFrame:
    """Count of unique medication types per patient-encounter."""
    meds = preop[preop["code"].str.startswith("MED//", na=False)].copy()
    if meds.empty:
        return pd.DataFrame()

    counts = meds.groupby(["subject_id", "encounter_csn"])["code"].nunique().reset_index()
    counts.columns = ["subject_id", "encounter_csn", "med_unique_count"]
    return counts


def _extract_diagnosis_features(preop: pd.DataFrame) -> pd.DataFrame:
    """
    ICD10 chapter presence flags (A–Z) from medical_history + problem_list.
    Each chapter becomes a binary feature: 1 if patient has any code in that chapter.
    """
    diag = preop[
        preop["code"].str.contains("ICD10", na=False) &
        preop["code"].str.startswith(("DIAGNOSIS//", "PROBLEM//"), na=False)
    ].copy()
    if diag.empty:
        return pd.DataFrame()

    # Extract ICD10 code: last segment after final '//'
    diag["icd10"] = diag["code"].str.split("//").str[-1]
    diag["chapter"] = diag["icd10"].str[0].str.upper()  # first letter = chapter

    chapters = sorted(diag["chapter"].dropna().unique())
    result = diag.groupby(["subject_id", "encounter_csn"]).apply(
        lambda g: pd.Series({f"diag_chapter_{c}": int(c in g["chapter"].values) for c in chapters})
    ).reset_index()
    return result


def _extract_adt_features(preop: pd.DataFrame) -> pd.DataFrame:
    """Prior hospital encounters: total ADT IN events, ICU flag."""
    adt = preop[preop["code"].str.startswith("ADT//", na=False) &
                ~preop["code"].str.endswith("//OUT", na=False)].copy()
    if adt.empty:
        return pd.DataFrame()

    adt_counts = adt.groupby(["subject_id", "encounter_csn"]).agg(
        adt_prior_visits=("code", "count"),
        adt_prior_icu=(  "code", lambda x: int(x.str.contains("ICU", case=False).any())),
    ).reset_index()
    return adt_counts


def _extract_lda_features(preop: pd.DataFrame) -> pd.DataFrame:
    """Prior LDA placement flags: arterial line, ETT, CVL, ECMO."""
    lda = preop[preop["code"].str.startswith("LDA//", na=False) &
                ~preop["code"].str.endswith("//REMOVED", na=False)].copy()
    if lda.empty:
        return pd.DataFrame()

    flags = {
        "lda_prior_art_line": "ARTERIAL",
        "lda_prior_ett":      "ETT",
        "lda_prior_cvl":      "CVL",
        "lda_prior_ecmo":     "ECMO",
    }
    result = lda.groupby(["subject_id", "encounter_csn"]).apply(
        lambda g: pd.Series({
            name: int(g["code"].str.contains(kw, case=False).any())
            for name, kw in flags.items()
        })
    ).reset_index()
    return result


def _extract_transfusion_features(preop: pd.DataFrame) -> pd.DataFrame:
    """Prior transfusion count per patient-encounter."""
    tx = preop[preop["code"].str.startswith("TRANSFUSION//", na=False) &
               ~preop["code"].str.endswith("//END", na=False)].copy()
    if tx.empty:
        return pd.DataFrame()

    counts = tx.groupby(["subject_id", "encounter_csn"])["code"].count().reset_index()
    counts.columns = ["subject_id", "encounter_csn", "transfusion_prior_count"]
    return counts


# ── Public API ─────────────────────────────────────────────────────────────────

def build_manual_features(
    task: pd.DataFrame,
    events: pd.DataFrame,
    window_days: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build Layer 2 manual features (demographics, vitals, ASA, procedure).

    Returns:
        X      : feature DataFrame
        y      : IoD label (boolean_value)
        groups : subject_id (for split-aware evaluation)
    """
    log.info(f"Extracting manual features (window_days={window_days}) …")
    preop = _preop_events(events, task, window_days=window_days)
    an_feats = _extract_an_features(preop)

    # One-hot encode primary procedure (top 50 by frequency)
    top_procs = an_feats["primary_procedure"].value_counts().head(50).index
    an_feats["primary_procedure"] = an_feats["primary_procedure"].where(
        an_feats["primary_procedure"].isin(top_procs), other="OTHER"
    )
    proc_dummies = pd.get_dummies(an_feats["primary_procedure"], prefix="proc")
    an_feats = pd.concat([an_feats.drop(columns=["primary_procedure"]), proc_dummies], axis=1)

    # Merge with task to get labels
    merged = task[["subject_id", "encounter_csn", "boolean_value"]].merge(
        an_feats, on=["subject_id", "encounter_csn"], how="left"
    )

    feature_cols = [c for c in merged.columns if c not in ("subject_id", "encounter_csn", "boolean_value")]
    X = merged[feature_cols].copy()
    y = merged["boolean_value"].astype(int)
    groups = merged["subject_id"]

    log.info(f"  Manual features: {X.shape[0]:,} encounters × {X.shape[1]} features")
    return X, y, groups


def build_meds_features(
    task: pd.DataFrame,
    events: pd.DataFrame,
    top_n_labs: int = 50,
    window_days: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build Layer 3 MEDS aggregate features (manual + labs + meds + diagnoses + ADT + LDAs + transfusions).

    Returns:
        X      : feature DataFrame
        y      : IoD label (boolean_value)
        groups : subject_id (for split-aware evaluation)
    """
    log.info(f"Extracting MEDS aggregate features (window_days={window_days}) …")
    preop = _preop_events(events, task, window_days=window_days)

    # AN features (base)
    an_feats = _extract_an_features(preop)
    top_procs = an_feats["primary_procedure"].value_counts().head(50).index
    an_feats["primary_procedure"] = an_feats["primary_procedure"].where(
        an_feats["primary_procedure"].isin(top_procs), other="OTHER"
    )
    proc_dummies = pd.get_dummies(an_feats["primary_procedure"], prefix="proc")
    an_feats = pd.concat([an_feats.drop(columns=["primary_procedure"]), proc_dummies], axis=1)

    # Additional MEDS feature tables
    extra_tables = [
        _extract_lab_features(preop, top_n_labs),
        _extract_med_features(preop),
        _extract_diagnosis_features(preop),
        _extract_adt_features(preop),
        _extract_lda_features(preop),
        _extract_transfusion_features(preop),
    ]

    base = an_feats
    for tbl in extra_tables:
        if tbl is not None and len(tbl) > 0:
            base = base.merge(tbl, on=["subject_id", "encounter_csn"], how="left")

    # Merge with task labels
    merged = task[["subject_id", "encounter_csn", "boolean_value"]].merge(
        base, on=["subject_id", "encounter_csn"], how="left"
    )

    feature_cols = [c for c in merged.columns if c not in ("subject_id", "encounter_csn", "boolean_value")]
    X = merged[feature_cols].copy()
    y = merged["boolean_value"].astype(int)
    groups = merged["subject_id"]

    log.info(f"  MEDS features: {X.shape[0]:,} encounters × {X.shape[1]} features")
    return X, y, groups
