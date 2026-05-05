"""
cardiology_encounters_to_meds.py
---------------------------------
Convert DR15201_Outpatient_Cardiology_Encounters_V.csv → two Parquet files:

  1. cardiology_encounters.parquet  (MEDS format)
       One event per unique (C MRN, Encounter CSN) — the primary diagnosis.
       patient_id, time (Visit Date), code, numeric_value, text_value

  2. cardiology_notes.parquet  (separate, not MEDS)
       One row per Note ID where note text is recoverable.
       patient_id, encounter_csn, time (Note Created → Visit Date), note_type, note_text

MEDS code:
    "ENCOUNTER//CARDIOLOGY//ICD10//<ICD10_CODE>"

─── CSV Column-Shift Problem ──────────────────────────────────────────────────
When Diagnosis Name contains commas (e.g. "Trisomy 21, Down syndrome"),
the CSV parser splits it across the ICD10 / Note ID / Note Type / Note Created
/ Note Text columns, shifting all downstream fields right by N places.

Shift detection: find the leftmost column among [ICD10, Note ID, Note Type,
Note Created, Note Text] whose value matches the ICD10 pattern ^[A-Z]\\d{2}.

Shift → ICD10 location:
  0 → ICD10 column (normal)
  1 → Note ID column
  2 → Note Type column
  3 → Note Created column
  4 → Note Text column

Diagnosis Name is reconstructed by joining the N extra comma-fragments that
were absorbed into [ICD10 … Note Text] before the true ICD10.

Note fields after each shift level (columns that survive):
  shift 0: Note ID, Note Type, Note Created, Note Text  — all present
  shift 1: Note ID=Note Type, Note Type=Note Created, Note Created=Note Text, Note Text=lost
  shift 2: Note ID=Note Created, Note Type=Note Text,  Note Created=lost,    Note Text=lost
  shift 3: Note ID=Note Text,   Note Type=lost,         Note Created=lost,    Note Text=lost
  shift 4: Note ID=lost,        all note fields lost

Usage:
    python cardiology_encounters_to_meds.py [--input PATH] [--output_dir DIR]
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
DEFAULT_INPUT      = "/path/to/CHOA_RAW/DR15201_Outpatient_Cardiology_Encounters_V.csv"
DEFAULT_OUTPUT_DIR = Path("/path/to/CHD_MEDS/data")

ICD_RE = re.compile(r"^[A-Z]\d{2}")

# Columns that may absorb the shifted ICD10 value (in order)
SHIFT_COLS = ["ICD10", "Note ID", "Note Type", "Note Created", "Note Text"]

# ── normalisation ──────────────────────────────────────────────────────────────
_MULTI_SPACE  = re.compile(r"\s+")
_UNSAFE_CHARS = re.compile(r"[^A-Z0-9_/]")

def normalise_code(s: str) -> str:
    s = str(s).upper().strip()
    s = _MULTI_SPACE.sub("_", s)
    s = _UNSAFE_CHARS.sub("", s)
    return re.sub(r"_+", "_", s).strip("_")


# ── column-shift detection & recovery ─────────────────────────────────────────
def detect_shifts(df: pd.DataFrame) -> pd.Series:
    """
    Return a Series of shift levels (0–4).
    -1 means the ICD10 could not be located (unrecoverable).
    """
    shifts = pd.Series(-1, index=df.index, dtype=np.int8)
    for level, col in enumerate(SHIFT_COLS):
        if col not in df.columns:
            continue
        is_icd = df[col].astype(str).str.strip().str.match(ICD_RE)
        unclassified = shifts == -1
        shifts[unclassified & is_icd] = level
    return shifts


def recover_fields(df: pd.DataFrame, shifts: pd.Series) -> pd.DataFrame:
    """
    Return a DataFrame with recovered columns:
        true_diag_name, true_icd10,
        true_note_id, true_note_type, true_note_created, true_note_text
    """
    # Raw column values as strings (NaN → "")
    def s(col):
        if col in df.columns:
            return df[col].fillna("").astype(str).str.strip()
        return pd.Series("", index=df.index)

    diag  = s("Diagnosis Name")
    icd10 = s("ICD10")
    nid   = s("Note ID")
    ntype = s("Note Type")
    ncre  = s("Note Created")
    ntxt  = s("Note Text")

    # ── true ICD10 ────────────────────────────────────────────────────────────
    true_icd = pd.Series("", index=df.index, dtype=str)
    true_icd[shifts == 0] = icd10[shifts == 0]
    true_icd[shifts == 1] = nid  [shifts == 1]
    true_icd[shifts == 2] = ntype[shifts == 2]
    true_icd[shifts == 3] = ncre [shifts == 3]
    true_icd[shifts == 4] = ntxt [shifts == 4]

    # ── true Diagnosis Name (rejoin absorbed fragments) ───────────────────────
    extra = {
        1: [icd10],
        2: [icd10, nid],
        3: [icd10, nid, ntype],
        4: [icd10, nid, ntype, ncre],
    }
    true_diag = diag.copy()
    for level, frag_cols in extra.items():
        mask = shifts == level
        fragments = [diag[mask]] + [fc[mask] for fc in frag_cols]
        true_diag[mask] = pd.concat(fragments, axis=1).apply(
            lambda row: ", ".join(v for v in row if v), axis=1
        )

    # ── true Note fields ──────────────────────────────────────────────────────
    # shift 0: nid / ntype / ncre / ntxt (all correct)
    # shift 1: ntype / ncre / ntxt / ""
    # shift 2: ncre  / ntxt / ""   / ""
    # shift 3: ntxt  / ""   / ""   / ""
    # shift 4: ""    / ""   / ""   / ""

    note_src = {
        "true_note_id":      [nid,   ntype, ncre,  ntxt,  pd.Series("", index=df.index)],
        "true_note_type":    [ntype, ncre,  ntxt,  pd.Series("", index=df.index), pd.Series("", index=df.index)],
        "true_note_created": [ncre,  ntxt,  pd.Series("", index=df.index), pd.Series("", index=df.index), pd.Series("", index=df.index)],
        "true_note_text":    [ntxt,  pd.Series("", index=df.index), pd.Series("", index=df.index), pd.Series("", index=df.index), pd.Series("", index=df.index)],
    }

    result = pd.DataFrame({"true_icd10": true_icd, "true_diag_name": true_diag}, index=df.index)
    for field, sources in note_src.items():
        col_out = pd.Series("", index=df.index, dtype=str)
        for level, src in enumerate(sources):
            mask = shifts == level
            col_out[mask] = src[mask]
        result[field] = col_out

    # Replace empty strings with NA
    for col in result.columns:
        result[col] = result[col].replace("", pd.NA)

    return result


# ── main transform ─────────────────────────────────────────────────────────────
def transform(df: pd.DataFrame):
    df = df[df["C MRN"].astype(str).str.startswith("C")].copy()

    shifts = detect_shifts(df)
    recovered = recover_fields(df, shifts)

    visit_date = pd.to_datetime(df["Visit Date"], errors="coerce")

    # ── encounters.parquet (MEDS) ─────────────────────────────────────────────
    # One event per unique (C MRN, Encounter CSN) — use first occurrence
    enc_df = pd.DataFrame({
        "patient_id":     df["C MRN"].values,
        "encounter_csn":  df["Encounter CSN"].values,
        "time":           visit_date.values,
        "true_icd10":     recovered["true_icd10"].values,
        "true_diag_name": recovered["true_diag_name"].values,
    })

    enc_dedup = (
        enc_df
        .dropna(subset=["time"])
        .sort_values("patient_id")
        .drop_duplicates(subset=["patient_id", "encounter_csn"])
        .copy()
    )

    enc_dedup["code"] = enc_dedup["true_icd10"].where(
        enc_dedup["true_icd10"].notna(),
        other="UNKNOWN"
    ).apply(lambda x: f"ENCOUNTER//CARDIOLOGY//ICD10//{x.strip()}" if pd.notna(x) else "ENCOUNTER//CARDIOLOGY//UNKNOWN")

    encounters = pd.DataFrame({
        "patient_id":    enc_dedup["patient_id"].values,
        "time":          enc_dedup["time"].values,
        "code":          enc_dedup["code"].values,
        "numeric_value": np.full(len(enc_dedup), np.nan, dtype="float32"),
        "text_value":    enc_dedup["true_diag_name"].values,
    })

    # ── notes.parquet (separate) ──────────────────────────────────────────────
    note_created = pd.to_datetime(recovered["true_note_created"], errors="coerce")
    note_time = note_created.fillna(visit_date)   # fallback to Visit Date

    has_text = recovered["true_note_text"].notna()

    notes = pd.DataFrame({
        "patient_id":    df.loc[has_text, "C MRN"].values,
        "encounter_csn": df.loc[has_text, "Encounter CSN"].values,
        "time":          note_time[has_text].values,
        "note_type":     recovered.loc[has_text, "true_note_type"].values,
        "note_text":     recovered.loc[has_text, "true_note_text"].values,
    })

    return encounters, notes, shifts


# ── main ───────────────────────────────────────────────────────────────────────
def main(input_path: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    enc_path   = output_dir / "cardiology_encounters.parquet"
    notes_path = output_dir / "cardiology_notes.parquet"

    log.info(f"Input : {input_path}")

    df = pd.read_csv(
        input_path,
        encoding="utf-8-sig",
        encoding_errors="replace",
        on_bad_lines="skip",
        low_memory=False,
    )
    df = df.iloc[:-2]
    log.info(f"Loaded {len(df):,} raw rows")

    encounters, notes, shifts = transform(df)

    # ── shift summary ─────────────────────────────────────────────────────────
    shift_counts = shifts.value_counts().sort_index()
    log.info("Column shift distribution:")
    labels = {0: "normal", 1: "1-col shift", 2: "2-col shift",
              3: "3-col shift", 4: "4-col shift", -1: "unrecoverable"}
    for lvl, cnt in shift_counts.items():
        log.info(f"  shift={lvl:2d} ({labels.get(lvl, '?')}): {cnt:>8,}  ({cnt/len(df)*100:.1f}%)")
    log.info("")

    # ── sort & write encounters ───────────────────────────────────────────────
    encounters.sort_values(["patient_id", "time"], inplace=True, ignore_index=True)
    log.info(f"Writing {len(encounters):,} encounter events → {enc_path}")
    encounters.to_parquet(enc_path, index=False, engine="pyarrow")

    # ── sort & write notes ────────────────────────────────────────────────────
    notes.sort_values(["patient_id", "time"], inplace=True, na_position="first", ignore_index=True)
    log.info(f"Writing {len(notes):,} notes → {notes_path}")
    notes.to_parquet(notes_path, index=False, engine="pyarrow")

    # ── summary ───────────────────────────────────────────────────────────────
    log.info("=" * 55)
    log.info("  ENCOUNTERS (MEDS)")
    log.info(f"    Events           : {len(encounters):>10,}")
    log.info(f"    Unique patients  : {encounters['patient_id'].nunique():>10,}")
    log.info(f"    Unique codes     : {encounters['code'].nunique():>10,}")
    timed = encounters["time"].dropna()
    if len(timed):
        log.info(f"    Time range       : {timed.min()}  →  {timed.max()}")
    log.info("  NOTES (separate)")
    log.info(f"    Notes            : {len(notes):>10,}")
    log.info(f"    Unique patients  : {notes['patient_id'].nunique():>10,}")
    nc_ok = pd.to_datetime(notes["time"], errors="coerce").notna()
    log.info(f"    Time from Note Created: {nc_ok.sum():>8,}  ({nc_ok.mean()*100:.1f}%)")
    log.info("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    main(args.input, Path(args.output_dir))
