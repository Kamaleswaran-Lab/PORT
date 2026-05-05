"""
df Integration Preprocessor
============================
Applies all 6 df streams to baseline MEDS parquets, producing the integrated input..

Input:  /path/to/CHD_MEDS/ethos_input_v3/{train,val,test}/shard_*.parquet
Output: /path/to/CHD_MEDS/ethos_input/{train,val,test}/shard_*.parquet

Transformations:
  1. Stream F: redirect MED→LDA leak, AN_EVENT typo merge, TRANSFUSION consolidate
  2. Stream B: ICD-10 hierarchical decomposition (DIAG+PROBLEM+ENCOUNTER//CARDIOLOGY)
  3. Stream A: MED → ATC hierarchical (uses atc_team_reviewed.csv)
  4. Stream C: LAB/PROCEDURE/ENCOUNTER//AN//PRIMARY/SDE top-N cutoff + OTHER fallback
  5. Stream D: DEMO atomic decomposition + LANGUAGE split (uses demo_language_tokens.parquet)
  6. Stream E: add new SES events (INSURANCE, POI, HOME_COUNTY) per encounter

Usage:
  python preprocess_v4_integrate.py --split train --shard 0    # single-shard pilot
  python preprocess_v4_integrate.py --all                      # all splits, all shards
"""
import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import numpy as np

VOCAB_DIR = Path("experiments/vocab")
V3_INPUT = Path("/path/to/CHD_MEDS/ethos_input_v3")
INPUT_DIR = Path("/path/to/CHD_MEDS/ethos_input")
OUT = VOCAB_DIR / "outputs"
LOG = VOCAB_DIR / "logs"


# ══════════════════════════════════════════════════════════════════════════════
# Load all stream outputs once
# ══════════════════════════════════════════════════════════════════════════════
print("Loading df stream outputs...")

# Stream F maps: v3_code → v4_code
f_an_event = pd.read_csv(OUT / "an_event_map.csv")
f_transfusion = pd.read_csv(OUT / "transfusion_map.csv")
f_med_lda = pd.read_csv(OUT / "med_lda_leak_map.csv")

MAP_F = {}
for df in [f_an_event, f_transfusion, f_med_lda]:
    MAP_F.update(dict(zip(df["v3_code"], df["v4_code"])))
print(f"  Stream F: {len(MAP_F)} direct code remaps")


# Stream A: drug → ATC
atc_reviewed = pd.read_csv(OUT / "atc_team_reviewed.csv")
# team-reviewed atc (corrected if CORRECT, else final_atc if KEEP)
atc_reviewed["final_v4_atc"] = atc_reviewed.apply(
    lambda r: r["corrected_atc"] if r["decision"] == "CORRECT" and pd.notna(r.get("corrected_atc"))
    else r.get("final_atc"),
    axis=1,
)
llm_atc_map = dict(zip(atc_reviewed["v3_med_code"], atc_reviewed["final_v4_atc"]))

# ETHOS gold mapping (1,731 drugs matched directly from MIMIC dict)
gold_atc = pd.read_csv(OUT / "atc_validation_set.csv")
for _, r in gold_atc.iterrows():
    llm_atc_map.setdefault(r["v3_med_code"], r["gold_atc"])
print(f"  Stream A: {len(llm_atc_map)} drug → ATC mappings")


def decompose_atc(atc_code):
    """ATC code → list of hierarchical tokens."""
    if not atc_code or pd.isna(atc_code):
        return None
    atc_code = str(atc_code).strip()
    if len(atc_code) < 3:
        return None
    tokens = [f"ATC//{atc_code[:3]}"]  # L1+L2
    if len(atc_code) >= 4:
        tokens.append(f"ATC//4//{atc_code[3]}")  # L3 differentiator
    if len(atc_code) >= 7:
        tokens.append(f"ATC//SFX//{atc_code[4:7]}")  # L5 leaf suffix
    elif len(atc_code) >= 5:
        tokens.append(f"ATC//SFX//{atc_code[4:]}")
    return tokens


# Stream B: ICD-10-CM hierarchical
def decompose_icd10(icd_code):
    """ICD-10-CM code (e.g. 'I25.119') → shared hierarchical tokens."""
    icd = icd_code.replace(".", "")
    tokens = []
    if len(icd) >= 3:
        tokens.append(f"ICD//CM//{icd[:3]}")
    if len(icd) >= 4:
        tokens.append(f"ICD//CM//3-6//{icd[3:6]}")
    if len(icd) >= 7:
        tokens.append(f"ICD//CM//SFX//{icd[6:]}")
    return tokens


# Stream C: frequency-cutoff allowed vocabs
def load_allowed(csv_file):
    df = pd.read_csv(OUT / csv_file)
    # First column has token names
    return set(df.iloc[:, 0].tolist())

LAB_ALLOWED = load_allowed("lab_top500_vocab.csv")
PROC_ALLOWED = load_allowed("procedure_top500_vocab.csv")
ENC_AN_PRIM_ALLOWED = load_allowed("encounter_an_primary_top500_vocab.csv")
SDE_ALLOWED = load_allowed("sde_top300_vocab.csv")
print(f"  Stream C: cutoff vocabs — LAB {len(LAB_ALLOWED)}, PROC {len(PROC_ALLOWED)}, "
      f"ENC_PRIM {len(ENC_AN_PRIM_ALLOWED)}, SDE {len(SDE_ALLOWED)}")


# Stream D: DEMO + LANGUAGE (per-patient-encounter)
demo_lang = pd.read_parquet(OUT / "demo_language_tokens.parquet")
# Index by patient_id for fast lookup; we need demographics per-patient
# Take first encounter's demo_tokens as the patient-level atomic DEMO
demo_by_patient = {}
lang_by_patient_enc = {}
for _, r in demo_lang.iterrows():
    pid = r["patient_id"]
    if pid not in demo_by_patient:
        demo_by_patient[pid] = list(r["demo_tokens"])
    # Language per encounter
    lang_by_patient_enc[(pid, str(r["encounter_csn"]))] = r["language_token"]
print(f"  Stream D: demo tokens for {len(demo_by_patient)} patients")


# Stream E: SES per encounter (INSURANCE, POI, COUNTY, AN_Start timestamp)
ses_events = pd.read_parquet(OUT / "ses_events.parquet")
# Index by patient
ses_by_patient = {}
for pid, grp in ses_events.groupby("patient_id"):
    ses_by_patient[pid] = grp.to_dict("records")
print(f"  Stream E: SES events for {len(ses_by_patient)} patients")


# c_mrn (string "C12345") → int subject_id for ethos (subject_id = int stripped of C prefix)
def cmrn_to_subject(cmrn):
    try:
        return int(str(cmrn).lstrip("C"))
    except (ValueError, TypeError):
        return None


# Build subject_id → patient_id lookup (reverse of cmrn_to_subject)
# demo_lang has patient_id as "CXXXXX" strings; ses has same
def pid_from_subject(subject_id):
    return f"C{subject_id}"


# ══════════════════════════════════════════════════════════════════════════════
# Per-row transformation logic
# ══════════════════════════════════════════════════════════════════════════════

PREFIX_CATEGORIES = [
    ("DIAGNOSIS//ICD10//", "icd10"),
    ("PROBLEM//ICD10//",    "icd10_problem"),  # need //END handling
    ("ENCOUNTER//CARDIOLOGY//ICD10//", "icd10_enc"),
    ("MED//", "med"),
    ("LAB//", "lab_cutoff"),
    ("PROCEDURE//", "proc_cutoff"),
    ("ENCOUNTER//AN//PRIMARY_PROCEDURE//", "enc_prim_cutoff"),
    ("SDE//", "sde_cutoff"),
    ("DEMO//", "demo_handle"),  # Stream D replaces DEMO wholesale
]


def transform_row(row):
    """Return list of rows (possibly 0, 1, or multiple) derived from input row.

    Row is a dict-like with keys: subject_id, time, numeric_value, text_value, code.
    """
    code = row["code"]

    # === Stream F: direct code remaps (AN_EVENT typo, TRANSFUSION, MED→LDA) ===
    if code in MAP_F:
        new_code = MAP_F[code]
        # If MED→LDA leak: drop numeric_value (LDA isn't dose-quantized)
        if code.startswith("MED//") and new_code.startswith("LDA//"):
            return [{**row, "code": new_code, "numeric_value": np.nan}]
        return [{**row, "code": new_code}]

    # === Stream B: ICD-10 decomposition ===
    if code.startswith("DIAGNOSIS//ICD10//"):
        raw = code.replace("DIAGNOSIS//ICD10//", "").replace("//END", "")
        if raw in ("UNKNOWN",) or raw.startswith("IMO"):
            # Edge case — keep as-is (they'll become singleton tokens if kept)
            return [row]
        tokens = decompose_icd10(raw)
        out = [{**row, "code": t} for t in tokens]
        return out

    if code.startswith("PROBLEM//ICD10//"):
        stripped = code.replace("PROBLEM//ICD10//", "")
        has_end = stripped.endswith("//END")
        raw = stripped.replace("//END", "")
        if raw in ("UNKNOWN",) or raw.startswith("IMO"):
            return [row]
        tokens = decompose_icd10(raw)
        out = [{**row, "code": t} for t in tokens]
        if has_end:
            out.append({**row, "code": "PROBLEM//END"})
        return out

    if code.startswith("ENCOUNTER//CARDIOLOGY//ICD10//"):
        raw = code.replace("ENCOUNTER//CARDIOLOGY//ICD10//", "")
        if raw in ("UNKNOWN",) or raw.startswith("IMO"):
            return [row]
        tokens = decompose_icd10(raw)
        return [{**row, "code": t} for t in tokens]

    # === Stream A: MED → ATC hierarchical ===
    if code.startswith("MED//Q//"):
        # Dose quantile row — keep code name per-drug (per original drug)
        # The Quantizator will still bin numeric_value by per-code.
        return [row]
    if code.startswith("MED//"):
        atc = llm_atc_map.get(code)
        if atc is None:
            # Unmapped drug — fallback preserves the code but marks it
            normalized = code.replace("MED//", "").upper()
            # Truncate overly long names to avoid vocab explosion
            normalized = re.sub(r"[_/]+", "_", normalized)[:60]
            return [{**row, "code": f"MED//UNMAPPED//{normalized}"}]
        tokens = decompose_atc(atc)
        if tokens is None:
            return [{**row, "code": f"MED//UNMAPPED//{code.replace('MED//', '')[:60]}"}]
        return [{**row, "code": t} for t in tokens]

    # === Stream C: frequency cutoff with OTHER fallback ===
    if code.startswith("LAB//"):
        if code in LAB_ALLOWED or code.startswith("LAB//Q//"):
            return [row]
        return [{**row, "code": "LAB//OTHER"}]
    if code.startswith("PROCEDURE//"):
        if code in PROC_ALLOWED:
            return [row]
        return [{**row, "code": "PROCEDURE//SURG_HX//OTHER"}]
    if code.startswith("ENCOUNTER//AN//PRIMARY_PROCEDURE//"):
        if code in ENC_AN_PRIM_ALLOWED:
            return [row]
        return [{**row, "code": "ENCOUNTER//AN//PRIMARY_PROCEDURE//OTHER"}]
    if code.startswith("SDE//"):
        if code in SDE_ALLOWED or code.startswith("SDE//Q//"):
            return [row]
        return [{**row, "code": "SDE//OTHER"}]

    # === Stream D: DEMO atomic replacement ===
    # DEMO events will be fully replaced at patient-level (see per-patient pass below).
    # Here we DROP all DEMO rows except DEMO//Q//GESTATIONAL_AGE/BMI quantized numerics.
    if code.startswith("DEMO//Q//"):
        return [row]  # keep quantized numerics (gestational age, BMI if exist)
    if code.startswith("DEMO//"):
        return []  # drop — replaced by atomic tokens in patient-level pass

    # Default: keep as-is (VITAL, LDA, ADT, AN_EVENT, TRANSFUSION non-mapped, ENCOUNTER//AN//flow, etc.)
    return [row]


# ══════════════════════════════════════════════════════════════════════════════
# Per-patient emission of Stream D/E static + per-encounter events
# ══════════════════════════════════════════════════════════════════════════════

def emit_patient_events(subject_id):
    """Emit new patient-level and per-encounter events (Streams D+E).

    Returns list of new rows (with subject_id, time, numeric_value, text_value, code).
    """
    pid = pid_from_subject(subject_id)
    out = []

    # Static (NaT) DEMO atomic tokens
    for tok in demo_by_patient.get(pid, []):
        out.append({
            "subject_id": subject_id,
            "time": pd.NaT,
            "numeric_value": np.nan,
            "text_value": None,
            "code": tok,
        })

    # Static (NaT) HOME_COUNTY — use first encounter's value
    ses_list = ses_by_patient.get(pid, [])
    if ses_list:
        first = ses_list[0]
        if first.get("home_county_token"):
            out.append({
                "subject_id": subject_id,
                "time": pd.NaT,
                "numeric_value": np.nan,
                "text_value": None,
                "code": first["home_county_token"],
            })

    # Per-encounter: INSURANCE + POI + LANGUAGE at AN_Start timestamp
    for se in ses_list:
        csn = str(se["encounter_csn"])
        # Use an_start if exists, else in_or_timestamp, else skip
        t = se.get("an_start_timestamp") or se.get("in_or_timestamp")
        if t is None or pd.isna(t):
            continue
        for key in ("insurance_token", "point_of_origin_token"):
            val = se.get(key)
            if val:
                out.append({
                    "subject_id": subject_id,
                    "time": pd.Timestamp(t),
                    "numeric_value": np.nan,
                    "text_value": None,
                    "code": val,
                })
        # Language per encounter
        lang = lang_by_patient_enc.get((pid, csn))
        if lang:
            out.append({
                "subject_id": subject_id,
                "time": pd.Timestamp(t),
                "numeric_value": np.nan,
                "text_value": None,
                "code": lang,
            })

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Process a single shard
# ══════════════════════════════════════════════════════════════════════════════

def process_shard(in_path: Path, out_path: Path, diagnostics=True):
    print(f"\n--- {in_path.name} → {out_path.name}")
    t0 = time.time()
    df = pd.read_parquet(in_path)
    in_rows = len(df)
    in_patients = df["subject_id"].nunique()

    # Row-by-row transformation (vectorization would require heavy refactor; this is acceptable for one-time run)
    out_rows = []
    for _, row in df.iterrows():
        rd = row.to_dict()
        out_rows.extend(transform_row(rd))

    # Emit per-patient new events (DEMO atomic + SES)
    for subject_id in df["subject_id"].unique():
        out_rows.extend(emit_patient_events(int(subject_id)))

    df = pd.DataFrame(out_rows)
    # Sort by (subject_id, time) with NaT first (static events at beginning)
    # Use nsmallest workaround for NaT handling
    df["_time_sort"] = df["time"].fillna(pd.Timestamp("1900-01-01"))
    df = df.sort_values(["subject_id", "_time_sort"]).drop(columns=["_time_sort"]).reset_index(drop=True)

    # Cast dtypes to match
    df["subject_id"] = df["subject_id"].astype("int64")
    df["numeric_value"] = df["numeric_value"].astype("float32")

    df.to_parquet(out_path, index=False)

    # Diagnostics
    if diagnostics:
        out_patients = df["subject_id"].nunique()
        row_expansion = len(df) / in_rows
        # Sequence length stats
        seq_lens = df.groupby("subject_id").size()
        # Code family counts
        fam = df["code"].str.split("//").str[0]
        top_fams = fam.value_counts().head(10).to_dict()
        n_other = int((df["code"].str.endswith("//OTHER") | (df["code"] == "LAB//OTHER") | (df["code"] == "SDE//OTHER")).sum())
        n_unmapped_med = int(df["code"].str.startswith("MED//UNMAPPED//").sum())
        n_atc = int(df["code"].str.startswith("ATC//").sum())
        n_icd = int(df["code"].str.startswith("ICD//CM//").sum())

        print(f"  in:  {in_rows:>10,} rows,  {in_patients:>6,} patients,  {df.code.nunique()} unique codes")
        print(f"  out: {len(df):>10,} rows,  {out_patients:>6,} patients,  {df.code.nunique()} unique codes")
        print(f"  expansion: {row_expansion:.2f}x")
        print(f"  seq len (rows/patient): median {seq_lens.median():.0f}, mean {seq_lens.mean():.0f}, p95 {seq_lens.quantile(0.95):.0f}, max {seq_lens.max():,}")
        print(f"  ATC events: {n_atc:,},  ICD hier events: {n_icd:,}")
        print(f"  OTHER fallback events: {n_other:,},  unmapped MED: {n_unmapped_med:,}")
        print(f"  top code families: {top_fams}")
        print(f"  time: {time.time()-t0:.1f}s")
    return len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "test"], default="train")
    ap.add_argument("--shard", type=int, default=None, help="single shard number (pilot)")
    ap.add_argument("--all", action="store_true", help="all splits, all shards")
    args = ap.parse_args()

    splits = ["train", "val", "test"] if args.all else [args.split]
    for split in splits:
        src = V3_INPUT / split
        dst = INPUT_DIR / split
        dst.mkdir(parents=True, exist_ok=True)

        shards = sorted(src.glob("shard_*.parquet"))
        if args.shard is not None and not args.all:
            shards = [s for s in shards if f"shard_{args.shard:03d}" in s.name]

        print(f"\n{'='*70}")
        print(f"Split: {split}  ({len(shards)} shards)")
        print('='*70)

        for shard in shards:
            process_shard(shard, dst / shard.name)


if __name__ == "__main__":
    main()
