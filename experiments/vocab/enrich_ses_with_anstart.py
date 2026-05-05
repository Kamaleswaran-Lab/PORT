"""
Enrich ses_events.parquet with an_start_timestamp column.
Option (a) design lock: emit SES events at AN_Start (all encounters, including non-surgical).
"""
import pandas as pd
import polars as pl
from pathlib import Path

OUT = Path("experiments/vocab/outputs")

# Load ses_events (already has patient_id, encounter_csn, in_or_timestamp)
ses = pd.read_parquet(OUT / "ses_events.parquet")
# Drop any previous AN_Start column (in case this script was re-run)
if "an_start_timestamp" in ses.columns:
    ses = ses.drop(columns=["an_start_timestamp"])
# Normalize encounter_csn: strip '.0' suffix from float-cast strings
ses["encounter_csn"] = ses["encounter_csn"].astype(str).str.replace(r"\.0$", "", regex=True)
print(f"SES rows: {len(ses):,}")

# Load AN_Patients with polars (streaming, memory-efficient)
print("Loading AN_Patients with polars...")
an_pl = pl.read_csv(
    "/path/to/CHOA_RAW_TABLES/CHOA_DATA_Tables_CHD/DR15201_AN_Patients.rpt",
    separator="|", columns=["C MRN", "Encounter CSN", "AN Start"],
    infer_schema_length=0,  # all strings
)
print(f"AN rows: {an_pl.height:,}")
an_pl = (an_pl
    .rename({"C MRN": "patient_id", "Encounter CSN": "encounter_csn"})
    .with_columns(
        pl.col("AN Start").str.to_datetime(format="%m/%d/%Y %H:%M", strict=False).alias("an_start_timestamp"),
    )
    .select(["patient_id", "encounter_csn", "an_start_timestamp"])
)
# Drop duplicates — keep first per (patient, csn)
an_pl = an_pl.unique(subset=["patient_id", "encounter_csn"], keep="first")
an = an_pl.to_pandas()
print(f"AN after dedup: {len(an):,}")

# Merge
ses["encounter_csn"] = ses["encounter_csn"].astype(str)
an["encounter_csn"] = an["encounter_csn"].astype(str)
merged = ses.merge(an, on=["patient_id", "encounter_csn"], how="left")
print(f"Merged rows: {len(merged):,}")
print(f"AN_Start available: {merged['an_start_timestamp'].notna().sum():,} / {len(merged):,}")
print(f"In_OR available:    {merged['in_or_timestamp'].notna().sum():,} / {len(merged):,}")

# Save
merged.to_parquet(OUT / "ses_events.parquet", index=False)
print(f"Saved enriched SES: {OUT/'ses_events.parquet'}")
