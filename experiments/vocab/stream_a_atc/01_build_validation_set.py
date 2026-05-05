"""
Stream A Step 1: Build validation set for ATC LLM mapping.

Approach (Chen 2024 methodology):
  - Ground truth: ETHOS MIMIC drug→ATC mapping (7,569 entries)
  - Our set: MED drug names (2,496 codes)
  - Overlap = validation set (expected ~900-1,000 drugs)

Outputs:
  outputs/atc_validation_set.csv — (our_drug_code, normalized_name, gold_atc, frequency)
  outputs/atc_unmapped_drugs.csv — drugs needing LLM classification
  logs/A1_validation_set_report.md
"""
import pandas as pd
import re
from pathlib import Path

OUT = Path("experiments/vocab/outputs")
OUT.mkdir(parents=True, exist_ok=True)
LOG = Path("experiments/vocab/logs")
LOG.mkdir(parents=True, exist_ok=True)


# ─── drug name normalization (for dictionary lookup) ────────────────────────
DOSE_PATTERNS = [
    r"\d+_?MG(/\d+_?ML)?",       # 500_MG, 2_MG/ML
    r"\d+_?MCG(/\d+_?ML)?",      # 100_MCG
    r"\d+_?UNIT(S)?(/\d+_?ML)?", # 2_UNITS/ML
    r"\d+_?PCT",                 # 5_PCT
    r"\d+_?MEQ(/\d+_?ML)?",      # 20_MEQ/ML
    r"\d+_?ML",                  # 500_ML
    r"\d+_?G(RAM)?",             # 10_GRAM
    r"\d+PCT\d+PCT\d+",
]
ROUTE_SUFFIXES = [
    "_DRIP", "_INTRAVENOUS", "_IV", "_ORAL_SOLUTION", "_ORAL_SUSPENSION",
    "_ORAL_LIQUID", "_ORAL", "_SYRINGE", "_TABLET", "_CAPSULE", "_INJECTION",
    "_IVPB_SOLUTION", "_IVPB", "_IV_SOLUTION", "_INJECTION_SYRINGE",
    "_INJECTION_SOLUTION", "_INFUSION", "_SUBCUTANEOUS", "_SUBCUT",
    "_INTRAMUSCULAR", "_IM", "_TOPICAL", "_OPHTHALMIC", "_OTIC",
    "_INHALATION", "_NEBULIZER", "_TRANSDERMAL", "_PF",
]

def normalize_drug(raw):
    """Strip MED// prefix, dose, route, formulation — keep core drug name."""
    name = raw.replace("MED//", "", 1).upper()
    # dose patterns
    for pat in DOSE_PATTERNS:
        name = re.sub(pat, "", name)
    # route suffixes
    for sfx in ROUTE_SUFFIXES:
        name = name.replace(sfx, "")
    # cleanup underscores and slashes
    name = re.sub(r"[_/]+", "_", name).strip("_")
    # remove empty concentration remnants
    name = re.sub(r"_+", "_", name)
    return name


# ─── load data ───────────────────────────────────────────────────────────────
print("Loading ETHOS MIMIC drug→ATC mapping...")
atc_map = pd.read_csv(
    "/path/to/ethos-ares/src/ethos/tokenize/maps/mimic_drug_to_atc.csv.gz"
)
atc_map["drug_upper"] = atc_map["drug"].str.upper().str.replace(" ", "_")
atc_map = atc_map.drop_duplicates(subset="drug_upper", keep="first")
# Deduplicate to one ATC per drug (take first if multiple)
atc_lookup = dict(zip(atc_map["drug_upper"], atc_map["atc_code"]))
print(f"  ETHOS map: {len(atc_lookup):,} unique drugs")

print("\nLoading ATC coding definitions...")
atc_defs = pd.read_csv(
    "/path/to/ethos-ares/src/ethos/tokenize/maps/atc_coding.csv.gz"
)
atc_defs = atc_defs.drop_duplicates(subset="atc_code", keep="last")
atc_name_lookup = dict(zip(atc_defs["atc_code"], atc_defs["atc_name"]))
print(f"  ATC ontology: {len(atc_name_lookup):,} entries across L1-L5")

print("\nLoading our MED codes...")
cc = pd.read_csv("/path/to/CHD_MEDS/tokenized_v3/train/code_counts.csv")
med = cc[cc.code.str.startswith("MED//")].copy()
med = med.sort_values("count", ascending=False).reset_index(drop=True)
print(f"  Our MED tokens: {len(med):,}  total events: {med['count'].sum():,}")


# ─── non-medication leak filter (Stream F handles this; we flag here) ───────
LEAK_PATTERNS = ["PERIPHERAL_ARTLINE", "CENTRAL_VENOUS_LINE", "PROC_SITE_LD",
                 "ARTLINE", "PICC_LINE", "CVL"]
leak_mask = med["code"].str.contains("|".join(LEAK_PATTERNS), case=False, regex=True)
n_leak = leak_mask.sum()
print(f"  Non-medication leak (flagged, excluded from ATC mapping): {n_leak}")
med_drugs = med[~leak_mask].copy()


# ─── normalize, attempt ETHOS dict match ────────────────────────────────────
print("\nNormalizing drug names and matching against ETHOS MIMIC dict...")
med_drugs["normalized"] = med_drugs["code"].apply(normalize_drug)

# Multi-strategy matching:
#  a) exact match of normalized full name
#  b) first-word match
med_drugs["atc_exact"] = med_drugs["normalized"].map(atc_lookup)
med_drugs["first_word"] = med_drugs["normalized"].str.split("_").str[0]
med_drugs["atc_first_word"] = med_drugs["first_word"].map(atc_lookup)
med_drugs["atc_gold"] = med_drugs["atc_exact"].fillna(med_drugs["atc_first_word"])

# Coverage
n_exact = med_drugs["atc_exact"].notna().sum()
n_first_only = ((med_drugs["atc_exact"].isna()) & (med_drugs["atc_first_word"].notna())).sum()
n_total_mapped = med_drugs["atc_gold"].notna().sum()
n_unmapped = med_drugs["atc_gold"].isna().sum()
n_total = len(med_drugs)

events_exact = med_drugs[med_drugs["atc_exact"].notna()]["count"].sum()
events_any = med_drugs[med_drugs["atc_gold"].notna()]["count"].sum()
events_total = med_drugs["count"].sum()

print(f"\n  Exact normalized match: {n_exact}/{n_total} ({100*n_exact/n_total:.1f}%)")
print(f"  First-word fallback:    {n_first_only}/{n_total} ({100*n_first_only/n_total:.1f}%)")
print(f"  Total mapped:           {n_total_mapped}/{n_total} ({100*n_total_mapped/n_total:.1f}%)")
print(f"  Unmapped (needs LLM):   {n_unmapped}/{n_total} ({100*n_unmapped/n_total:.1f}%)")
print(f"  Event coverage: exact {100*events_exact/events_total:.1f}%, any {100*events_any/events_total:.1f}%")


# ─── Save outputs ────────────────────────────────────────────────────────────
# Validation set = drugs mapped via ETHOS (ground truth for LLM testing)
val_set = med_drugs[med_drugs["atc_gold"].notna()][
    ["code", "normalized", "atc_gold", "count"]
].copy()
val_set.columns = ["v3_med_code", "normalized_name", "gold_atc", "frequency"]
val_set.to_csv(OUT / "atc_validation_set.csv", index=False)
print(f"\n  Saved validation set: {len(val_set):,} drugs → {OUT/'atc_validation_set.csv'}")

# Unmapped drugs (need LLM classification in Stream A step 3)
unmapped = med_drugs[med_drugs["atc_gold"].isna()][["code", "normalized", "count"]].copy()
unmapped.columns = ["v3_med_code", "normalized_name", "frequency"]
unmapped.to_csv(OUT / "atc_unmapped_drugs.csv", index=False)
print(f"  Saved unmapped set: {len(unmapped):,} drugs → {OUT/'atc_unmapped_drugs.csv'}")


# ─── Report ──────────────────────────────────────────────────────────────────
report = f"""# Stream A Step 1 — ATC Validation Set Construction

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

## Purpose

Build a ground-truth validation set for the LLM-based ATC classifier (Chen 2024 methodology), using the ETHOS MIMIC drug→ATC dictionary as gold standard for drugs that overlap with our pediatric CHD cohort.

## Data sources

- ETHOS MIMIC drug→ATC map: `ethos-ares/src/ethos/tokenize/maps/mimic_drug_to_atc.csv.gz` — {len(atc_lookup):,} unique drugs
- ATC ontology: `ethos-ares/src/ethos/tokenize/maps/atc_coding.csv.gz` — {len(atc_name_lookup):,} codes across L1-L5
- Our MED codes: `/path/to/CHD_MEDS/tokenized_v3/train/code_counts.csv` — {len(med):,} codes

## Normalization

Raw MED codes have dose/route/formulation concatenated, e.g.:
- `MED//HEPARIN_PF_IN_NS_2_UNITS/ML_IV_SYRINGE` → normalized `HEPARIN`
- `MED//POTASSIUM_CHLORIDE_20_MEQ/L_IN_DEXTROSE_5_PCT045_PCT_SODIUM_CHLORIDE_IV` → `POTASSIUM_CHLORIDE`
- `MED//FENTANYL_DRIP` → `FENTANYL`

Applied: dose regexes (`\\d+_?MG`, `\\d+_?MCG`, etc.), route suffixes (`_DRIP`, `_IV`, `_INJECTION_SYRINGE`, etc.).

## Non-medication leak (to be handled by Stream F)

Tokens flagged for removal from MED category: `{LEAK_PATTERNS}`
Count flagged: **{n_leak}**

## Matching results ({n_total} non-leak MED codes, {events_total:,} events)

| Match strategy | Codes matched | Events matched |
|----------------|---------------|----------------|
| Exact normalized name | {n_exact} ({100*n_exact/n_total:.1f}%) | {events_exact:,} ({100*events_exact/events_total:.1f}%) |
| First-word fallback (additional) | {n_first_only} | |
| **Total in validation set** | **{n_total_mapped} ({100*n_total_mapped/n_total:.1f}%)** | **{events_any:,} ({100*events_any/events_total:.1f}%)** |
| **Needs LLM classification** | **{n_unmapped} ({100*n_unmapped/n_total:.1f}%)** | {events_total-events_any:,} ({100*(events_total-events_any)/events_total:.1f}%) |

## Outputs

1. `outputs/atc_validation_set.csv` — {n_total_mapped} drugs with ETHOS gold-standard ATC codes, used to measure LLM accuracy.
2. `outputs/atc_unmapped_drugs.csv` — {n_unmapped} drugs the LLM will classify.

## Sample validation entries

```
{val_set.head(10).to_string(index=False)}
```

## Sample unmapped drugs (top 10 by frequency)

```
{unmapped.head(10).to_string(index=False)}
```

## Next step

Step 2: Build ATC ontology tree from `atc_coding.csv` for hierarchical LLM prompting (level-by-level constrained classification, per Chen 2024).

Step 3: Run pilot LLM classification on validation set, measure L1-L5 accuracy vs gold labels.

Step 4: If L3 accuracy ≥70% (Chen 2024 baseline), proceed to full unmapped-drug classification.
Step 5: Expert review of top-100 final mappings.
"""
(LOG / "A1_validation_set_report.md").write_text(report)
print(f"\n  Report: {LOG/'A1_validation_set_report.md'}")
print("Done.")
