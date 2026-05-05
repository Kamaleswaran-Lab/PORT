"""
Stream A Step 5: Build final drug → ATC mapping + spot-check sheet.

Inputs:
  - outputs/atc_validation_set.csv  (1,731 ETHOS gold mappings)
  - outputs/atc_full_dose_Llama-3p3-70B.csv  (762 LLM mappings)
  - outputs/atc_name_lookup.json  (ATC code → generic name)

Outputs:
  - outputs/drug_name_to_atc.csv  (final 2,493 drug mappings)
  - outputs/atc_spot_check_top100.csv  (top-100 LLM-mapped by frequency, for expert review)
"""
import pandas as pd
import json
from pathlib import Path

OUT = Path("experiments/vocab/outputs")

# Load sources
gold = pd.read_csv(OUT / "atc_validation_set.csv")
llm  = pd.read_csv(OUT / "atc_full_dose_Llama-3p3-70B.csv")
with open(OUT / "atc_name_lookup.json") as f:
    atc_name = json.load(f)

print(f"Gold (ETHOS MIMIC): {len(gold):,} drugs")
print(f"LLM (Llama 3.3 70B dose-aware): {len(llm):,} drugs")

# LLM classification stats
print(f"\nLLM classification depth stats:")
for L in [1, 2, 3, 4, 5]:
    col = f"L{L}"
    n_reached = llm[col].notna().sum()
    print(f"  L{L} reached: {n_reached}/{len(llm)} ({100*n_reached/len(llm):.1f}%)")

n_failed = llm["final_atc"].isna().sum()
print(f"\nClassification failures (no final_atc): {n_failed}")
n_partial = (llm["final_atc"].notna() & llm["L5"].isna()).sum()
print(f"Partial classifications (stopped before L5): {n_partial}")

# Build unified final mapping
# Gold: has gold_atc (L5 code from MIMIC dict)
gold_unified = pd.DataFrame({
    "v3_med_code": gold["v3_med_code"],
    "normalized_name": gold["normalized_name"],
    "atc_code": gold["gold_atc"],
    "source": "ETHOS_MIMIC_dict",
    "confidence": "high",
    "frequency": gold["frequency"],
})

# LLM: has L1-L5 predictions, final_atc = deepest level reached
llm_unified = pd.DataFrame({
    "v3_med_code": llm["v3_med_code"],
    "normalized_name": llm["drug_name"],
    "atc_code": llm["final_atc"],
    "source": "Llama_3.3_70B_dose_aware",
    "confidence": "medium",
    "frequency": llm["frequency"],
})

# Decompose ATC code → L1-L5 + names
def decompose(code):
    if pd.isna(code) or not isinstance(code, str):
        return pd.Series([None]*10)
    n = len(code)
    L1 = code[:1] if n >= 1 else None
    L2 = code[:3] if n >= 3 else None
    L3 = code[:4] if n >= 4 else None
    L4 = code[:5] if n >= 5 else None
    L5 = code if n == 7 else None
    return pd.Series([L1, L2, L3, L4, L5,
                      atc_name.get(L1), atc_name.get(L2), atc_name.get(L3),
                      atc_name.get(L4), atc_name.get(L5)])

for df in (gold_unified, llm_unified):
    df[["L1","L2","L3","L4","L5","L1_name","L2_name","L3_name","L4_name","L5_name"]] = \
        df["atc_code"].apply(decompose)

# Combine
final = pd.concat([gold_unified, llm_unified], ignore_index=True)
final = final.sort_values("frequency", ascending=False).reset_index(drop=True)
final.to_csv(OUT / "drug_name_to_atc.csv", index=False)
print(f"\nSaved unified mapping: {OUT/'drug_name_to_atc.csv'}  ({len(final)} drugs)")

# Source distribution
print("\nFinal mapping source distribution:")
print(final["source"].value_counts().to_string())
print(f"\nConfidence distribution:")
print(final["confidence"].value_counts().to_string())

# Build top-100 spot-check sheet (LLM-mapped only)
llm_with_ctx = llm.merge(llm_unified[["v3_med_code","L1","L2","L3","L4","L5",
                                        "L1_name","L2_name","L3_name","L4_name","L5_name"]],
                          on="v3_med_code", how="left")
spot = llm_with_ctx.sort_values("frequency", ascending=False).head(100).copy()
spot_cols = ["v3_med_code", "drug_name", "route", "dose", "purpose", "frequency",
             "L1", "L1_name", "L2", "L2_name", "L3", "L3_name",
             "L4", "L4_name", "L5", "L5_name", "final_atc"]
spot = spot[[c for c in spot_cols if c in spot.columns]]
# Add empty review columns
spot["reviewer_decision"] = ""  #  /  / ?
spot["correct_atc"] = ""         # if , the correct ATC code
spot["reviewer_comment"] = ""
spot.to_csv(OUT / "atc_spot_check_top100.csv", index=False)
print(f"\nSaved spot-check sheet: {OUT/'atc_spot_check_top100.csv'}  (top-100 LLM drugs)")

# Summary of spot-check coverage
spot_events = spot["frequency"].sum()
all_llm_events = llm["frequency"].sum()
print(f"Top-100 covers {100*spot_events/all_llm_events:.1f}% of LLM-mapped events "
      f"({spot_events:,} / {all_llm_events:,})")

# Print top 10 samples
print("\nTop 10 LLM mappings (by frequency):")
for _, r in spot.head(10).iterrows():
    lname = str(r.get("L3_name", ""))[:30]
    print(f"  {r.drug_name[:35]:<35s}  freq={r.frequency:>7,d}  → {r.final_atc or 'N/A':<8s} ({lname})")

print("\nDone.")
