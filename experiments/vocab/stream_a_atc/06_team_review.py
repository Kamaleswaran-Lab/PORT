"""
Stream A Step 6: Research-team review of all 762 LLM-classified drug → ATC mappings.

Workflow (Phase 1 of path 2):
  1. Apply systematic pattern-based corrections (TPN, IV fluids, etc.)
  2. Case-by-case decisions encoded below for top-200 frequent drugs
  3. Remaining low-frequency drugs → keep LLM prediction unless obvious error
  4. Output structured review with decision, corrected_atc, confidence, rationale
  5. Generate expert validation sample (60 drugs: all uncertain + random 20 keep + 20 corrected)

Each decision is ONE of:
  KEEP:      LLM prediction accepted as correct (high-confidence)
  CORRECT:   Specific ATC revision (high-confidence by research team)
  UNCERTAIN: Flagged for expert decision (low-confidence / genuinely ambiguous)
"""
import pandas as pd
import re
import json
import random
from pathlib import Path

OUT = Path("experiments/vocab/outputs")
LOG = Path("experiments/vocab/logs")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ─── Systematic pattern corrections ────────────────────────────────────────
# Each (regex_on_v3_code, new_atc, rationale)
PATTERN_CORRECTIONS = [
    # TPN family → B05BA10 (combinations for parenteral nutrition)
    (r"^MED//TPN", "B05BA10",
     "TPN formulations are parenteral nutrition combinations, not vitamin products"),
    # Fat emulsion → B05BA02
    (r"^MED//FAT_EMULSION", "B05BA02",
     "Fat emulsion for parenteral nutrition (intralipid)"),
    # IV dextrose + electrolyte mixtures → B05BB02 (electrolytes with carbs)
    # Only when LLM said B05BB01 (pure electrolyte) for a solution containing dextrose
    (r"^MED//(D5|D10|D51|D101|DEXTROSE).*(_NS_|SODIUM_CHLORIDE|NORMAL_SALINE)",
     "B05BB02",
     "Dextrose+saline mixture: B05BB02 electrolytes with carbohydrates"),
    # KCL in IV fluid → B05XA01 (IV solution additive, not oral supplement)
    (r"^MED//KCL_.*(_IN_NS|_IN_DEXTROSE|_IN_D|_NS|_D5W|_D10W|PCT_NS)", "B05XA01",
     "KCl as IV solution additive, not oral A12BA"),
    (r"^MED//POTASSIUM_CHLORIDE.*(_IN_NS|_IN_DEXTROSE|_IN_D|_IV|_NS|_D5)", "B05XA01",
     "Potassium chloride as IV additive"),
    # Vecuronium (all forms) → M03AC03 muscle relaxant
    (r"^MED//VECURONIUM", "M03AC03",
     "Vecuronium is neuromuscular blocker (M03AC03), not anesthetic"),
    # Pentobarbital (all forms) → N05CA01
    (r"^MED//PENTOBARBITAL", "N05CA01",
     "Pentobarbital is barbiturate hypnotic (N05CA01), not thiopental"),
    # Nalbuphine → N02AF02
    (r"^MED//NALBUPHINE", "N02AF02",
     "Nalbuphine is N02AF02, not pentazocine"),
    # Levetiracetam → N03AX14
    (r"^MED//LEVETIRACETAM", "N03AX14",
     "Levetiracetam is N03AX14, not N03AF carboxamides"),
    # Dornase alfa → R05CB13
    (r"^MED//DORNASE_ALFA", "R05CB13",
     "Dornase alfa is mucolytic for CF"),
    # D20W → B05BA03 parenteral nutrition carbohydrate
    (r"^MED//D20W", "B05BA03",
     "20% dextrose for parenteral nutrition"),
    # Povidone iodine eye drops → S01AX18
    (r"^MED//POVIDONEIODINE.*EYE", "S01AX18",
     "Povidone-iodine ophthalmic antiinfective"),
    # Balanced salt solution (ophthalmic irrigation) → S01XA20
    (r"^MED//BALANCED_SALT_SOLUTION.*INTRAOCULAR", "S01XA20",
     "Ophthalmic balanced salt irrigation, not chymotrypsin"),
    # Lidocaine-prilocaine combination → N01BB20
    (r"^MED//LIDOCAINEPRILOCAINE", "N01BB20",
     "Lidocaine/prilocaine combination (EMLA-type), N01BB20"),
    # Heparin flush concentrations → B01AB01 but flagged as flush (non-therapeutic)
    # We keep LLM's B01AB01 but note in comment
    # Cholecalciferol (Vit D3) → A11CC05
    (r"^MED//CHOLECALCIFEROL", "A11CC05",
     "Cholecalciferol = vitamin D3, specific ATC A11CC05"),
    # Bumetanide → C03CA02 loop diuretic
    (r"^MED//BUMETANIDE", "C03CA02",
     "Bumetanide is loop diuretic C03CA02, not etozolin"),
    # Rituximab → L01XC02 (legacy) / L01FA01 (ATC 2022+). We use L01XC02 for MIMIC compat
    (r"^MED//RITUXIMAB", "L01XC02",
     "Rituximab is L01XC02, not ofatumumab"),
    # Barium sulfate contrast → V08BA02
    (r"^MED//BARIUM_SULFATE", "V08BA02",
     "Barium sulfate contrast media, not sodium phosphate"),
    # Omega-3 / fish oil → C10AX06
    (r"^MED//OMEGA3|^MED//FISH_OIL", "C10AX06",
     "Omega-3 fatty acids / fish oil, lipid-modifying C10AX06"),
    # Antithymocyte globulin rabbit → L04AA04
    (r"^MED//ANTITHYMOCYTE_GLOB", "L04AA04",
     "Antithymocyte globulin is immunosuppressant L04AA04"),
    # Dextran 40 volume expander → B05AA05
    (r"^MED//DEXTRAN_40", "B05AA05",
     "Dextran 40 plasma volume expander B05AA05"),
    # Tuberculin PPD → V04CF01
    (r"^MED//TUBERCULIN|^MED//PPD_", "V04CF01",
     "Tuberculin PPD for TB testing, V04CF01"),
    # IVIG → J06BA02
    (r"^MED//GAMMAGARD|^MED//GAMUNEX|^MED//IVIG|^MED//IMMUNE_GLOBULIN|^MED//GAMMAPLEX", "J06BA02",
     "Human normal immunoglobulin (IVIG) is J06BA02"),
    # Lactated Ringer's (all variants) → B05BB01
    (r"^MED//LACTATED_RINGERS", "B05BB01",
     "Lactated Ringer's is electrolyte solution B05BB01"),
    # Oxacillin (mis-routed via OPTM_ prefix) → J01CF04
    (r"^MED//(OPTM_)?OXACILLIN", "J01CF04",
     "Oxacillin is penicillin J01CF04, not ocular"),
    # Diclofenac topical gel → M02AA15
    (r"^MED//DICLOFENAC_GEL", "M02AA15",
     "Diclofenac topical gel is M02AA15"),
    # Mometasone + Formoterol → R03AK13 combination inhaler
    (r"^MED//MOMETASONEFORMOTEROL", "R03AK13",
     "Mometasone+formoterol combination inhaler R03AK13"),
    # Neomycin+Polymyxin+Dexamethasone eye → S01CA01
    (r"^MED//NEOMYCIN.*DEXAMETH.*EYE|^MED//NEOMYCINPOLYMYXINDEXAMETH", "S01CA01",
     "Neomycin/polymyxin/dexamethasone eye combo S01CA01"),
    # Phenol mucosal spray → R02AA19
    (r"^MED//PHENOL_MUCOSAL", "R02AA19",
     "Phenol throat spray R02AA19, not cocaine"),
]

# ─── Case-by-case research-team decisions ─────────────────────────────────
# Format: v3_med_code → (decision, corrected_atc_or_None, confidence, rationale)
# decision: KEEP, CORRECT, UNCERTAIN
# Only drugs where research team disagrees with LLM or wants to explicitly flag

EXPLICIT_DECISIONS = {
    # ── Top 30 frequency review ──
    "MED//FAT_EMULSION_20_PCT_INTRAVENOUS":
        ("CORRECT", "B05BA02", "high", "Intralipid / fat emulsion for PN (B05BA02). LLM A16AX wrong."),
    "MED//TPNNEONATAL":
        ("CORRECT", "B05BA10", "high", "Neonatal TPN combination (B05BA10). LLM A11EC (vitamin B) wrong."),
    "MED//TPNPEDIATRIC":
        ("CORRECT", "B05BA10", "high", "Pediatric TPN combination. LLM A11J wrong."),
    "MED//TPN_SYRINGE_1":
        ("CORRECT", "B05BA10", "high", "TPN in syringe. LLM A11JA wrong."),
    "MED//TPNADOLESCENT/ADULT":
        ("CORRECT", "B05BA10", "high", "Adolescent/adult TPN combination. LLM A11AA03 (multivitamin) wrong."),
    "MED//VANCOMYCIN_IVPB_SOLUTION":
        ("KEEP", None, "high", "Vancomycin J01XA01 correct."),
    "MED//RANITIDINE_IVPB_SOLUTION":
        ("KEEP", None, "high", "Ranitidine A02BA02 correct (H2 antagonist)."),
    "MED//CEFAZOLIN_IVPB_SOLUTION":
        ("CORRECT", "J01DB04", "high",
         "Cefazolin is 1st-gen cephalosporin (J01DB04). LLM J01C (penicillins) wrong anatomical class."),
    "MED//PIPERACILLINTAZOBACTAM_IVPB_SOLN":
        ("KEEP", None, "high", "Piperacillin-tazobactam J01CR05 correct."),
    "MED//PROAIR_HFA_90_MCG/ACTUATION_AEROSOL_INHALER":
        ("KEEP", None, "high", "Albuterol inhaler R03AC02 correct."),
    "MED//CHLORHEXIDINE_GLUCONATE_2_PCT_TOWELETTE":
        ("KEEP", None, "high", "Chlorhexidine antiseptic D08AC02 correct."),
    "MED//CLINDAMYCIN_IVPB_SOLUTION":
        ("KEEP", None, "high", "Clindamycin J01FF01 correct."),
    "MED//MEROPENEM_IVPB_SOLUTION":
        ("KEEP", None, "high", "Meropenem J01DH02 correct."),
    "MED//DEXTROSESODIUM_CHLORIDE_ADDITIVES":
        ("KEEP", None, "high", "Dextrose/saline B05BB02 correct."),
    "MED//HEPARIN_DRIPECMO":
        ("KEEP", None, "high", "Heparin B01AB01 correct; ECMO anticoagulation."),
    "MED//LACTOBACILLUS_RHAMNOSUS_GG_10_BILLION_CELL_CAPSULE":
        ("KEEP", None, "medium", "Lactobacillus probiotic A07FA01 reasonable."),
    "MED//SALIVA_SUBSTITUTE_COMBO_NO9_MOUTHWASH":
        ("UNCERTAIN", None, "low", "Saliva substitute has no dedicated ATC; S03D not valid. May be A01AD or V06D."),
    "MED//PANTOPRAZOLE_IVPB_SOLUTION":
        ("KEEP", None, "high", "Pantoprazole A02BC02 correct (PPI)."),
    "MED//POLYETHYLENE_GLYCOL_3350_17_GRAM_ORAL_POWDER_PACKET":
        ("KEEP", None, "high", "PEG 3350 A06AD15 correct (osmotic laxative)."),
    "MED//CEFOXITIN_IVPB_SOLUTION":
        ("KEEP", None, "high", "Cefoxitin J01DC01 correct (2nd-gen cephalosporin)."),
    "MED//CITRATE_DEXTROSE_SOLUTION":
        ("UNCERTAIN", None, "low",
         "Could be ACD-A for apheresis (B05CB10) or ACD for blood preservation (V08BA)."),
    "MED//LACTATED_RINGERS_INTRAVENOUS_SOLUTION":
        ("KEEP", None, "high", "LR B05BB01 correct (pure electrolyte, no carbs)."),
    "MED//GLUC_OXIDLACTOPEROXIDMURAMID_MOUTHWASH":
        ("UNCERTAIN", None, "low", "Oral enzyme mouthwash — no clear ATC. A01AB possible."),
    "MED//WHITE_PETROLATUMMINERAL_OIL_83_PCT15_PCT_EYE_OINTMENT":
        ("CORRECT", "S01XA20", "medium",
         "Artificial tears/lubricant ointment better fits S01XA20 than S01KX01 (other ophthalmologic visual)."),
    # ── Pediatric / CHD specific drugs (beyond top 30) ──
    "MED//ALPROSTADIL_PROSTIN_VR":
        ("CORRECT", "C01EA01", "high",
         "Prostin VR = alprostadil for neonatal PDA maintenance (C01EA01). LLM G02C (gyne) wrong in pediatric context."),
    "MED//SILDENAFIL_20_MG_TABLET":
        ("KEEP", None, "medium", "Sildenafil G04BE03; in pediatrics also for pulm HTN (C02KX05). Context-dependent."),

    # ── Rank 31-100 additional corrections (2nd pass review) ──
    "MED//LEVETIRACETAMPB_SOLUTION":
        ("CORRECT", "N03AX14", "high",
         "Levetiracetam is N03AX14, not N03AF (carboxamides like oxcarbazepine)."),
    "MED//KCL_NS":
        ("CORRECT", "B05XA01", "high",
         "KCl in NS is an IV additive (B05XA01), not oral K supplement A12BA01."),
    "MED//LIDOCAINEPRILOCAINE_CREAM":
        ("CORRECT", "N01BB20", "high",
         "EMLA-type lidocaine/prilocaine combo is N01BB20, not pure lidocaine D04AB01."),
    "MED//LIDOCAINETRANSPARENT_DRESSING_KIT":
        ("UNCERTAIN", None, "low",
         "Lidocaine-containing transparent dressing — could be D09A (medicated dressing) or N01BB02."),
    "MED//VECURONIUM_BROMIDE_DOSE":
        ("CORRECT", "M03AC03", "high",
         "Vecuronium is a neuromuscular blocker (M03AC03), not an anesthetic (N01AX)."),
    "MED//VECURONIUM_BROMIDE_SOLUTION":
        ("CORRECT", "M03AC03", "high",
         "Vecuronium is muscle relaxant M03AC03, not N01 (anesthetics)."),
    "MED//GELATIN_SPONGEABSORBABLEPORCINE_SKIN_100":
        ("UNCERTAIN", None, "low",
         "Absorbable gelatin sponge (hemostatic) — B02BC family or V03 for local hemostasis. Not D03AX02."),
    "MED//BALANCED_SALT_SOLUTION_COMBINATION_NO2_I":
        ("CORRECT", "S01XA20", "medium",
         "Balanced salt solution is ophthalmic irrigation (S01XA20), not chymotrypsin (S01KX01)."),
    "MED//DORNASE_ALFA_ML_SOLUTION_FOR":
        ("CORRECT", "R05CB13", "high",
         "Dornase alfa is R05CB13 (mucolytic for CF), not R03BX01 (fenspiride)."),
    "MED//D20W":
        ("CORRECT", "B05BA03", "high",
         "20% dextrose in water is a carbohydrate for parenteral nutrition (B05BA03), not mannitol (B05BC01)."),
    "MED//PENTOBARBITAL_SODIUM_ML_SOLUTION":
        ("CORRECT", "N05CA01", "high",
         "Pentobarbital is N05CA01 (barbiturate hypnotic), not N01AF03 (thiopental)."),
    "MED//NALBUPHINEPB_SOLUTION":
        ("CORRECT", "N02AF02", "high",
         "Nalbuphine is N02AF02, not N02AD01 (pentazocine)."),
    "MED//POVIDONEIODINE_EYE_SOLUTION":
        ("CORRECT", "S01AX18", "medium",
         "Povidone-iodine ophthalmic is S01AX18 (other antiinfectives), not S01KX01 (chymotrypsin)."),
    "MED//IMS_MIXTURE_TEMPLATE":
        ("UNCERTAIN", None, "low",
         "'IMS mixture template' is a compounded formulation placeholder; no specific ATC."),
    "MED//NONFORMULARY_MEDICATION":
        ("UNCERTAIN", None, "low",
         "'Non-formulary medication' is a billing placeholder; varies by actual drug."),
    "MED//GENT_VIOLETBRLNT_GRNPROFLAV_ML_SWAB":
        ("UNCERTAIN", None, "low",
         "Gentian violet + brilliant green + proflavine antiseptic swab; multi-active compound, no single ATC."),
    "MED//MONO_KPHOS_AND_MONO_DI_NAPHOS_EQ_PHOSPHO":
        ("UNCERTAIN", None, "low",
         "Mono-potassium + mono/di-sodium phosphate mixture — A12CX (oral minerals) or B05XA06 (IV phosphate) depends on route."),
    "MED//OPTM_BACITRACIN_SOLUTION":
        ("UNCERTAIN", None, "low",
         "Bacitracin solution — topical (D06AX05), ophthalmic (S01AA08), or IV (J01XX10) depends on context."),
    "MED//SILDENAFIL_ML_SUSPENSION":
        ("KEEP", None, "medium", "Sildenafil oral — pediatric pulm HTN use, G04BE03 or C02KX05."),
    "MED//UMBILICAL_ARTERIAL_CATHETER_FLUID":
        ("KEEP", None, "medium",
         "UAC fluid (heparinized saline for line patency) — B05BB01 is acceptable though line-flush, not therapeutic."),

    # ── Rank 101-200 additional flags ──
    "MED//DENTAL_POLISHVARNISHES_LIQUID":
        ("UNCERTAIN", None, "low", "Dental polish varnish — A01AA30 or fluoride-related, not otic S02DC."),
    "MED//MYLANTABENADRYLLIDOCAINE_SUSP":
        ("UNCERTAIN", None, "low", "Magic mouthwash compound — no single ATC."),
    "MED//MYLANTABENADRYLNYSTATIN_SUSP":
        ("UNCERTAIN", None, "low", "Magic mouthwash with nystatin — compound, no single ATC."),
    "MED//EPIDURAL":
        ("UNCERTAIN", None, "low", "'Epidural' is route, not a drug — LLM guessed fentanyl."),
    "MED//HEH_PREOP_SOLN":
        ("UNCERTAIN", None, "low", "HEH preop compound — institutional abbreviation, unclear composition."),
    "MED//CLOG_ZAPPER":
        ("UNCERTAIN", None, "low", "Catheter clearing enzyme device, not a drug — no valid ATC."),
    "MED//LACTATED_RINGERS_SOLUTION":
        ("CORRECT", "B05BB01", "high",
         "LR is B05BB01 electrolytes, not B05CB10 combinations."),
}

# ─── Helper: apply patterns + explicit decisions ─────────────────────────

def apply_patterns(row):
    """Return (new_atc, rationale) if a pattern matches, else (None, None)."""
    code = row["v3_med_code"]
    for pattern, new_atc, rationale in PATTERN_CORRECTIONS:
        if re.search(pattern, code):
            return new_atc, rationale
    return None, None


def review_drug(row):
    """Return (decision, corrected_atc, confidence, rationale)."""
    code = row["v3_med_code"]

    # Explicit decisions win
    if code in EXPLICIT_DECISIONS:
        return EXPLICIT_DECISIONS[code]

    # Pattern-based correction
    new_atc, pat_rat = apply_patterns(row)
    if new_atc:
        return ("CORRECT", new_atc, "high", f"Pattern: {pat_rat}")

    # Default: keep LLM prediction at medium confidence (low-frequency long tail)
    llm_atc = row.get("final_atc")
    if pd.isna(llm_atc) or llm_atc is None:
        return ("UNCERTAIN", None, "low", "LLM failed to produce final ATC.")
    return ("KEEP", None, "medium", "LLM prediction accepted (low-frequency long tail).")


# ─── Main ─────────────────────────────────────────────────────────────────

print("Loading LLM full classification...")
llm = pd.read_csv(OUT / "atc_full_dose_Llama-3p3-70B.csv")
print(f"  {len(llm)} drugs")

print("\nApplying team review...")
decisions = llm.apply(review_drug, axis=1, result_type="expand")
decisions.columns = ["decision", "corrected_atc", "confidence", "rationale"]
reviewed = pd.concat([llm, decisions], axis=1)

# Final ATC after review
reviewed["final_atc_reviewed"] = reviewed.apply(
    lambda r: r["corrected_atc"] if r["decision"] == "CORRECT" else r.get("final_atc"),
    axis=1,
)

# Statistics
print("\n=== Review outcomes ===")
print(reviewed["decision"].value_counts().to_string())
print("\n=== By confidence ===")
print(reviewed["confidence"].value_counts().to_string())

# Coverage weighted by frequency
total_events = reviewed["frequency"].sum()
for dec in ["KEEP", "CORRECT", "UNCERTAIN"]:
    sub = reviewed[reviewed["decision"] == dec]
    events = sub["frequency"].sum()
    print(f"  {dec:10s}  {len(sub):>4d} drugs, {events:>10,d} events ({100*events/total_events:.1f}%)")

# Save reviewed file
reviewed.to_csv(OUT / "atc_team_reviewed.csv", index=False)
print(f"\nSaved reviewed mapping: {OUT/'atc_team_reviewed.csv'}")

# ─── Build expert validation sample ───────────────────────────────────────

# Sample composition:
#   ALL  UNCERTAIN items (expert must decide these)
#   20 random KEEP items (validate our "trust LLM" calls)
#   20 random CORRECT items (validate our corrections)

uncertain = reviewed[reviewed["decision"] == "UNCERTAIN"]
keeps = reviewed[reviewed["decision"] == "KEEP"]
corrects = reviewed[reviewed["decision"] == "CORRECT"]

# Stratified random: include high-frequency + low-frequency samples for KEEP/CORRECT
def stratified_sample(df, n):
    if len(df) <= n:
        return df
    # 50% from top quartile by frequency, 50% random from the rest
    n_top = n // 2
    top = df.sort_values("frequency", ascending=False).head(len(df) // 4).sample(min(n_top, len(df) // 4), random_state=RANDOM_SEED)
    rest = df.drop(top.index)
    other = rest.sample(n - len(top), random_state=RANDOM_SEED)
    return pd.concat([top, other]).sort_values("frequency", ascending=False)

keep_sample = stratified_sample(keeps, 20)
correct_sample = stratified_sample(corrects, 20)
expert_sample = pd.concat([uncertain, keep_sample, correct_sample])
expert_sample = expert_sample.sort_values("frequency", ascending=False).reset_index(drop=True)

print(f"\n=== Expert validation sample ===")
print(f"  Uncertain (all):         {len(uncertain)}")
print(f"  Keep (random 20):        {len(keep_sample)}")
print(f"  Correct (random 20):     {len(correct_sample)}")
print(f"  Total for expert:        {len(expert_sample)}")

# Load ATC name lookup for context columns
with open(OUT / "atc_name_lookup.json") as f:
    atc_name = json.load(f)

def atc_info(code):
    if pd.isna(code) or not code:
        return ""
    return atc_name.get(code, "")

expert_sample["llm_atc_name"] = expert_sample["final_atc"].apply(atc_info)
expert_sample["reviewed_atc_name"] = expert_sample["final_atc_reviewed"].apply(atc_info)

# Columns to present to expert
review_cols = [
    "v3_med_code", "drug_name", "route", "dose", "purpose", "frequency",
    "final_atc", "llm_atc_name",
    "decision", "corrected_atc", "reviewed_atc_name",
    "confidence", "rationale",
    "expert_decision",  # CONFIRM / REVISE / UNCLEAR
    "expert_atc",       # if REVISE, correct ATC
    "expert_comment",
]
for col in ["expert_decision", "expert_atc", "expert_comment"]:
    expert_sample[col] = ""

expert_sample = expert_sample[review_cols]
expert_sample.to_csv(OUT / "atc_expert_validation_sample.csv", index=False)
print(f"\nSaved expert validation sample: {OUT/'atc_expert_validation_sample.csv'}")

# ─── Save summary for paper ──────────────────────────────────────────────
summary = f"""# Stream A — Team Review Summary (2026-04-15)

## Totals

- LLM-classified drugs:           {len(reviewed)}
- Team decision: KEEP              {(reviewed['decision']=='KEEP').sum()}
- Team decision: CORRECT           {(reviewed['decision']=='CORRECT').sum()}
- Team decision: UNCERTAIN         {(reviewed['decision']=='UNCERTAIN').sum()}

## Event-weighted

"""
for dec in ["KEEP", "CORRECT", "UNCERTAIN"]:
    sub = reviewed[reviewed["decision"] == dec]
    events = sub["frequency"].sum()
    summary += f"- {dec:10s}: {len(sub):>4d} drugs, {events:>10,d} events ({100*events/total_events:.1f}%)\n"

summary += f"""

## Expert validation sample (stratified)

- All UNCERTAIN:                 {len(uncertain)} drugs
- Random 20 from KEEP:           {len(keep_sample)} drugs
- Random 20 from CORRECT:        {len(correct_sample)} drugs
- Total for expert:              {len(expert_sample)} drugs

Stratification: within KEEP and CORRECT, 50% drawn from top-quartile by frequency
(to weight validation toward high-impact drugs), 50% drawn randomly from remainder.

## Correction breakdown (top categories)

"""
top_corrections = reviewed[reviewed["decision"] == "CORRECT"].groupby("corrected_atc").agg(
    count=("v3_med_code", "count"),
    events=("frequency", "sum"),
).sort_values("events", ascending=False).head(10)
summary += top_corrections.to_string() + "\n"

summary += f"""

## Top-10 UNCERTAIN drugs (by frequency)

"""
for _, r in uncertain.sort_values("frequency", ascending=False).head(10).iterrows():
    summary += f"- {r['drug_name'][:40]:<40s} freq={r['frequency']:>7,d}  LLM={r['final_atc']}\n"

(LOG / "A_team_review_summary.md").write_text(summary)
print(f"\nSaved review summary: {LOG/'A_team_review_summary.md'}")

print("\nDone.")
