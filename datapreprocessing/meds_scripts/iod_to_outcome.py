"""
iod_to_outcome.py
-----------------
Reproduce the IoD (Intraoperative Deterioration) label dataframe from IoD_label.ipynb
and save as:

  /path/to/CHD_MEDS/outcome/iod_labels.csv      — full encounter-level label table
  /path/to/CHD_MEDS/outcome/iod_task.parquet    — MEDS task format

Label definitions (binary, per encounter):
  IoD1  CPR event during OR window (AN_Events)
  IoD2  Quick Note keyword match during OR window (AN_Events)
  IoD3  IV vasoactive bolus (epinephrine, phenylephrine>2, ephedrine>2,
         vasopressin, adenosine, atropine-age>3mo) during OR window
  IoD4  Vasoactive infusion (dopamine, epinephrine, milrinone, norepinephrine,
         phenylephrine, vasopressin) started OR rate-doubled during AN Start→Out OR
  IoD5  Arterial line placed intraprocedurally after surgical incision
  IoD   OR of IoD1–IoD5

MEDS task format (iod_task.parquet):
  patient_id      : str       — C MRN
  prediction_time : datetime  — AN Start (time at which prediction is made)
  boolean_value   : bool      — Intraoperative_Deterioration (IoD1|...|IoD5)

  Sub-label columns also included for analysis:
  IoD1 … IoD5     : bool

Usage:
    python iod_to_outcome.py [--output_dir DIR]
"""

import re
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── file paths ─────────────────────────────────────────────────────────────────
FILE_AN_EVENTS   = (
    "/path/to/CHOA_RAW_TABLES/"
    "CHOA_DATA_Tables_CHD/DR15201_AN_Events.rpt"
)
FILE_AN_PATIENTS = (
    "/path/to/CHOA_RAW_TABLES/"
    "CHOA_DATA_Tables_CHD/DR15201_AN_Patients.rpt"
)
FILE_MEDICATIONS = (
    "/path/to/CHOA_RAW/DR15201_Medication_Administration/"
    "DR15201_Medication_Administration/DR15201_Medication_Administration.rpt"
)
FILE_LDAS = (
    "/path/to/CHOA_RAW_TABLES/"
    "CHOA_DATA_Tables_CHD/DR15201_LDAs.rpt"
)
FILE_SURGICAL_HISTORY = "/path/to/CHOA_RAW/DR15201_Surgical_History_V.csv"

DEFAULT_OUTPUT_DIR = Path("/path/to/CHD_MEDS/outcome")


# ── IoD2 keyword pattern ────────────────────────────────────────────────────────
_IOD2_KEYWORDS = [
    "chest compression", "CPR", "code", "PALS", "arrest",
    "deteriorated", "deterioration", "hypotension",
    "epinephrine", "epi", "dopamine", "phenylephrine",
    "norepi", "norepinephrine", "emergently", "shock",
    "defib", "defibrillation", "ECMO", "cardioversion", "DCCV",
    "pulseless", "no pulse", "help", "anesthesia now", "overhead",
    "echo", "cardiac output", "arrhythmia", "SVT",
    "Vtach", "v tach", "v-tach", "v-fib", "vfib", "v fib",
    "ventricular tachycardia", "ventricular fibrillation",
    "brady", "fibrillation",
]
_IOD2_PATTERN = re.compile(
    "|".join(re.escape(k) for k in _IOD2_KEYWORDS), re.IGNORECASE
)

# ── IoD2 refined exclusion patterns (per-keyword false positive filters) ─────────
# Applied AFTER keyword match to remove high-confidence false positives.
# Refined iteratively via full-dataset audit (2026-03-29):
#   - Pass 1 (initial): per-keyword exclusion for epi/echo/help/brady/defib/
#     hypotension/phenyl-ophthalmic/intranasal-epi/ICD-test
#   - Pass 2 (audit-refined): expanded echo logistical patterns, added circ arrest
#     filter, added brady negation filter, added code-red filter, expanded ICD test,
#     added override for ICD failure, expanded help emergency patterns, expanded
#     epi IV override, added clinical-severity override for help-removed notes.
# Design principle: remove HIGH-CONFIDENCE FPs only. When ambiguous, keep (prefer FN
# over FP). All additions documented with evidence from 6,535-note audit.

# epi/epinephrine: local anesthetic injections, epidural, Epic EMR, concentrations
_IOD2_EPI_EXCLUDE = re.compile(
    r"epidural"                                                   # epidural procedure
    r"|epiglott"                                                  # epiglottis anatomy
    r"|epicardial"                                                # epicardial pacemaker leads (cardiac surgery routine)
    r"|\bepic\b"                                                  # Epic EMR system
    r"|cefepime"                                                  # antibiotic (cefepime)
    r"|(?:lido(?:caine)?|marcaine|bupiv(?:acaine)?|bupi\b|ropivacaine)"  # local anesthetic agents (incl. "bupi" abbrev)
    r"|\bepi\s*1\s*:[0-9]|1\s*:\s*(?:100|200|400)[,kK]?(?:,?000)?\s*(?:epi|epinephrine)"  # concentration ratios
    r"|(?:racemic|nebulized)\s+epi"                               # inhaled/nebulized routes
    r"|\bcaudal\b"                                                # caudal block (neuraxial, not IV)
    r"|pledg[ae]?t|plegett"                                       # epi pledget/plegette = topical nasal hemostasis
    r"|(?:surgeon|dentist|physician|interventional|\bENT\b)\s+inject"  # provider-injected local (incl. ENT)
    r"|inject(?:ed|ing|s)?\s+(?:local|by\s+(?:surgeon|dentist|interventional|\bENT\b))"  # injection by provider
    r"|inject(?:ed|ing|s)?\s+(?:into\s+)?(?:nose|nasal|nare)"    # nasal injection = topical
    r"|local\s+(?:\w+\s+){0,3}(?:infiltr|inject|with\s+epi)"     # local infiltration (words in between ok)
    r"|\bLA\s+(?:w[/\\]|with)\s*(?:epi|epinephrine)"             # "LA w/Epi" = local anesthetic with epi by surgeon
    r"|(?:ophthalm|eye(?:s|\s)|drop(?:s|\s)|conjunctiv)"         # ophthalmic route
    r"|inject(?:ed|ing|s)?\s+(?:\w+\s+){0,3}(?:by\s+)?(?:surgeon|dentist|interventional|\bENT\b|dr\.?\s+\w+)",  # injected by provider
    re.IGNORECASE,
)

# echo: routine intraoperative TEE markers (not emergency findings)
# Two tiers:
#   Tier 1 (_IOD2_ECHO_ROUTINE): original standalone start/stop/probe markers
#   Tier 2 (_IOD2_ECHO_LOGISTICAL): broader logistical/scheduling notes
#     (audit showed 223/231 echo-kept notes had no emergency language)
# Clinical echo findings ("echo shows LV distended", "chest reopened per echo") are
# NOT matched by either pattern and are correctly kept.
# Safety: both tiers are overridden by _IOD2_ALWAYS_KEEP if the note also contains
# emergency language (CPR, compressions, pulseless, code cart, etc.).
_IOD2_ECHO_ROUTINE = re.compile(
    r"^\s*echo\s*(?:start|started|end|ended|stop|stopped|begin|began|complete|done|off|on|probe|removed?)\s*[.!]?\s*$"
    r"|^\s*(?:start|stop|end|begin)\s+(?:of\s+)?echo\s*[.!]?\s*$"
    r"|echo\s+probe\s+(?:placed|removed|inserted|pulled|in|out)"
    r"|(?:tee|echo)\s+(?:monitoring|monitor(?:ing)?|surveillance)\b"
    r"|(?:begin|start|end|stop|finish)\s+(?:transthoracic|epicardial|intraoperative|tee)\s+echo"
    r"|echo\s+procedure\s+(?:begin|start|end|stop|complete)",
    re.IGNORECASE,
)

# Audit-derived logistical echo patterns (no clinical finding = no IoD signal)
_IOD2_ECHO_LOGISTICAL = re.compile(
    # "waiting for echo" / "wait on echo" — scheduling note, not clinical
    r"(?:waiting|wait(?:ing)?)\s+for\s+echo"
    # "echo team present to do echo while under GA" / "echo team present" standalone
    r"|echo\s+team\s+(?:present|arrived?|here|coming)\s+(?:to|for|at)?\s*(?:do|perform|obtain)?"
    # "echo tech at bedside performing ordered echo by cards for routine f/u"
    r"|echo\s+tech(?:nician)?\s+(?:at\s+bedside\s+)?performing\s+(?:ordered|routine|scheduled|requested)"
    # "doing echo in lab" / "echo in lab/radiology/cath lab/MRI/radiation room"
    r"|doing\s+echo\s+in\s+(?:lab|OR|room|cath)"
    r"|echo\s+in\s+(?:lab|radiology|PACU|recovery\s+room|radiation|cath|MRI)\b"
    # "begin EKG and echo, in conjunction with ABR" — routine workup bundle
    r"|begin\s+(?:EKG|ECG|ekg)\s+and\s+echo\b"
    r"|echo\s+in\s+conjunction\s+with\b"
    # "new echo machine and probe" — equipment note
    r"|new\s+echo\s+(?:machine|probe|equipment)"
    # "echo ordered by cards" / "echo for routine f/u"
    r"|echo\s+(?:for\s+routine|ordered\s+by\s+cards?|per\s+cards?|study\s+start)"
    # Standalone "echo at bedside" note (no other clinical content)
    r"|^\s*(?:echo|TTE|TEE)\s+at\s+bedside\s*[.,!]?\s*$"
    # Sedation purely to facilitate echo (not clinical deterioration)
    r"|precedex\s+to\s+facilitate\s+echo"
    # "dental finished - echo start" / "cath complete - echo start prior to cath"
    r"|(?:dental|cath|procedure|surgery)\s+(?:complete|done|finished)[^.]*echo\s+start"
    r"|echo\s+start\s+prior\s+to\s+(?:cath|procedure)",
    re.IGNORECASE,
)

# help: routine task-help vs emergency calls for help
# Expanded (audit 2026-03-29): added "call for additional help" pattern — catches
# notes like "call for additional help and administered first dose epinephrine IV"
# that were incorrectly removed as non-emergency (n=8 clinical FNs found in audit).
_IOD2_HELP_EMERGENCY = re.compile(
    r"(?:called?|call(?:ing)?|paged?|overhead|stat)\s+(?:for\s+)?help"
    r"|help\s+(?:called|arrives?|paged|needed\s+stat|to\s+OR)"
    r"|overhead\s+page.*help"
    r"|anesthesia\s+(?:help|now\b)"
    r"|call(?:ed|ing)?\s+for\s+additional\s+(?:anesthesia\s+)?help"  # "call for additional help"
    r"|additional\s+help\s+(?:call(?:ed)?|request(?:ed)?|arrived?|needed\s+stat)",
    re.IGNORECASE,
)

# brady: person's name "Dr. Brady" is not bradycardia
_IOD2_BRADY_EXCLUDE = re.compile(r"\bdr\.?\s+brady\b", re.IGNORECASE)

# brady: explicit negation of bradycardia = note is documenting ABSENCE of the finding.
# "no bradycardia", "never bradycardic" = clinical all-clear, not IoD.
# Applied only when note does not also contain clinical severity override (see below).
# Evidence: audit identified notes with "No bradycardia noted", "Never bradycardic",
# "no bradycardia throughout" that were FPs.
_IOD2_BRADY_NEGATION = re.compile(
    r"\bno\s+(?:significant\s+|further\s+|notable\s+)?bradycardia\b"
    r"|never\s+bradycardic\b"
    r"|without\s+(?:significant\s+)?bradycardia\b"
    r"|bradycardia\s+(?:absent|denied|none\b|not\s+(?:observed|noted|present|seen))",
    re.IGNORECASE,
)

# defib: equipment testing is not an emergency defibrillation
_IOD2_DEFIB_EXCLUDE = re.compile(r"test(?:ing|ed)?\s+defib|defib.*test", re.IGNORECASE)

# hypotension: deliberate/planned/surgical hypotension OR BP measurement artifact is not IoD
_IOD2_HYPOTENSION_EXCLUDE = re.compile(
    r"(?:deliberate|intentional|planned|controlled|induced|surgical)\s+hypotension"
    r"|hypotension\s+(?:for|to\s+reduce|requested)"
    r"|(?:surgeon|surgical\s+team|surgeon\s+request(?:ed|s)?)\s+(?:moderate|mild|permissive|controlled)?\s*hypotension"
    r"|(?:request(?:ed|s)?|ask(?:ed|s)?|want(?:ed|s)?|prefer(?:red|s)?)\s+(?:moderate|mild|permissive|controlled)?\s*hypotension"
    r"|hypotension\s+(?:due\s+to\s+retraction|from\s+retraction|secondary\s+to\s+retraction|with\s+retraction)"
    r"|hypotension\s+(?:to\s+limit|to\s+reduce|to\s+minimize)\s+blood\s+loss"
    r"|(?:NIBP|BP)\s+cuff\s+(?:not\s+reading|error|difficult|inaccurate|repositioned)"
    r"|(?:difficulty|having\s+difficulty)\s+with\s+(?:NIBP|BP)\s+cuff"
    r"|hypotension\s+(?:associated\s+with|due\s+to)\s+(?:tourniquet|cuff)"
    r"|(?:likely|probably)\s+(?:an?\s+)?(?:error|artifact)\s+(?:in\s+)?(?:NIBP|BP|blood\s+pressure)",
    re.IGNORECASE,
)

# phenylephrine ophthalmic: eye drops/swabs/solution by surgeon/ophthalmologist = topical, not IV
_IOD2_PHENYL_OPHTHALMIC_EXCLUDE = re.compile(
    r"(?:eye\s*drops?|ophthalmic|eyebrow|conjunctiv|ocular|eye\s*brow|instill(?:ed|s)?)"
    r"|(?:eye|eyes|both\s+eyes|right\s+eye|left\s+eye)\s+(?:by|per)\s+(?:surgeon|opthalmologist|ophthalmologist|optho|ophthal)"
    r"|phenylephrine\s+(?:eye\s*drops?|ophthalmic|drops?|swabs?|solution)\s+(?:applied|placed|instilled|given|started|administ)"
    r"|(?:applied|placed|instilled)\s+(?:to|in|into)\s+(?:eye|eyes|both\s+eyes|eyebrow)"
    r"|(?:phenylephrine|neo-synephrine)\s+(?:cotton\s+swabs?|pledgets?|gauze)\s+(?:applied|placed)",
    re.IGNORECASE,
)

# intranasal/topical epi: nasal epi, epi pledgets = topical hemostasis, not IV
_IOD2_NASAL_EPI_EXCLUDE = re.compile(
    r"(?:nasal|intranasally|intra-?nasal|nare|nostril)\s+epi(?:nephrine)?"
    r"|epi(?:nephrine)?\s+(?:intranasally|nasal(?:ly)?|via\s+nose|to\s+nose|intra-?nasal)"
    r"|epi(?:nephrine)?\s+(?:pledgets?|plegetts?|cotton\s+pledgets?|gauze\s+pledgets?)"
    r"|pledgets?\s+(?:placed|soaked|dipped|with)\s+(?:\w+\s+){0,3}epi(?:nephrine)?"
    r"|(?:nasal|intranasal)\s+(?:decongest|vasoconstric)",
    re.IGNORECASE,
)

# ICD/defibrillator functional testing (planned procedure step, not emergency)
# Expanded (audit 2026-03-29): added "fibrillated intentionally to test ICD" pattern —
# catches notes like "Fibrillated intentionally to test ICD. ICD successfully
# defibrillated..." which were FPs. Previous pattern required "ICD tested" exact phrase.
_IOD2_ICD_TEST_EXCLUDE = re.compile(
    r"ICD\s+(?:tested?|testing|check(?:ed)?)\s*(?:x\s*\d+)?"
    r"|(?:internal|implant(?:able)?|device)\s+(?:shock|defib|defibrillat)"
    r"|induced?\s+(?:VT|V-?fib|ventricular\s+(?:fibrillation|tachycardia))\s+(?:for\s+)?(?:test|check|ICD|defib)"
    r"|(?:test|check(?:ing)?)\s+(?:ICD|defibrillator|device)\s+(?:function|threshold|sensing|output)"
    r"|(?:DFT|defibrillation\s+threshold)\s+(?:test|check)"
    r"|fibrillat(?:ed|ing)\s+intentionally\s+(?:to\s+test|for\s+(?:ICD|DFT|test))"  # "fibrillated intentionally to test ICD"
    r"|(?:intentionally|deliberately)\s+fibrillat(?:ed|ing)",                         # "intentionally fibrillated"
    re.IGNORECASE,
)

# Circulatory arrest during cardiopulmonary bypass (DHCA) = planned cardiac surgery
# technique, NOT intraoperative deterioration. "Circ arrest" is used exclusively in
# cardiac surgery to denote deep hypothermic circulatory arrest (DHCA) — a controlled
# step where the heart is deliberately stopped. Distinct from true "cardiac arrest".
# Evidence: audit found multiple "Circ arrest" / "circulatory arrest" notes that were
# FPs triggered by the "arrest" keyword (e.g., "CPB pump off, Circ arrest",
# "circ arrest 12min", "Cardioplegia given during brief period of circ arrest").
# Safety: _IOD2_ALWAYS_KEEP (CPR, compressions, pulseless) overrides this filter
# for any note that also documents a real emergency alongside a circ arrest note.
_IOD2_CIRC_ARREST_EXCLUDE = re.compile(
    r"\bcirc(?:ulatory)?\s+arrest\b"
    r"|(?:deep\s+hypothermic|moderate\s+hypothermic|hypothermic)\s+circulatory\s+arrest"
    r"|DHCA\b"                                                      # deep hypothermic circ arrest abbrev
    r"|(?:on|off|start|stop)\s+(?:full\s+)?circ\b"                 # "on circ" / "off circ" bypass shorthand
    r"|cardioplegia.*\barrest\b"                                    # "cardioplegia given ... arrest"
    r"|\barrest\b.*\bcardioplegia\b",                               # "arrest ... cardioplegia"
    re.IGNORECASE,
)

# Facility/hospital code types that are NOT cardiac arrest (fire, security, etc.)
# Evidence: audit found "code red called, therefore elevators would not run. Vss entire
# time." — the "code" keyword triggered IoD2 but the event was a building fire drill.
_IOD2_CODE_FACILITY_EXCLUDE = re.compile(
    r"\bcode\s+red\b"           # fire code
    r"|\bcode\s+orange\b"       # hazmat
    r"|\bcode\s+silver\b"       # active shooter
    r"|\bcode\s+white\b",       # some hospitals use for behavior/child
    re.IGNORECASE,
)

# ── Override patterns: always KEEP regardless of exclusion filters ────────────
# Applied last — any note matching these is restored even if an exclusion fired.

# GLOBAL override: unambiguous intraoperative emergency language.
# "chest compressions / code cart / all anesthesia alert / called stat" are
# never present in routine notes; they always signal a true IoD event.
_IOD2_ALWAYS_KEEP = re.compile(
    r"chest\s+compressions|compressions\s+(?:done|started|performed|given)"
    r"|code\s+cart"
    r"|all\s+anesthesia\s+alert"
    r"|\bcalled\s+stat\b"
    r"|pulseless"
    r"|\bcpr\b",
    re.IGNORECASE,
)

# EPI override: systemic IV epinephrine indicators that distinguish emergency
# use from local anesthetic / field application.
# Key discriminators (from 500-note analysis):
#   local epi  → volume in cc/mL + concentration (%) or ratio (1:200K)
#   systemic IV → mcg dosing, "epi x N", "epi drip/gtt", "bolus of epi"
# Expanded (audit 2026-03-29): added "IV epi"/"IV epinephrine" pattern to catch
# notes like "low dose IV epi given" that were missed (only "epinephrine IV" was covered).
_IOD2_EPI_OVERRIDE = re.compile(
    r"\d+\s*mcg\s*(?:of\s+)?(?:epi|epinephrine)"       # mcg dosing: "5mcg epi"
    r"|\bepi(?:nephrine)?\b\s*\d+\s*mcg"                # "epi 5 mcg"
    r"|\bepi\s+[xX×]\s*\d+\s*(?!(?:ml|cc)\b)"          # "epi x 2" (NOT "epi x 2ml" volume)
    r"|\bepi(?:nephrine)?\s+(?:drip|gtt)\b"             # "epi drip", "epinephrine gtt"
    r"|epi(?:nephrine)?\s+(?:drip|gtt)\s+(?:ordered|started|initiated|increased|running)"
    r"|\bbolus\s+of\s+epi(?:nephrine)?"                 # "bolus of epi"
    r"|\bepi(?:nephrine)?\s+(?:infusion|drip)\s+(?:order|start|increas|titrat)"
    r"|epinephrine\s+(?:IV\b|intravenous)"               # "epinephrine IV"
    r"|\bIV\s+epi(?:nephrine)?\b",                       # "IV epi", "IV epinephrine" (audit addition)
    re.IGNORECASE,
)

# ICD test FAILURE override: "without successful conversion", "multiple defibrillations"
# indicate a real emergency despite ICD test language.
# Evidence: audit found "Internal defibrillation for V. Fibrillation, multiple
# defibrillations at 50J without successful conversion of rhythm" removed by ICD test
# filter — this is a true IoD (failed rescue defibrillation).
_IOD2_ICD_FAIL_OVERRIDE = re.compile(
    r"without\s+successful\s+conversion"
    r"|multiple\s+defibrillat(?:ion|s)\b.*(?:without|fail|unsuccess)"
    r"|unable\s+to\s+(?:convert|defibrillat)"
    r"|fail(?:ed|ure)\s+(?:to\s+)?(?:convert|defibrillat)",
    re.IGNORECASE,
)

# Clinical severity override for help-removed notes: notes with explicit hemodynamic
# collapse language that were removed by the help-not-emergency filter.
# Evidence: "blood pressure dropped precipitously along with HR (to 40's) after
# suctioning... Anesthetist available to help with Epi" — "available to help" is not
# an emergency call, but the hemodynamic event IS IoD.
_IOD2_HELP_CLINICAL_OVERRIDE = re.compile(
    r"blood\s+pressure\s+(?:dropped|fell?|drop(?:ped)?|decreased)\s+(?:precipitously|rapidly|suddenly|significantly|dramatically)"
    r"|(?:HR|heart\s+rate)\s+(?:to|down\s+to)\s+(?:[123][0-9]|4[0-5])\b"
    r"|MAP[s]?\s+(?:in\s+(?:the\s+)?)?(?:[123][0-9]|4[0-5])\b"
    r"|desaturat(?:ion|ing|ed)\s+to\s+(?:[3-6][0-9]|[7][0-4])\b"
    r"|\bpulseless\b"
    r"|cardiac\s+arrest\b",
    re.IGNORECASE,
)


# ── helpers ────────────────────────────────────────────────────────────────────
def read_rpt(path, **kwargs):
    return pd.read_csv(
        path, delimiter="|", encoding="utf-8-sig",
        encoding_errors="replace", on_bad_lines="skip",
        low_memory=False, **kwargs
    )


def in_or_window(df, time_col, an_patients, left="In OR", right="Out OR"):
    """
    Merge df with an_patients on C MRN, filter to rows where time_col
    is within [left, right].

    AN_Patients' Encounter CSN is always included as '_enc_csn' so callers
    can identify which encounter matched — regardless of whether df also has
    its own (different) Encounter CSN (AN_Events and AN_Patients use different
    CSN spaces, so AN_Patients' CSN is the authoritative per-encounter key).
    """
    an_cols = an_patients[["C MRN", "Encounter CSN", left, right]].rename(
        columns={"Encounter CSN": "_enc_csn"}
    )
    merged = pd.merge(df, an_cols, on="C MRN").drop_duplicates()
    merged = merged[merged[time_col].notna() &
                    (merged[left].notna() | merged[right].notna())]
    merged[time_col] = pd.to_datetime(merged[time_col], errors="coerce")
    merged[left]     = pd.to_datetime(merged[left],  errors="coerce")
    merged[right]    = pd.to_datetime(merged[right], errors="coerce")
    return merged[(merged[time_col] >= merged[left]) &
                  (merged[time_col] <= merged[right])]


# ── label functions ─────────────────────────────────────────────────────────────
def compute_iod1(an_events, an_patients):
    """CPR event during OR window. Returns set of positive Encounter CSNs (AN_Patients)."""
    cpr = an_events[an_events["Event"] == "CPR"]
    intra = in_or_window(cpr, "Recorded Time", an_patients)
    return set(intra["_enc_csn"].unique())


def compute_iod2(an_events, an_patients):
    """
    Quick Note keyword match during OR window with refined false-positive filtering.

    Three-pass approach:
      Pass 1 — broad keyword match (_IOD2_PATTERN)
      Pass 2 — per-keyword exclusion filters (remove high-confidence FPs):
        - epi/epinephrine: local anesthetic, epidural, Epic EMR, concentration ratios,
                           nasal/topical application
        - echo: routine TEE start/stop markers + logistical scheduling notes
        - help: keep only emergency calls; drop routine task-help
        - brady: exclude Dr. Brady (name) + explicit negation ("no bradycardia")
        - defib: equipment testing
        - hypotension: planned/deliberate hypotension, BP cuff artifact
        - phenylephrine: ophthalmic (topical) application
        - arrest: circulatory arrest during planned CPB (DHCA) — not true cardiac arrest
        - ICD: planned device testing (including "intentionally fibrillated")
        - code: facility codes (code red = fire, not cardiac)
      Pass 3 — overrides (restore notes incorrectly excluded):
        - ALWAYS_KEEP: chest compressions, code cart, CPR, pulseless
        - EPI_OVERRIDE: systemic IV epi markers (mcg dosing, drip, bolus)
        - ICD_FAIL_OVERRIDE: ICD test filter fired but conversion failed = real emergency
        - HELP_CLINICAL_OVERRIDE: help filter fired but note documents hemodynamic collapse

    Returns set of positive Encounter CSNs (AN_Patients).
    """
    qn = an_events[an_events["Event"].fillna("").str.strip() == "Quick Note"].copy()
    txt = qn["Event Comment"].fillna("")

    # Pass 1: broad keyword match
    matched = qn[txt.str.contains(_IOD2_PATTERN, regex=True)].copy()
    t = matched["Event Comment"].fillna("")

    # Pass 2: per-keyword false positive removal
    keep = pd.Series(True, index=matched.index)

    # epi/epinephrine rows: catch any word starting with "epi" — covers "epi", "epinephrine",
    # "epidural", "Epic", "epiglottis" etc. so exclusion patterns can fire on all of them.
    epi_rows = t.str.contains(r"\bepi", case=False, regex=True)
    keep[epi_rows & t.str.contains(_IOD2_EPI_EXCLUDE, regex=True)] = False

    # echo rows: remove routine TEE monitoring markers (tier 1: standalone start/stop)
    echo_rows = t.str.contains(r"\becho\b", case=False, regex=True)
    keep[echo_rows & t.str.contains(_IOD2_ECHO_ROUTINE, regex=True)] = False

    # echo rows: remove logistical echo notes (tier 2: scheduling/equipment, no clinical finding)
    keep[echo_rows & t.str.contains(_IOD2_ECHO_LOGISTICAL, regex=True)] = False

    # help rows: keep only emergency-call language; drop routine task-help
    help_rows = t.str.contains(r"\bhelp\b", case=False, regex=True)
    help_excluded = help_rows & ~t.str.contains(_IOD2_HELP_EMERGENCY, regex=True)
    keep[help_excluded] = False

    # brady rows: (a) remove person's name "Dr. Brady" when no clinical brady language
    brady_rows = t.str.contains(r"\bbrady\b", case=False, regex=True)
    is_dr_brady        = brady_rows & t.str.contains(_IOD2_BRADY_EXCLUDE, regex=True)
    has_clinical_brady = t.str.contains(r"bradycardia|bradycardic", case=False, regex=True)
    keep[is_dr_brady & ~has_clinical_brady] = False

    # brady rows: (b) remove explicit negations ("no bradycardia", "never bradycardic")
    # Conservative guard: only remove when the note lacks OTHER respiratory/hemodynamic
    # events that would independently justify IoD2 classification.
    # Technical note: _IOD2_BRADY_NEGATION is matched on ALL text (not just \bbrady\b)
    # because "bradycardia" contains "brady" without a trailing word boundary.
    has_other_iod_signal = t.str.contains(
        r"epi(?:nephrine)?\s+(?:given|IV|drip|bolus|gtt|\d+\s*mcg)"
        r"|vasopressor|compressions|CPR|pulseless|code\s+(?:blue|cart)"
        r"|laryngospasm|bronchospasm"
        r"|desaturat(?:ion|ed|ing)"
        r"|arrest\b|emergent"
        r"|\bbradycardia\b"                 # note has actual brady event (resolved notes still had it)
        r"|unable\s+to\s+ventilate"         # critical airway event
        r"|atropine|glycopyrrolate"         # pharmacologic treatment for bradycardia
        r"|succinylcholine|\bsux\b",        # emergency neuromuscular blockade
        case=False, regex=True
    )
    brady_negated = t.str.contains(_IOD2_BRADY_NEGATION, regex=True)
    keep[brady_negated & ~has_other_iod_signal] = False

    # defib rows: remove equipment testing
    defib_rows = t.str.contains(r"\bdefib", case=False, regex=True)
    keep[defib_rows & t.str.contains(_IOD2_DEFIB_EXCLUDE, regex=True)] = False

    # hypotension rows: remove deliberate/planned hypotension
    hypo_rows = t.str.contains(r"hypotension", case=False, regex=True)
    keep[hypo_rows & t.str.contains(_IOD2_HYPOTENSION_EXCLUDE, regex=True)] = False

    # phenylephrine ophthalmic rows: remove eye drops/swabs (not IV)
    phenyl_rows = t.str.contains(r"phenylephrine", case=False, regex=True)
    keep[phenyl_rows & t.str.contains(_IOD2_PHENYL_OPHTHALMIC_EXCLUDE, regex=True)] = False

    # intranasal/topical epi rows: remove nasal epi, pledgets (not IV)
    nasal_epi_rows = t.str.contains(r"\bepi", case=False, regex=True)
    keep[nasal_epi_rows & t.str.contains(_IOD2_NASAL_EPI_EXCLUDE, regex=True)] = False

    # ICD/defibrillator functional testing rows: not emergency (includes planned DHCA)
    icd_rows = t.str.contains(
        r"\bICD\b|defibrillat|induced?\s+(?:VT|V-?fib)|intentionally\s+fibrillat",
        case=False, regex=True
    )
    icd_excluded = icd_rows & t.str.contains(_IOD2_ICD_TEST_EXCLUDE, regex=True)
    keep[icd_excluded] = False

    # Circulatory arrest during CPB (DHCA) — planned bypass technique, not cardiac arrest
    arrest_rows = t.str.contains(r"\barrest\b", case=False, regex=True)
    keep[arrest_rows & t.str.contains(_IOD2_CIRC_ARREST_EXCLUDE, regex=True)] = False

    # Facility codes (code red = fire, not cardiac code)
    code_rows = t.str.contains(r"\bcode\b", case=False, regex=True)
    keep[code_rows & t.str.contains(_IOD2_CODE_FACILITY_EXCLUDE, regex=True)] = False

    # Pass 3: override — restore notes that were excluded above but clearly ARE IoD.

    # (a) ALWAYS_KEEP: unambiguous emergency phrases — override ALL exclusion filters.
    always_keep_mask = t.str.contains(_IOD2_ALWAYS_KEEP, regex=True)
    keep[always_keep_mask] = True

    # (b) EPI_OVERRIDE: systemic IV epinephrine indicators (mcg dosing, epi drip,
    #     "epi x N", "bolus of epi", "IV epi") incorrectly excluded by local-anesthetic
    #     epi filter.
    epi_was_excluded = epi_rows & t.str.contains(_IOD2_EPI_EXCLUDE, regex=True)
    epi_override_mask = epi_was_excluded & t.str.contains(_IOD2_EPI_OVERRIDE, regex=True)
    keep[epi_override_mask] = True

    # (c) ICD_FAIL_OVERRIDE: ICD test filter fired but "without successful conversion" /
    #     "multiple defibrillations" indicates a true emergency resuscitation.
    icd_fail_mask = icd_excluded & t.str.contains(_IOD2_ICD_FAIL_OVERRIDE, regex=True)
    keep[icd_fail_mask] = True

    # (d) HELP_CLINICAL_OVERRIDE: help-not-emergency filter fired but note documents
    #     explicit hemodynamic collapse (precipitous BP drop, HR to 30s-40s, severe desat).
    help_clinical_mask = help_excluded & t.str.contains(_IOD2_HELP_CLINICAL_OVERRIDE, regex=True)
    keep[help_clinical_mask] = True

    restored = (always_keep_mask | epi_override_mask | icd_fail_mask | help_clinical_mask).sum()
    filtered = matched[keep]
    log.info(
        f"  IoD2 keyword match: {len(matched):,} notes"
        f" → after FP filter: {len(filtered):,} notes"
        f" ({len(matched)-len(filtered):,} removed, {restored} overridden back)"
    )

    intra = in_or_window(filtered[["C MRN", "Recorded Time"]], "Recorded Time", an_patients)
    return set(intra["_enc_csn"].unique())


def compute_iod3(meds, an_patients):
    """
    IV vasoactive bolus during OR window.
    Drugs: epinephrine, phenylephrine (dose>2), ephedrine (dose>2),
           vasopressin, adenosine, atropine (age>3 months).
    Returns set of positive Encounter CSNs.
    """
    iv_routes = ["IV", "intravenous"]
    route_pat = "|".join(iv_routes)
    iv = meds[meds["Route"].fillna("").str.contains(route_pat, case=False)].copy()
    # Exclude non-IV routes
    iv = iv[~iv["Route"].isin([
        "Code/Sedation/Trauma IV", "Subconjunctival", "Intrasalivary gland"
    ])]
    iv["Dose"] = pd.to_numeric(iv["Dose"], errors="coerce")

    epi    = iv[iv["Medication"].str.contains("epinephrine", case=False, na=False)]
    phe    = iv[iv["Medication"].str.contains("phenylephrine", case=False, na=False) &
                (iv["Dose"] > 2)]
    eph    = iv[iv["Medication"].str.contains(r"\bephedrine\b", case=False, na=False) &
                (iv["Dose"] > 2)]
    vaso   = iv[iv["Medication"].str.contains("vasopressin", case=False, na=False)]
    aden   = iv[iv["Medication"].str.contains("adenosine", case=False, na=False)]

    # Atropine: only patients > 3 months old
    atr = iv[iv["Medication"].str.contains("atropine", case=False, na=False)].copy()
    atr["Date of Birth"] = pd.to_datetime(atr["Date of Birth"], errors="coerce")
    atr["MAR Time"]      = pd.to_datetime(atr["MAR Time"],      errors="coerce")
    atr["age_months"]    = (atr["MAR Time"] - atr["Date of Birth"]) / pd.Timedelta(days=30)
    atr = atr[atr["age_months"] > 3]

    combined = pd.concat([epi, phe, eph, vaso, aden, atr])
    intra = in_or_window(combined, "MAR Time", an_patients)
    return set(intra["_enc_csn"].unique())


def compute_iod4(meds, an_patients):
    """
    Vasoactive infusion started or dose-doubled during AN Start → Out OR.
    Drugs: dopamine, epinephrine (not norepi), milrinone,
           norepinephrine, phenylephrine, vasopressin.
    Dose unit must include min/hr/day (infusion marker).
    Returns set of positive Encounter CSNs.
    """
    # Encounter window: AN Start within In OR → Out OR
    an_pt = an_patients.copy()
    for col in ["AN Start", "In OR", "Out OR"]:
        an_pt[col] = pd.to_datetime(an_pt[col], errors="coerce")
    intra_enc = an_pt[
        an_pt["AN Start"].notna() &
        (an_pt["AN Start"] >= an_pt["In OR"]) &
        (an_pt["AN Start"] <= an_pt["Out OR"])
    ]

    inf = meds[meds["Dose Unit"].fillna("").str.contains("min|hr|day", case=False)].copy()
    inf["Infusion Rate"] = pd.to_numeric(inf["Infusion Rate"], errors="coerce")
    inf["MAR Time"]      = pd.to_datetime(inf["MAR Time"],     errors="coerce")

    drugs = {
        "dopamine":       inf["Medication"].str.contains("dopamine", case=False, na=False),
        "epinephrine":    (inf["Medication"].str.contains("epinephrine", case=False, na=False) &
                           ~inf["Medication"].str.contains("norepinephrine", case=False, na=False)),
        "milrinone":      inf["Medication"].str.contains("milrinone", case=False, na=False),
        "norepinephrine": inf["Medication"].str.contains("norepinephrine", case=False, na=False),
        "phenylephrine":  inf["Medication"].str.contains("phenylephrine", case=False, na=False),
        "vasopressin":    inf["Medication"].str.contains("vasopressin", case=False, na=False),
    }
    vasoactive = inf[pd.concat(drugs.values(), axis=1).any(axis=1)]

    # Merge with intraprocedure AN encounter window (AN Start → Out OR)
    # Bring AN_Patients Encounter CSN as _enc_csn for encounter-level labeling
    merged = pd.merge(
        vasoactive,
        intra_enc[["C MRN", "Encounter CSN", "AN Start", "Out OR"]].rename(
            columns={"Encounter CSN": "_enc_csn"}),
        on="C MRN"
    ).drop_duplicates()
    merged = merged[
        merged["MAR Time"].notna() &
        (merged["AN Start"].notna() | merged["Out OR"].notna())
    ]
    merged = merged[
        (merged["MAR Time"] >= merged["AN Start"]) &
        (merged["MAR Time"] <= merged["Out OR"])
    ]

    # Keep encounters where infusion was newly started (1 row) or rate doubled
    positive_csns = set()
    grouped = merged.groupby(["_enc_csn", "Medication"])
    for (enc_csn, med), group in grouped:
        group = group.sort_values("MAR Time")
        if len(group) == 1:
            positive_csns.add(enc_csn)
        else:
            rates = group["Infusion Rate"].dropna().values
            if len(rates) > 1:
                rate_diffs = np.diff(rates)
                if len(rate_diffs) > 0 and rates[0] > 0:
                    if any(rate_diffs > rates[0]):   # >100% increase
                        positive_csns.add(enc_csn)
            else:
                positive_csns.add(enc_csn)

    return positive_csns


def compute_iod5(ldas, an_patients, surgical_history):
    """
    Arterial line placed intraprocedurally after surgical incision.
    Returns set of positive Encounter CSNs.
    """
    an_pt = an_patients.copy()
    for col in ["AN Start", "In OR", "Out OR"]:
        an_pt[col] = pd.to_datetime(an_pt[col], errors="coerce")
    intra_enc = an_pt[
        an_pt["AN Start"].notna() &
        (an_pt["AN Start"] >= an_pt["In OR"]) &
        (an_pt["AN Start"] <= an_pt["Out OR"])
    ]

    art = ldas[ldas["LDA Type"] == "Arterial Lin"].copy()
    art["LDA Placed"] = pd.to_datetime(art["LDA Placed"], errors="coerce")

    # Arterial lines placed during OR window — use in_or_window to get _enc_csn
    art_intra = in_or_window(art, "LDA Placed", intra_enc)

    # Surgical incision records (verify incision exists for patient)
    incision = surgical_history[
        surgical_history["Proc Name"].str.contains("incision", case=False, na=False)
    ]

    merged2 = pd.merge(
        art_intra, incision[["C MRN", "Proc Name"]].drop_duplicates("C MRN"), on="C MRN"
    ).drop_duplicates()

    return set(merged2["_enc_csn"].unique())


# ── main ───────────────────────────────────────────────────────────────────────
def main(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── load raw data ─────────────────────────────────────────────────────────
    log.info("Loading AN_Patients …")
    an_patients = read_rpt(FILE_AN_PATIENTS)
    an_patients = an_patients[
        an_patients["C MRN"].astype(str).str.startswith("C")
    ].iloc[:-2]

    log.info("Loading AN_Events …")
    an_events = read_rpt(FILE_AN_EVENTS)
    an_events = an_events[
        an_events["C MRN"].astype(str).str.startswith("C")
    ].iloc[:-2]

    log.info("Loading Medication_Administration …")
    meds = read_rpt(FILE_MEDICATIONS,
                    usecols=["C MRN", "Encounter CSN", "Medication", "Route",
                             "Dose", "Dose Unit", "Infusion Rate", "MAR Time",
                             "Date of Birth"])
    meds = meds[meds["C MRN"].astype(str).str.startswith("C")].iloc[:-2]
    meds["Dose"] = pd.to_numeric(meds["Dose"], errors="coerce")

    log.info("Loading LDAs …")
    ldas = read_rpt(FILE_LDAS,
                    usecols=["C MRN", "LDA Type", "LDA Placed"])
    ldas = ldas[ldas["C MRN"].astype(str).str.startswith("C")].iloc[:-2]

    log.info("Loading Surgical_History …")
    surg = pd.read_csv(FILE_SURGICAL_HISTORY, on_bad_lines="skip",
                       low_memory=False,
                       usecols=["C MRN", "Proc Name", "Surg Hx Start Date"])
    surg = surg[surg["C MRN"].astype(str).str.startswith("C")]

    # ── build base encounter DataFrame ───────────────────────────────────────
    log.info("Building encounter-level df …")
    df = an_patients[[
        "C MRN", "Z Patient ID", "Encounter CSN",
        "In OR", "Out OR", "Hospital Admission Date", "Hospital Discharge Date",
        "Weight in kg", "Height in cm", "BMI Percentile", "AN Start", "AN End",
    ]].drop_duplicates(subset=["C MRN", "Z Patient ID", "Encounter CSN"]).copy()
    df = df.reset_index(drop=True)
    log.info(f"  {len(df):,} unique encounters · {df['C MRN'].nunique():,} patients")

    # ── compute labels ────────────────────────────────────────────────────────
    # Labels are assigned per Encounter CSN (not per C MRN)
    # so a patient's later encounters are not contaminated by earlier IoD events
    log.info("Computing IoD1 (CPR during OR) …")
    iod1_csns = compute_iod1(an_events, an_patients)
    df["Intraoperative_Deterioration1"] = df["Encounter CSN"].isin(iod1_csns).astype(int)
    log.info(f"  IoD1 positive encounters: {df['Intraoperative_Deterioration1'].sum():,}")

    log.info("Computing IoD2 (Quick Note keywords during OR) …")
    iod2_csns = compute_iod2(an_events, an_patients)
    df["Intraoperative_Deterioration2"] = df["Encounter CSN"].isin(iod2_csns).astype(int)
    log.info(f"  IoD2 positive encounters: {df['Intraoperative_Deterioration2'].sum():,}")

    log.info("Computing IoD3 (IV vasoactive bolus during OR) …")
    iod3_csns = compute_iod3(meds, an_patients)
    df["Intraoperative_Deterioration3"] = df["Encounter CSN"].isin(iod3_csns).astype(int)
    log.info(f"  IoD3 positive encounters: {df['Intraoperative_Deterioration3'].sum():,}")

    log.info("Computing IoD4 (vasoactive infusion started/escalated during OR) …")
    iod4_csns = compute_iod4(meds, an_patients)
    df["Intraoperative_Deterioration4"] = df["Encounter CSN"].isin(iod4_csns).astype(int)
    log.info(f"  IoD4 positive encounters: {df['Intraoperative_Deterioration4'].sum():,}")

    log.info("Computing IoD5 (arterial line after incision during OR) …")
    iod5_csns = compute_iod5(ldas, an_patients, surg)
    df["Intraoperative_Deterioration5"] = df["Encounter CSN"].isin(iod5_csns).astype(int)
    log.info(f"  IoD5 positive encounters: {df['Intraoperative_Deterioration5'].sum():,}")

    # Composite label
    iod_cols = [f"Intraoperative_Deterioration{i}" for i in range(1, 6)]
    df["Intraoperative_Deterioration"] = df[iod_cols].any(axis=1).astype(int)

    # ── save raw CSV ──────────────────────────────────────────────────────────
    csv_path = output_dir / "iod_labels.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"Saved raw labels → {csv_path}")

    # ── save MEDS task parquet ────────────────────────────────────────────────
    # MEDS task format: patient_id, prediction_time, boolean_value
    # prediction_time = In OR: only preop data (time < In OR) is used for prediction.
    # This correctly excludes intraoperative data from the feature window.
    in_or = pd.to_datetime(df["In OR"], errors="coerce")
    valid_time = in_or.notna()

    task = pd.DataFrame({
        "patient_id":      df["C MRN"].values,
        "encounter_csn":   df["Encounter CSN"].values,
        "prediction_time": in_or.values,      # ← In OR (preop cutoff)
        "boolean_value":   df["Intraoperative_Deterioration"].astype(bool).values,
        "IoD1":            df["Intraoperative_Deterioration1"].astype(bool).values,
        "IoD2":            df["Intraoperative_Deterioration2"].astype(bool).values,
        "IoD3":            df["Intraoperative_Deterioration3"].astype(bool).values,
        "IoD4":            df["Intraoperative_Deterioration4"].astype(bool).values,
        "IoD5":            df["Intraoperative_Deterioration5"].astype(bool).values,
    })

    parquet_path = output_dir / "iod_task.parquet"
    task.to_parquet(parquet_path, index=False, engine="pyarrow")
    log.info(f"Saved MEDS task parquet → {parquet_path}")

    # ── summary ───────────────────────────────────────────────────────────────
    n_total    = len(df)
    n_positive = df["Intraoperative_Deterioration"].sum()
    n_valid_t  = valid_time.sum()

    log.info("=" * 55)
    log.info(f"  Total encounters       : {n_total:>10,}")
    log.info(f"  With valid In OR       : {n_valid_t:>10,}  ({n_valid_t/n_total*100:.1f}%)")
    log.info(f"  IoD positive           : {n_positive:>10,}  ({n_positive/n_total*100:.1f}%)")
    log.info(f"  IoD negative           : {n_total-n_positive:>10,}  ({(n_total-n_positive)/n_total*100:.1f}%)")
    log.info("  Sub-label breakdown:")
    for col in iod_cols:
        cnt = df[col].sum()
        log.info(f"    {col:<40}: {cnt:>6,}  ({cnt/n_total*100:.1f}%)")
    log.info("=" * 55)
    log.info(f"Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute IoD labels and save as CSV + MEDS task parquet")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    main(Path(args.output_dir))
