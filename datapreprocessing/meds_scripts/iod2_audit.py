"""
iod2_audit.py
-------------
Comprehensive audit of IoD2 Quick Note filtering.

For each Quick Note that matched the IoD2 keyword pattern, categorize it as:
  KEPT    — passed all filters (potential FP if not actually IoD)
  REMOVED — caught by a filter (potential FN if actually IoD)

Outputs:
  iod2_audit_kept.csv     — all kept notes with matched keywords
  iod2_audit_removed.csv  — all removed notes with removal reason + matched keywords
  iod2_audit_report.txt   — summary statistics + sampled notes for review

Usage:
    conda activate ethos
    python datapreprocessing/meds_scripts/iod2_audit.py
"""

import re
import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── copy filter definitions from iod_to_outcome.py ──────────────────────────
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

_IOD2_EPI_EXCLUDE = re.compile(
    r"epidural"
    r"|epiglott"
    r"|epicardial"
    r"|\bepic\b"
    r"|cefepime"
    r"|(?:lido(?:caine)?|marcaine|bupiv(?:acaine)?|bupi\b|ropivacaine)"
    r"|\bepi\s*1\s*:[0-9]|1\s*:\s*(?:100|200|400)[,kK]?(?:,?000)?\s*(?:epi|epinephrine)"
    r"|(?:racemic|nebulized)\s+epi"
    r"|\bcaudal\b"
    r"|pledg[ae]?t|plegett"
    r"|(?:surgeon|dentist|physician|interventional|\bENT\b)\s+inject"
    r"|inject(?:ed|ing|s)?\s+(?:local|by\s+(?:surgeon|dentist|interventional|\bENT\b))"
    r"|inject(?:ed|ing|s)?\s+(?:into\s+)?(?:nose|nasal|nare)"
    r"|local\s+(?:\w+\s+){0,3}(?:infiltr|inject|with\s+epi)"
    r"|\bLA\s+(?:w[/\\]|with)\s*(?:epi|epinephrine)"
    r"|(?:ophthalm|eye(?:s|\s)|drop(?:s|\s)|conjunctiv)"
    r"|inject(?:ed|ing|s)?\s+(?:\w+\s+){0,3}(?:by\s+)?(?:surgeon|dentist|interventional|\bENT\b|dr\.?\s+\w+)",
    re.IGNORECASE,
)

_IOD2_ECHO_ROUTINE = re.compile(
    r"^\s*echo\s*(?:start|started|end|ended|stop|stopped|begin|began|complete|done|off|on|probe|removed?)\s*[.!]?\s*$"
    r"|^\s*(?:start|stop|end|begin)\s+(?:of\s+)?echo\s*[.!]?\s*$"
    r"|echo\s+probe\s+(?:placed|removed|inserted|pulled|in|out)"
    r"|(?:tee|echo)\s+(?:monitoring|monitor(?:ing)?|surveillance)\b"
    r"|(?:begin|start|end|stop|finish)\s+(?:transthoracic|epicardial|intraoperative|tee)\s+echo"
    r"|echo\s+procedure\s+(?:begin|start|end|stop|complete)",
    re.IGNORECASE,
)

_IOD2_HELP_EMERGENCY = re.compile(
    r"(?:called?|call(?:ing)?|paged?|overhead|stat)\s+(?:for\s+)?help"
    r"|help\s+(?:called|arrives?|paged|needed\s+stat|to\s+OR)"
    r"|overhead\s+page.*help"
    r"|anesthesia\s+(?:help|now\b)",
    re.IGNORECASE,
)

_IOD2_BRADY_EXCLUDE = re.compile(r"\bdr\.?\s+brady\b", re.IGNORECASE)

_IOD2_DEFIB_EXCLUDE = re.compile(r"test(?:ing|ed)?\s+defib|defib.*test", re.IGNORECASE)

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

_IOD2_PHENYL_OPHTHALMIC_EXCLUDE = re.compile(
    r"(?:eye\s*drops?|ophthalmic|eyebrow|conjunctiv|ocular|eye\s*brow|instill(?:ed|s)?)"
    r"|(?:eye|eyes|both\s+eyes|right\s+eye|left\s+eye)\s+(?:by|per)\s+(?:surgeon|opthalmologist|ophthalmologist|optho|ophthal)"
    r"|phenylephrine\s+(?:eye\s*drops?|ophthalmic|drops?|swabs?|solution)\s+(?:applied|placed|instilled|given|started|administ)"
    r"|(?:applied|placed|instilled)\s+(?:to|in|into)\s+(?:eye|eyes|both\s+eyes|eyebrow)"
    r"|(?:phenylephrine|neo-synephrine)\s+(?:cotton\s+swabs?|pledgets?|gauze)\s+(?:applied|placed)",
    re.IGNORECASE,
)

_IOD2_NASAL_EPI_EXCLUDE = re.compile(
    r"(?:nasal|intranasally|intra-?nasal|nare|nostril)\s+epi(?:nephrine)?"
    r"|epi(?:nephrine)?\s+(?:intranasally|nasal(?:ly)?|via\s+nose|to\s+nose|intra-?nasal)"
    r"|epi(?:nephrine)?\s+(?:pledgets?|plegetts?|cotton\s+pledgets?|gauze\s+pledgets?)"
    r"|pledgets?\s+(?:placed|soaked|dipped|with)\s+(?:\w+\s+){0,3}epi(?:nephrine)?"
    r"|(?:nasal|intranasal)\s+(?:decongest|vasoconstric)",
    re.IGNORECASE,
)

_IOD2_ICD_TEST_EXCLUDE = re.compile(
    r"ICD\s+(?:tested?|testing|check(?:ed)?)\s*(?:x\s*\d+)?"
    r"|(?:internal|implant(?:able)?|device)\s+(?:shock|defib|defibrillat)"
    r"|induced?\s+(?:VT|V-?fib|ventricular\s+(?:fibrillation|tachycardia))\s+(?:for\s+)?(?:test|check|ICD|defib)"
    r"|(?:test|check(?:ing)?)\s+(?:ICD|defibrillator|device)\s+(?:function|threshold|sensing|output)"
    r"|(?:DFT|defibrillation\s+threshold)\s+(?:test|check)",
    re.IGNORECASE,
)

_IOD2_ALWAYS_KEEP = re.compile(
    r"chest\s+compressions|compressions\s+(?:done|started|performed|given)"
    r"|code\s+cart"
    r"|all\s+anesthesia\s+alert"
    r"|\bcalled\s+stat\b"
    r"|pulseless"
    r"|\bcpr\b",
    re.IGNORECASE,
)

_IOD2_EPI_OVERRIDE = re.compile(
    r"\d+\s*mcg\s*(?:of\s+)?(?:epi|epinephrine)"
    r"|\bepi(?:nephrine)?\b\s*\d+\s*mcg"
    r"|\bepi\s+[xX×]\s*\d+\s*(?!(?:ml|cc)\b)"
    r"|\bepi(?:nephrine)?\s+(?:drip|gtt)\b"
    r"|epi(?:nephrine)?\s+(?:drip|gtt)\s+(?:ordered|started|initiated|increased|running)"
    r"|\bbolus\s+of\s+epi(?:nephrine)?"
    r"|\bepi(?:nephrine)?\s+(?:infusion|drip)\s+(?:order|start|increas|titrat)"
    r"|epinephrine\s+(?:IV\b|intravenous)",
    re.IGNORECASE,
)


def get_matched_keywords(text: str) -> list:
    """Return list of keywords that matched in the text."""
    found = []
    for kw in _IOD2_KEYWORDS:
        if re.search(re.escape(kw), text, re.IGNORECASE):
            found.append(kw)
    return found


def classify_note(text: str) -> tuple[str, str]:
    """
    Returns (action, reason) where:
      action = 'kept' or 'removed' or 'kept_override'
      reason = which filter triggered (or 'passed' / 'override_always' / 'override_epi')
    """
    t = text

    # Start as keep=True
    keep = True
    reason = "passed"

    epi_excluded = False

    # epi/epinephrine exclusion
    if re.search(r"\bepi", t, re.IGNORECASE):
        if re.search(_IOD2_EPI_EXCLUDE, t):
            keep = False
            reason = "epi_FP"
            epi_excluded = True

    # echo routine
    if re.search(r"\becho\b", t, re.IGNORECASE):
        if re.search(_IOD2_ECHO_ROUTINE, t):
            keep = False
            reason = "echo_routine" if keep is True or reason == "passed" else reason

    # help: keep only emergency
    if re.search(r"\bhelp\b", t, re.IGNORECASE):
        if not re.search(_IOD2_HELP_EMERGENCY, t):
            keep = False
            reason = "help_not_emergency" if reason == "passed" else reason

    # brady name
    if re.search(r"\bbrady\b", t, re.IGNORECASE):
        if re.search(_IOD2_BRADY_EXCLUDE, t) and not re.search(r"bradycardia|bradycardic", t, re.IGNORECASE):
            keep = False
            reason = "dr_brady_name" if reason == "passed" else reason

    # defib test
    if re.search(r"\bdefib", t, re.IGNORECASE):
        if re.search(_IOD2_DEFIB_EXCLUDE, t):
            keep = False
            reason = "defib_test" if reason == "passed" else reason

    # hypotension planned
    if re.search(r"hypotension", t, re.IGNORECASE):
        if re.search(_IOD2_HYPOTENSION_EXCLUDE, t):
            keep = False
            reason = "planned_hypotension" if reason == "passed" else reason

    # phenylephrine ophthalmic
    if re.search(r"phenylephrine", t, re.IGNORECASE):
        if re.search(_IOD2_PHENYL_OPHTHALMIC_EXCLUDE, t):
            keep = False
            reason = "ophthalmic_phenylephrine" if reason == "passed" else reason

    # nasal/topical epi
    if re.search(r"\bepi", t, re.IGNORECASE):
        if re.search(_IOD2_NASAL_EPI_EXCLUDE, t):
            keep = False
            reason = "intranasal_topical_epi" if reason == "passed" else reason

    # ICD test
    if re.search(r"\bICD\b|defibrillat|induced?\s+(?:VT|V-?fib)", t, re.IGNORECASE):
        if re.search(_IOD2_ICD_TEST_EXCLUDE, t):
            keep = False
            reason = "icd_device_test" if reason == "passed" else reason

    # Pass 3: overrides
    if re.search(_IOD2_ALWAYS_KEEP, t):
        keep = True
        reason = "override_always_keep"

    if epi_excluded and re.search(_IOD2_EPI_OVERRIDE, t):
        keep = True
        reason = "override_epi_systemic"

    if keep:
        return "kept", reason
    else:
        return "removed", reason


def main():
    FILE_AN_EVENTS = (
        "/path/to/CHOA_RAW_TABLES/"
        "CHOA_DATA_Tables_CHD/DR15201_AN_Events.rpt"
    )
    OUTPUT_DIR = Path("/path/to/CHD_MEDS/outcome")

    log.info("Loading AN_Events …")
    an_events = pd.read_csv(
        FILE_AN_EVENTS, delimiter="|", encoding="utf-8-sig",
        encoding_errors="replace", on_bad_lines="skip", low_memory=False
    )

    # Filter to Quick Notes
    qn = an_events[an_events["Event"].fillna("").str.strip() == "Quick Note"].copy()
    log.info(f"Total Quick Notes: {len(qn):,}")

    txt = qn["Event Comment"].fillna("")

    # Pass 1: broad keyword match
    matched_mask = txt.str.contains(_IOD2_PATTERN, regex=True)
    matched = qn[matched_mask].copy()
    log.info(f"Quick Notes matching keywords: {len(matched):,}")

    # Classify each note
    results = []
    for _, row in matched.iterrows():
        text = str(row["Event Comment"]) if pd.notna(row["Event Comment"]) else ""
        action, reason = classify_note(text)
        keywords = get_matched_keywords(text)
        results.append({
            "C MRN": row.get("C MRN", ""),
            "Recorded Time": row.get("Recorded Time", ""),
            "Event Comment": text,
            "action": action,
            "reason": reason,
            "matched_keywords": "|".join(keywords),
        })

    df_results = pd.DataFrame(results)

    # Split into kept / removed
    kept = df_results[df_results["action"] == "kept"].copy()
    removed = df_results[df_results["action"] == "removed"].copy()

    log.info(f"Kept notes: {len(kept):,}")
    log.info(f"Removed notes: {len(removed):,}")

    # ── Save full audit files ────────────────────────────────────────────────
    kept_path = OUTPUT_DIR / "iod2_audit_kept.csv"
    removed_path = OUTPUT_DIR / "iod2_audit_removed.csv"
    kept.to_csv(kept_path, index=False)
    removed.to_csv(removed_path, index=False)
    log.info(f"Saved: {kept_path}")
    log.info(f"Saved: {removed_path}")

    # ── Generate audit report ────────────────────────────────────────────────
    report_path = OUTPUT_DIR / "iod2_audit_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("IoD2 QUICK NOTE FILTER AUDIT REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total Quick Notes: {len(qn):,}\n")
        f.write(f"Matched keywords (broad): {len(matched):,}\n")
        f.write(f"Kept after filters: {len(kept):,}\n")
        f.write(f"Removed by filters: {len(removed):,}\n")
        f.write(f"Removal rate: {len(removed)/len(matched)*100:.1f}%\n\n")

        f.write("── REMOVAL REASON BREAKDOWN ──\n")
        rc = removed["reason"].value_counts()
        for reason, cnt in rc.items():
            f.write(f"  {reason}: {cnt:,}\n")

        f.write("\n── KEPT NOTES: TOP MATCHED KEYWORDS ──\n")
        all_kws = []
        for kws in kept["matched_keywords"]:
            all_kws.extend(kws.split("|"))
        from collections import Counter
        kw_counts = Counter(k for k in all_kws if k)
        for kw, cnt in kw_counts.most_common(20):
            f.write(f"  '{kw}': {cnt:,}\n")

        # ── Sample kept notes for FP analysis ──────────────────────────────
        f.write("\n" + "=" * 80 + "\n")
        f.write("KEPT NOTES — SAMPLE FOR FALSE POSITIVE REVIEW\n")
        f.write("(Are these truly IoD? If not, we need more filters)\n")
        f.write("=" * 80 + "\n\n")

        # Sample by keyword group for systematic review
        keyword_groups = {
            "echo": r"\becho\b",
            "help": r"\bhelp\b",
            "hypotension": r"hypotension",
            "code": r"\bcode\b",
            "shock": r"\bshock\b",
            "dopamine": r"\bdopamine\b",
            "norepi/norepinephrine": r"nore?pi",
            "phenylephrine": r"phenylephrine",
            "arrhythmia/SVT/Vtach": r"arrhythmia|SVT|[Vv]tach|v.tach|v.fib",
            "brady/fibrillation": r"\bbrady|fibrillation",
            "arrest/cardiac output": r"\barrest\b|cardiac\s+output",
        }

        for group_name, pat in keyword_groups.items():
            group_kept = kept[kept["Event Comment"].str.contains(pat, case=False, regex=True, na=False)]
            if len(group_kept) == 0:
                continue
            sample = group_kept.sample(min(20, len(group_kept)), random_state=42)
            f.write(f"\n--- Kept notes with keyword '{group_name}' (n={len(group_kept)}, showing {len(sample)}) ---\n")
            for _, row in sample.iterrows():
                comment = str(row["Event Comment"])[:200]
                f.write(f"  [{row['reason']}] {comment}\n")

        # ── Sample removed notes for FN analysis ───────────────────────────
        f.write("\n" + "=" * 80 + "\n")
        f.write("REMOVED NOTES — SAMPLE FOR FALSE NEGATIVE REVIEW\n")
        f.write("(Are any of these actually IoD? If so, we need override patterns)\n")
        f.write("=" * 80 + "\n\n")

        for reason, group_df in removed.groupby("reason"):
            sample = group_df.sample(min(30, len(group_df)), random_state=42)
            f.write(f"\n--- Removed by '{reason}' (n={len(group_df)}, showing {len(sample)}) ---\n")
            for _, row in sample.iterrows():
                comment = str(row["Event Comment"])[:250]
                f.write(f"  {comment}\n")

        # ── Special: help filter analysis ──────────────────────────────────
        f.write("\n" + "=" * 80 + "\n")
        f.write("HELP FILTER: REMOVED NOTES WITH CLINICAL DISTRESS SIGNALS\n")
        f.write("(Notes removed by 'help' filter but contain vasopressor/hemodynamic language)\n")
        f.write("=" * 80 + "\n\n")

        help_removed = removed[removed["reason"] == "help_not_emergency"].copy()
        clinical_signals = re.compile(
            r"(?:MAP|SBP|BP|blood\s+pressure)\s*(?:in\s+the)?\s*[0-9]"
            r"|vasopressor|vasopressors"
            r"|phenylephrine|epinephrine|dopamine|norepinephrine|vasopressin|ephedrine"
            r"|hypotension|hypotensive"
            r"|desaturation|O2\s+sat|SpO2"
            r"|bradycardia|bradycardic"
            r"|CPR|code|arrest"
            r"|emergent|stat\b",
            re.IGNORECASE,
        )
        help_clinical = help_removed[
            help_removed["Event Comment"].str.contains(clinical_signals, regex=True, na=False)
        ]
        f.write(f"Help-removed notes with clinical distress signals: {len(help_clinical)} / {len(help_removed)}\n\n")
        for _, row in help_clinical.iterrows():
            f.write(f"  {str(row['Event Comment'])[:300]}\n\n")

        # ── Special: echo kept notes (not just standalone) ─────────────────
        f.write("\n" + "=" * 80 + "\n")
        f.write("ECHO: KEPT NOTES — ARE THESE TRULY IoD?\n")
        f.write("(Routine 'Waiting for echo', 'Used ECHO to confirm' etc. may be FP)\n")
        f.write("=" * 80 + "\n\n")

        echo_kept = kept[kept["Event Comment"].str.contains(r"\becho\b", case=False, regex=True, na=False)]
        non_emergency_echo = echo_kept[
            ~echo_kept["Event Comment"].str.contains(
                r"ECMO|defib|VF|VT|arrest|CPR|compressions|deteriorat|emergent|stat\b",
                case=False, regex=True, na=False
            )
        ]
        f.write(f"Echo-kept notes without clear emergency keywords: {len(non_emergency_echo)} / {len(echo_kept)}\n\n")
        for _, row in non_emergency_echo.sample(min(30, len(non_emergency_echo)), random_state=42).iterrows():
            f.write(f"  {str(row['Event Comment'])[:250]}\n")

        # ── Special: negation analysis ─────────────────────────────────────
        f.write("\n" + "=" * 80 + "\n")
        f.write("NEGATION PATTERNS IN KEPT NOTES\n")
        f.write("(Notes with 'without X', 'no X', 'denied X' for the matched keyword)\n")
        f.write("=" * 80 + "\n\n")

        negation_pat = re.compile(
            r"\bwithout\s+(?:\w+\s+){0,3}(?:hypotension|arrest|CPR|shock|defib|deteriorat)"
            r"|\bno\s+(?:hypotension|arrest|CPR|shock|defib|deteriorat|arrhythmia|bradycardia|tachycardia)"
            r"|\bdenied?\s+(?:\w+\s+){0,3}(?:hypotension|arrest|shock|deteriorat)"
            r"|\bhemodynamically\s+stable\b"
            r"|\bVSS\b|\bvitals?\s+stable\b",
            re.IGNORECASE,
        )
        negation_kept = kept[
            kept["Event Comment"].str.contains(negation_pat, regex=True, na=False)
        ]
        f.write(f"Kept notes with negation patterns: {len(negation_kept)}\n\n")
        for _, row in negation_kept.sample(min(40, len(negation_kept)), random_state=42).iterrows():
            f.write(f"  {str(row['Event Comment'])[:250]}\n")

    log.info(f"Saved: {report_path}")
    print(f"\nAudit complete. Check:\n  {report_path}\n  {kept_path}\n  {removed_path}")


if __name__ == "__main__":
    main()
