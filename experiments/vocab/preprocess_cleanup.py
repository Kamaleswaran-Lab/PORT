"""
Stream F — vocab cleanup mappings.

Three small but important bug fixes on top of previous tokenization:

  F1. AN_EVENT typo duplicate
       has both `AN_EVENT//HYPERVENTALATION` (typo, high count)
      and `AN_EVENT//HYPERVENTILATION` (correct spelling, low count).
      Merge both -> correct spelling.

  F2. TRANSFUSION dose-in-name
      44 TRANSFUSION tokens in  — the long tail encodes dose into the
      token name (e.g. TRANSFUSE_CONVALESCENT_COVID19_PLASMA_ML_DOSE_1_TOTAL_AMT_MLS_360).
      Consolidate to ~10 canonical TRANSFUSION products; keep //END markers.
      Dose should be carried via numeric_value at MEDS-conversion time.

  F3. MED/LDA leak
      Raw Epic MAR records LDA-placement "orders" in the Medication column,
      so `medications_to_meds.py` emits codes like MED//PERIPHERAL_ARTLINE
      that are semantically LDAs. Root cause is the Epic source, not our
      preprocessing code (`normalise_med` has no domain knowledge to split
      these out). Build a redirect map MED -> LDA for integration into the integrated MEDS.

Inputs:
  /path/to/CHD_MEDS/tokenized_v3/train/vocab_t47691.csv
  /path/to/CHD_MEDS/tokenized_v3/train/code_counts.csv

Outputs:
  outputs/an_event_map.csv
  outputs/transfusion_map.csv
  outputs/med_lda_leak_map.csv

All CSVs have columns: v3_code, code, reason, v3_count
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

ROOT = Path("experiments/vocab")
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

V3_VOCAB = Path("/path/to/CHD_MEDS/tokenized_v3/train/vocab_t47691.csv")
V3_COUNTS = Path("/path/to/CHD_MEDS/tokenized_v3/train/code_counts.csv")


# ── Load vocab + counts ─────────────────────────────────────────────────────
def load_v3() -> tuple[pd.DataFrame, dict[str, int]]:
    # vocab file is a bare csv of "<idx>,<code>" -> treat as list
    vocab_rows = []
    with V3_VOCAB.open() as f:
        reader = csv.reader(f)
        for row in reader:
            # rows look like "6185,AN_EVENT//HYPERVENTALATION"
            # but vocab file had plain list — detect format
            if len(row) == 1:
                vocab_rows.append(row[0])
            elif len(row) >= 2:
                vocab_rows.append(row[-1])
    vocab_df = pd.DataFrame({"code": vocab_rows})

    counts: dict[str, int] = {}
    with V3_COUNTS.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            code, cnt = row[0], row[1]
            try:
                counts[code] = int(cnt)
            except ValueError:
                # header or malformed
                continue
    return vocab_df, counts


# ── F1: HYPERVENT typo ─────────────────────────────────────────────────────────
def build_an_event_typo_map(counts: dict[str, int]) -> pd.DataFrame:
    v3_typo = "AN_EVENT//HYPERVENTALATION"
    v4_target = "AN_EVENT//HYPERVENTILATION"
    rows = [{
        "v3_code": v3_typo,
        "code": v4_target,
        "reason": "merge misspelling into correct spelling",
        "v3_count": counts.get(v3_typo, 0),
    }, {
        "v3_code": v4_target,
        "code": v4_target,
        "reason": "canonical (identity mapping kept for completeness)",
        "v3_count": counts.get(v4_target, 0),
    }]
    return pd.DataFrame(rows)


# ── F2: TRANSFUSION consolidation ──────────────────────────────────────────────
# All 44 TRANSFUSION tokens (22 base + 22 //END) collapse into a canonical
# set of 9 TRANSFUSION products × 2 (base/END) = 18 tokens.
#
# Canonical products:
#   RBC, PLATELETS, FFP, CRYO, EXCHANGE_RBC, GRANULOCYTES, CONVALESCENT_PLASMA
# (CONVALESCENT_PLASMA collapses the 16 dose-in-name tokens + their ENDs;
#  GRANULOCYTES drops the "_UNITS" suffix.)
def _consolidate_transfusion(code: str) -> tuple[str, str] | None:
    """Return (code, reason) for a TRANSFUSION code, or None if unchanged."""
    if not code.startswith("TRANSFUSION//"):
        return None

    # Detect //END marker
    end = code.endswith("//END")
    body = code[len("TRANSFUSION//"):-len("//END")] if end else code[len("TRANSFUSION//"):]

    # Covid convalescent plasma (with dose encoded in name)
    if "CONVALESCENT_COVID19_PLASMA" in body:
        base = "CONVALESCENT_PLASMA"
        reason = "consolidate covid convalescent-plasma dose/amount variants into one product token"
    elif body.startswith("TRANSFUSE_GRANULOCYTES"):
        base = "GRANULOCYTES"
        reason = "strip '_UNITS' and 'TRANSFUSE_' prefix"
    elif body == "EXCHANGE_RBC":
        base = "EXCHANGE_RBC"
        reason = "identity (already clean)"
    elif body in {"RBC", "PLATELETS", "FFP", "CRYO"}:
        base = body
        reason = "identity (already clean)"
    else:
        # Unknown transfusion variant — fall back to as-is (documented below).
        base = body
        reason = "unrecognized variant — kept as-is (review)"

    code = f"TRANSFUSION//{base}" + ("//END" if end else "")
    return code, reason


def build_transfusion_map(vocab_df: pd.DataFrame, counts: dict[str, int]) -> pd.DataFrame:
    trans = vocab_df[vocab_df["code"].str.startswith("TRANSFUSION//")]["code"].tolist()
    rows = []
    for code in trans:
        res = _consolidate_transfusion(code)
        if res is None:
            continue
        code, reason = res
        rows.append({
            "v3_code": code,
            "code": code,
            "reason": reason,
            "v3_count": counts.get(code, 0),
        })
    return pd.DataFrame(rows).sort_values(["code", "v3_count"], ascending=[True, False])


# ── F3: MED/LDA leak ───────────────────────────────────────────────────────────
# Deliberate redirects: MED tokens that are actually LDA-placement entries
# in Epic MAR. Verified targets exist in  LDA vocab.
MED_LDA_REDIRECTS: dict[str, tuple[str, str]] = {
    "MED//PERIPHERAL_ARTLINE": (
        "LDA//ARTERIAL_LIN",
        "LDA-placement order recorded in Epic MAR as medication (peripheral arterial line)",
    ),
    "MED//PERIPHERAL_ARTLINE_SODIUM_ACETATE": (
        "LDA//ARTERIAL_LIN",
        "arterial line maintenance fluid — treat as LDA placement",
    ),
    "MED//CENTRAL_VENOUS_LINE": (
        "LDA//CVL",
        "LDA-placement order recorded in MAR as medication (CVL)",
    ),
    "MED//UMBILICAL_ARTERIAL_CATHETER_FLUID": (
        "LDA//UVC/UAC",
        "UAC maintenance fluid — treat as LDA placement",
    ),
    "MED//UMBILICAL_VENOUS_CATHETER_FLUID": (
        "LDA//UVC/UAC",
        "UVC maintenance fluid — treat as LDA placement",
    ),
}


def build_med_lda_map(vocab_df: pd.DataFrame, counts: dict[str, int]) -> pd.DataFrame:
    # Verify every target exists in  LDA vocab so the redirect is sound.
    lda_set = set(vocab_df[vocab_df["code"].str.startswith("LDA//")]["code"].tolist())
    rows = []
    for v3_code, (code, reason) in MED_LDA_REDIRECTS.items():
        assert code in lda_set, f"LDA target {code} not in  LDA vocab"
        rows.append({
            "v3_code": v3_code,
            "code": code,
            "reason": reason,
            "v3_count": counts.get(v3_code, 0),
        })
    return pd.DataFrame(rows).sort_values("v3_count", ascending=False)


# ── Runner ─────────────────────────────────────────────────────────────────────
def main() -> None:
    vocab_df, counts = load_v3()

    an_event_map = build_an_event_typo_map(counts)
    transfusion_map = build_transfusion_map(vocab_df, counts)
    med_lda_map = build_med_lda_map(vocab_df, counts)

    an_event_map.to_csv(OUT_DIR / "an_event_map.csv", index=False)
    transfusion_map.to_csv(OUT_DIR / "transfusion_map.csv", index=False)
    med_lda_map.to_csv(OUT_DIR / "med_lda_leak_map.csv", index=False)

    # Compact console summary
    print("=== F1 AN_EVENT typo ===")
    print(an_event_map.to_string(index=False))
    print()
    print("=== F2 TRANSFUSION consolidation ===")
    print(f"TRANSFUSION tokens: {len(transfusion_map)}")
    print(f"TRANSFUSION tokens: {transfusion_map['code'].nunique()}")
    print(transfusion_map.groupby("code")["v3_count"].sum().to_string())
    print()
    print("=== F3 MED/LDA leak ===")
    print(med_lda_map.to_string(index=False))


if __name__ == "__main__":
    main()
