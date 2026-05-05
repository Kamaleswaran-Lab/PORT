"""
Stream E — SES / care-continuity tokenization for.

Parses DR15201_AN_Patients.rpt and emits per-encounter SES tokens:
  INSURANCE//{MEDICAID, COMMERCIAL, MEDICARE, TRICARE,
              UNKNOWN, OTHER}                                   (6 tokens)
  POINT_OF_ORIGIN//{OFFICE_REFERRAL, HOSPITAL_TRANSFER, HOME,
                    EMERGENCY_DEPT, OUTPATIENT_CLINIC, OTHER}   (6 tokens)
  HOME_COUNTY//{top-50 county names, OTHER}                     (~51 tokens)

Outputs CSV vocabs + ses_events.parquet, plus a validation report.

Advisor motivation: capture SES / care-continuity signals for pediatric
cardiac surgery IoD prediction.  lacked these features despite raw data
containing them.
"""

import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd

RAW = Path(
    "/path/to/CHOA_RAW_TABLES/CHOA_DATA_Tables_CHD/DR15201_AN_Patients.rpt"
)
OUT_DIR = Path("experiments/vocab/outputs")
LOG_DIR = Path("experiments/vocab/logs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

COUNTY_TOP_K = 50


# --- Payor Name normalization / mapping ------------------------------------

def _norm_payer(val: str) -> str:
    """Uppercase, collapse whitespace, strip punctuation to a canonical key."""
    if val is None:
        return ""
    s = str(val).strip().upper()
    if s in ("", "NAN", "NONE", "NULL"):
        return ""
    s = re.sub(r"[^A-Z0-9]+", "_", s).strip("_")
    return s


# Keyword-based mapping; applied in declared order.
# NOTE: the canonical mapping rule is keyword inclusion on the normalized
# (uppercase, underscore) name. `OUT_OF_STATE_MEDICAID` will match MEDICAID
# keyword. We handle OUT_OF_STATE_MEDICAID separately as advisor requested,
# so it still maps to MEDICAID.
MEDICAID_KEYWORDS = [
    "MEDICAID",
    "WELLCARE",
    "AMERIGROUP",
    "PEACHSTATE",
    "PEACH_STATE",
    "CARESOURCE",
    "CARE_SOURCE",
]
COMMERCIAL_KEYWORDS = [
    "BLUE_CROSS",
    "BLUE_SHIELD",
    "BCBS",
    "ANTHEM",
    "CIGNA",
    "UNITED",
    "UHC",
    "AETNA",
    "HUMANA",
    "KAISER",
    "UNITED_MEDICAL_RESOURCE",
    "UMR",
    "OPTUMA",
    "OPTUM",
    "GEHA",
    "MERITAIN",
    "ALLIED",
    "BENEFIT",
    "COMMERCIAL",
    "SELF_PAY",  # out-of-pocket; treat as commercial (non-government)
    "HMO",
    "PPO",
    "COVENTRY",
    "MULTIPLAN",
    "AMBETTER",
    "ALLSTATE",
    "PROGRESSIVE",
    "STATE_FARM",
    "AUTO",
    "WORKERS_COMP",
    "WORKERS",
    "LIBERTY",
    "METLIFE",
    "PRUDENTIAL",
    "GUARDIAN",
    "MUTUAL",
    "PRINCIPAL",
    "ASSURANT",
    "GOLDEN_RULE",
    "FIRST_HEALTH",
    "PHCS",
]
TRICARE_KEYWORDS = ["TRICARE", "CHAMPUS"]
# Explicit routing overrides (checked before keyword matching).
# CHAMPVA = disabled-veteran family coverage; distinct from commercial.
OTHER_EXPLICIT = ["CHAMPVA"]


def map_payer(raw: str) -> tuple[str, str]:
    """Return (token, normalized_key). Token is INSURANCE//*.

    Priority order:
      1. Explicit OTHER (e.g. CHAMPVA — disabled-veteran family, distinct from commercial)
      2. TRICARE / CHAMPUS
      3. MEDICAID variants (MEDICAID keyword + managed-Medicaid plans). Checked
         BEFORE MEDICARE so payers like OUT_OF_STATE_MEDICAID do not collide.
      4. MEDICARE — any payer containing "MEDICARE" that did not match MEDICAID.
         In pediatrics, Medicare signals ESRD / SSDI-qualifying chronic disease.
      5. COMMERCIAL keywords
      6. OTHER (default)
    """
    norm = _norm_payer(raw)
    if norm == "":
        return "INSURANCE//UNKNOWN", norm
    # Explicit OTHER (CHAMPVA) — must come before COMMERCIAL
    for kw in OTHER_EXPLICIT:
        if kw in norm:
            return "INSURANCE//OTHER", norm
    # TRICARE before others (has its own bucket)
    for kw in TRICARE_KEYWORDS:
        if kw in norm:
            return "INSURANCE//TRICARE", norm
    # MEDICAID must be checked BEFORE MEDICARE so OUT_OF_STATE_MEDICAID etc.
    # are not accidentally pulled into MEDICARE by a substring match.
    for kw in MEDICAID_KEYWORDS:
        if kw in norm:
            return "INSURANCE//MEDICAID", norm
    # MEDICARE: any remaining payer whose normalized name contains "MEDICARE".
    # Covers MEDICARE, MEDICARE_ONLY, MEDICARE_ADV(ANTAGE), MEDICARE_PART_B/D,
    # MEDICARE_REPLACEMENT, RAILROAD_MEDICARE, etc. Any MEDICAID-variant payer
    # has already been captured above.
    if "MEDICARE" in norm:
        return "INSURANCE//MEDICARE", norm
    for kw in COMMERCIAL_KEYWORDS:
        if kw in norm:
            return "INSURANCE//COMMERCIAL", norm
    return "INSURANCE//OTHER", norm


# --- Point of Origin mapping -----------------------------------------------

POI_MAP = {
    "Physician Office Referral/HMO": "POINT_OF_ORIGIN//OFFICE_REFERRAL",
    "Another Hospital Transfer": "POINT_OF_ORIGIN//HOSPITAL_TRANSFER",
    "Non-Health Care Facility Point of Origin": "POINT_OF_ORIGIN//HOME",
    "Other Hospital Emergency Dept": "POINT_OF_ORIGIN//EMERGENCY_DEPT",
    "Outpatient Provider, Clinic or Non ED": "POINT_OF_ORIGIN//OUTPATIENT_CLINIC",
}


def map_poi(raw: str) -> str:
    if raw is None:
        return "POINT_OF_ORIGIN//OTHER"
    s = str(raw).strip()
    if s == "" or s.upper() in ("NAN", "NONE", "NULL"):
        return "POINT_OF_ORIGIN//OTHER"
    return POI_MAP.get(s, "POINT_OF_ORIGIN//OTHER")


# --- Home County mapping ---------------------------------------------------

def norm_county(raw: str) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()
    if s == "" or s.upper() in ("NAN", "NONE", "NULL"):
        return ""
    s = s.upper()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Z0-9_]", "", s)
    return s


# ---------------------------------------------------------------------------


def main():
    print(f"Loading {RAW} ...")
    # Read only needed columns; treat file as pipe-delimited with utf-8-sig.
    cols = [
        "C MRN",
        "Encounter CSN",
        "In OR",
        "Point of Origin",
        "Home County ",
        "Payor Name",
    ]
    df = pd.read_csv(
        RAW,
        sep="|",
        usecols=cols,
        dtype=str,
        encoding="utf-8-sig",
        low_memory=False,
        on_bad_lines="warn",
    )
    print(f"Loaded {len(df):,} encounter rows, {df['C MRN'].nunique():,} patients.")

    # ---- Payor Name analysis ----
    payor_counts = df["Payor Name"].fillna("").value_counts(dropna=False)
    payor_rows = []
    for val, cnt in payor_counts.items():
        token, norm = map_payer(val)
        payor_rows.append(
            {"payor_name_raw": val, "payor_name_norm": norm, "token": token, "count": int(cnt)}
        )
    payor_audit = pd.DataFrame(payor_rows).sort_values("count", ascending=False)
    payor_audit.to_csv(OUT_DIR / "insurance_audit.csv", index=False)

    # Final 6-token vocab (MEDICAID / COMMERCIAL / MEDICARE / TRICARE / UNKNOWN / OTHER)
    insurance_vocab = (
        payor_audit.groupby("token")["count"].sum().sort_values(ascending=False).reset_index()
    )
    insurance_vocab.to_csv(OUT_DIR / "insurance_vocab.csv", index=False)

    # ---- Point of Origin analysis ----
    poi_counts = df["Point of Origin"].fillna("").value_counts(dropna=False)
    poi_rows = []
    for val, cnt in poi_counts.items():
        poi_rows.append(
            {"poi_raw": val, "token": map_poi(val), "count": int(cnt)}
        )
    poi_audit = pd.DataFrame(poi_rows).sort_values("count", ascending=False)
    poi_audit.to_csv(OUT_DIR / "point_of_origin_audit.csv", index=False)
    poi_vocab = (
        poi_audit.groupby("token")["count"].sum().sort_values(ascending=False).reset_index()
    )
    poi_vocab.to_csv(OUT_DIR / "point_of_origin_vocab.csv", index=False)

    # ---- Home County analysis (patient-level) ----
    pat_county = (
        df.groupby("C MRN")["Home County "].agg(
            lambda s: next((x for x in s if isinstance(x, str) and x.strip()), "")
        )
    )
    county_norm_by_patient = pat_county.map(norm_county)
    # Frequency over encounters (more representative for coverage %):
    county_enc_norm = df["Home County "].map(norm_county)
    county_enc_counts = county_enc_norm.value_counts(dropna=False)
    # top-K excluding empty (empty => OTHER)
    top_non_empty = [c for c in county_enc_counts.index if c != ""][:COUNTY_TOP_K]
    top_set = set(top_non_empty)

    def map_county(raw: str) -> str:
        n = norm_county(raw)
        if n == "" or n not in top_set:
            return "HOME_COUNTY//OTHER"
        return f"HOME_COUNTY//{n}"

    county_token_enc = county_enc_norm.map(
        lambda n: f"HOME_COUNTY//{n}" if n in top_set else "HOME_COUNTY//OTHER"
    )
    home_county_vocab = (
        county_token_enc.value_counts()
        .rename_axis("token")
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    home_county_vocab.to_csv(OUT_DIR / "home_county_vocab.csv", index=False)

    # Audit of the full county distribution
    county_audit = (
        county_enc_counts.rename_axis("county_norm")
        .reset_index(name="count")
        .assign(
            token=lambda d: d["county_norm"].map(
                lambda n: f"HOME_COUNTY//{n}" if n in top_set else "HOME_COUNTY//OTHER"
            )
        )
    )
    county_audit.to_csv(OUT_DIR / "home_county_audit.csv", index=False)

    # ---- Build ses_events.parquet ----
    out = pd.DataFrame(
        {
            "patient_id": df["C MRN"].astype(str),
            "encounter_csn": pd.to_numeric(df["Encounter CSN"], errors="coerce"),
            "in_or_timestamp": pd.to_datetime(df["In OR"], errors="coerce"),
            "insurance_token": df["Payor Name"].map(lambda v: map_payer(v)[0]),
            "point_of_origin_token": df["Point of Origin"].map(map_poi),
            "home_county_token": df["Home County "].map(map_county),
        }
    )
    out.to_parquet(OUT_DIR / "ses_events.parquet", index=False)

    # ---- Validation report ----
    n = len(df)
    ins_null = int((df["Payor Name"].isna() | (df["Payor Name"].astype(str).str.strip() == "")).sum())
    poi_null = int(
        (df["Point of Origin"].isna() | (df["Point of Origin"].astype(str).str.strip() == "")).sum()
    )
    co_null = int(county_enc_norm.eq("").sum())

    top50_cov_pct = 100.0 * (
        county_enc_counts[county_enc_counts.index.isin(top_set)].sum() / n
    )
    bottom10_at_cutoff = (
        county_enc_counts[county_enc_counts.index != ""]
        .iloc[COUNTY_TOP_K - 10 : COUNTY_TOP_K + 0]
        .reset_index()
    )

    ins_tok_cov = 100.0 * (
        insurance_vocab["count"].sum() / n
    )  # always 100 by construction
    poi_6tok_cov = 100.0 * (poi_vocab["count"].sum() / n)  # always 100

    # Explicit: % of encounters that fall into OTHER for each
    ins_other_pct = 100.0 * insurance_vocab.query("token == 'INSURANCE//OTHER'")[
        "count"
    ].sum() / n
    ins_unknown_pct = 100.0 * insurance_vocab.query("token == 'INSURANCE//UNKNOWN'")[
        "count"
    ].sum() / n
    poi_other_pct = 100.0 * poi_vocab.query("token == 'POINT_OF_ORIGIN//OTHER'")[
        "count"
    ].sum() / n
    co_other_pct = 100.0 * home_county_vocab.query("token == 'HOME_COUNTY//OTHER'")[
        "count"
    ].sum() / n

    lines = []
    lines.append("# Stream E — SES / care-continuity tokenization report\n")
    lines.append(f"- Source: `{RAW}`")
    lines.append(f"- Total encounter rows: **{n:,}**")
    lines.append(f"- Unique patients: **{df['C MRN'].nunique():,}**\n")

    lines.append("## 1. Payor Name")
    lines.append(f"- NULL/empty encounters: **{ins_null:,}** ({100*ins_null/n:.2f}%)")
    lines.append(f"- 6-token coverage: **{ins_tok_cov:.2f}%** (all encounters by construction)")
    lines.append(f"- `INSURANCE//OTHER` share: **{ins_other_pct:.2f}%**")
    lines.append(f"- `INSURANCE//UNKNOWN` share: **{ins_unknown_pct:.2f}%**\n")
    lines.append("### Final insurance vocab (6 tokens)")
    lines.append(insurance_vocab.to_markdown(index=False))
    lines.append("\n### Complete unique Payor Name values with counts and mapping\n")
    lines.append(payor_audit.to_markdown(index=False))

    lines.append("\n## 2. Point of Origin")
    lines.append(f"- NULL/empty encounters: **{poi_null:,}** ({100*poi_null/n:.2f}%)")
    lines.append(f"- 6-token coverage: **{poi_6tok_cov:.2f}%** (all encounters by construction)")
    lines.append(f"- `POINT_OF_ORIGIN//OTHER` share: **{poi_other_pct:.2f}%**\n")
    lines.append("### Final POI vocab (6 tokens)")
    lines.append(poi_vocab.to_markdown(index=False))
    lines.append("\n### Complete unique Point of Origin values with counts and mapping\n")
    lines.append(poi_audit.to_markdown(index=False))

    n_unique_counties = int((county_enc_counts.index != "").sum())
    lines.append("\n## 3. Home County")
    lines.append(f"- Unique county values (non-empty): **{n_unique_counties}**")
    lines.append(f"- NULL/empty encounters: **{co_null:,}** ({100*co_null/n:.2f}%)")
    lines.append(f"- Top-{COUNTY_TOP_K} named coverage: **{top50_cov_pct:.2f}%**")
    lines.append(f"- `HOME_COUNTY//OTHER` share: **{co_other_pct:.2f}%**")
    lines.append(f"- Home county vocab size: **{len(home_county_vocab)}** tokens\n")
    lines.append("### Top-50 counties retained")
    top50_tbl = (
        county_enc_counts[county_enc_counts.index.isin(top_set)]
        .rename_axis("county_norm")
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    lines.append(top50_tbl.to_markdown(index=False))
    lines.append("\n### Counties near cutoff (ranks 41–50 — bottom-10 of retained set)")
    lines.append(bottom10_at_cutoff.to_markdown(index=False))
    lines.append("\n### First 20 counties just below cutoff (rolled into OTHER)")
    below = (
        county_enc_counts[county_enc_counts.index != ""]
        .iloc[COUNTY_TOP_K : COUNTY_TOP_K + 20]
        .reset_index()
    )
    lines.append(below.to_markdown(index=False))

    lines.append("\n## 4. NULL / missing handling summary\n")
    lines.append(
        f"| Field | NULL encounters | % | Tokenized as |\n|---|---|---|---|\n"
        f"| Payor Name | {ins_null:,} | {100*ins_null/n:.2f}% | `INSURANCE//UNKNOWN` |\n"
        f"| Point of Origin | {poi_null:,} | {100*poi_null/n:.2f}% | `POINT_OF_ORIGIN//OTHER` |\n"
        f"| Home County | {co_null:,} | {100*co_null/n:.2f}% | `HOME_COUNTY//OTHER` |\n"
    )

    lines.append("\n## 5. Sample output (first 10 rows of ses_events.parquet)\n")
    lines.append(out.head(10).to_markdown(index=False))

    report_path = LOG_DIR / "E_ses_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {report_path}")
    print(f"Wrote {OUT_DIR}/insurance_vocab.csv ({len(insurance_vocab)} tokens)")
    print(f"Wrote {OUT_DIR}/point_of_origin_vocab.csv ({len(poi_vocab)} tokens)")
    print(f"Wrote {OUT_DIR}/home_county_vocab.csv ({len(home_county_vocab)} tokens)")
    print(f"Wrote {OUT_DIR}/ses_events.parquet ({len(out):,} rows)")


if __name__ == "__main__":
    main()
