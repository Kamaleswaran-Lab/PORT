"""
vocab fix — Stream D: DEMO/LANGUAGE atomic tokenization.

Fixes three  bugs:
  1. Combinatorial race explosion (multi-race → single concat token).
  2. Duplicate concat (same race repeated N times in raw; kept by norm).
  3. Language mixed into DEMO//<LANG> namespace.

schema (atomic):
  DEMO//RACE//<canonical>
  DEMO//SEX//<M|F|OTHER|UNKNOWN>
  DEMO//ETHNICITY//<HISPANIC|NON_HISPANIC|DECLINED|UNKNOWN>
  LANGUAGE//<ENGLISH|SPANISH|VIETNAMESE|SOMALI|ARABIC|OTHER>

Outputs:
  outputs/demo_language_tokens.parquet   per-encounter token assignments
  outputs/demo_vocab.csv                 atomic DEMO vocab
  outputs/language_vocab.csv             atomic LANGUAGE vocab
  logs/D_demo_language_report.md         validation report
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_PATH = Path(
    "/path/to/CHOA_RAW_TABLES/"
    "CHOA_DATA_Tables_CHD/DR15201_AN_Patients.rpt"
)

ROOT = Path("experiments/vocab")
OUT_DIR = ROOT / "outputs"
LOG_DIR = ROOT / "logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Atomic canonical race vocabulary (order used by CSV export).
RACE_CANON = {
    "white": "WHITE",
    "black or african american": "BLACK_OR_AFRICAN_AMERICAN",
    "asian": "ASIAN",
    "american indian or alaska native": "AMERICAN_INDIAN_OR_ALASKA_NATIVE",
    "native hawaiian or other pacific islander":
        "NATIVE_HAWAIIAN_OR_OTHER_PACIFIC_ISLANDER",
    "other": "OTHER",
    "declined": "DECLINED",
    "unknown": "UNKNOWN",
}
RACE_VOCAB = [f"DEMO//RACE//{v}" for v in dict.fromkeys(RACE_CANON.values())]

SEX_CANON = {
    "male": "M",
    "female": "F",
    "other": "OTHER",
    "unknown": "UNKNOWN",
}
SEX_VOCAB = [f"DEMO//SEX//{v}" for v in dict.fromkeys(SEX_CANON.values())]

ETH_CANON = {
    "hispanic or latino": "HISPANIC",
    "non hispanic or latino": "NON_HISPANIC",
    "declined": "DECLINED",
    "unknown": "UNKNOWN",
    "parent not present": "UNKNOWN",
}
ETH_VOCAB = [f"DEMO//ETHNICITY//{v}" for v in dict.fromkeys(ETH_CANON.values())]

# Spec-mandated top-5 + OTHER.
LANG_TOP5 = {"english", "spanish", "vietnamese", "somali", "arabic"}
LANG_CANON_MAP = {
    "english": "ENGLISH",
    "spanish": "SPANISH",
    "vietnamese": "VIETNAMESE",
    "somali": "SOMALI",
    "arabic": "ARABIC",
}
LANG_VOCAB = [
    "LANGUAGE//ENGLISH",
    "LANGUAGE//SPANISH",
    "LANGUAGE//VIETNAMESE",
    "LANGUAGE//SOMALI",
    "LANGUAGE//ARABIC",
    "LANGUAGE//OTHER",
]


def _norm(x):
    if pd.isna(x):
        return None
    return str(x).strip().lower()


def parse_race(raw):
    """Return list of atomic race tokens (no duplicates per encounter)."""
    v = _norm(raw)
    if v is None or v == "":
        return ["DEMO//RACE//UNKNOWN"]
    parts = [p.strip() for p in v.split(";") if p.strip()]
    seen = []
    for p in parts:
        canon = RACE_CANON.get(p, "OTHER")
        tok = f"DEMO//RACE//{canon}"
        if tok not in seen:
            seen.append(tok)
    if not seen:
        seen = ["DEMO//RACE//UNKNOWN"]
    return seen


def parse_sex(raw):
    v = _norm(raw)
    if v is None:
        return "DEMO//SEX//UNKNOWN"
    return f"DEMO//SEX//{SEX_CANON.get(v, 'UNKNOWN')}"


def parse_eth(raw):
    v = _norm(raw)
    if v is None:
        return "DEMO//ETHNICITY//UNKNOWN"
    return f"DEMO//ETHNICITY//{ETH_CANON.get(v, 'UNKNOWN')}"


def parse_lang(raw):
    v = _norm(raw)
    if v is None:
        return "LANGUAGE//OTHER"
    if v in LANG_TOP5:
        return f"LANGUAGE//{LANG_CANON_MAP[v]}"
    return "LANGUAGE//OTHER"


def main():
    print(f"[load] {RAW_PATH}")
    df = pd.read_csv(RAW_PATH, sep="|", dtype=str, on_bad_lines="skip")
    df = df.iloc[:-2].reset_index(drop=True)  # strip footer
    print(f"[load] raw rows = {len(df):,}")

    keep = ["C MRN", "Encounter CSN", "Race", "Ethnicity",
            "Legal Sex ", "Language "]
    df = df[keep].copy()
    df.columns = ["patient_id", "encounter_csn",
                  "race_raw", "eth_raw", "sex_raw", "lang_raw"]

    # Filter rows with valid patient_id.
    df = df[df["patient_id"].astype(str).str.startswith("C")].reset_index(
        drop=True)
    print(f"[filter] valid patient rows = {len(df):,}")

    # === Raw distribution analytics ===
    race_vc = df["race_raw"].value_counts(dropna=False)
    eth_vc = df["eth_raw"].value_counts(dropna=False)
    sex_vc = df["sex_raw"].value_counts(dropna=False)
    lang_vc = df["lang_raw"].value_counts(dropna=False)

    # Multi-race stats.
    def _n_races(v):
        if pd.isna(v) or v == "":
            return 0
        return len([p for p in str(v).split(";") if p.strip()])

    n_race = df["race_raw"].apply(_n_races)
    multi_race_count = int((n_race >= 2).sum())

    # Bug-2 confirmation: find duplicate-in-raw examples (same race listed ≥2×).
    def _has_dup(v):
        if pd.isna(v) or v == "":
            return False
        parts = [p.strip().lower() for p in str(v).split(";") if p.strip()]
        return len(parts) != len(set(parts))

    dup_mask = df["race_raw"].apply(_has_dup)
    dup_count = int(dup_mask.sum())
    dup_examples = df.loc[dup_mask, "race_raw"].value_counts().head(10)

    # === Parse per-encounter tokens ===
    print("[parse] tokenizing per encounter …")
    df["demo_tokens_race"] = df["race_raw"].apply(parse_race)
    df["demo_token_sex"] = df["sex_raw"].apply(parse_sex)
    df["demo_token_eth"] = df["eth_raw"].apply(parse_eth)
    df["language_token"] = df["lang_raw"].apply(parse_lang)

    # Flatten combined demo token list (race tokens + sex + eth) per encounter.
    df["demo_tokens"] = [
        list(r) + [s, e]
        for r, s, e in zip(df["demo_tokens_race"],
                           df["demo_token_sex"],
                           df["demo_token_eth"])
    ]

    out_cols = ["patient_id", "encounter_csn", "demo_tokens", "language_token"]
    df_out = df[out_cols].copy()
    out_parquet = OUT_DIR / "demo_language_tokens.parquet"
    df_out.to_parquet(out_parquet, index=False)
    print(f"[write] {out_parquet}  rows={len(df_out):,}")

    # === Vocab CSVs ===
    demo_vocab = RACE_VOCAB + SEX_VOCAB + ETH_VOCAB
    pd.DataFrame({"token": demo_vocab, "category": (
        ["RACE"] * len(RACE_VOCAB)
        + ["SEX"] * len(SEX_VOCAB)
        + ["ETHNICITY"] * len(ETH_VOCAB))}).to_csv(
            OUT_DIR / "demo_vocab.csv", index=False)
    pd.DataFrame({"token": LANG_VOCAB, "category": ["LANGUAGE"] * len(
        LANG_VOCAB)}).to_csv(OUT_DIR / "language_vocab.csv", index=False)
    print(f"[write] demo_vocab.csv  ({len(demo_vocab)})")
    print(f"[write] language_vocab.csv  ({len(LANG_VOCAB)})")

    # === Validation stats for report ===
    # Token-level counts (per encounter, each race token counted once).
    race_tok_counts = {}
    for toks in df["demo_tokens_race"]:
        for t in toks:
            race_tok_counts[t] = race_tok_counts.get(t, 0) + 1

    lang_tok_counts = df["language_token"].value_counts()
    sex_tok_counts = df["demo_token_sex"].value_counts()
    eth_tok_counts = df["demo_token_eth"].value_counts()

    # Top-5 language coverage.
    top5_coverage = 0
    total = len(df)
    for lang in LANG_TOP5:
        top5_coverage += int(lang_vc.get(lang.capitalize(), 0))
    # Case-insensitive fallback.
    lang_lower = df["lang_raw"].str.lower()
    top5_mask = lang_lower.isin(LANG_TOP5)
    top5_coverage_ci = int(top5_mask.sum())

    # 5 multi-race examples raw → parsed.
    multi_rows = df[n_race >= 2].head(5)
    sample_examples = []
    for _, r in multi_rows.iterrows():
        sample_examples.append((r["race_raw"], parse_race(r["race_raw"])))

    # === Write report ===
    report = LOG_DIR / "D_demo_language_report.md"
    with report.open("w") as f:
        f.write("# Stream D — DEMO/LANGUAGE vocab report\n\n")
        f.write(f"Raw file: `{RAW_PATH}`  \n")
        f.write(f"Rows (valid patient, post footer strip): **{total:,}**  \n\n")

        f.write("## 1. Raw data analysis\n\n")
        f.write(f"- Race: **{df['race_raw'].nunique(dropna=False)}** unique "
                f"raw values, nulls={df['race_raw'].isna().sum()}  \n")
        f.write(f"- Ethnicity: **{df['eth_raw'].nunique(dropna=False)}** "
                f"unique, nulls={df['eth_raw'].isna().sum()}  \n")
        f.write(f"- Sex: **{df['sex_raw'].nunique(dropna=False)}** unique, "
                f"nulls={df['sex_raw'].isna().sum()}  \n")
        f.write(f"- Language: **{df['lang_raw'].nunique(dropna=False)}** "
                f"unique, nulls={df['lang_raw'].isna().sum()}  \n\n")

        for name, vc in [("Race", race_vc), ("Ethnicity", eth_vc),
                         ("Sex", sex_vc), ("Language", lang_vc)]:
            f.write(f"### Top-20 raw values — {name}\n\n")
            f.write("| value | count |\n|---|---|\n")
            for v, c in vc.head(20).items():
                f.write(f"| `{v}` | {c:,} |\n")
            f.write("\n")

        f.write("## 2. Race delimiter discovery\n\n")
        f.write("- Delimiter: **`; `** (semicolon-space) — verified by "
                "inspecting raw value counts.  \n")
        f.write(f"- Multi-race encounters (≥2 races after split): "
                f"**{multi_race_count:,}** "
                f"({100*multi_race_count/total:.2f}%)  \n\n")

        f.write("## 3. Bug-2 confirmation (duplicate concat)\n\n")
        f.write(f"- Encounters whose raw Race contains the same race listed "
                f"≥2×: **{dup_count:,}**  \n")
        f.write("- Root cause: duplicates already exist in the raw EHR Race "
                "field (e.g. `Black or African American; Black or African "
                "American`). The previous converter (`an_patients_to_meds.py:113`) "
                "applies `norm(...)` to the whole string without splitting on "
                "`;`, so `_norm` drops non-alphanumerics and yields a single "
                "mega-token like `DEMO//BLACK_OR_AFRICAN_AMERICAN_BLACK_OR_"
                "AFRICAN_AMERICAN`. **Bug 2 is therefore a combination of "
                "dirty-source data AND the converter not splitting multi-race "
                "lists.**  \n\n")
        f.write("Top raw duplicate-in-source examples:\n\n")
        f.write("| raw value | count |\n|---|---|\n")
        for v, c in dup_examples.items():
            f.write(f"| `{v}` | {c:,} |\n")
        f.write("\n")

        f.write("## 4. atomic vocab\n\n")
        f.write(f"- **DEMO (atomic)**: **{len(demo_vocab)}** tokens = "
                f"{len(RACE_VOCAB)} race + {len(SEX_VOCAB)} sex + "
                f"{len(ETH_VOCAB)} ethnicity  \n")
        f.write(f"- **LANGUAGE**: **{len(LANG_VOCAB)}** tokens (top-5 + "
                "OTHER)  \n")
        f.write("- DEMO count (pre-fix, includes languages + "
                "combinatorial race): **190**  \n")
        f.write(f"- Reduction: 190 → {len(demo_vocab) + len(LANG_VOCAB)} "
                f"({len(demo_vocab)} DEMO + {len(LANG_VOCAB)} LANGUAGE)  \n\n")

        f.write("### Final DEMO vocab\n\n")
        for t in demo_vocab:
            f.write(f"- `{t}`\n")
        f.write("\n### Final LANGUAGE vocab\n\n")
        for t in LANG_VOCAB:
            f.write(f"- `{t}`\n")
        f.write("\n")

        f.write("## 5. Multi-race patient examples (raw → parsed)\n\n")
        f.write(f"- Multi-race encounter count: **{multi_race_count:,}**  \n\n")
        f.write("| raw | parsed tokens |\n|---|---|\n")
        for raw, toks in sample_examples:
            f.write(f"| `{raw}` | {', '.join('`'+t+'`' for t in toks)} |\n")
        f.write("\n")

        f.write("## 6. Language coverage\n\n")
        f.write(f"- Encounters whose language is in top-5 "
                f"(English/Spanish/Vietnamese/Somali/Arabic): "
                f"**{top5_coverage_ci:,}** "
                f"({100*top5_coverage_ci/total:.3f}%)  \n")
        f.write(f"- Remaining → `LANGUAGE//OTHER`: "
                f"**{total-top5_coverage_ci:,}** "
                f"({100*(total-top5_coverage_ci)/total:.3f}%)  \n\n")

        f.write("### Token-level distribution\n\n")
        f.write("#### Race tokens (per-encounter, multi-race counted once "
                "per race)\n\n")
        f.write("| token | count |\n|---|---|\n")
        for t in RACE_VOCAB:
            f.write(f"| `{t}` | {race_tok_counts.get(t,0):,} |\n")
        f.write("\n#### Sex tokens\n\n| token | count |\n|---|---|\n")
        for t, c in sex_tok_counts.items():
            f.write(f"| `{t}` | {c:,} |\n")
        f.write("\n#### Ethnicity tokens\n\n| token | count |\n|---|---|\n")
        for t, c in eth_tok_counts.items():
            f.write(f"| `{t}` | {c:,} |\n")
        f.write("\n#### Language tokens\n\n| token | count |\n|---|---|\n")
        for t, c in lang_tok_counts.items():
            f.write(f"| `{t}` | {c:,} |\n")
        f.write("\n")

    print(f"[write] {report}")
    print("[done]")


if __name__ == "__main__":
    main()
