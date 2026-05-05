"""
Stream C: vocab cutoff preprocessing.

For each high-cardinality category in the previous tokenization vocab, keep the top-N
codes by training-set frequency and replace all dropped codes with a single
fallback token `{CATEGORY}//OTHER`.

Inputs:
    /path/to/CHD_MEDS/tokenized_v3/train/code_counts.csv
Outputs:
    experiments/vocab/outputs/{lab_top500,procedure_top500,
        encounter_an_primary_top500,sde_top300}_vocab.csv
    experiments/vocab/logs/C_cutoffs_report.md

This is a *vocab-only* step. Tokenized shards are NOT touched here.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CODE_COUNTS = Path("/path/to/CHD_MEDS/tokenized_v3/train/code_counts.csv")
ROOT = Path("experiments/vocab")
OUT_DIR = ROOT / "outputs"
LOG_DIR = ROOT / "logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# (prefix, top_n, fallback_token, output_filename, human_label)
CATEGORIES = [
    ("LAB//", 500, "LAB//OTHER",
     "lab_top500_vocab.csv", "LAB"),
    ("PROCEDURE//", 500, "PROCEDURE//SURG_HX//OTHER",
     "procedure_top500_vocab.csv", "PROCEDURE"),
    ("ENCOUNTER//AN//PRIMARY_PROCEDURE//", 500,
     "ENCOUNTER//AN//PRIMARY_PROCEDURE//OTHER",
     "encounter_an_primary_top500_vocab.csv",
     "ENCOUNTER//AN//PRIMARY_PROCEDURE"),
    ("SDE//", 300, "SDE//OTHER",
     "sde_top300_vocab.csv", "SDE"),
]


def select_top_n(df: pd.DataFrame, prefix: str, top_n: int):
    sub = df[df["code"].str.startswith(prefix)].copy()
    sub = sub.sort_values("count", ascending=False).reset_index(drop=True)
    total = sub["count"].sum()
    kept = sub.head(top_n).copy()
    kept["cumulative_pct"] = 100.0 * kept["count"].cumsum() / total
    dropped = sub.iloc[top_n:]
    return sub, kept, dropped, total


def procedure_name(code: str, prefix: str) -> str:
    """Strip prefix and (for PROCEDURE) the SURG_HX// middle segment."""
    name = code[len(prefix):]
    # PROCEDURE codes are PROCEDURE//SURG_HX//<NAME>
    # so when prefix='PROCEDURE//', name is 'SURG_HX//<NAME>'
    if name.startswith("SURG_HX//"):
        name = name[len("SURG_HX//"):]
    return name


def main():
    df = pd.read_csv(CODE_COUNTS)

    # --- Per-category cutoff ---
    summaries = []
    per_cat_tables = {}
    for prefix, top_n, fallback, fname, label in CATEGORIES:
        sub, kept, dropped, total = select_top_n(df, prefix, top_n)

        # append fallback row: count = sum of dropped counts
        fallback_count = int(dropped["count"].sum())
        fb_row = pd.DataFrame(
            [{"code": fallback,
              "count": fallback_count,
              "cumulative_pct": 100.0}]
        )
        out = pd.concat([kept, fb_row], ignore_index=True)
        out.to_csv(OUT_DIR / fname, index=False)

        retained = kept["count"].sum()
        summaries.append({
            "label": label,
            "prefix": prefix,
            "input_tokens": len(sub),
            "output_tokens": len(out),           # top-N + 1 (OTHER)
            "total_events": int(total),
            "retained_events": int(retained),
            "retained_pct": 100.0 * retained / total if total else 0.0,
            "dropped_tokens": len(dropped),
            "fallback_count": fallback_count,
            "fallback": fallback,
        })
        per_cat_tables[label] = {
            "all": sub, "kept": kept, "dropped": dropped,
            "total": total, "prefix": prefix,
        }

    # --- PROCEDURE vs ENCOUNTER//AN//PRIMARY_PROCEDURE name overlap ---
    proc_all = per_cat_tables["PROCEDURE"]["all"].copy()
    enc_all = per_cat_tables["ENCOUNTER//AN//PRIMARY_PROCEDURE"]["all"].copy()

    proc_all["name"] = proc_all["code"].apply(
        lambda c: procedure_name(c, "PROCEDURE//"))
    enc_all["name"] = enc_all["code"].apply(
        lambda c: procedure_name(c, "ENCOUNTER//AN//PRIMARY_PROCEDURE//"))

    # PROCEDURE uses "/" inside names (e.g. TONSILLECTOMY/ADENOIDECTOMY)
    # ENCOUNTER uses "_" (TONSILLECTOMY_ADENOIDECTOMY). Normalize for matching.
    def norm(s):
        return s.replace("/", "_").upper()

    proc_all["name_norm"] = proc_all["name"].map(norm)
    enc_all["name_norm"] = enc_all["name"].map(norm)

    overlap_names = set(proc_all["name_norm"]) & set(enc_all["name_norm"])
    merged = (
        proc_all[proc_all["name_norm"].isin(overlap_names)][
            ["name_norm", "code", "count"]]
        .rename(columns={"code": "procedure_code", "count": "procedure_count"})
        .merge(
            enc_all[enc_all["name_norm"].isin(overlap_names)][
                ["name_norm", "code", "count"]]
            .rename(columns={"code": "encounter_code",
                             "count": "encounter_count"}),
            on="name_norm",
            how="inner",
        )
    )
    merged["total"] = merged["procedure_count"] + merged["encounter_count"]
    merged = merged.sort_values("total", ascending=False).reset_index(drop=True)

    # --- SDE subtype breakdown ---
    sde_kept = per_cat_tables["SDE"]["kept"].copy()

    def sde_subtype(code: str) -> str:
        # SDE//PRE//..., SDE//POST//..., SDE//PROC//..., else OTHER
        rest = code[len("SDE//"):]
        head = rest.split("//", 1)[0] if "//" in rest else rest
        if head in ("PRE", "POST", "PROC"):
            return head
        return f"OTHER({head})"

    sde_kept["subtype"] = sde_kept["code"].map(sde_subtype)
    sde_breakdown = (
        sde_kept.groupby("subtype")
        .agg(tokens=("code", "count"), events=("count", "sum"))
        .reset_index()
        .sort_values("events", ascending=False)
    )

    # --- Write report ---
    lines = []
    lines.append("# Stream C — Vocab Cutoff Report")
    lines.append("")
    lines.append(f"Source: `{CODE_COUNTS}`")
    lines.append("")
    lines.append("## 1. Per-category cutoff summary")
    lines.append("")
    lines.append("| Category | input_tokens | output_tokens | total_events | retained_events | retained_% | dropped_tokens | fallback_events |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in summaries:
        lines.append(
            f"| {s['label']} | {s['input_tokens']} | {s['output_tokens']} | "
            f"{s['total_events']:,} | {s['retained_events']:,} | "
            f"{s['retained_pct']:.2f}% | {s['dropped_tokens']} | "
            f"{s['fallback_count']:,} |"
        )
    lines.append("")

    for s in summaries:
        label = s["label"]
        kept = per_cat_tables[label]["kept"]
        lines.append(f"### {label} — top 10")
        lines.append("")
        lines.append("| rank | code | count | cum_% |")
        lines.append("|---|---|---|---|")
        for i, row in kept.head(10).iterrows():
            lines.append(f"| {i+1} | `{row['code']}` | {int(row['count']):,} | "
                         f"{row['cumulative_pct']:.2f}% |")
        lines.append("")
        lines.append(f"### {label} — bottom 10 of kept (cutoff boundary)")
        lines.append("")
        lines.append("| rank | code | count | cum_% |")
        lines.append("|---|---|---|---|")
        bottom = kept.tail(10)
        for i, row in bottom.iterrows():
            lines.append(f"| {i+1} | `{row['code']}` | {int(row['count']):,} | "
                         f"{row['cumulative_pct']:.2f}% |")
        lines.append("")
        dropped = per_cat_tables[label]["dropped"]
        if len(dropped):
            lines.append(f"### {label} — top 10 dropped codes (folded into "
                         f"`{s['fallback']}`)")
            lines.append("")
            lines.append("| code | count |")
            lines.append("|---|---|")
            for _, row in dropped.head(10).iterrows():
                lines.append(f"| `{row['code']}` | {int(row['count']):,} |")
            lines.append("")

    # --- Procedure overlap ---
    lines.append("## 2. PROCEDURE vs ENCOUNTER//AN//PRIMARY_PROCEDURE overlap")
    lines.append("")
    lines.append(f"- Distinct procedure names in PROCEDURE//: "
                 f"{proc_all['name_norm'].nunique()}")
    lines.append(f"- Distinct procedure names in "
                 f"ENCOUNTER//AN//PRIMARY_PROCEDURE//: "
                 f"{enc_all['name_norm'].nunique()}")
    lines.append(f"- Names appearing in BOTH categories: "
                 f"{len(overlap_names)}")
    lines.append(f"- PROCEDURE-only names: "
                 f"{proc_all['name_norm'].nunique() - len(overlap_names)}")
    lines.append(f"- ENCOUNTER-only names: "
                 f"{enc_all['name_norm'].nunique() - len(overlap_names)}")
    lines.append("")
    lines.append("Merge rationale: rejected — PROCEDURE = **surgical history** "
                 "(past procedures, patient-level durable record), ENCOUNTER//AN//"
                 "PRIMARY_PROCEDURE = **current-encounter** primary surgery. Name "
                 "string may match but semantic role differs; keeping separate.")
    lines.append("")
    lines.append("### Top 20 overlapping procedure names (by combined frequency)")
    lines.append("")
    lines.append("| name (normalized) | PROCEDURE count | ENCOUNTER count | combined |")
    lines.append("|---|---|---|---|")
    for _, row in merged.head(20).iterrows():
        lines.append(
            f"| `{row['name_norm']}` | {int(row['procedure_count']):,} | "
            f"{int(row['encounter_count']):,} | {int(row['total']):,} |"
        )
    lines.append("")

    # --- SDE subtype ---
    lines.append("## 3. SDE subtype breakdown of top-300")
    lines.append("")
    lines.append("| subtype | tokens | events |")
    lines.append("|---|---|---|")
    for _, row in sde_breakdown.iterrows():
        lines.append(
            f"| {row['subtype']} | {int(row['tokens'])} | "
            f"{int(row['events']):,} |"
        )
    lines.append("")

    # --- Output file manifest ---
    lines.append("## 4. Output files")
    lines.append("")
    for _, _, _, fname, _ in CATEGORIES:
        lines.append(f"- `outputs/{fname}`")
    lines.append("")

    (LOG_DIR / "C_cutoffs_report.md").write_text("\n".join(lines))

    # --- Console summary ---
    print("=== Stream C cutoff summary ===")
    for s in summaries:
        print(f"  {s['label']:40s} "
              f"{s['input_tokens']:5d} -> {s['output_tokens']:5d} "
              f"retained {s['retained_pct']:6.2f}%")
    print(f"  PROCEDURE/ENCOUNTER overlap names: {len(overlap_names)}")
    print(f"  Report: {LOG_DIR / 'C_cutoffs_report.md'}")


if __name__ == "__main__":
    main()
