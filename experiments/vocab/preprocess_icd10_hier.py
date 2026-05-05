"""
Stream B — tokenization overhaul: ICD-10 hierarchical decomposition.

Reads vocab and code_counts. For every DIAGNOSIS//ICD10//..., PROBLEM//ICD10//...,
and ENCOUNTER//CARDIOLOGY//ICD10//... code, strip optional //END suffix, remove the
dot, and decompose into up to three hierarchical tokens following ETHOS convention:

    raw_code -> [0:3] [3:6] [6:]
    -> ICD//CM//{part1}            (always)
    -> ICD//CM//3-6//{part2}       (if non-empty)
    -> ICD//CM//SFX//{part3}       (if non-empty)

If the original vocab entry carried //END (problem resolution), a single shared
marker token PROBLEM//END is emitted in addition to the decomposed tokens.

Shared vocab across all three sources (no source prefix in hierarchical tokens).

Outputs:
    outputs/icd10_hier_vocab.csv  — sorted unique hierarchical vocab
    logs/B_icd10_hier_report.md   — validation report
"""

from __future__ import annotations

import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("experiments/vocab")
VOCAB_FILE = Path("/path/to/CHD_MEDS/tokenized_v3/train/vocab_t47691.csv")
COUNTS_FILE = Path("/path/to/CHD_MEDS/tokenized_v3/train/code_counts.csv")

OUT_VOCAB = ROOT / "outputs" / "icd10_hier_vocab.csv"
OUT_REPORT = ROOT / "logs" / "B_icd10_hier_report.md"

ICD_SOURCE_PREFIXES = (
    "DIAGNOSIS//ICD10//",
    "PROBLEM//ICD10//",
    "ENCOUNTER//CARDIOLOGY//ICD10//",
)
END_SUFFIX = "//END"
END_MARKER = "PROBLEM//END"


def read_vocab(path: Path) -> list[str]:
    tokens: list[str] = []
    with path.open() as f:
        for line in f:
            tok = line.strip()
            if tok:
                tokens.append(tok)
    return tokens


def read_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            counts[row["code"]] = int(row["count"])
    return counts


def is_icd_source(token: str) -> bool:
    return any(token.startswith(p) for p in ICD_SOURCE_PREFIXES)


def strip_source_and_end(token: str) -> tuple[str, bool]:
    """Return (raw_icd_code_with_dot, had_end_suffix).

    had_end_suffix True iff original token ended with //END.
    raw code keeps its dot as it appears in source (e.g., 'Z78.9', 'IMO0001', 'Q21.0').
    """
    had_end = token.endswith(END_SUFFIX)
    if had_end:
        token = token[: -len(END_SUFFIX)]
    for p in ICD_SOURCE_PREFIXES:
        if token.startswith(p):
            return token[len(p) :], had_end
    # should not happen because we filtered upstream
    return token, had_end


def decompose(raw_code: str) -> list[str]:
    """Decompose a raw ICD-10 code into up to 3 hierarchical tokens.

    Dots are removed before slicing (match ETHOS MIMIC dotless convention).
    Returns list with 1-3 tokens depending on length.
    """
    flat = raw_code.replace(".", "")
    part1 = flat[0:3]
    part2 = flat[3:6]
    part3 = flat[6:]

    out: list[str] = []
    if part1:
        out.append(f"ICD//CM//{part1}")
    if part2:
        out.append(f"ICD//CM//3-6//{part2}")
    if part3:
        out.append(f"ICD//CM//SFX//{part3}")
    return out


def main() -> None:
    OUT_VOCAB.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    all_vocab = read_vocab(VOCAB_FILE)
    counts = read_counts(COUNTS_FILE)

    # Filter ICD tokens (inputs to our decomposition)
    icd_tokens = [t for t in all_vocab if is_icd_source(t)]

    # Per-source bucket counts (for report)
    source_counts = Counter()
    for t in icd_tokens:
        for p in ICD_SOURCE_PREFIXES:
            if t.startswith(p):
                source_counts[p] += 1
                break

    # Decompose every token and track outputs + expansions
    expansions: list[int] = []  # tokens emitted per input code
    parse_failures: list[str] = []
    parent_children: dict[str, set[str]] = defaultdict(set)  # ICD//CM//{part1} -> {raw_code}
    sample_conversions: list[tuple[str, list[str]]] = []
    seen_end = False

    output_vocab: set[str] = set()

    # Sort inputs deterministically for sampling
    icd_tokens_sorted = sorted(icd_tokens)

    # Collect 10 representative samples spanning length/source diversity
    sample_buckets: dict[str, list[str]] = {
        "short_3": [],      # Z78.9
        "medium_5": [],     # J45.909
        "long_7": [],       # (rare) 8+ char post-dot
        "with_end": [],
        "encounter": [],
        "imo": [],
    }

    for tok in icd_tokens_sorted:
        raw, had_end = strip_source_and_end(tok)
        pieces = decompose(raw)
        if not pieces:
            parse_failures.append(tok)
            continue

        emitted: list[str] = list(pieces)
        if had_end:
            emitted.append(END_MARKER)
            seen_end = True

        # Bucket the sample
        flat = raw.replace(".", "")
        if tok.startswith("ENCOUNTER//CARDIOLOGY//ICD10//") and len(sample_buckets["encounter"]) < 2:
            sample_buckets["encounter"].append(tok)
        elif "IMO" in raw and len(sample_buckets["imo"]) < 1:
            sample_buckets["imo"].append(tok)
        elif had_end and len(sample_buckets["with_end"]) < 2:
            sample_buckets["with_end"].append(tok)
        elif len(flat) <= 3 and len(sample_buckets["short_3"]) < 2:
            sample_buckets["short_3"].append(tok)
        elif 4 <= len(flat) <= 5 and len(sample_buckets["medium_5"]) < 2:
            sample_buckets["medium_5"].append(tok)
        elif len(flat) >= 7 and len(sample_buckets["long_7"]) < 1:
            sample_buckets["long_7"].append(tok)

        output_vocab.update(emitted)
        expansions.append(len(emitted) - (1 if had_end else 0))  # ignore END from expansion ratio
        parent_children[pieces[0]].add(raw)

    # Build the 10 sample conversions from buckets (pad with remaining if short)
    picked: list[str] = []
    for toks in sample_buckets.values():
        picked.extend(toks)
    # fill to 10 from sorted input
    i = 0
    while len(picked) < 10 and i < len(icd_tokens_sorted):
        t = icd_tokens_sorted[i]
        if t not in picked:
            picked.append(t)
        i += 1
    picked = picked[:10]
    for t in picked:
        raw, had_end = strip_source_and_end(t)
        pieces = decompose(raw)
        out = list(pieces) + ([END_MARKER] if had_end else [])
        sample_conversions.append((t, out))

    # Sort and write output vocab
    sorted_vocab = sorted(output_vocab)
    with OUT_VOCAB.open("w") as f:
        f.write("token\n")
        for t in sorted_vocab:
            f.write(t + "\n")

    # Compute stats
    n_in = len(icd_tokens)
    n_out = len(sorted_vocab)
    reduction_ratio = n_in / n_out if n_out > 0 else float("nan")

    mean_exp = statistics.mean(expansions) if expansions else 0.0
    median_exp = statistics.median(expansions) if expansions else 0.0
    max_exp = max(expansions) if expansions else 0

    # Top-5 most-shared parent tokens (by number of distinct child raw codes)
    top_parents = sorted(parent_children.items(), key=lambda kv: -len(kv[1]))[:5]

    # Count hierarchical token type breakdown
    n_part1 = sum(1 for t in sorted_vocab if t.startswith("ICD//CM//") and not t.startswith(("ICD//CM//3-6//", "ICD//CM//SFX//")))
    n_part2 = sum(1 for t in sorted_vocab if t.startswith("ICD//CM//3-6//"))
    n_part3 = sum(1 for t in sorted_vocab if t.startswith("ICD//CM//SFX//"))
    n_end = 1 if END_MARKER in output_vocab else 0

    # Write markdown report
    lines: list[str] = []
    lines.append("# Stream B — ICD-10 Hierarchical Decomposition Validation Report\n")
    lines.append(f"Generated from: `{VOCAB_FILE}`\n")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("Decomposes flat ICD-10 tokens (DIAGNOSIS, PROBLEM, ENCOUNTER//CARDIOLOGY) into")
    lines.append("shared hierarchical tokens following ETHOS `process_icd10` convention:")
    lines.append("`ICD//CM//{part1}` + optional `ICD//CM//3-6//{part2}` + optional `ICD//CM//SFX//{part3}`.")
    lines.append("The `//END` problem-resolution marker is preserved as a single shared `PROBLEM//END` token.")
    lines.append("")
    lines.append("## 1. Input / Output Vocabulary Counts")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Input ICD tokens (total) | {n_in:,} |")
    lines.append(f"| &nbsp;&nbsp;DIAGNOSIS//ICD10//* | {source_counts['DIAGNOSIS//ICD10//']:,} |")
    lines.append(f"| &nbsp;&nbsp;PROBLEM//ICD10//* (incl. //END) | {source_counts['PROBLEM//ICD10//']:,} |")
    lines.append(f"| &nbsp;&nbsp;ENCOUNTER//CARDIOLOGY//ICD10//* | {source_counts['ENCOUNTER//CARDIOLOGY//ICD10//']:,} |")
    lines.append(f"| Output hierarchical unique tokens | {n_out:,} |")
    lines.append(f"| &nbsp;&nbsp;ICD//CM//{{part1}} | {n_part1:,} |")
    lines.append(f"| &nbsp;&nbsp;ICD//CM//3-6//{{part2}} | {n_part2:,} |")
    lines.append(f"| &nbsp;&nbsp;ICD//CM//SFX//{{part3}} | {n_part3:,} |")
    lines.append(f"| &nbsp;&nbsp;PROBLEM//END marker | {n_end} |")
    lines.append(f"| **Reduction ratio (input / unique output)** | **{reduction_ratio:.2f}×** |")
    lines.append(f"| Parse failures | {len(parse_failures)} |")
    lines.append("")
    lines.append("## 2. Token Expansion per Input Code")
    lines.append("")
    lines.append("How many hierarchical tokens are emitted per input code (excluding `PROBLEM//END`):")
    lines.append("")
    lines.append("| Statistic | Value |")
    lines.append("|---|---|")
    lines.append(f"| Mean | {mean_exp:.3f} |")
    lines.append(f"| Median | {median_exp:.1f} |")
    lines.append(f"| Max | {max_exp} |")
    exp_hist = Counter(expansions)
    for k in sorted(exp_hist):
        lines.append(f"| &nbsp;&nbsp;tokens/code = {k} | {exp_hist[k]:,} codes |")
    lines.append("")
    lines.append("## 3. Top-5 Most-Shared Parent Tokens")
    lines.append("")
    lines.append("Parent tokens (`ICD//CM//{part1}`) grouped across all three sources —")
    lines.append("each raw child code collapses onto one shared parent.")
    lines.append("")
    lines.append("| Parent | # distinct raw children | Example children |")
    lines.append("|---|---|---|")
    for parent, children in top_parents:
        example = ", ".join(sorted(children)[:6])
        lines.append(f"| `{parent}` | {len(children)} | `{example}` … |")
    lines.append("")
    lines.append("## 4. Sample Conversions (10 examples)")
    lines.append("")
    lines.append("| Input token | → | Output tokens |")
    lines.append("|---|---|---|")
    for inp, out in sample_conversions:
        lines.append(f"| `{inp}` | → | " + " + ".join(f"`{o}`" for o in out) + " |")
    lines.append("")
    lines.append("## 5. Parse Failures")
    lines.append("")
    if parse_failures:
        lines.append(f"{len(parse_failures)} tokens failed to decompose:")
        lines.append("")
        for f in parse_failures[:25]:
            lines.append(f"- `{f}`")
    else:
        lines.append("None — every ICD source token decomposed to ≥ 1 hierarchical token.")
    lines.append("")
    lines.append("## 6. Notes on Methodology")
    lines.append("")
    lines.append("- Dots are stripped before slicing to match ETHOS MIMIC dotless convention.")
    lines.append("- `//END` suffix (problem-list resolution marker) is detached before decomposition")
    lines.append("  and re-added as a single shared `PROBLEM//END` marker token appended after the")
    lines.append("  decomposed tokens.")
    lines.append("- No source prefix retained: one shared ICD-10-CM vocab regardless of whether the")
    lines.append("  code originated in DIAGNOSIS, PROBLEM, or ENCOUNTER//CARDIOLOGY. Temporal source")
    lines.append("  context is preserved at event ordering level.")
    lines.append("- Unusual codes handled:")
    lines.append("  - `Z3A.xx`, `D3A.xxx`: letter in pos 3 → part1 keeps letter (e.g., `Z3A`).")
    lines.append("  - `IMO00xx` (Intelligent Medical Objects placeholders): slice mechanically →")
    lines.append("    part1=`IMO`, part2=`000`, part3=`1/2/…`.")
    lines.append("  - `UNKNOWN` (ENCOUNTER//CARDIOLOGY placeholder): part1=`UNK`, part2=`NOW`,")
    lines.append("    part3=`N`. Preserved as-is; a real pipeline integration may choose to filter.")
    lines.append("")
    lines.append("## 7. Outputs")
    lines.append("")
    lines.append(f"- Sorted unique vocab: `{OUT_VOCAB}`")
    lines.append(f"- This report: `{OUT_REPORT}`")
    lines.append("")

    with OUT_REPORT.open("w") as f:
        f.write("\n".join(lines))

    # Console summary (so srun log captures it)
    print(f"[B] ICD source tokens in:   {n_in:,}")
    print(f"[B]   DIAGNOSIS:             {source_counts['DIAGNOSIS//ICD10//']:,}")
    print(f"[B]   PROBLEM:               {source_counts['PROBLEM//ICD10//']:,}")
    print(f"[B]   ENCOUNTER//CARDIO:     {source_counts['ENCOUNTER//CARDIOLOGY//ICD10//']:,}")
    print(f"[B] Hierarchical vocab out: {n_out:,}")
    print(f"[B]   part1:                 {n_part1:,}")
    print(f"[B]   3-6 (part2):           {n_part2:,}")
    print(f"[B]   SFX (part3):           {n_part3:,}")
    print(f"[B]   PROBLEM//END marker:   {n_end}")
    print(f"[B] Reduction ratio:        {reduction_ratio:.2f}x")
    print(f"[B] Tokens per code: mean={mean_exp:.3f} median={median_exp} max={max_exp}")
    print(f"[B] Parse failures:         {len(parse_failures)}")
    print(f"[B] Wrote vocab -> {OUT_VOCAB}")
    print(f"[B] Wrote report -> {OUT_REPORT}")


if __name__ == "__main__":
    main()
