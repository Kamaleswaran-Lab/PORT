"""
Post-process fix: apply top-N cutoff to PROBLEM//NAME// and DIAGNOSIS//MEDHX// free-text codes.
These were missed in the initial integration (only PROBLEM//ICD10// and DIAGNOSIS//ICD10// were decomposed).

Input:  /path/to/CHD_MEDS/ethos_input/{train,val,test}/shard_*.parquet
Output: overwrites in-place with fixed codes
"""
import pandas as pd
import polars as pl
from pathlib import Path

IN = Path("/path/to/CHD_MEDS/ethos_input")
OUT = Path("experiments/vocab/outputs")

# Build top-N cutoffs from train split frequency
print("Building cutoff vocabs from train split...")
train_counts = {}
for shard in sorted((IN / "train").glob("shard_*.parquet")):
    df = pl.read_parquet(shard, columns=["code"])
    for prefix in ["PROBLEM//NAME//", "DIAGNOSIS//MEDHX//"]:
        sub = df.filter(pl.col("code").str.starts_with(prefix))
        counts = sub.group_by("code").agg(pl.len().alias("n"))
        for c, n in zip(counts["code"].to_list(), counts["n"].to_list()):
            train_counts.setdefault(prefix, {})
            train_counts[prefix][c] = train_counts[prefix].get(c, 0) + n

TOP_N = {"PROBLEM//NAME//": 200, "DIAGNOSIS//MEDHX//": 200}
ALLOWED = {}
for prefix, N in TOP_N.items():
    freq = train_counts.get(prefix, {})
    top = sorted(freq.items(), key=lambda x: -x[1])[:N]
    ALLOWED[prefix] = set(c for c, _ in top)
    total_codes = len(freq)
    top_events = sum(n for _, n in top)
    total_events = sum(freq.values()) or 1
    print(f"  {prefix:<25s} {total_codes} codes -> top-{N} ({100*top_events/total_events:.1f}% events)")

# Save vocabs
for prefix, codes in ALLOWED.items():
    name = prefix.replace("//", "_").strip("_").lower()
    with open(OUT / f"{name}_top200_vocab.csv", "w") as f:
        f.write("code\n")
        for c in sorted(codes):
            f.write(c + "\n")

# Remap function
def remap(code):
    # Handle //END suffix — strip, test, re-append
    for prefix in TOP_N:
        if code.startswith(prefix):
            had_end = code.endswith("//END")
            base = code[:-5] if had_end else code
            if base in ALLOWED[prefix]:
                return code  # keep as-is
            fallback = prefix.rstrip("/") + "//OTHER"
            return fallback + ("//END" if had_end else "")
    return code

# Process each shard in-place
print("\nProcessing shards...")
for split in ["train", "val", "test"]:
    for shard in sorted((IN / split).glob("shard_*.parquet")):
        df = pl.read_parquet(shard)
        # Apply remap via python map — polars map_elements
        df = df.with_columns(pl.col("code").map_elements(remap, return_dtype=pl.String).alias("code"))
        df.write_parquet(shard)
        print(f"  {split}/{shard.name} done ({df.height:,} rows)")

print("\nVerification — count unique codes by family after fix:")
all_codes = set()
for split in ["train", "val", "test"]:
    for shard in sorted((IN / split).glob("shard_*.parquet")):
        df = pl.read_parquet(shard, columns=["code"])
        all_codes |= set(df["code"].unique().to_list())

fam_counts = {}
for c in all_codes:
    fam = c.split("//")[0]
    fam_counts[fam] = fam_counts.get(fam, 0) + 1

for fam, n in sorted(fam_counts.items(), key=lambda x: -x[1]):
    print(f"  {fam:<30s} {n:>5,d}")
print(f"\nTotal unique codes: {len(all_codes):,}")
