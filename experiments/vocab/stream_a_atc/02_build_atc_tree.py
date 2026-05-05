"""
Stream A Step 2: Build ATC ontology tree for hierarchical LLM prompting.

Per Chen 2024 (arXiv:2412.07743):
  - Traverse ATC 5 levels: L1 (1 char) → L2 (3 chars) → L3 (4 chars) → L4 (5 chars) → L5 (7 chars)
  - At each level, constrain LLM answer to valid children of previously chosen parent
  - Present each candidate with its generic name as definition (knowledge grounding)

ATC code structure:
  Full code:   C07AB03
  L1:          C           (main anatomical group: cardiovascular)
  L2:          C07         (therapeutic subgroup: beta blockers)
  L3:          C07A        (pharmacological subgroup: selective)
  L4:          C07AB       (chemical subgroup: beta-1 selective w/o intrinsic sympathomimetic)
  L5:          C07AB03     (chemical substance: metoprolol)

We output a tree dict: {parent_code: [(child_code, child_name), ...]}

Outputs:
  outputs/atc_tree.json — full hierarchical tree
  outputs/atc_level_counts.csv — # codes per level
"""
import pandas as pd
import json
from collections import defaultdict
from pathlib import Path

OUT = Path("experiments/vocab/outputs")
LOG = Path("experiments/vocab/logs")

print("Loading ATC ontology...")
df = pd.read_csv(
    "/path/to/ethos-ares/src/ethos/tokenize/maps/atc_coding.csv.gz"
)
df = df.drop_duplicates(subset="atc_code", keep="last").reset_index(drop=True)
print(f"  Loaded: {len(df):,} codes")

# Classify each code by length → ATC level
def atc_level(code):
    n = len(code)
    if n == 1: return 1    # e.g., "C"
    if n == 3: return 2    # e.g., "C07"
    if n == 4: return 3    # e.g., "C07A"
    if n == 5: return 4    # e.g., "C07AB"
    if n == 7: return 5    # e.g., "C07AB03"
    return None

df["level"] = df["atc_code"].str.len().map({1:1, 3:2, 4:3, 5:4, 7:5})
lvl_counts = df["level"].value_counts().sort_index()
print(f"  Per-level counts:\n{lvl_counts.to_string()}")

# Build parent→children index
# Parent of a level-L code = the longest prefix that exists in the dictionary
def parent_of(code):
    n = len(code)
    if n == 1: return None     # L1 root
    if n == 3: return code[:1]     # L2→L1
    if n == 4: return code[:3]     # L3→L2
    if n == 5: return code[:4]     # L4→L3
    if n == 7: return code[:5]     # L5→L4
    return None

df["parent"] = df["atc_code"].apply(parent_of)

# Tree: {parent_code_or_ROOT: [(child_code, child_name), ...]}
tree = defaultdict(list)
for _, r in df.iterrows():
    key = r["parent"] if r["parent"] is not None else "ROOT"
    tree[key].append((r["atc_code"], r["atc_name"]))

# Stats: mean children per parent at each level
print("\nTree structure:")
for L in range(1, 5):
    parents_at_L = df[df["level"] == L]["atc_code"].tolist()
    fanout = [len(tree.get(p, [])) for p in parents_at_L]
    if fanout:
        import statistics
        print(f"  L{L} → L{L+1}:  {len(parents_at_L):>4d} parents,  mean fanout {statistics.mean(fanout):4.1f},  median {statistics.median(fanout):4.1f},  max {max(fanout)}")

# Save tree
tree_serializable = {k: v for k, v in tree.items()}
with open(OUT / "atc_tree.json", "w") as f:
    json.dump(tree_serializable, f, indent=2)
print(f"\nSaved ATC tree: {OUT/'atc_tree.json'}")

# Level counts CSV
lvl_counts.to_csv(OUT / "atc_level_counts.csv", header=["count"])

# Full flat dict for reverse lookup (code → name)
atc_name_lookup = dict(zip(df["atc_code"], df["atc_name"]))
with open(OUT / "atc_name_lookup.json", "w") as f:
    json.dump(atc_name_lookup, f, indent=2)

# Quick validation: pick a known ATC chain and walk it
example_chain = ["C", "C07", "C07A", "C07AB", "C07AB03"]
print("\nExample chain traversal (metoprolol, C07AB03):")
for code in example_chain:
    name = atc_name_lookup.get(code, "NOT FOUND")
    siblings = tree.get(parent_of(code) or "ROOT", [])
    print(f"  {code} = {name}   (siblings at this level: {len(siblings)})")

print("Done.")
