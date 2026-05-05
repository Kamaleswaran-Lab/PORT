"""
Stream A Step 3: Hierarchical LLM ATC classification.

Methodology (Chen et al., NAACL 2025, arXiv:2412.07743):
  - Traverse ATC levels 1→5
  - At each level, prompt LLM with drug name + list of valid children of current parent
  - Parse LLM response, constrain to valid options (fallback to string-match if invalid)
  - Knowledge grounding: each option presented with its ATC generic name

We adapt for our pediatric cohort with dose-aware CoT per Xu et al. (JMIR AI 2025):
  - Include drug dose range (from token name) in prompt for polytherapeutic drugs

This script uses vLLM for batch inference. Model: Llama 3.1 8B Instruct (already downloaded).

Usage:
  python 03_llm_classify_atc.py --mode pilot --n 100   # validation accuracy
  python 03_llm_classify_atc.py --mode full            # classify all unmapped drugs
"""
import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd

OUT = Path("experiments/vocab/outputs")
LOG = Path("experiments/vocab/logs")

MODEL_PATHS = {
    "Llama-3.1-8B-Instruct":  "${HF_HOME}/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots",
    "Llama-3.3-70B-Instruct": "${HF_HOME}/models--meta-llama--Llama-3.3-70B-Instruct/snapshots",
}


# ─── Prompt templates ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a pharmacology expert specializing in the Anatomical Therapeutic Chemical (ATC) classification system. Classify the given drug by selecting exactly one code from the list of options. Respond with only the ATC code, nothing else."""

def build_level_prompt(drug_name, level, options, dose_context=""):
    """Build a prompt for one ATC level.

    options: list of (code, name) tuples
    """
    opts_str = "\n".join([f"  {code}: {name}" for code, name in options])
    dose_line = f"\nDose context: {dose_context}" if dose_context else ""

    prompt = f"""Drug: {drug_name}{dose_line}

Classify this drug into one of the following ATC level {level} categories:
{opts_str}

Respond with only the ATC code from the list above. No explanation."""
    return prompt


def extract_atc_code(response, valid_codes):
    """Parse LLM response; fall back to string-match against valid codes if parse fails."""
    resp = response.strip().upper()
    # Try exact match first
    for code in valid_codes:
        if code.upper() == resp:
            return code
    # Try substring match (LLM may have added whitespace or brackets)
    for code in valid_codes:
        if code.upper() in resp:
            return code
    # Give up
    return None


# ─── Main classification loop ──────────────────────────────────────────────

def classify_drug(llm, drug_name, atc_tree, sampling_params, max_level=5):
    """Classify one drug via hierarchical LLM prompting.

    Returns: dict with L1-L5 predictions + full final code, or None if failed early.
    """
    predictions = {}
    current = "ROOT"

    for level in range(1, max_level + 1):
        options = atc_tree.get(current, [])
        if not options:
            break

        # Single-option shortcut: no LLM call needed
        if len(options) == 1:
            code, name = options[0]
            predictions[f"L{level}"] = code
            current = code
            continue

        # Build prompt
        prompt_text = build_level_prompt(
            drug_name,
            level,
            options,
            dose_context="",  # no dose for now; can extend later
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]

        # Generate
        outputs = llm.chat(messages, sampling_params)
        response = outputs[0].outputs[0].text

        # Parse & validate
        valid_codes = [c for c, _ in options]
        chosen = extract_atc_code(response, valid_codes)

        if chosen is None:
            # LLM output invalid; stop here, return partial
            predictions[f"L{level}_raw"] = response
            break

        predictions[f"L{level}"] = chosen
        current = chosen

    # Final full code = the deepest level reached
    predictions["final_atc"] = current if current != "ROOT" else None
    return predictions


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--n", type=int, default=100, help="# drugs for pilot")
    ap.add_argument("--model", default="Llama-3.1-8B-Instruct",
                    choices=list(MODEL_PATHS.keys()))
    ap.add_argument("--tensor-parallel", type=int, default=1)
    args = ap.parse_args()

    # Locate model
    snap_root = Path(MODEL_PATHS[args.model])
    snapshots = [p for p in snap_root.glob("*") if p.is_dir()] if snap_root.exists() else []
    if not snapshots:
        print(f"ERROR: {args.model} not found at {snap_root}")
        return
    model_dir = snapshots[0]
    print(f"Model: {args.model}  |  dir: {model_dir}  |  TP={args.tensor_parallel}")

    # Load ATC tree
    with open(OUT / "atc_tree.json") as f:
        atc_tree = json.load(f)

    # Load drugs to classify
    if args.mode == "pilot":
        val = pd.read_csv(OUT / "atc_validation_set.csv")
        sample = val.sort_values("frequency", ascending=False).head(args.n)
        print(f"Pilot mode: {len(sample)} drugs from validation set")
    else:
        sample = pd.read_csv(OUT / "atc_unmapped_drugs.csv")
        print(f"Full mode: {len(sample)} unmapped drugs")

    # Load LLM
    from vllm import LLM, SamplingParams
    print(f"Loading {args.model}...")
    t0 = time.time()
    llm = LLM(
        model=str(model_dir),
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        tensor_parallel_size=args.tensor_parallel,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=50, top_p=1.0)
    print(f"Loaded in {time.time()-t0:.0f}s")

    # Classify each drug
    results = []
    t0 = time.time()
    for i, row in sample.iterrows():
        drug = row["normalized_name"]
        res = classify_drug(llm, drug, atc_tree, sampling_params)
        row_out = {
            "v3_med_code": row["v3_med_code"],
            "normalized_name": drug,
            "frequency": row.get("frequency", row.get("count", 0)),
        }
        if args.mode == "pilot":
            row_out["gold_atc"] = row["gold_atc"]
        row_out.update(res)
        results.append(row_out)
        if (i+1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(sample)} drugs, {elapsed:.0f}s elapsed, {elapsed/(i+1):.1f}s/drug")

    res_df = pd.DataFrame(results)
    model_tag = args.model.replace("Meta-", "").replace("-Instruct", "").replace(".", "p")
    if args.mode == "pilot":
        outfile = OUT / f"atc_pilot_{model_tag}_n{args.n}.csv"
    else:
        outfile = OUT / f"atc_full_{model_tag}.csv"
    res_df.to_csv(outfile, index=False)
    print(f"Saved: {outfile}")

    # Pilot accuracy
    if args.mode == "pilot":
        gold = res_df["gold_atc"]
        for L in [1, 2, 3, 4, 5]:
            pred = res_df.get(f"L{L}", pd.Series([None]*len(res_df)))
            gold_L = gold.str[:[1, 3, 4, 5, 7][L-1]]
            correct = (pred == gold_L).sum()
            print(f"  L{L} accuracy: {correct}/{len(res_df)} ({100*correct/len(res_df):.1f}%)")

    print("Done.")


if __name__ == "__main__":
    main()
