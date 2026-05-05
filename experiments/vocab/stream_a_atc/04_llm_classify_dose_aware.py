"""
Stream A Step 4: Dose-aware hierarchical ATC classification.

Improvement over 03_llm_classify_atc.py:
  - Extract route/formulation/dose context from raw MED code
  - Include that context in the prompt (Xu 2025 style)
  - Helps disambiguate polytherapeutic drugs:
      fentanyl drip  → N01AH (anesthetic use)
      fentanyl bolus → N02AB (acute analgesic)
      aspirin 81 mg  → B01AC (antiplatelet)
      aspirin 650 mg → N02BA (analgesic)

Prompt structure (per level L):
  System: "You are a pharmacology expert specializing in ATC classification."
  User:   Drug: FENTANYL
          Route: continuous IV infusion (drip)
          Formulation: —
          Common doses: —

          Classify into one of the following ATC level {L} categories:
            <code>: <generic-name>
            ...
          Respond with only the ATC code.

Usage:
  sbatch with --gres=gpu:4 for 70B TP=4
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


# ─── Context extraction from raw MED code ────────────────────────────────

# Route/administration tokens
ROUTE_MAP = {
    "DRIP": "continuous IV infusion (drip)",
    "CONTINUOUS": "continuous IV infusion",
    "INFUSION": "continuous IV infusion",
    "BOLUS": "IV bolus (one-time acute dose)",
    "IV": "IV administration",
    "INTRAVENOUS": "IV administration",
    "IVPB": "IV piggyback (short infusion)",
    "ORAL_SOLUTION": "oral solution (liquid)",
    "ORAL_SUSPENSION": "oral suspension",
    "ORAL_LIQUID": "oral liquid",
    "ORAL": "oral",
    "TABLET": "oral tablet",
    "CAPSULE": "oral capsule",
    "SYRINGE": "IV syringe",
    "INJECTION": "injection",
    "SUBCUT": "subcutaneous",
    "SUBCUTANEOUS": "subcutaneous",
    "INTRAMUSCULAR": "intramuscular",
    "INHALATION": "inhaled",
    "NEBULIZER": "nebulized (inhaled)",
    "NEBULIZATION": "nebulized (inhaled)",
    "TOPICAL": "topical",
    "OPHTHALMIC": "ophthalmic (eye drop)",
    "OTIC": "otic (ear drop)",
    "TRANSDERMAL": "transdermal patch",
    "SUPPOSITORY": "rectal suppository",
    "RECTAL": "rectal",
    "NASAL": "nasal",
    "DROPS": "drops",
}

# Purpose/intent markers
PURPOSE_MAP = {
    "FLUSH": "line flush (not therapeutic dose)",
    "DILUTE": "diluted solution",
    "CONCENTRATE": "concentrate (must be diluted)",
    "IRRIGATION": "irrigation solution",
    "IRRIGATING": "irrigation solution",
    "PF": "preservative-free",
    "ADDITIVES": "IV fluid with additives",
}

DOSE_RE = re.compile(r"(\d+(?:\.\d+)?)_?(MG|MCG|UNIT|UNITS|G|GRAM|MEQ|PCT|MMOL|MMOL/L)(?:/(\d+)_?ML)?")


def extract_context(raw_code):
    """Parse MED code to extract route/form/dose context.

    Returns a dict with keys: drug_name (normalized), route, purpose, dose_strings
    """
    name = raw_code.replace("MED//", "", 1).upper()

    # Extract route/form tokens (order matters — more specific first)
    route_hits = []
    for tok, desc in ROUTE_MAP.items():
        if f"_{tok}" in f"_{name}_" or name.startswith(tok + "_") or name.endswith("_" + tok):
            route_hits.append(desc)
            name = name.replace(f"_{tok}_", "_").replace(f"_{tok}", "").replace(f"{tok}_", "")

    # Purpose markers
    purpose_hits = []
    for tok, desc in PURPOSE_MAP.items():
        if f"_{tok}" in f"_{name}_":
            purpose_hits.append(desc)

    # Dose extraction
    dose_hits = []
    for m in DOSE_RE.finditer(name):
        val, unit, per = m.group(1), m.group(2), m.group(3)
        if per:
            dose_hits.append(f"{val} {unit}/{per} ML")
        else:
            dose_hits.append(f"{val} {unit}")

    # Strip dose patterns + route tokens from name for clean drug name
    name = DOSE_RE.sub("", name)
    for tok in list(ROUTE_MAP.keys()) + list(PURPOSE_MAP.keys()):
        name = name.replace(f"_{tok}_", "_").replace(f"_{tok}", "").replace(f"{tok}_", "")
    name = re.sub(r"[_/]+", "_", name).strip("_")
    name = re.sub(r"^[\d_]+", "", name)  # strip leading digits

    return {
        "drug_name": name,
        "route": ", ".join(route_hits) if route_hits else None,
        "purpose": ", ".join(purpose_hits) if purpose_hits else None,
        "dose": "; ".join(dose_hits) if dose_hits else None,
    }


# ─── Prompts ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a pharmacology expert specializing in the Anatomical Therapeutic Chemical (ATC) classification system. Given a drug with clinical context (route, dose, formulation), select the single best ATC code from the provided options. The same active ingredient may have different ATC codes depending on indication and dosing — use the provided context to disambiguate. Respond with only the ATC code, nothing else."""


def build_prompt(ctx, level, options):
    context_lines = [f"Drug: {ctx['drug_name']}"]
    if ctx.get("route"):
        context_lines.append(f"Route/administration: {ctx['route']}")
    if ctx.get("dose"):
        context_lines.append(f"Typical dose: {ctx['dose']}")
    if ctx.get("purpose"):
        context_lines.append(f"Notes: {ctx['purpose']}")

    context_str = "\n".join(context_lines)
    opts_str = "\n".join([f"  {code}: {name}" for code, name in options])

    return f"""{context_str}

Classify this drug into one of the following ATC level {level} categories (use the route/dose context to disambiguate ambiguous drugs):
{opts_str}

Respond with only the ATC code from the list above. No explanation."""


def extract_atc_code(response, valid_codes):
    resp = response.strip().upper()
    for code in valid_codes:
        if code.upper() == resp:
            return code
    for code in valid_codes:
        if code.upper() in resp:
            return code
    return None


# ─── Classification ─────────────────────────────────────────────────────────

def classify_drug(llm, ctx, atc_tree, sampling_params, max_level=5):
    predictions = {}
    current = "ROOT"

    for level in range(1, max_level + 1):
        options = atc_tree.get(current, [])
        if not options:
            break

        if len(options) == 1:
            code, _ = options[0]
            predictions[f"L{level}"] = code
            current = code
            continue

        prompt_text = build_prompt(ctx, level, options)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt_text},
        ]
        outputs = llm.chat(messages, sampling_params)
        response = outputs[0].outputs[0].text

        valid_codes = [c for c, _ in options]
        chosen = extract_atc_code(response, valid_codes)

        if chosen is None:
            predictions[f"L{level}_raw"] = response
            break

        predictions[f"L{level}"] = chosen
        current = chosen

    predictions["final_atc"] = current if current != "ROOT" else None
    return predictions


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model", default="Llama-3.3-70B-Instruct", choices=list(MODEL_PATHS.keys()))
    ap.add_argument("--tensor-parallel", type=int, default=4)
    args = ap.parse_args()

    snap_root = Path(MODEL_PATHS[args.model])
    snapshots = [p for p in snap_root.glob("*") if p.is_dir()]
    if not snapshots:
        print(f"ERROR: {args.model} not found at {snap_root}")
        return
    model_dir = snapshots[0]
    print(f"Model: {args.model}  |  dir: {model_dir}  |  TP={args.tensor_parallel}")

    with open(OUT / "atc_tree.json") as f:
        atc_tree = json.load(f)

    if args.mode == "pilot":
        val = pd.read_csv(OUT / "atc_validation_set.csv")
        sample = val.sort_values("frequency", ascending=False).head(args.n).reset_index(drop=True)
        print(f"Pilot: {len(sample)} drugs from validation set")
    else:
        sample = pd.read_csv(OUT / "atc_unmapped_drugs.csv").reset_index(drop=True)
        print(f"Full: {len(sample)} unmapped drugs")

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

    results = []
    t0 = time.time()
    for i, row in sample.iterrows():
        ctx = extract_context(row["v3_med_code"])
        res = classify_drug(llm, ctx, atc_tree, sampling_params)
        row_out = {
            "v3_med_code": row["v3_med_code"],
            "drug_name": ctx["drug_name"],
            "route": ctx.get("route"),
            "dose": ctx.get("dose"),
            "purpose": ctx.get("purpose"),
            "frequency": row.get("frequency", row.get("count", 0)),
        }
        if args.mode == "pilot":
            row_out["gold_atc"] = row["gold_atc"]
        row_out.update(res)
        results.append(row_out)
        if (i+1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(sample)} drugs, {elapsed:.0f}s elapsed, {elapsed/(i+1):.2f}s/drug")

    res_df = pd.DataFrame(results)
    model_tag = args.model.replace("Meta-", "").replace("-Instruct", "").replace(".", "p")
    outfile = OUT / (
        f"atc_pilot_dose_{model_tag}_n{args.n}.csv" if args.mode == "pilot"
        else f"atc_full_dose_{model_tag}.csv"
    )
    res_df.to_csv(outfile, index=False)
    print(f"Saved: {outfile}")

    if args.mode == "pilot":
        gold = res_df["gold_atc"]
        print("\nAccuracy vs ETHOS gold:")
        for L, n_chars in [(1, 1), (2, 3), (3, 4), (4, 5), (5, 7)]:
            pred = res_df.get(f"L{L}", pd.Series([None]*len(res_df)))
            gold_L = gold.str[:n_chars]
            correct = (pred == gold_L).sum()
            print(f"  L{L}: {correct}/{len(res_df)} ({100*correct/len(res_df):.1f}%)")

        # Compare to non-dose baseline (atc_pilot_Llama-3p3-70B_n100.csv)
        baseline_path = OUT / f"atc_pilot_{model_tag}_n{args.n}.csv"
        if baseline_path.exists():
            base = pd.read_csv(baseline_path)
            print(f"\nvs baseline (non-dose-aware):")
            for L, n_chars in [(1, 1), (2, 3), (3, 4), (4, 5), (5, 7)]:
                b_pred = base.get(f"L{L}")
                b_gold = base["gold_atc"].str[:n_chars]
                b_correct = (b_pred == b_gold).sum()
                d_correct = (res_df.get(f"L{L}") == res_df["gold_atc"].str[:n_chars]).sum()
                print(f"  L{L}: baseline {b_correct}% → dose-aware {d_correct}%  ({d_correct-b_correct:+d}pp)")

    print("Done.")


if __name__ == "__main__":
    main()
