"""
IoD2 LLM Reclassification Script
----------------------------------
Step 1: Apply existing keyword filter (produces ~3340 pts / ~12586 events)
Step 2: Run each Quick Note Event Comment through local Llama 3.1-8B-Instruct
         to verify whether it truly represents intraoperative deterioration
Step 3: Report final patient count after LLM filtering
Step 4: Save 50 example notes that were DROPPED by the LLM (false positives)

Usage:
    python iod2_llm_filter.py

Assumes the following variables are already defined / loaded in the session,
or adjust the CSV paths at the top of main() to load them fresh:
    - AN_Events   : DataFrame
    - AN_Patients : DataFrame
    - df          : master patient DataFrame with 'C MRN' column
"""

import re
import json
import time
import logging
import warnings
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH = "${HF_HOME}/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
BATCH_SIZE = 16          # adjust based on VRAM (8B model needs ~16 GB; reduce if OOM)
MAX_NEW_TOKENS = 10      # we only need YES / NO
OUTPUT_DIR = Path("./iod2_llm_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# KEYWORD LIST (same as original IoD2)
# ─────────────────────────────────────────────
VALUES_TO_CHECK = [
    "chest compression", "CPR", "code", "PALS", "arrest", "deteriorated",
    "deterioration", "hypotension", "epinephrine", "epi", "dopamine",
    "phenylephrine", "norepi", "norepinephrine", "emergently", "shock",
    "defib", "defibrillation", "ECMO", "cardioversion", "DCCV", "pulseless",
    "no pulse", "help", "anesthesia now", "overhead", "echo", "cardiac output",
    "arrhythmia", "SVT", "Vtach", "v tach", "v-tach", "v-fib", "vfib",
    "v fib", "ventricular tachycardia", "ventricular fibrillation", "brady",
    "fibrillation",
]

# ─────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a pediatric cardiac anesthesiologist reviewing intraoperative Quick Notes.

Your job is to flag notes that could indicate an INTRAOPERATIVE DETERIORATION event.
When in doubt, classify as YES.

=== Classify as NO ONLY IF the note clearly falls into one of these categories ===

1. The note has NO clinical content — purely administrative, IT/equipment issues,
   or documentation errors.
   (e.g. "IT fixed data connection", "Bair hugger gave error code", "Epic charting not current")

2. The keyword appears ONLY in a negated form with no other concerning content.
   (e.g. "no CPR", "no arrest", "no pulse", "no bradycardia", "no complications")

3. The word "help" appears only in a routine sentence context, not as an emergency call.
   (e.g. "help secure the tube", "help with positioning" — NOT "called for help overhead")

=== Classify as YES for everything else, including ===
- Any mention of echo, SVT, arrhythmia, hypotension, bradycardia, shock, epi, dopamine,
  defibrillation, ECMO, cardioversion, CPR, arrest — regardless of context or severity
- Epinephrine/epi given by surgeon (local or intracardiac injection counts)
- Even mild or self-resolved events with any of the above keywords
- If ANY of the following words appear, classify as YES regardless of context:
  "chest compression", "CPR", "cardiac arrest", "ECMO", "defibrillation",
  "cardioversion", "pulseless", "anesthesia now", "overhead", "help" (when not clearly routine), "deterioration", "deteriorated"

Respond with exactly one word: YES or NO."""


def build_user_message(comment: str) -> str:
    return f'Quick Note:\n"""\n{comment.strip()}\n"""\n\nIs this an intraoperative deterioration event?'


# ─────────────────────────────────────────────
# KEYWORD FILTER (replicates original IoD2)
# ─────────────────────────────────────────────
def apply_keyword_filter(AN_Events: pd.DataFrame, AN_Patients: pd.DataFrame) -> pd.DataFrame:
    log.info("Applying keyword filter …")
    values_lower = [v.lower() for v in VALUES_TO_CHECK]
    pattern = "|".join(re.escape(v) for v in values_lower)

    quick_note = AN_Events[
        AN_Events["Event"].fillna("").str.strip().eq("Quick Note")
    ].copy()
    quick_note["Event Comment"] = quick_note["Event Comment"].fillna("")
    mask = quick_note["Event Comment"].str.lower().str.contains(pattern, regex=True)
    quick_note_matched = quick_note[mask].copy()

    merged = pd.merge(
        quick_note_matched[["C MRN", "Recorded Time", "Event Comment"]],
        AN_Patients[["C MRN", "In OR", "Out OR"]],
        on="C MRN",
    ).drop_duplicates()

    merged = merged[merged["Recorded Time"].notnull()]
    merged = merged[(merged["In OR"].notnull()) | (merged["Out OR"].notnull())]

    for col in ["Recorded Time", "In OR", "Out OR"]:
        merged[col] = pd.to_datetime(merged[col], errors="coerce")

    iod2 = merged[
        (merged["Recorded Time"] >= merged["In OR"])
        & (merged["Recorded Time"] <= merged["Out OR"])
    ].copy()

    log.info(
        f"Keyword filter → {iod2['C MRN'].nunique():,} patients | {len(iod2):,} events"
    )
    return iod2.reset_index(drop=True)


# ─────────────────────────────────────────────
# LLM SETUP
# ─────────────────────────────────────────────
def load_pipeline(model_path: str):
    log.info(f"Loading model from {model_path} …")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}  |  GPUs: {torch.cuda.device_count()}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,        # greedy → deterministic YES/NO
        return_full_text=False,
    )
    log.info("Model loaded.")
    return pipe


# ─────────────────────────────────────────────
# PARSE YES / NO FROM MODEL OUTPUT
# ─────────────────────────────────────────────
def parse_label(raw_text: str) -> str:
    """Return 'YES', 'NO', or 'UNCLEAR'."""
    t = raw_text.strip().upper()
    if t.startswith("YES"):
        return "YES"
    if t.startswith("NO"):
        return "NO"
    # fallback: scan first 30 chars
    if "YES" in t[:30]:
        return "YES"
    if "NO" in t[:30]:
        return "NO"
    return "UNCLEAR"


# ─────────────────────────────────────────────
# BATCH CLASSIFICATION
# ─────────────────────────────────────────────
def classify_events(pipe, events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'llm_label' column ('YES'/'NO'/'UNCLEAR') to events_df.
    Processes in batches and logs progress.
    """
    comments = events_df["Event Comment"].tolist()
    n = len(comments)
    labels = []

    log.info(f"Classifying {n:,} events in batches of {BATCH_SIZE} …")
    t0 = time.time()

    for start in range(0, n, BATCH_SIZE):
        batch_comments = comments[start : start + BATCH_SIZE]

        # Build chat-formatted prompts using the tokenizer's template
        prompts = []
        for comment in batch_comments:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_message(comment)},
            ]
            # apply_chat_template returns a string
            prompt_str = pipe.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt_str)

        outputs = pipe(prompts, batch_size=len(prompts))

        for out in outputs:
            # pipeline returns list of dicts; first element has 'generated_text'
            raw = out[0]["generated_text"] if isinstance(out, list) else out["generated_text"]
            labels.append(parse_label(raw))

        elapsed = time.time() - t0
        done = min(start + BATCH_SIZE, n)
        rate = done / elapsed
        eta = (n - done) / rate if rate > 0 else 0
        log.info(
            f"  {done:>6,}/{n:,}  ({done/n*100:.1f}%)  "
            f"elapsed {elapsed/60:.1f} min  ETA {eta/60:.1f} min"
        )

    events_df = events_df.copy()
    events_df["llm_label"] = labels
    return events_df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main(AN_Events, AN_Patients, df):
    # ── Step 1: keyword filter ──────────────────
    iod2_keyword = apply_keyword_filter(AN_Events, AN_Patients)

    # ── Step 2: LLM classification ─────────────
    pipe = load_pipeline(MODEL_PATH)
    iod2_classified = classify_events(pipe, iod2_keyword)

    # ── Step 3: save full classification result ─
    full_out = OUTPUT_DIR / "iod2_all_classified.csv"
    iod2_classified.to_csv(full_out, index=False)
    log.info(f"Full classified table saved → {full_out}")

    # ── Step 4: patient-level aggregation ──────
    # A patient is IoD2-positive if AT LEAST ONE of their events is YES
    patient_yes = (
        iod2_classified.groupby("C MRN")["llm_label"]
        .apply(lambda s: "YES" in s.values)
    )
    iod2_patients_llm = patient_yes[patient_yes].index

    # ── Step 5: update master df ────────────────
    df = df.copy()
    df["Intraoperative_Deterioration2_LLM"] = df["C MRN"].isin(iod2_patients_llm).astype(int)

    # ── Step 6: report ──────────────────────────
    n_patients_keyword = iod2_keyword["C MRN"].nunique()
    n_events_keyword   = len(iod2_keyword)
    n_patients_llm     = int(df["Intraoperative_Deterioration2_LLM"].sum())
    n_events_llm       = (iod2_classified["llm_label"] == "YES").sum()
    n_filtered_pts     = n_patients_keyword - n_patients_llm

    label_counts = iod2_classified["llm_label"].value_counts().to_dict()

    print("\n" + "=" * 55)
    print("  IoD2 — KEYWORD vs LLM FILTER COMPARISON")
    print("=" * 55)
    print(f"  After keyword filter   : {n_patients_keyword:>5,} patients | {n_events_keyword:>6,} events")
    print(f"  After LLM filter       : {n_patients_llm:>5,} patients | {n_events_llm:>6,} events")
    print(f"  Patients filtered out  : {n_filtered_pts:>5,}")
    print(f"  LLM label distribution : {label_counts}")
    print("=" * 55 + "\n")

    # ── Step 7: false-positive examples (dropped by LLM) ──
    # Events where keyword said YES but LLM said NO/UNCLEAR
    # for patients who have NO confirmed YES event
    dropped_patients = set(iod2_keyword["C MRN"].unique()) - set(iod2_patients_llm)
    dropped_events = iod2_classified[
        iod2_classified["C MRN"].isin(dropped_patients)
    ].copy()

    # Among those, prioritise events the LLM explicitly labelled NO
    dropped_no   = dropped_events[dropped_events["llm_label"] == "NO"]
    dropped_unc  = dropped_events[dropped_events["llm_label"] == "UNCLEAR"]
    sample_pool  = pd.concat([dropped_no, dropped_unc]).drop_duplicates()

    # Deduplicate by comment text so we get diverse examples
    sample_pool = sample_pool.drop_duplicates(subset="Event Comment")
    examples = sample_pool.sample(n=min(50, len(sample_pool)), random_state=42)

    examples_out = OUTPUT_DIR / "iod2_filtered_examples_50.csv"
    examples[["C MRN", "Recorded Time", "Event Comment", "llm_label"]].to_csv(
        examples_out, index=False
    )
    log.info(f"50 filtered examples saved → {examples_out}")

    # Also save a quick human-readable text version
    txt_out = OUTPUT_DIR / "iod2_filtered_examples_50.txt"
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write("IoD2 — Notes Dropped by LLM (sample of up to 50)\n")
        f.write("=" * 60 + "\n\n")
        for i, (_, row) in enumerate(examples.iterrows(), 1):
            f.write(f"[{i:02d}]  MRN: {row['C MRN']}  |  Time: {row['Recorded Time']}  |  LLM: {row['llm_label']}\n")
            f.write(f"      Comment: {row['Event Comment']}\n")
            f.write("-" * 60 + "\n")
    log.info(f"Readable examples saved → {txt_out}")

    return df, iod2_classified


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # ── Data file paths ────────────────────────────────────────────────────
    file_AN_Events   = "/path/to/CHOA_RAW_TABLES/CHOA_DATA_Tables_CHD/DR15201_AN_Events.rpt"
    file_AN_Patients = "/path/to/CHOA_RAW_TABLES/CHOA_DATA_Tables_CHD/DR15201_AN_Patients.rpt"

    # ── Load AN_Events & AN_Patients ───────────────────────────────────────
    log.info("Loading AN_Events …")
    AN_Events = pd.read_csv(file_AN_Events, delimiter="|", on_bad_lines="skip")
    AN_Events = AN_Events.iloc[:-2]

    log.info("Loading AN_Patients …")
    AN_Patients = pd.read_csv(file_AN_Patients, delimiter="|", on_bad_lines="skip")
    AN_Patients = AN_Patients.iloc[:-2]

    # ── Build minimal master df from AN_Patients ───────────────────────────
    # 'df' only needs a 'C MRN' column for the IoD2 flag assignment.
    # If you have a pre-built master df, load it here instead.
    df = AN_Patients[["C MRN"]].drop_duplicates().reset_index(drop=True)

    # ── Run pipeline ───────────────────────────────────────────────────────
    df_updated, classified = main(AN_Events, AN_Patients, df)
    print("Done. Results in:", OUTPUT_DIR.resolve())