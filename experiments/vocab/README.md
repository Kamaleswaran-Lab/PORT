# Vocabulary reconstruction (ETHOS-aligned)

Reduces a flat-code MEDS vocabulary derived directly from the source tables to a compact ETHOS-aligned vocabulary by combining four strategies tailored to each source table:

1. **Hierarchical decomposition of structured ontology codes** (ICD-10-CM and ATC).
2. **LLM-assisted mapping of free-text medication names to ATC**, with manual review of all LLM-assigned mappings.
3. **Frequency-based long-tail truncation** with per-category `OTHER` fallbacks for laboratory tests, procedures, and free-text problem-list and medical-history items, retaining the most frequent codes covering ≥95% of training events.
4. **Atomic decomposition of demographic context** augmented with previously absent socioeconomic fields (primary language, insurance, point of origin, home county).

The resulting 6,170-token vocabulary approaches the ~4,400-token scale of the original ETHOS model while adding ~70 new socioeconomic-context tokens.

## Final vocabulary

| Family | Count | Strategy |
|--------|-------|----------|
| ICD-10 hierarchical (DIAG + PROBLEM + CARDIOLOGY) | 3,216 | ETHOS 3-level decomposition |
| ATC hierarchical (MED) | 277 | Chen 2024 LLM + manual review |
| LAB | 501 | top-500 + OTHER |
| PROCEDURE | 500 | top-499 + OTHER |
| ENCOUNTER (AN flow + AN//PRIMARY top-500 + admission / class) | 520 | mixed |
| SDE | 301 | top-300 + OTHER |
| PROBLEM // NAME (free-text) | 272 | top-200 + OTHER |
| DIAGNOSIS // MEDHX (free-text) | 248 | top-200 + OTHER |
| ADT | 106 | 54 departments × in/out |
| LDA | 80 | 40 device types × place/remove |
| HOME_COUNTY | 51 | top-50 Georgia + OTHER (SES) |
| Time-interval tokens | 19 | 5 min – 6 mo, quasi-logarithmic |
| Demographics (race / sex / ethnicity, atomic) | 14 | atomic decomposition |
| TRANSFUSION | 14 | product-type only (dose stripped) |
| AN_EVENT | 12 | atomic event categories |
| Quantile tokens (Q1 – Q10) | 10 | shared across all numeric codes |
| VITAL (preprocedure) | 8 | HR / SBP / DBP / SpO₂ / RR / temp / ht / wt |
| INSURANCE | 6 | Medicaid / commercial / Medicare / TRICARE / OTHER / unknown (SES) |
| POINT_OF_ORIGIN | 6 | top-5 + OTHER (SES) |
| LANGUAGE | 6 | top-5 + OTHER |
| Sentinel control tokens | 5 | `TIMELINE_END`, `OUTCOME//DEATH`, `MED//UNMAPPED`, ... |
| **Total** | **6,170** | |

## Pipeline

The full pipeline runs as a sequence of preprocessing scripts that operate on per-source-table MEDS shards and merge into a single integrated MEDS dataset:

```
preprocess_icd10_hier.py    ICD-10-CM 3-level hierarchical decomposition
preprocess_cutoffs.py       Frequency-based long-tail truncation (LAB, PROC, ENC, SDE)
preprocess_demo_fix.py      Atomic demographics (race / sex / ethnicity)
preprocess_ses.py           New socioeconomic fields (insurance, language, county, point of origin)
preprocess_cleanup.py       AN_EVENT typo dedup, TRANSFUSION dose strip
preprocess_integrate.py     Merge all streams into a single MEDS parquet
fix_free_text_cutoffs.py    Apply frequency cutoffs to free-text PROBLEM and MEDHX names
```

The medication-to-ATC mapping pipeline (`stream_a_atc/`) classifies each unique drug name with a hierarchical ATC code via zero-shot prompting of Llama 3.3 70B Instruct, with a manual top-200 review tier and a regex-based pattern review for the long tail.

## Outputs

After running `preprocess_integrate.py` and `fix_free_text_cutoffs.py`, the integrated MEDS parquet is written to the data directory configured in your environment (`/path/to/CHD_MEDS/ethos_input/`); the vocabulary CSV is written to `outputs/`.

## Downstream

After vocabulary reconstruction, run the standard ETHOS tokenization, training, and inference scripts (`ethos/tokenize.sh`, `ethos/train_slurm.sh`, `ethos/infer.sh`), then PORT fine-tuning (`slurm_finetune.sh`) and evaluation.
