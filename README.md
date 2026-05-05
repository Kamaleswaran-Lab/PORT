# PORT — Pediatric Operative Risk Transformer

A generative EHR foundation model for predicting **intraoperative deterioration (IoD)** in pediatric cardiac surgery, adapted from [ETHOS](https://github.com/ipolharvard/ethos-ares) on Children's Healthcare of Atlanta (CHOA) data via the [MEDS](https://medical-event-data-standard.github.io/) format and fine-tuned with [LoRA](https://arxiv.org/abs/2106.09685).

> **Paper.** *PORT: A Generative EHR Foundation Model for Pediatric Intraoperative Risk Prediction.* Manuscript prepared for *JAMIA*; main text and supplement are tracked in `paper/overleaf/` (separate Overleaf-Git repository).

## Headline result

| Model | AUROC | AUPRC | Brier | ECE |
|---|---|---|---|---|
| ASA score (clinical standard) | 0.703 | 0.035 | 0.212 | 0.420 |
| Logistic Regression (MEDS aggregates) | 0.750 | 0.043 | 0.183 | 0.379 |
| XGBoost (MEDS aggregates) | 0.755 | 0.062 | 0.204 | 0.437 |
| BiLSTM on raw MEDS sequences (tuned) | 0.798 | 0.105 | 0.019 | 0.047 |
| Backbone, zero-shot trajectory | 0.502 | 0.026 | 0.101 | 0.290 |
| Backbone + linear probe | 0.673 | 0.048 | 0.016 | 0.059 |
| **PORT (LoRA fine-tuned)** | **0.833** | **0.144** | **0.014** | **0.037** |

Bootstrap test, PORT vs tuned BiLSTM: $\Delta$AUROC $= +0.031$, 95 % CI $[+0.012, +0.050]$, $p = 0.002$.
Three-seed mean (seeds 42 / 123 / 456): AUROC 0.820 ± 0.009, AUPRC 0.130 ± 0.010.

## Cohort

- 210,274 surgical encounters, 133,755 unique patients (CHOA AIMS, March 2007 to October 2022).
- IoD prevalence 0.94 % (composite of five intraoperative events: CPR, Quick Note keyword deterioration, IV vasoactive bolus, vasopressor infusion escalation, emergent arterial line).
- Patient-level split, stratified on IoD: train 70 % / val 10 % / test 20 %.

Detailed cohort flow and evaluation set sizes appear in the manuscript Figure 1.

## Repository layout

```
.
├── README.md                  project overview and headline results
├── LICENSE                    MIT
├── requirements.txt           pinned Python dependencies
│
├── datapreprocessing/         raw .rpt to MEDS parquet (12 source tables)
│   └── meds_scripts/
│
├── pipeline/                  MEDS to ETHOS-ready shards
│   ├── merge_meds.py          merge per-table parquets
│   ├── create_splits.py       patient-level splits stratified on IoD
│   └── prepare_ethos_data.py  shard for tokenization
│
├── ethos/                     PORT model code
│   ├── configs/dataset/chd.yaml
│   ├── datasets/iod_dataset.py        prediction-time-aware dataset
│   ├── datasets/iod_window_dataset.py context-window-aware variant
│   ├── tokenize.sh, train.sh, train_slurm.sh, infer.sh
│   ├── run_infer.py           zero-shot N-trajectory inference
│   ├── analyze_infer.py       per-encounter risk + uncertainty
│   ├── aggregate_zeroshot.py  trajectory aggregation
│   └── finetune.py            LoRA fine-tuning -> PORT
│
├── baselines/                 conventional baselines
│   ├── asa_baseline.py        ASA-score-only logistic regression
│   ├── features.py            manual + MEDS-aggregated feature builders
│   ├── logreg_xgb.py          logistic regression + XGBoost (base)
│   ├── logreg_xgb_tuned.py    LR/XGB hyperparameter sweep
│   ├── lstm.py                BiLSTM model
│   ├── lstm_tuned.py          BiLSTM hyperparameter sweep
│   └── slurm_tune_*.sh
│
├── evaluation/
│   ├── evaluate.py            AUROC / AUPRC / Brier / ECE / calibration
│   ├── occlusion_analysis.py  category-level token masking
│   ├── ppv_subgroup_analysis.py
│   └── plot_history_distribution.py
│
├── experiments/
│   └── vocab/                 vocabulary reconstruction pipeline
│
└── paper/
    ├── overleaf/              main.tex + bibliography + figures (separate remote)
    ├── benchmark_inference.py inference latency / GPU memory benchmark
    └── gen_*.py               figure-generation scripts
```

`ethos-ares/` is a clone of the upstream [ETHOS](https://github.com/ipolharvard/ethos-ares) repository and is excluded from this project; install it separately (`pip install -e ethos-ares/`) to expose the `ethos_tokenize`, `ethos.train.run_training`, and `ethos.utils.load_model_checkpoint` APIs used by `ethos/finetune.py`.

## Site-specific paths

Scripts use placeholder paths that you must adjust to your environment before running:

| Placeholder | Meaning |
|---|---|
| `/path/to/CHOA_RAW` | Raw CHOA `.rpt` / `.csv` extracts (root of `DR15201_*` files) |
| `/path/to/CHD_MEDS` | Working directory for derived MEDS parquets, splits, model outputs |
| `/path/to/ethos-ares` | Local clone of `ipolharvard/ethos-ares` |
| `${HF_HOME}` | HuggingFace model cache (used only by the LLM ATC step) |

A simple search-and-replace pattern adapts a fresh clone to your environment, for example:

```bash
grep -rl '/path/to/CHD_MEDS' . | xargs sed -i 's|/path/to/CHD_MEDS|/your/actual/path|g'
```

## Quick start

Reproducing the published results requires (i) CHOA EHR data under a data use agreement and (ii) a SLURM-managed cluster with H200 / A100 GPUs.

```bash
conda activate ethos                                 # Python 3.12, PyTorch 2.7
git clone https://github.com/ipolharvard/ethos-ares.git
pip install -e ethos-ares/

# 1. Raw to MEDS (12 scripts, run once)
for f in datapreprocessing/meds_scripts/*_to_meds.py; do python "$f"; done

# 2. Merge to splits to ETHOS shards
python pipeline/merge_meds.py
python pipeline/create_splits.py
python pipeline/prepare_ethos_data.py

# 3. Vocabulary reconstruction (ICD-10 hierarchical, ATC mapping, SES enrichment)
python experiments/vocab/preprocess_integrate.py

# 4. ETHOS tokenize, train, zero-shot IoD inference
bash ethos/tokenize.sh
sbatch ethos/train_slurm.sh                          # 8 x H200, about 17 h
bash ethos/infer.sh

# 5. PORT fine-tuning (LoRA, 3 seeds)
sbatch experiments/vocab/slurm_finetune.sh

# 6. Baselines, evaluation, occlusion analysis, figures
python -m baselines.asa_baseline
python -m baselines.logreg_xgb_tuned
python -m baselines.lstm_tuned
python -m evaluation.evaluate
python -m evaluation.occlusion_analysis
python experiments/vocab/regenerate_figures.py
```

## Citation

Manuscript under preparation. A preprint reference and DOI will be added upon submission.

## License

Code is released under the MIT License (see `LICENSE`). The CHOA EHR data are not redistributable; access requires a data use agreement with Children's Healthcare of Atlanta.

## Acknowledgements

PORT is built on top of the open-source [ETHOS](https://github.com/ipolharvard/ethos-ares) and [MEDS](https://medical-event-data-standard.github.io/) projects and uses Llama 3.3 70B Instruct for ATC drug-class mapping. The composite IoD label was developed in collaboration with a pediatric cardiac anesthesiologist at CHOA.
