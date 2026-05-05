#!/bin/bash
# infer.sh — Run ETHOS zero-shot inference for IoD prediction
#
# Usage (from project root):
#   bash ethos/infer.sh [rep_num=100]
#
# Prereqs:
#   conda activate ethos
#   bash ethos/train.sh    (model trained)
#   bash ethos/tokenize.sh (test split tokenized)
#
# Calls ethos/run_infer.py — our own inference runner.
# Does NOT use ethos_infer CLI (which would require modifying ethos-ares).

set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ethos

DATA_ROOT="/path/to/CHD_MEDS"
MODEL_NAME="chd_layer6_do0.3"
MODEL_FP="${DATA_ROOT}/tokenized/models/${MODEL_NAME}/recent_model.pt"
INPUT_DIR="${DATA_ROOT}/tokenized/test"
OUTPUT_DIR="${DATA_ROOT}/results/ethos/iod"

REP_NUM=${1:-100}  # number of trajectory samples per encounter

# Respect CUDA_VISIBLE_DEVICES if set; otherwise count all GPUs
if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
    NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
else
    NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
fi

if [[ ! -f "${MODEL_FP}" ]]; then
    echo "ERROR: Model not found at ${MODEL_FP}"
    echo "Run bash ethos/train.sh first."
    exit 1
fi

echo "=== ETHOS Zero-Shot Inference: IoD ==="
echo "  Model:   ${MODEL_FP}"
echo "  Rep num: ${REP_NUM}"
echo "  GPUs:    ${NUM_GPUS}"

INFER_SUBDIR="iod_rep${REP_NUM}_$(date +%Y-%m-%d_%H-%M-%S)"

python ethos/run_infer.py \
    --model_fp    "${MODEL_FP}" \
    --input_dir   "${INPUT_DIR}" \
    --output_dir  "${OUTPUT_DIR}" \
    --output_fn   "${INFER_SUBDIR}" \
    --rep_num     ${REP_NUM} \
    --n_gpus      ${NUM_GPUS}

echo ""
echo "=== Post-processing: uncertainty + attribution ==="
python ethos/analyze_infer.py \
    --infer_dir  "${OUTPUT_DIR}/${INFER_SUBDIR}" \
    --output_dir "${OUTPUT_DIR}"

echo "=== Inference complete ==="
echo "Raw trajectories: ${OUTPUT_DIR}/${INFER_SUBDIR}/"
echo "Final predictions: ${OUTPUT_DIR}/iod_predictions.parquet"
