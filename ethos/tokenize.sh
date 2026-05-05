#!/bin/bash
# tokenize.sh — Run ethos_tokenize on CHD data
#
# Usage (from project root .):
#   bash ethos/tokenize.sh
#
# Prereqs:
#   conda activate ethos
#   python pipeline/prepare_ethos_data.py   (run first)

set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ethos

PROJ_ROOT="."
DATA_ROOT="/path/to/CHD_MEDS"

INPUT_TRAIN="${DATA_ROOT}/ethos_input/train"
INPUT_VAL="${DATA_ROOT}/ethos_input/val"
INPUT_TEST="${DATA_ROOT}/ethos_input/test"
OUTPUT_DIR="${DATA_ROOT}/tokenized"

# Copy CHD dataset config into ethos package config dir
ETHOS_CFG_DIR=$(python -c "import ethos; import pathlib; print(pathlib.Path(ethos.__file__).parent / 'configs' / 'dataset')")
cp "${PROJ_ROOT}/ethos/configs/dataset/chd.yaml" "${ETHOS_CFG_DIR}/chd.yaml"
echo "Copied chd.yaml → ${ETHOS_CFG_DIR}/chd.yaml"

echo "=== Tokenizing TRAIN split ==="
ethos_tokenize -m worker="range(0,8)" \
    dataset=chd \
    input_dir="${INPUT_TRAIN}" \
    output_dir="${OUTPUT_DIR}" \
    out_fn=train

echo "=== Tokenizing VAL split ==="
ethos_tokenize -m worker="range(0,2)" \
    dataset=chd \
    input_dir="${INPUT_VAL}" \
    vocab="${OUTPUT_DIR}/train" \
    output_dir="${OUTPUT_DIR}" \
    out_fn=val

echo "=== Tokenizing TEST split ==="
ethos_tokenize -m worker="range(0,4)" \
    dataset=chd \
    input_dir="${INPUT_TEST}" \
    vocab="${OUTPUT_DIR}/train" \
    output_dir="${OUTPUT_DIR}" \
    out_fn=test

echo "=== Tokenization complete ==="
echo "Output: ${OUTPUT_DIR}"
