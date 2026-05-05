#!/bin/bash
# train.sh — Train ETHOS transformer on CHD tokenized data
#
# Usage (from project root):
#   bash ethos/train.sh [extra torchrun args]
#
# Model: 6-layer decoder-only transformer, 45M params (same as ETHOS paper)
# Adjusted for CHD dataset size (~19M tokens vs MIMIC's 321M):
#   - max_iters reduced to 200,000
#   - lr_decay_iters reduced to 100,000
#
# Prereqs:
#   conda activate ethos
#   bash ethos/tokenize.sh   (tokenization complete)

set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ethos

DATA_ROOT="/path/to/CHD_MEDS"
DATA_PATH="${DATA_ROOT}/tokenized/train"
MODEL_DIR="${DATA_ROOT}/tokenized/models"

if [[ ! -d "${DATA_PATH}" ]]; then
    echo "ERROR: Tokenized train data not found at ${DATA_PATH}"
    echo "Run bash ethos/tokenize.sh first."
    exit 1
fi

# Respect CUDA_VISIBLE_DEVICES if set; otherwise count all GPUs
if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
    NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
else
    NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
fi
echo "Detected ${NUM_GPUS} GPU(s) (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all})"

# Model hyperparameters (same architecture as ETHOS paper)
BATCH_SIZE=32
N_POSITIONS=2048
N_LAYER=6
N_HEAD=12
N_EMBD=768
DROPOUT=0.3
LR=0.0006
MIN_LR=0.00001

# Training duration: scaled down from MIMIC (1M iters) proportional to dataset size
MAX_ITERS=200000
LR_DECAY_ITERS=100000
WARMUP_ITERS=2000

MODEL_NAME="chd_layer${N_LAYER}_do${DROPOUT}"

echo "=== Training ETHOS on CHD data ==="
echo "  Model: ${MODEL_NAME}"
echo "  GPUs:  ${NUM_GPUS}"
echo "  Iters: ${MAX_ITERS}"

torchrun \
    --standalone \
    --nproc_per_node=${NUM_GPUS} \
    -m ethos.train.run_training \
    data_fp="${DATA_PATH}" \
    val_size=6 \
    batch_size=${BATCH_SIZE} \
    n_positions=${N_POSITIONS} \
    n_layer=${N_LAYER} \
    n_head=${N_HEAD} \
    n_embd=${N_EMBD} \
    dropout=${DROPOUT} \
    lr=${LR} \
    min_lr=${MIN_LR} \
    log_interval=10 \
    eval_interval=1000 \
    gradient_accumulation_steps=16 \
    warmup_iters=${WARMUP_ITERS} \
    max_iters=${MAX_ITERS} \
    lr_decay_iters=${LR_DECAY_ITERS} \
    wandb_log=false \
    out_dir="${MODEL_DIR}/${MODEL_NAME}" \
    "$@"

echo "=== Training complete ==="
echo "Model saved to: ${MODEL_DIR}/${MODEL_NAME}"
