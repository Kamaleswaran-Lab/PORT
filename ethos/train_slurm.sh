#!/bin/bash
#SBATCH --job-name=ethos_chd_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/path/to/CHD_MEDS/tokenized/models/train_%j.log
#SBATCH --error=/path/to/CHD_MEDS/tokenized/models/train_%j.err

set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ethos

DATA_ROOT="/path/to/CHD_MEDS"
DATA_PATH="${DATA_ROOT}/tokenized/train"
MODEL_DIR="${DATA_ROOT}/tokenized/models"
mkdir -p "${MODEL_DIR}"

NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
echo "Detected ${NUM_GPUS} GPU(s)"

BATCH_SIZE=32
N_POSITIONS=2048
N_LAYER=6
N_HEAD=12
N_EMBD=768
DROPOUT=0.3
LR=0.0006
MIN_LR=0.00001
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
    -m ethos_train \
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
    out_dir="${MODEL_DIR}/${MODEL_NAME}"

echo "=== Training complete ==="
echo "Model saved to: ${MODEL_DIR}/${MODEL_NAME}"
