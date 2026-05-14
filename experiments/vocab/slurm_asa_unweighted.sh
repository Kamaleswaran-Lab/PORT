#!/bin/bash
# ASA-only baseline with unweighted log-loss (fair comparison with PORT BCE).
# CPU-only; load events (~230MB) requires non-trivial memory.
#
#SBATCH --job-name=asa_uw
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0:30:00
#SBATCH --output=/path/to/CHD_MEDS/results_v4/slurm_asa_uw_%j.log
#SBATCH --error=/path/to/CHD_MEDS/results_v4/slurm_asa_uw_%j.err

set -e
cd /path/to/CHDLLM
export PATH=/path/to/miniforge3/envs/ethos/bin:$PATH
export PYTHONPATH=/path/to/CHDLLM:$PYTHONPATH
export CHD_DATA_ROOT=/path/to/CHD_MEDS

echo "=== ASA-only LR (unweighted) ==="
python -m baselines.asa_baseline --unweighted --suffix _unweighted
echo "=== Done ==="
