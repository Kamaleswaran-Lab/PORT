#!/bin/bash
#SBATCH --job-name=bench_inf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=/path/to/CHD_MEDS/results/baselines_tuned/slurm_bench_%j.log
#SBATCH --error=/path/to/CHD_MEDS/results/baselines_tuned/slurm_bench_%j.err

set -euo pipefail
mkdir -p /path/to/CHD_MEDS/results/baselines_tuned
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ethos
cd .
python -u paper/benchmark_inference.py --gpu 0 --n_samples 500
echo "BENCH DONE"

# After benchmark completes, append the Supp Table to main.tex and push.
# Idempotent: if the label already exists, the python helper replaces in-place.
python -u <<'PY'
from pathlib import Path
import re, subprocess

OVERLEAF = Path("paper/overleaf")
MAIN = OVERLEAF / "main.tex"
TEX = Path("/path/to/CHD_MEDS/results/baselines_tuned/inference_benchmark_supp_table.tex")
if not TEX.exists():
    print("benchmark Supp Table .tex not found; skipping insertion")
    raise SystemExit(0)

block = TEX.read_text().rstrip() + "\n"
text = MAIN.read_text()

if r"\label{tab:inference_cost}" in text:
    new_text = re.sub(
        r"\\begin\{table\}\[h\]\n[^\\]*?\\label\{tab:inference_cost\}.*?\\end\{table\}",
        block.rstrip(),
        text, flags=re.DOTALL,
    )
    print("Replaced existing tab:inference_cost block")
else:
    # Insert just before \end{document}
    new_text = text.replace(r"\end{document}", block + "\n" + r"\end{document}")
    print("Appended new tab:inference_cost block before \\end{document}")

MAIN.write_text(new_text)

# git pull-rebase + commit + push
subprocess.run(["git", "add", "main.tex"], cwd=OVERLEAF, check=True)
subprocess.run(["git", "pull", "--rebase"], cwd=OVERLEAF, check=False)
msg = (
    "Add Supp Table tab:inference_cost (PORT vs BiLSTM per-encounter latency + GPU mem)\n\n"
    "500 test encounters, batch size 1 on a single NVIDIA H200; warmup 5 encounters.\n"
    "Reports mean ± std latency, throughput, and peak GPU memory for each model.\n\n"
)
r = subprocess.run(["git", "commit", "-m", msg], cwd=OVERLEAF, capture_output=True, text=True)
print(r.stdout); print(r.stderr)
subprocess.run(["git", "push"], cwd=OVERLEAF, check=False)
print("BENCH PUSH DONE")
PY
