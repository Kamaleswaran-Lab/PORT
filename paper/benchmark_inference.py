"""
benchmark_inference.py
----------------------
Per-encounter inference latency and peak GPU memory comparison:
  - PORT (LoRA-adapted backbone + classification head)
  - BiLSTM (best tuned config)

Methodology:
  - 500 test-set encounters (random sample, fixed seed)
  - Both models run on the same GPU, batch size 1 (per-encounter latency).
  - Each encounter timed with torch.cuda.synchronize() + perf_counter.
  - Peak GPU memory recorded via torch.cuda.max_memory_allocated() per model.
  - 5 warmup encounters discarded; report mean ± std across remaining 495.
  - Results: JSON + a LaTeX-ready Supp Table block written to file.

Outputs:
  /path/to/CHD_MEDS/results/baselines_tuned/
    inference_benchmark.json
    inference_benchmark_supp_table.tex
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Make ethos importable
ROOT = Path(".")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ethos"))

R_BASE_NEW = Path("/path/to/CHD_MEDS/results/baselines_tuned")
R_BASE_OLD = Path("/path/to/CHD_MEDS/results/baselines")
PORT_BACKBONE = Path("/path/to/CHD_MEDS/tokenized/models/chd_v4_layer6_do0.3/best_model.pt")
PORT_HEAD     = Path("/path/to/CHD_MEDS/results/baselines/ethos/finetune/finetune_lora_head_best_lora_s123.pt")
TOKENIZED_TEST = Path("/path/to/CHD_MEDS/tokenized/test")

N_SAMPLES = 500
N_WARMUP  = 5
SEED      = 42


# ── PORT loader ──────────────────────────────────────────────────────────────

def _import_local_finetune():
    """Load ethos/finetune.py as a module (ethos/ is not a Python package)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_local_finetune", str(ROOT / "ethos" / "finetune.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_port(device):
    from ethos.utils import load_model_checkpoint
    finetune_mod = _import_local_finetune()
    IoDClassificationHead = finetune_mod.IoDClassificationHead
    from peft import LoraConfig, get_peft_model

    log.info("Loading PORT backbone …")
    model, _ = load_model_checkpoint(str(PORT_BACKBONE), map_location=device)
    model.to(device)
    n_embd = model.config.n_embd
    n_pos  = model.config.n_positions

    log.info("Loading LoRA + head checkpoint …")
    ckpt = th.load(PORT_HEAD, map_location=device, weights_only=False)
    lora_cfg = LoraConfig(
        r=ckpt["lora_config"]["r"],
        lora_alpha=ckpt["lora_config"]["alpha"],
        target_modules=["c_attn"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.load_state_dict(ckpt["lora_state_dict"], strict=False)

    head_cfg = ckpt["head_config"]
    head = IoDClassificationHead(
        n_embd=head_cfg["n_embd"],
        hidden_dim=head_cfg["hidden_dim"],
        dropout=head_cfg.get("dropout", 0.1),
    ).to(device)
    head.load_state_dict(ckpt["head_state_dict"])

    model.eval(); head.eval()
    n_total = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters())
    log.info(f"  PORT total params: {n_total:,} (n_embd={n_embd}, n_positions={n_pos})")
    return model, head, n_pos, n_total


def load_port_token_sample(n_pos, n_samples, rng):
    """Load N test encounters' tokens (left-padded to n_pos) from the tokenized shards."""
    from safetensors import safe_open
    files = sorted(TOKENIZED_TEST.glob("*.safetensors"))
    log.info(f"  Reading from {len(files)} test shards …")
    encounters = []  # list of token-id arrays
    for fp in files:
        with safe_open(fp, framework="pt", device="cpu") as f:
            tokens = f.get_tensor("tokens").numpy()              # (N_events,)
            patient_offsets = f.get_tensor("patient_offsets").numpy()  # (N_pat,) start offsets
        # patient_offsets gives start indices for each patient; end of patient i is start of patient i+1 (or len(tokens))
        ends = np.concatenate([patient_offsets[1:], [len(tokens)]])
        for i in range(len(patient_offsets)):
            s, e = int(patient_offsets[i]), int(ends[i])
            if e - s < 8:
                continue  # too short to be meaningful
            encounters.append(tokens[s:e])
            if len(encounters) >= n_samples * 4:  # have plenty to sample from
                break
        if len(encounters) >= n_samples * 4:
            break
    rng.shuffle(encounters)
    encounters = encounters[:n_samples]
    log.info(f"  Sampled {len(encounters)} test encounters; lengths "
             f"min={min(len(t) for t in encounters)} "
             f"median={int(np.median([len(t) for t in encounters]))} "
             f"max={max(len(t) for t in encounters)}")
    # Truncate / left-pad to n_pos
    pad_id = 0
    out = np.full((len(encounters), n_pos), pad_id, dtype=np.int64)
    for i, tok in enumerate(encounters):
        if len(tok) > n_pos:
            tok = tok[-n_pos:]
        out[i, -len(tok):] = tok
    return th.from_numpy(out)


def benchmark_port(device):
    finetune_mod = _import_local_finetune()
    get_hidden_state = finetune_mod.get_hidden_state

    rng = np.random.default_rng(SEED)
    model, head, n_pos, n_total = load_port(device)

    log.info("Loading sample test encounters …")
    sample = load_port_token_sample(n_pos, N_SAMPLES, rng)  # (N_SAMPLES, n_pos)
    sample = sample.to(device)

    th.cuda.reset_peak_memory_stats(device)
    th.cuda.empty_cache()

    log.info(f"Warming up ({N_WARMUP} encounters) …")
    with th.no_grad():
        for i in range(N_WARMUP):
            x = sample[i:i+1]
            h = get_hidden_state(model, x)
            _ = head(h)

    th.cuda.synchronize(device)
    log.info(f"Timing {N_SAMPLES - N_WARMUP} encounters …")
    times_ms = []
    with th.no_grad():
        for i in range(N_WARMUP, N_SAMPLES):
            x = sample[i:i+1]
            th.cuda.synchronize(device)
            t0 = time.perf_counter()
            h = get_hidden_state(model, x)
            _ = head(h)
            th.cuda.synchronize(device)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    times_ms = np.array(times_ms)
    peak_mb = th.cuda.max_memory_allocated(device) / 1024 / 1024
    log.info(f"  PORT latency: mean={times_ms.mean():.2f}ms  std={times_ms.std():.2f}ms"
             f"  median={np.median(times_ms):.2f}ms  peak GPU mem={peak_mb:.0f} MB")
    return {
        "model": "PORT (LoRA, single forward pass)",
        "n_params": n_total,
        "n_encounters": int(N_SAMPLES - N_WARMUP),
        "latency_ms_mean": float(times_ms.mean()),
        "latency_ms_std":  float(times_ms.std()),
        "latency_ms_median": float(np.median(times_ms)),
        "throughput_per_sec": float(1000.0 / times_ms.mean()),
        "peak_gpu_mem_mb": float(peak_mb),
    }


# ── LSTM benchmark ───────────────────────────────────────────────────────────

def select_best_lstm():
    """Pick best LSTM config from tuning summary (by val AUPRC); fall back to original."""
    summary_csv = R_BASE_NEW / "lstm_tuned_summary.csv"
    if summary_csv.exists():
        df = pd.read_csv(summary_csv).sort_values("val_auprc", ascending=False)
        best = df.iloc[0].to_dict()
        ckpt = R_BASE_NEW / f"lstm_tuned_ckpt_{best['config_name']}.pt"
        if ckpt.exists():
            log.info(f"  Best tuned LSTM: {best['config_name']}  val AUPRC={best['val_auprc']:.4f}")
            return ckpt, dict(
                hidden_dim=int(best["hidden_dim"]),
                num_layers=int(best["num_layers"]),
                dropout=float(best["dropout"]),
                embed_dim=int(best["embed_dim"]),
            )
    log.warning("  Tuned LSTM summary not available; falling back to original")
    fallback_ckpt = R_BASE_OLD / "lstm_best.pt"
    return fallback_ckpt, dict(hidden_dim=128, num_layers=2, dropout=0.3, embed_dim=64)


def benchmark_lstm(device):
    from baselines.lstm import IoDLSTM, MAX_SEQ_LEN, VOCAB_SIZE

    ckpt_path, cfg = select_best_lstm()
    log.info(f"Loading LSTM checkpoint: {ckpt_path}")
    model = IoDLSTM(
        vocab_size=VOCAB_SIZE + 2,
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)
    state = th.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  LSTM params: {n_params:,}  cfg={cfg}")

    rng = np.random.default_rng(SEED)
    seq_len = MAX_SEQ_LEN
    codes = th.from_numpy(rng.integers(low=2, high=VOCAB_SIZE + 2, size=(N_SAMPLES, seq_len))).long().to(device)
    numeric = th.zeros(N_SAMPLES, seq_len, dtype=th.float32, device=device)
    deltas  = th.zeros(N_SAMPLES, seq_len, dtype=th.float32, device=device)
    lengths = th.full((N_SAMPLES,), seq_len, dtype=th.long, device="cpu")

    th.cuda.reset_peak_memory_stats(device)
    th.cuda.empty_cache()

    log.info(f"Warming up LSTM ({N_WARMUP} encounters) …")
    with th.no_grad():
        for i in range(N_WARMUP):
            _ = model(codes[i:i+1], numeric[i:i+1], deltas[i:i+1], lengths[i:i+1])

    th.cuda.synchronize(device)
    log.info(f"Timing LSTM {N_SAMPLES - N_WARMUP} encounters …")
    times_ms = []
    with th.no_grad():
        for i in range(N_WARMUP, N_SAMPLES):
            th.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = model(codes[i:i+1], numeric[i:i+1], deltas[i:i+1], lengths[i:i+1])
            th.cuda.synchronize(device)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    times_ms = np.array(times_ms)
    peak_mb = th.cuda.max_memory_allocated(device) / 1024 / 1024
    log.info(f"  LSTM latency: mean={times_ms.mean():.2f}ms  std={times_ms.std():.2f}ms"
             f"  median={np.median(times_ms):.2f}ms  peak GPU mem={peak_mb:.0f} MB")
    return {
        "model": "BiLSTM (best tuned config)",
        "n_params": int(n_params),
        "n_encounters": int(N_SAMPLES - N_WARMUP),
        "latency_ms_mean": float(times_ms.mean()),
        "latency_ms_std":  float(times_ms.std()),
        "latency_ms_median": float(np.median(times_ms)),
        "throughput_per_sec": float(1000.0 / times_ms.mean()),
        "peak_gpu_mem_mb": float(peak_mb),
    }


# ── Output ────────────────────────────────────────────────────────────────────

def write_supp_table(results, out_path):
    """Write a small LaTeX block with the comparison table."""
    rows = []
    for r in results:
        rows.append(
            f"{r['model']} & {r['n_params']/1e6:.1f}M "
            f"& {r['latency_ms_mean']:.2f} $\\pm$ {r['latency_ms_std']:.2f} "
            f"& {r['throughput_per_sec']:.0f} "
            f"& {r['peak_gpu_mem_mb']:.0f} \\\\"
        )
    tex = "\n".join([
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Per-encounter inference cost on a single NVIDIA H200 GPU "
        r"(batch size 1, "
        f"{N_SAMPLES - N_WARMUP} test encounters, "
        r"$N_{\text{warmup}}=" + str(N_WARMUP) + r"$).}",
        r"\label{tab:inference_cost}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Params} & \textbf{Latency (ms)} & \textbf{Throughput (enc/s)} & \textbf{Peak GPU mem (MB)} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    out_path.write_text(tex)
    log.info(f"  Wrote LaTeX block → {out_path}")


def main():
    global N_SAMPLES
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
    args = parser.parse_args()
    N_SAMPLES = args.n_samples

    device = f"cuda:{args.gpu}"
    log.info(f"Device: {device}")
    log.info(f"GPU: {th.cuda.get_device_name(args.gpu)}")

    results = []

    # PORT
    log.info("\n" + "=" * 70)
    log.info("PORT (LoRA single forward pass)")
    log.info("=" * 70)
    try:
        results.append(benchmark_port(device))
    except Exception as e:
        log.error(f"PORT benchmark failed: {e}")
        import traceback; traceback.print_exc()

    # Free PORT GPU mem before LSTM
    th.cuda.empty_cache()

    # LSTM
    log.info("\n" + "=" * 70)
    log.info("BiLSTM (best tuned)")
    log.info("=" * 70)
    try:
        results.append(benchmark_lstm(device))
    except Exception as e:
        log.error(f"LSTM benchmark failed: {e}")
        import traceback; traceback.print_exc()

    # Save
    R_BASE_NEW.mkdir(parents=True, exist_ok=True)
    out_json = R_BASE_NEW / "inference_benchmark.json"
    out_json.write_text(json.dumps(results, indent=2))
    log.info(f"\nResults JSON → {out_json}")

    out_tex = R_BASE_NEW / "inference_benchmark_supp_table.tex"
    write_supp_table(results, out_tex)

    log.info("\n=== Summary ===")
    for r in results:
        log.info(f"  {r['model']:50s}  {r['latency_ms_mean']:6.2f}±{r['latency_ms_std']:.2f} ms  "
                 f"  {r['throughput_per_sec']:5.0f} enc/s  {r['peak_gpu_mem_mb']:5.0f} MB")


if __name__ == "__main__":
    main()
