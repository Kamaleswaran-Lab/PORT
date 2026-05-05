"""
run_infer.py — ETHOS zero-shot inference for IoD prediction
------------------------------------------------------------
Standalone inference runner for the CHD IoD task.
Does NOT modify ethos-ares; imports only public utilities from the installed
ethos package and uses IoDDataset from our project (ethos/datasets/iod_dataset.py).

Usage (from project root):
    conda activate ethos
    python ethos/run_infer.py \\
        --model_fp  /path/to/CHD_MEDS/tokenized/models/chd_layer6_do0.3/recent_model.pt \\
        --input_dir /path/to/CHD_MEDS/tokenized/test \\
        --output_dir /path/to/CHD_MEDS/results/ethos/iod \\
        --rep_num 100 \\
        --n_gpus 4

Or via:
    bash ethos/infer.sh [rep_num]
"""

import argparse
import sys
from copy import copy
from multiprocessing import Manager, Process, set_start_method
from pathlib import Path
from queue import Empty

import numpy as np
import torch as th
from loguru import logger
from tqdm import tqdm

# Make ethos/datasets importable when running as `python ethos/run_infer.py`
_ETHOS_PROJECT_DIR = Path(__file__).parent
if str(_ETHOS_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_ETHOS_PROJECT_DIR))

from datasets.iod_dataset import IOD_STOP_STOKENS, IoDDataset  # noqa: E402

# Utilities from the installed ethos package (ethos-ares, read-only)
from ethos.inference.constants import Reason  # noqa: E402
from ethos.inference.utils import (  # noqa: E402
    build_token_time_cache,
    get_next_token,
    get_token_time,
    write_results_to_parquet,
)
from ethos.utils import load_model_checkpoint, setup_torch  # noqa: E402


# ---------------------------------------------------------------------------
# Inference worker (runs in a subprocess, one per GPU)
# ---------------------------------------------------------------------------

def _inference_worker(
    job_queue,
    model_fp: str,
    dataset_kwargs: dict,
    progress_queue,
    temperature: float,
    rep_num: int,
    device: str,
    no_compile: bool,
    save_generated_tokens: bool,
):
    # Re-add project ethos/ dir to sys.path for subprocess (spawn creates fresh process)
    _proj_dir = str(Path(__file__).parent)
    if _proj_dir not in sys.path:
        sys.path.insert(0, _proj_dir)
    from datasets.iod_dataset import IoDDataset  # noqa: E402 (subprocess re-import)

    if "cuda" in device:
        th.cuda.set_device(device)
        th.set_float32_matmul_precision("high")
    autocast_ctx = setup_torch(device, dtype="bfloat16" if "cuda" in device else "float32")

    model, _ = load_model_checkpoint(model_fp, map_location=device)
    model.to(device)
    model = th.compile(model, disable=no_compile)

    dataset          = IoDDataset(**dataset_kwargs)
    max_timeline_size = dataset_kwargs["n_positions"]
    ctx_size         = dataset.context_size
    vocab            = dataset.vocab
    stop_stokens     = dataset.stop_stokens
    stop_tokens      = th.tensor(vocab.encode(stop_stokens), dtype=th.long)
    time_limit       = th.tensor(dataset.time_limit.total_seconds() * 1e6)
    token_time_cache = build_token_time_cache(vocab)
    # Pre-build reverse vocab dict for fast decode in hot loop
    vocab_itos       = vocab.itos  # trigger lazy build once

    while True:
        indices = job_queue.get()
        if indices is None:
            break

        for idx in indices:
            timeline, ground_truth = dataset[idx]
            timeline = timeline.to(device).unsqueeze(0).repeat(rep_num, 1)

            gen_token_num = 0
            offset        = 0
            gen_times     = th.zeros(rep_num, dtype=th.float64)
            generated_tokens = [] if save_generated_tokens else None

            while timeline.size(0):
                with autocast_ctx:
                    next_token, probs, _ = get_next_token(
                        model, timeline, return_probs=True, temperature=temperature
                    )

                if generated_tokens is not None:
                    generated_tokens.append(next_token)

                if not offset and timeline.size(1) == max_timeline_size:
                    offset = 1

                timeline = th.cat(
                    (timeline[:, :ctx_size], timeline[:, ctx_size + offset:], next_token),
                    dim=1,
                )
                gen_token_num += 1

                new_token  = next_token.cpu().view(-1)
                gen_times += get_token_time(new_token, vocab, cache=token_time_cache)

                completed = th.isin(new_token, stop_tokens) | (gen_times > time_limit)

                if not completed.any():
                    continue

                for i in th.nonzero(completed).view(-1):
                    actual_token = next_token[i].item()
                    token_time   = gen_times[i]
                    stop_reason  = Reason.GOT_TOKEN

                    if token_time > time_limit:
                        stop_reason = Reason.TIME_LIMIT

                    if th.isinf(token_time):
                        actual_stoken = str(actual_token)
                        stop_reason   = Reason.KEY_ERROR
                        token_time    = None
                    else:
                        actual_stoken = vocab_itos.get(actual_token, str(actual_token))
                        token_time    = round(token_time.item())

                    gt = copy(ground_truth)
                    result = {
                        "expected":        gt.pop("expected"),
                        "actual":          actual_stoken,
                        "stop_reason":     stop_reason,
                        "actual_prob":     probs[i, actual_token].item(),
                        **dict(zip(stop_stokens, probs[i, stop_tokens].tolist())),
                        "true_token_time": gt.pop("true_token_time"),
                        "token_time":      token_time,
                        "true_token_dist": gt.pop("true_token_dist"),
                        "token_dist":      gen_token_num,
                        **gt,
                    }
                    if generated_tokens is not None:
                        result["generated_tokens"] = [t[i].item() for t in generated_tokens]
                    progress_queue.put(result)

                if completed.all():
                    break

                mask      = ~completed
                timeline  = timeline[mask]
                gen_times = gen_times[mask]
                if generated_tokens is not None:
                    generated_tokens = [t[mask] for t in generated_tokens]


def _producer(subsets, queue, num_proc):
    for subset in subsets:
        queue.put(subset)
    for _ in range(num_proc):
        queue.put(None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ETHOS IoD zero-shot inference")
    parser.add_argument("--model_fp",             required=True,       help="Path to model .pt checkpoint")
    parser.add_argument("--input_dir",            required=True,       help="Path to tokenized test split dir")
    parser.add_argument("--output_dir",           required=True,       help="Results output directory")
    parser.add_argument("--output_fn",            default=None,        help="Optional subdirectory name for results")
    parser.add_argument("--rep_num",              type=int,   default=100,  help="Trajectories per encounter (default 100)")
    parser.add_argument("--n_gpus",               type=int,   default=1,    help="Number of GPUs")
    parser.add_argument("--n_jobs",               type=int,   default=1,    help="Workers per GPU")
    parser.add_argument("--chunksize",            type=int,   default=32,   help="Encounters per job chunk")
    parser.add_argument("--temperature",          type=float, default=1.0,  help="Sampling temperature")
    parser.add_argument("--seed",                 type=int,   default=42)
    parser.add_argument("--timeout",              type=int,   default=600,  help="Worker timeout (seconds)")
    parser.add_argument("--result_chunk_size",    type=int,   default=1000, help="Flush to parquet every N results")
    parser.add_argument("--no_compile",           action="store_true",  help="Disable torch.compile")
    parser.add_argument("--save_generated_tokens",action="store_true",  help="Save full generated token sequences")
    args = parser.parse_args()

    # Load model config to get n_positions
    ckpt = th.load(args.model_fp, map_location="cpu", mmap=True, weights_only=False)
    model_config = ckpt["model_config"]
    n_positions  = (
        model_config.decoder.n_positions
        if model_config.is_encoder_decoder
        else model_config.n_positions
    )

    dataset_kwargs = {
        "input_dir":          args.input_dir,
        "n_positions":        n_positions,
        "is_encoder_decoder": model_config.is_encoder_decoder,
    }

    # Initialize dataset on main process for logging
    dataset = IoDDataset(**dataset_kwargs)
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Time limit: {dataset.time_limit}")
    logger.info(f"Stop tokens ({len(dataset.stop_stokens)}): {dataset.stop_stokens[:5]} ...")

    n_samples = len(dataset)
    logger.info(f"Samples: {n_samples:,}  |  rep_num: {args.rep_num}  |  total trajectories: {n_samples * args.rep_num:,}")

    result_dir = Path(args.output_dir)
    if args.output_fn:
        result_dir = result_dir / args.output_fn
    result_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results → {result_dir}")

    # Shuffle and chunk indices
    np.random.seed(args.seed)
    indices    = np.random.choice(np.arange(n_samples), n_samples, replace=False)
    chunk_num  = max(n_samples // args.chunksize, 1)
    subsets    = list(np.array_split(indices, chunk_num))

    num_proc   = min(args.n_jobs * args.n_gpus, len(subsets))
    use_cpu    = args.n_gpus == 0
    logger.info(f"Launching {num_proc} worker(s) on {'CPU' if use_cpu else f'{args.n_gpus} GPU(s)'}")

    set_start_method("spawn")
    with Manager() as mgr:
        job_queue  = mgr.Queue(maxsize=num_proc * 2)
        prog_queue = mgr.Queue()

        processes = [
            Process(target=_producer, args=(subsets, job_queue, num_proc), name="producer")
        ]
        processes += [
            Process(
                target=_inference_worker,
                args=(
                    job_queue,
                    args.model_fp,
                    dataset_kwargs,
                    prog_queue,
                    args.temperature,
                    args.rep_num,
                    "cpu" if use_cpu else f"cuda:{i % args.n_gpus}",
                    args.no_compile,
                    args.save_generated_tokens,
                ),
                name=f"Worker_{i}",
            )
            for i in range(num_proc)
        ]

        for p in processes:
            p.start()

        results          = []
        total            = n_samples * args.rep_num
        generated_count  = 0
        pbar = tqdm(total=total, desc="Inference", unit="traj", smoothing=0.1)

        try:
            for _ in range(total):
                results.append(prog_queue.get(timeout=args.timeout))
                generated_count += results[-1]["token_dist"]
                pbar.set_postfix_str(f"tokens: {generated_count:,}")
                pbar.update()

                if len(results) >= args.result_chunk_size:
                    write_results_to_parquet(result_dir, results, pbar.format_dict["n"])
                    results = []

        except Empty:
            logger.error("Timed out waiting for workers. Saving partial results.")
            for p in processes:
                if p.is_alive():
                    p.terminate()

        for p in processes:
            p.join()

    if results:
        write_results_to_parquet(result_dir, results, pbar.format_dict["n"])

    logger.info(f"Done. Results saved to {result_dir}")


if __name__ == "__main__":
    main()
