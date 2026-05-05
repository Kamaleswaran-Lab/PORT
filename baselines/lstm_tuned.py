"""
lstm_tuned.py
-------------
Hyperparameter-tuned BiLSTM baseline for IoD prediction on MEDS sequences.

Tuning protocol:
  - Build datasets once (vocab_size=2000, seq_len=512), reuse across configs.
  - For each architecture/training config, train up to N epochs with
    early stopping (patience=5 on val AUPRC).
  - Pick best config by best-epoch val AUPRC.
  - Final eval on test using the best checkpoint.

Usage:
    conda activate ethos
    python -m baselines.lstm_tuned [--max_epochs 25] [--patience 5]
                                   [--output_dir DIR]
"""

import argparse
import json
import logging
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
)

from baselines.lstm import (
    IoDSequenceDataset, IoDLSTM, build_code_vocab, collate_fn, run_epoch,
    MAX_SEQ_LEN, VOCAB_SIZE, EMBED_DIM,
)
from baselines.features import load_task, load_events, load_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("/path/to/CHD_MEDS/results/baselines_tuned")


# Config grid (vocab=2000, seq_len=512 fixed; vary architecture + training)
CONFIGS = [
    {"hidden_dim": 128, "num_layers": 2, "dropout": 0.3, "lr": 1e-3, "embed_dim": 64,  "name": "h128_l2_do0.3_lr1e-3_e64"},
    {"hidden_dim": 256, "num_layers": 2, "dropout": 0.3, "lr": 1e-3, "embed_dim": 64,  "name": "h256_l2_do0.3_lr1e-3_e64"},
    {"hidden_dim": 512, "num_layers": 2, "dropout": 0.3, "lr": 1e-3, "embed_dim": 64,  "name": "h512_l2_do0.3_lr1e-3_e64"},
    {"hidden_dim": 256, "num_layers": 3, "dropout": 0.3, "lr": 1e-3, "embed_dim": 64,  "name": "h256_l3_do0.3_lr1e-3_e64"},
    {"hidden_dim": 256, "num_layers": 2, "dropout": 0.5, "lr": 1e-3, "embed_dim": 64,  "name": "h256_l2_do0.5_lr1e-3_e64"},
    {"hidden_dim": 256, "num_layers": 2, "dropout": 0.1, "lr": 1e-3, "embed_dim": 64,  "name": "h256_l2_do0.1_lr1e-3_e64"},
    {"hidden_dim": 256, "num_layers": 2, "dropout": 0.3, "lr": 5e-4, "embed_dim": 64,  "name": "h256_l2_do0.3_lr5e-4_e64"},
    {"hidden_dim": 256, "num_layers": 2, "dropout": 0.3, "lr": 1e-3, "embed_dim": 128, "name": "h256_l2_do0.3_lr1e-3_e128"},
]


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_config(
    cfg, train_loader, val_loader, test_loader, device, pos_weight,
    max_epochs=25, patience=5, ckpt_path=None,
):
    log.info(f"\n--- Config: {cfg['name']} ---")
    log.info(f"    hidden={cfg['hidden_dim']} layers={cfg['num_layers']} dropout={cfg['dropout']}"
             f" lr={cfg['lr']} embed={cfg['embed_dim']}")

    model = IoDLSTM(
        vocab_size=VOCAB_SIZE + 2,
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"    Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auprc = -1.0
    best_val_auroc = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        tr_loss, tr_auroc, tr_auprc, _, _ = run_epoch(
            model, train_loader, optimizer, criterion, device, train=True,
        )
        val_loss, val_auroc, val_auprc, _, _ = run_epoch(
            model, val_loader, optimizer, criterion, device, train=False,
        )
        scheduler.step(val_auprc)
        dt = time.time() - t0

        log.info(
            f"    Epoch {epoch:02d}/{max_epochs}  ({dt:.0f}s)  "
            f"tr_loss={tr_loss:.4f} tr_auroc={tr_auroc:.4f}  |  "
            f"val_loss={val_loss:.4f} val_auroc={val_auroc:.4f} val_auprc={val_auprc:.4f}"
        )
        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_auroc": tr_auroc, "train_auprc": tr_auprc,
            "val_loss": val_loss, "val_auroc": val_auroc, "val_auprc": val_auprc,
        })

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_val_auroc = val_auroc
            best_epoch = epoch
            if ckpt_path is not None:
                torch.save(model.state_dict(), ckpt_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                log.info(f"    Early stop at epoch {epoch} (no improvement for {patience} epochs)")
                break

    log.info(f"    → Best epoch {best_epoch}: val AUROC={best_val_auroc:.4f}  AUPRC={best_val_auprc:.4f}")

    # Load best ckpt and evaluate on test
    if ckpt_path is not None and ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    _, test_auroc, test_auprc, test_probs, test_labels = run_epoch(
        model, test_loader, optimizer, criterion, device, train=False,
    )
    test_brier = brier_score_loss(test_labels, test_probs)
    log.info(f"    TEST: AUROC={test_auroc:.4f}  AUPRC={test_auprc:.4f}  Brier={test_brier:.4f}")

    return {
        "config": cfg,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "val_auroc": best_val_auroc,
        "val_auprc": best_val_auprc,
        "test_auroc": test_auroc,
        "test_auprc": test_auprc,
        "test_brier": test_brier,
        "test_probs": test_probs,
        "test_labels": test_labels,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=25)
    parser.add_argument("--patience",   type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Load data
    log.info("Loading task and events …")
    task = load_task()
    events = load_events()
    splits = load_splits()
    task = task.merge(splits[["subject_id", "split"]], on="subject_id", how="left")
    task["split"] = task["split"].fillna("train")

    train_task = task[task["split"] == "train"].reset_index(drop=True)
    val_task   = task[task["split"] == "val"].reset_index(drop=True)
    test_task  = task[task["split"] == "test"].reset_index(drop=True)
    log.info(f"  Train: {len(train_task):,} | Val: {len(val_task):,} | Test: {len(test_task):,}")

    # Vocab + datasets (shared across all configs)
    log.info("Building code vocabulary …")
    code_vocab = build_code_vocab(events, train_task, top_n=VOCAB_SIZE)
    with open(output_dir / "lstm_tuned_code_vocab.pkl", "wb") as f:
        pickle.dump(code_vocab, f)

    log.info("Building datasets …")
    train_ds = IoDSequenceDataset(train_task, events, code_vocab, args.max_seq_len)
    val_ds   = IoDSequenceDataset(val_task,   events, code_vocab, args.max_seq_len)
    test_ds  = IoDSequenceDataset(test_task,  events, code_vocab, args.max_seq_len)

    pos_rate = train_task["boolean_value"].mean()
    pos_weight = torch.tensor([(1 - pos_rate) / pos_rate], dtype=torch.float, device=device)
    log.info(f"  pos_weight: {pos_weight.item():.1f}x")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # Run all configs
    all_results = []
    for i, cfg in enumerate(CONFIGS, 1):
        log.info(f"\n{'='*70}")
        log.info(f"[{i}/{len(CONFIGS)}] Config: {cfg['name']}")
        log.info(f"{'='*70}")
        ckpt_path = output_dir / f"lstm_tuned_ckpt_{cfg['name']}.pt"
        try:
            res = train_one_config(
                cfg, train_loader, val_loader, test_loader, device, pos_weight,
                max_epochs=args.max_epochs, patience=args.patience, ckpt_path=ckpt_path,
            )
            all_results.append(res)
        except Exception as e:
            log.error(f"    Config {cfg['name']} failed: {e}")
            continue

    # Sort by val AUPRC and pick best
    all_results.sort(key=lambda r: r["val_auprc"], reverse=True)
    best = all_results[0]

    log.info("\n" + "=" * 70)
    log.info("Tuning complete. Summary (sorted by val AUPRC):")
    log.info("=" * 70)
    summary_rows = []
    for r in all_results:
        cfg = r["config"]
        log.info(f"  {cfg['name']:35s}  val: AUROC={r['val_auroc']:.4f} AUPRC={r['val_auprc']:.4f}"
                 f"  | test: AUROC={r['test_auroc']:.4f} AUPRC={r['test_auprc']:.4f}")
        summary_rows.append({
            "config_name": cfg["name"],
            "hidden_dim": cfg["hidden_dim"], "num_layers": cfg["num_layers"],
            "dropout": cfg["dropout"], "lr": cfg["lr"], "embed_dim": cfg["embed_dim"],
            "n_params": r["n_params"], "best_epoch": r["best_epoch"],
            "val_auroc": r["val_auroc"], "val_auprc": r["val_auprc"],
            "test_auroc": r["test_auroc"], "test_auprc": r["test_auprc"],
            "test_brier": r["test_brier"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "lstm_tuned_summary.csv", index=False)

    log.info(f"\n→ Best: {best['config']['name']}")
    log.info(f"  Test AUROC={best['test_auroc']:.4f} AUPRC={best['test_auprc']:.4f}")

    # Save best test predictions
    test_enc = test_task.reset_index(drop=True)
    pred_df = pd.DataFrame({
        "subject_id":    test_enc["subject_id"],
        "encounter_csn": test_enc["encounter_csn"],
        "y_true":        best["test_labels"].astype(int),
        "y_prob":        best["test_probs"],
    })
    pred_df.to_parquet(output_dir / "lstm_tuned_test_predictions.parquet", index=False)

    with open(output_dir / "lstm_tuned_best.json", "w") as f:
        json.dump({
            "best_config": best["config"],
            "best_epoch": best["best_epoch"],
            "val_auroc": best["val_auroc"],
            "val_auprc": best["val_auprc"],
            "test_auroc": best["test_auroc"],
            "test_auprc": best["test_auprc"],
            "test_brier": best["test_brier"],
        }, f, indent=2)

    log.info(f"\nResults saved → {output_dir}")


if __name__ == "__main__":
    main()
