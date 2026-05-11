"""
finetune.py — ETHOS Linear Probe Fine-tuning for IoD Classification
-------------------------------------------------------------------
Option A (default): Frozen ETHOS backbone + classification head (linear probe)
  - Extract hidden state at OR_ENTRY token (last position before generation)
  - Train 2-layer MLP classifier head (~200K trainable params vs 75M frozen)
  - Expected AUROC: 0.72–0.78

Option B: Full fine-tuning with mixed loss (--full_finetune)
  - L = L_LM + lambda * L_BCE
  - Unfreezes all backbone layers
  - Expected AUROC: 0.78–0.83

Usage:
    conda activate ethos
    CUDA_VISIBLE_DEVICES=0 python ethos/finetune.py
    CUDA_VISIBLE_DEVICES=0 python ethos/finetune.py --full_finetune --lm_lambda 0.1

Output:
    /path/to/CHD_MEDS/results/ethos/finetune/finetune_head_best.pt
    /path/to/CHD_MEDS/results/baselines/ethos_finetune_test_predictions.parquet
    Updates results_summary.csv and re-runs evaluate.py
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_ETHOS_PROJECT_DIR = Path(__file__).parent
_DATASETS_DIR = _ETHOS_PROJECT_DIR / "datasets"
if str(_DATASETS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATASETS_DIR))

from iod_dataset import IoDDataset  # noqa: E402
from ethos.utils import load_model_checkpoint  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── constants (overridable via env vars or --data_root) ──────────────────────
DATA_ROOT     = Path(os.environ.get("CHD_DATA_ROOT", "/path/to/CHD_MEDS"))
TASK_PATH     = DATA_ROOT / "outcome" / "iod_task.parquet"
RESULTS_DIR   = DATA_ROOT / "results" / "ethos" / "finetune"
BASELINES_DIR = DATA_ROOT / "results" / "baselines"
SUMMARY_PATH  = BASELINES_DIR / "results_summary.csv"

DEFAULT_MODEL_FP = DATA_ROOT / "tokenized" / "models" / "chd_layer6_do0.3" / "best_model.pt"
TRAIN_DIR        = DATA_ROOT / "tokenized" / "train"
VAL_DIR          = DATA_ROOT / "tokenized" / "val"
TEST_DIR         = DATA_ROOT / "tokenized" / "test"


# ── label lookup ─────────────────────────────────────────────────────────────

def build_label_dict() -> dict:
    """
    Build {(patient_id_int, prediction_time_us): boolean_value} lookup
    from iod_task.parquet (authoritative ground truth).
    """
    task = pd.read_parquet(TASK_PATH, engine="pyarrow")
    task["patient_id_int"] = (
        pd.to_numeric(task["patient_id"].str.lstrip("C"), errors="coerce")
        .astype("Int64")
    )
    pt = pd.to_datetime(task["prediction_time"])
    if str(pt.dtype) == "datetime64[ns]":
        task["prediction_time_us"] = pt.astype("int64") // 1000
    else:
        task["prediction_time_us"] = pt.astype("int64")

    label_dict = {
        (int(row.patient_id_int), int(row.prediction_time_us)): int(row.boolean_value)
        for row in task.itertuples(index=False)
        if pd.notna(row.patient_id_int)
    }
    log.info(f"Label dict: {len(label_dict):,} encounters")
    return label_dict


# ── hidden-state extraction ───────────────────────────────────────────────────

def get_hidden_state(model, input_ids: th.Tensor) -> th.Tensor:
    """
    Extract hidden state at last-token position from GPT2LMNoBiasModel.
    Supports both raw model and peft-wrapped (LoRA) model.

    Args:
        model:     GPT2LMNoBiasModel or PeftModel wrapping it
        input_ids: (B, T) token tensor

    Returns:
        (B, n_embd) hidden state at the last token position (OR_ENTRY)
    """
    # Get the base transformer (handles both raw and peft-wrapped models)
    base = getattr(model, "base_model", model)
    base = getattr(base, "model", base)  # peft wraps as base_model.model
    transformer = base.transformer

    _, t = input_ids.size()
    tok_emb = transformer.wte(input_ids)
    pos_emb = transformer.wpe(base.pos[:t])
    x = transformer.drop(tok_emb + pos_emb)
    for block in transformer.h:
        out = block(x)
        x = out[0] if isinstance(out, tuple) else out
    x = transformer.ln_f(x)
    return x[:, -1, :]  # (B, n_embd)


# ── dataset ───────────────────────────────────────────────────────────────────

class IoDFinetuneDataset(Dataset):
    """
    Wraps IoDDataset to provide (input_ids, label, patient_id, prediction_time_us)
    for fine-tuning.

    Ground truth comes from iod_task.parquet (boolean_value), not the tokenized
    iod_label (which inflates positives via LDA//ARTERIAL_LIN false positives).

    Only samples with a known label in label_dict are included.

    If window_days is set, only tokens within (OR_entry - window_days) to OR_entry
    are kept; older tokens are replaced with zero-padding (context window ablation).
    """

    def __init__(self, input_dir: Path, n_positions: int, label_dict: dict,
                 window_days: int | None = None):
        self.base = IoDDataset(str(input_dir), n_positions=n_positions)
        self.window_days = window_days

        valid_indices, labels, pids, pts = [], [], [], []
        for idx in range(len(self.base)):
            _, y = self.base[idx]
            pid   = y["patient_id"]
            pt_us = y["prediction_time"] // 1000   # safetensors stores ns → convert to us
            label = label_dict.get((pid, pt_us))
            if label is not None:
                valid_indices.append(idx)
                labels.append(label)
                pids.append(pid)
                pts.append(pt_us)

        self.valid_indices = valid_indices
        self.labels        = np.array(labels,  dtype=np.int32)
        self.pids          = np.array(pids,    dtype=np.int64)
        self.pts           = np.array(pts,     dtype=np.int64)

        # Total sequence length for padding
        self.n_positions = self.base.timeline_size + self.base.context_size
        self.context_size = self.base.context_size

        n_pos  = self.labels.sum()
        n_tot  = len(self.labels)
        log.info(
            f"  {input_dir.name}: {n_tot:,} samples  "
            f"IoD+ {n_pos:,} ({n_pos/max(n_tot,1)*100:.2f}%)  "
            f"n_positions={self.n_positions}  window_days={window_days}"
        )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple[th.Tensor, int, int, int]:
        base_idx = self.valid_indices[idx]
        input_ids, _ = self.base[base_idx]
        # .clone() detaches from memory-mapped safetensors storage
        input_ids = input_ids.clone()

        # Base dataset now returns [pad... | pt_ctx | pre-op timeline | OR_ENTRY]
        # with shape = (n_positions,), OR_ENTRY at position -1, left-padded with zeros.

        # Apply context window: zero out tokens older than window_days before OR_ENTRY
        if self.window_days is not None:
            or_idx = self.base.start_indices[base_idx].item()
            or_time = self.base.times[or_idx].item()  # nanoseconds
            window_ns = int(self.window_days * 24 * 3600 * 1e9)
            cutoff_time = or_time - window_ns

            # input_ids from base dataset: [pad(0s) | patient tokens | OR_ENTRY]
            # OR_ENTRY is always at position -1.
            # We need to map each non-zero position back to a global token index
            # to check its timestamp.
            #
            # The base dataset truncates to n_positions, keeping the most recent
            # tokens. So the rightmost tokens in input_ids correspond to the
            # tokens just before and including OR_ENTRY in the global timeline.

            seq_len = input_ids.size(0)
            # Walk backwards from OR_ENTRY (position -1) to find token times
            for i in range(seq_len - 1, -1, -1):
                if input_ids[i].item() == 0:
                    break  # hit padding, stop
                # Global index: OR_ENTRY is at or_idx, position (seq_len-1)
                # So position i corresponds to global index: or_idx - (seq_len - 1 - i)
                global_idx = or_idx - (seq_len - 1 - i)
                if global_idx < 0 or global_idx >= len(self.base.times):
                    continue
                t = self.base.times[global_idx].item()
                if t != 0 and t < cutoff_time:
                    input_ids[i] = 0

        return input_ids, int(self.labels[idx]), int(self.pids[idx]), int(self.pts[idx])


# ── classification head ───────────────────────────────────────────────────────

class IoDClassificationHead(nn.Module):
    """2-layer MLP classification head on top of ETHOS hidden states."""

    def __init__(self, n_embd: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.xavier_uniform_(self.net[3].weight)

    def forward(self, hidden: th.Tensor) -> th.Tensor:
        return self.net(hidden).squeeze(-1)  # (B,)


# ── training loop ─────────────────────────────────────────────────────────────

def run_epoch_train(model, head, loader, criterion, optimizer, device,
                    full_finetune=False, use_lora=False):
    head.train()
    backbone_trainable = full_finetune or use_lora
    if backbone_trainable:
        model.train()
    total_loss = 0.0

    for input_ids, labels, _, _ in tqdm(loader, desc="  train", leave=False):
        input_ids = input_ids.to(device)
        labels    = labels.float().to(device)

        if backbone_trainable:
            hidden = get_hidden_state(model, input_ids)
        else:
            with th.no_grad():
                hidden = get_hidden_state(model, input_ids)

        logits = head(hidden)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        trainable_params = list(head.parameters())
        if backbone_trainable:
            trainable_params += [p for p in model.parameters() if p.requires_grad]
        nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@th.no_grad()
def run_epoch_eval(model, head, loader, device):
    head.eval()
    model.eval()
    all_probs, all_labels, all_pids, all_pts = [], [], [], []

    for input_ids, labels, pids, pts in loader:
        input_ids = input_ids.to(device)
        hidden    = get_hidden_state(model, input_ids)
        probs     = th.sigmoid(head(hidden)).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())
        all_pids.extend(pids.numpy())
        all_pts.extend(pts.numpy())

    return (
        np.array(all_labels),
        np.array(all_probs),
        np.array(all_pids),
        np.array(all_pts),
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ETHOS fine-tuning for IoD classification")
    parser.add_argument("--model_fp",      type=Path,  default=DEFAULT_MODEL_FP)
    parser.add_argument("--gpu",           type=int,   default=0,
                        help="GPU index among CUDA_VISIBLE_DEVICES (0-based)")
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--hidden_dim",    type=int,   default=256)
    parser.add_argument("--head_dropout",  type=float, default=0.1)
    parser.add_argument("--full_finetune", action="store_true",
                        help="Unfreeze backbone and fine-tune end-to-end with mixed loss")
    parser.add_argument("--lm_lambda",     type=float, default=0.1,
                        help="Weight for LM loss in mixed fine-tuning (Option B)")
    parser.add_argument("--patience",      type=int,   default=10,
                        help="Early stopping patience (epochs without val AUROC improvement)")
    parser.add_argument("--window_days",  type=int,   default=None,
                        help="Context window in days before OR entry (None=all history)")
    parser.add_argument("--pos_weight",   type=float, default=None,
                        help="Override pos_weight (None=auto from train set)")
    parser.add_argument("--suffix",       type=str,   default="",
                        help="Suffix for output filenames (e.g., _window_30d)")
    parser.add_argument("--seed",         type=int,   default=None,
                        help="Random seed for reproducibility (None=random)")
    parser.add_argument("--train_dir",   type=Path,  default=None,
                        help="Override tokenized train dir (default: DATA_ROOT/tokenized/train)")
    parser.add_argument("--val_dir",     type=Path,  default=None,
                        help="Override tokenized val dir")
    parser.add_argument("--test_dir",    type=Path,  default=None,
                        help="Override tokenized test dir")
    parser.add_argument("--results_dir", type=Path,  default=None,
                        help="Override results baselines dir for output parquets")
    parser.add_argument("--lora",        action="store_true",
                        help="Use LoRA adaptation instead of linear probe (backbone partially trainable)")
    parser.add_argument("--lora_r",      type=int,   default=8,
                        help="LoRA rank (default: 8)")
    parser.add_argument("--lora_alpha",  type=int,   default=16,
                        help="LoRA alpha scaling (default: 16)")
    # ── Ablation flags (added 2026-05-11) ──
    parser.add_argument("--train_frac",  type=float, default=1.0,
                        help="Stratified subsample fraction of training set (1.0=all)")
    parser.add_argument("--loss_type",   type=str,   default="weighted_bce",
                        choices=["weighted_bce", "bce", "focal"],
                        help="Loss function (weighted_bce=current, bce=unweighted, focal=alpha-balanced focal)")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Gamma exponent for focal loss")
    parser.add_argument("--focal_alpha", type=float, default=0.25,
                        help="Alpha class-balance weight for focal loss (1=positive class weight)")
    parser.add_argument("--sampler",     type=str,   default="none",
                        choices=["none", "oversample_pos", "undersample_neg"],
                        help="Mini-batch sampling strategy")
    parser.add_argument("--sampler_ratio", type=float, default=5.0,
                        help="Oversample positive multiplier or undersample neg:pos ratio")
    args = parser.parse_args()

    # Override global paths if CLI args provided
    global TRAIN_DIR, VAL_DIR, TEST_DIR, BASELINES_DIR, RESULTS_DIR
    if args.train_dir:
        TRAIN_DIR = args.train_dir
    if args.val_dir:
        VAL_DIR = args.val_dir
    if args.test_dir:
        TEST_DIR = args.test_dir
    if args.results_dir:
        BASELINES_DIR = args.results_dir
        RESULTS_DIR = args.results_dir / "ethos" / "finetune"

    # Set random seed if specified
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        th.manual_seed(args.seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(args.seed)
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False
        log.info(f"Random seed set to {args.seed}")

    device = f"cuda:{args.gpu}" if th.cuda.is_available() and args.gpu >= 0 else "cpu"
    log.info(f"Device: {device}  |  full_finetune: {args.full_finetune}")

    # ── backbone ──
    log.info(f"Loading backbone: {args.model_fp}")
    model, _ = load_model_checkpoint(args.model_fp, map_location=device)
    model.to(device)

    n_embd      = model.config.n_embd
    n_positions = model.config.n_positions
    log.info(f"Backbone params: {model.num_parameters():,}  n_embd={n_embd}  n_pos={n_positions}")

    if args.lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["c_attn"],  # GPT-2 QKV projection
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log.info(f"LoRA mode: r={args.lora_r}, alpha={args.lora_alpha}, "
                 f"trainable={trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    elif not args.full_finetune:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        log.info("Backbone frozen (linear probe mode)")
    else:
        log.info("Backbone unfrozen (full fine-tuning mode)")

    # ── labels + datasets ──
    label_dict = build_label_dict()

    log.info(f"Building datasets (window_days={args.window_days}) …")
    train_ds = IoDFinetuneDataset(TRAIN_DIR, n_positions, label_dict, window_days=args.window_days)
    val_ds   = IoDFinetuneDataset(VAL_DIR,   n_positions, label_dict, window_days=args.window_days)
    test_ds  = IoDFinetuneDataset(TEST_DIR,  n_positions, label_dict, window_days=args.window_days)

    # ── Stratified subsample of training set (data-efficiency ablation) ──
    if args.train_frac < 1.0:
        rng = np.random.RandomState(args.seed if args.seed is not None else 42)
        pos_idx = np.where(train_ds.labels == 1)[0]
        neg_idx = np.where(train_ds.labels == 0)[0]
        n_pos_keep = max(1, int(round(len(pos_idx) * args.train_frac)))
        n_neg_keep = max(1, int(round(len(neg_idx) * args.train_frac)))
        keep_pos = rng.choice(pos_idx, n_pos_keep, replace=False)
        keep_neg = rng.choice(neg_idx, n_neg_keep, replace=False)
        keep = np.sort(np.concatenate([keep_pos, keep_neg]))
        train_ds.valid_indices = [train_ds.valid_indices[int(i)] for i in keep]
        train_ds.labels = train_ds.labels[keep]
        train_ds.pids   = train_ds.pids[keep]
        train_ds.pts    = train_ds.pts[keep]
        log.info(f"Subsampled train to {args.train_frac*100:.1f}%: "
                 f"{n_pos_keep} pos + {n_neg_keep} neg = {len(keep)} total")

    # ── Undersample negatives if requested ──
    if args.sampler == "undersample_neg":
        rng_u = np.random.RandomState(args.seed if args.seed is not None else 42)
        pos_idx = np.where(train_ds.labels == 1)[0]
        neg_idx = np.where(train_ds.labels == 0)[0]
        n_neg_keep = min(len(neg_idx), int(round(len(pos_idx) * args.sampler_ratio)))
        keep_neg = rng_u.choice(neg_idx, n_neg_keep, replace=False)
        keep = np.sort(np.concatenate([pos_idx, keep_neg]))
        train_ds.valid_indices = [train_ds.valid_indices[int(i)] for i in keep]
        train_ds.labels = train_ds.labels[keep]
        train_ds.pids   = train_ds.pids[keep]
        train_ds.pts    = train_ds.pts[keep]
        log.info(f"Undersampled neg to {args.sampler_ratio}:1 → "
                 f"{len(pos_idx)} pos + {n_neg_keep} neg = {len(keep)} total")

    # ── pos_weight policy ──
    if args.loss_type == "bce":
        pos_weight = th.tensor([1.0], dtype=th.float32, device=device)
    elif args.pos_weight is not None:
        pos_weight = th.tensor([args.pos_weight], dtype=th.float32, device=device)
    else:
        pos_weight = th.tensor(
            [(train_ds.labels == 0).sum() / max((train_ds.labels == 1).sum(), 1)],
            dtype=th.float32, device=device,
        )
    log.info(f"loss_type={args.loss_type}, pos_weight={pos_weight.item():.1f}")

    # ── Oversampling sampler (mutually exclusive with undersample, applied above) ──
    if args.sampler == "oversample_pos":
        from torch.utils.data import WeightedRandomSampler
        weights = np.where(train_ds.labels == 1, args.sampler_ratio, 1.0).astype(np.float64)
        sampler = WeightedRandomSampler(weights.tolist(), num_samples=len(weights), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler,
            num_workers=0, pin_memory="cuda" in device,
        )
        log.info(f"Oversampling positives {args.sampler_ratio}× via WeightedRandomSampler")
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory="cuda" in device,
        )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory="cuda" in device,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory="cuda" in device,
    )

    # ── head + optimizer ──
    head = IoDClassificationHead(n_embd, args.hidden_dim, args.head_dropout).to(device)
    n_head_params = sum(p.numel() for p in head.parameters())
    log.info(f"Classification head: {n_head_params:,} trainable params")

    if args.loss_type == "focal":
        class FocalLoss(nn.Module):
            def __init__(self, gamma: float, alpha: float):
                super().__init__()
                self.gamma = gamma
                self.alpha = alpha
            def forward(self, logits, targets):
                bce = nn.functional.binary_cross_entropy_with_logits(
                    logits, targets, reduction="none")
                p  = th.sigmoid(logits)
                pt = th.where(targets == 1, p, 1 - p)
                a  = th.where(targets == 1,
                              th.full_like(p, self.alpha),
                              th.full_like(p, 1 - self.alpha))
                return (a * (1 - pt).pow(self.gamma) * bce).mean()
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
        log.info(f"FocalLoss(gamma={args.focal_gamma}, alpha={args.focal_alpha})")
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if args.full_finetune:
        optimizer = th.optim.AdamW(
            [
                {"params": model.parameters(),  "lr": args.lr * 0.01},  # lower LR for backbone
                {"params": head.parameters(),   "lr": args.lr},
            ],
            weight_decay=1e-2,
        )
    elif args.lora:
        # LoRA: train adapter params (in model) + head params
        lora_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = th.optim.AdamW(
            [
                {"params": lora_params,         "lr": args.lr * 0.1},   # lower LR for LoRA adapters
                {"params": head.parameters(),   "lr": args.lr},
            ],
            weight_decay=1e-2,
        )
        log.info(f"LoRA optimizer: {sum(p.numel() for p in lora_params):,} adapter params + {n_head_params:,} head params")
    else:
        optimizer = th.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-2)

    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)

    best_val_auroc  = 0.0
    best_head_state = None
    best_lora_state = None
    patience_count  = 0

    # ── training loop ──
    mode_name = "LoRA" if args.lora else ("full fine-tune" if args.full_finetune else "linear probe")
    log.info(f"\n=== Training ({mode_name}) ===")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch_train(model, head, train_loader, criterion, optimizer, device,
                                     full_finetune=args.full_finetune, use_lora=args.lora)
        val_labels, val_probs, _, _ = run_epoch_eval(model, head, val_loader, device)
        scheduler.step()

        val_auroc = roc_auc_score(val_labels, val_probs)
        val_auprc = average_precision_score(val_labels, val_probs)

        log.info(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"loss={train_loss:.4f}  "
            f"val_AUROC={val_auroc:.4f}  val_AUPRC={val_auprc:.4f}"
        )

        if val_auroc > best_val_auroc:
            best_val_auroc  = val_auroc
            best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            # For LoRA: also save adapter state at best epoch
            if args.lora:
                best_lora_state = {k: v.cpu().clone() for k, v in model.state_dict().items()
                                   if "lora_" in k}
            patience_count  = 0
            log.info(f"  → Best model saved (val_AUROC={best_val_auroc:.4f})")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                log.info(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # ── test evaluation ──
    log.info("\n=== Test Evaluation ===")
    head.load_state_dict(best_head_state)
    # For LoRA: restore adapter state from best epoch
    if args.lora and best_lora_state:
        model.load_state_dict(best_lora_state, strict=False)
    test_labels, test_probs, test_pids, test_pts = run_epoch_eval(model, head, test_loader, device)

    auroc = roc_auc_score(test_labels, test_probs)
    auprc = average_precision_score(test_labels, test_probs)
    brier = brier_score_loss(test_labels, test_probs)
    n_pos = int(test_labels.sum())
    n_tot = len(test_labels)

    log.info(
        f"\nETHOS (fine-tuned): AUROC={auroc:.4f}  AUPRC={auprc:.4f}  Brier={brier:.4f}"
        f"  (n={n_tot:,}, pos={n_pos:,}, {n_pos/n_tot*100:.2f}%)"
    )

    # ── save predictions ──
    preds_df = pd.DataFrame({
        "y_true":            test_labels.astype(int),
        "y_prob":            test_probs,
        "subject_id":        test_pids,
        "prediction_time_us": test_pts,
    })
    mode_tag  = "fullft" if args.full_finetune else ("lora" if args.lora else "probe")
    suffix = args.suffix
    preds_path = BASELINES_DIR / f"ethos_finetune_{mode_tag}_test_predictions{suffix}.parquet"
    preds_df.to_parquet(preds_path, index=False)
    # Also save as canonical name (only if no suffix, i.e., default run)
    if not suffix:
        canonical_path = BASELINES_DIR / "ethos_finetune_test_predictions.parquet"
        preds_df.to_parquet(canonical_path, index=False)
    log.info(f"Predictions saved → {preds_path}")

    # ── save head checkpoint (timestamped to prevent overwrites) ──
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_payload = {
        "head_state_dict": best_head_state,
        "head_config": {
            "n_embd":      n_embd,
            "hidden_dim":  args.hidden_dim,
            "dropout":     args.head_dropout,
        },
        "lora_state_dict": best_lora_state if args.lora else None,
        "lora_config": {"r": args.lora_r, "alpha": args.lora_alpha} if args.lora else None,
        "backbone_fp":  str(args.model_fp),
        "full_finetune": args.full_finetune,
        "val_auroc":    best_val_auroc,
        "test_auroc":   auroc,
        "test_auprc":   auprc,
        "test_brier":   brier,
        "args":         vars(args),
        "timestamp":    ts,
    }
    # Always save timestamped version (never overwritten)
    ckpt_ts_path = RESULTS_DIR / f"finetune_{mode_tag}_head_{ts}{suffix}.pt"
    th.save(ckpt_payload, ckpt_ts_path)
    log.info(f"Head checkpoint (timestamped) → {ckpt_ts_path}")
    # Also save canonical name for convenience (may be overwritten by future runs)
    ckpt_path = RESULTS_DIR / f"finetune_{mode_tag}_head_best{suffix}.pt"
    th.save(ckpt_payload, ckpt_path)
    log.info(f"Head checkpoint (canonical) → {ckpt_path}")

    # ── update results summary ──
    model_label = "ETHOS (fine-tuned, full)" if args.full_finetune else "ETHOS (fine-tuned)"
    new_row = pd.DataFrame([{
        "model":      model_label,
        "split":      "test",
        "auroc":      auroc,
        "auprc":      auprc,
        "brier":      brier,
        "n_total":    n_tot,
        "n_positive": n_pos,
    }])
    if SUMMARY_PATH.exists():
        summary = pd.read_csv(SUMMARY_PATH)
        summary = summary[summary["model"] != model_label]
        summary = pd.concat([summary, new_row], ignore_index=True)
    else:
        summary = new_row
    summary.to_csv(SUMMARY_PATH, index=False)
    log.info(f"Updated results_summary.csv")

    # ── trigger evaluate.py for updated plots ──
    log.info("Re-running evaluation/evaluate.py …")
    result = subprocess.run(
        [sys.executable, "-m", "evaluation.evaluate"],
        cwd=str(Path(__file__).parent.parent),
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        log.info("evaluate.py completed successfully.")
    else:
        log.warning(f"evaluate.py stderr:\n{result.stderr[-500:]}")

    log.info("\n=== Done ===")


if __name__ == "__main__":
    main()
