"""
lstm.py
-------
LSTM baseline for IoD prediction on MEDS event sequences.

Architecture:
  - Input: sequence of (code_id, numeric_value, time_delta) per encounter
  - Embedding: code → learned embedding (dim=64)
  - LSTM: 2-layer bidirectional LSTM (hidden=128)
  - Output: sigmoid over final hidden state → IoD probability

Sequence construction (per encounter):
  - Take all events with time <= prediction_time (In OR)
  - Sort by time (NaT/static events first)
  - Truncate/pad to max_seq_len=512
  - Code vocabulary: top 2000 codes by frequency (from train set)

Usage:
    conda activate tccc
    python baselines/lstm.py [--epochs 20] [--batch_size 64] [--output_dir DIR]
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from baselines.features import load_task, load_events, load_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

import os
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("CHD_DATA_ROOT", "/path/to/CHD_MEDS")
) / "results" / "baselines"

MAX_SEQ_LEN   = 512
VOCAB_SIZE    = 2000   # top-N codes from train set
EMBED_DIM     = 64
HIDDEN_DIM    = 128
NUM_LAYERS    = 2
DROPOUT       = 0.3
PAD_IDX       = 0      # padding token
UNK_IDX       = 1      # unknown code token


# ── Dataset ───────────────────────────────────────────────────────────────────

class IoDSequenceDataset(Dataset):
    """
    Each item is a single encounter (subject_id, encounter_csn).
    Returns:
      codes        : LongTensor [seq_len]  — code indices (padded)
      numeric_vals : FloatTensor [seq_len] — numeric value (0.0 if NaN)
      time_deltas  : FloatTensor [seq_len] — hours since previous event (0 for first)
      lengths      : int — actual sequence length (before padding)
      label        : float — IoD boolean_value
    """
    def __init__(
        self,
        encounters: pd.DataFrame,   # task rows for this split
        events: pd.DataFrame,       # all events
        code_vocab: dict,           # code → int index
        max_seq_len: int = MAX_SEQ_LEN,
        window_days: int | None = None,
    ):
        self.encounters = encounters.reset_index(drop=True)
        self.events = events
        self.code_vocab = code_vocab
        self.max_seq_len = max_seq_len
        self.window_days = window_days

        # Pre-filter events to only relevant patients
        valid_ids = set(encounters["subject_id"].unique())
        self.events = events[events["subject_id"].isin(valid_ids)].copy()
        self.events["time"] = pd.to_datetime(self.events["time"])

    def __len__(self):
        return len(self.encounters)

    def __getitem__(self, idx):
        row = self.encounters.iloc[idx]
        sid = row["subject_id"]
        pred_time = pd.to_datetime(row["prediction_time"])
        label = float(row["boolean_value"])

        # Get pre-op events for this patient
        pat_events = self.events[self.events["subject_id"] == sid].copy()
        mask = pat_events["time"].isna() | (pat_events["time"] <= pred_time)
        if self.window_days is not None:
            window_start = pred_time - pd.Timedelta(days=self.window_days)
            mask = mask & (pat_events["time"].isna() | (pat_events["time"] >= window_start))
        pat_events = pat_events[mask].copy()

        # Sort: NaT (static) first, then chronological
        pat_events = pat_events.sort_values("time", na_position="first").reset_index(drop=True)

        # Truncate to max_seq_len (keep most recent)
        if len(pat_events) > self.max_seq_len:
            n_static = pat_events["time"].isna().sum()
            pat_events = pd.concat([
                pat_events.iloc[:n_static],
                pat_events.iloc[-(self.max_seq_len - n_static):]
            ]).reset_index(drop=True)

        seq_len = len(pat_events)

        # Code indices
        codes = np.array([
            self.code_vocab.get(c, UNK_IDX) for c in pat_events["code"]
        ], dtype=np.int64)

        # Numeric values (normalize: clip to [-5, 5] z-score range)
        numeric = pat_events["numeric_value"].fillna(0.0).values.astype(np.float32)
        numeric = np.clip(numeric, -5.0, 5.0)

        # Time deltas in hours (NaT → 0)
        times = pat_events["time"].values  # ns timestamps or NaT
        time_deltas = np.zeros(seq_len, dtype=np.float32)
        prev_time = None
        for i, t in enumerate(pat_events["time"]):
            if pd.isna(t):
                time_deltas[i] = 0.0
            elif prev_time is None or pd.isna(prev_time):
                time_deltas[i] = 0.0
                prev_time = t
            else:
                delta_hrs = (t - prev_time).total_seconds() / 3600.0
                time_deltas[i] = min(delta_hrs, 8760.0)  # cap at 1 year
                prev_time = t

        # Pad to max_seq_len
        pad_len = self.max_seq_len - seq_len
        codes        = np.pad(codes,        (0, pad_len), constant_values=PAD_IDX)
        numeric      = np.pad(numeric,      (0, pad_len), constant_values=0.0)
        time_deltas  = np.pad(time_deltas,  (0, pad_len), constant_values=0.0)

        return (
            torch.tensor(codes,       dtype=torch.long),
            torch.tensor(numeric,     dtype=torch.float),
            torch.tensor(time_deltas, dtype=torch.float),
            seq_len,
            torch.tensor(label,       dtype=torch.float),
        )


def collate_fn(batch):
    codes, numeric, deltas, lengths, labels = zip(*batch)
    return (
        torch.stack(codes),
        torch.stack(numeric),
        torch.stack(deltas),
        torch.tensor(lengths, dtype=torch.long),
        torch.stack(labels),
    )


# ── Model ─────────────────────────────────────────────────────────────────────

class IoDLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE + 2,  # +2 for PAD and UNK
        embed_dim:  int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        dropout:    float = DROPOUT,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)

        # Project (embed + numeric + time_delta) → input to LSTM
        self.input_proj = nn.Linear(embed_dim + 2, embed_dim)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional

    def forward(self, codes, numeric, deltas, lengths):
        # codes: [B, L], numeric/deltas: [B, L]
        emb = self.embedding(codes)                          # [B, L, E]
        x = torch.cat([emb, numeric.unsqueeze(-1), deltas.unsqueeze(-1)], dim=-1)  # [B, L, E+2]
        x = torch.relu(self.input_proj(x))                  # [B, L, E]

        # Pack padded sequences for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, (h_n, _) = self.lstm(packed)

        # Concatenate final forward and backward hidden states
        # h_n: [num_layers * 2, B, hidden_dim]
        h_fwd = h_n[-2]   # last layer, forward
        h_bwd = h_n[-1]   # last layer, backward
        h = torch.cat([h_fwd, h_bwd], dim=-1)   # [B, hidden_dim*2]

        h = self.dropout(h)
        logit = self.classifier(h).squeeze(-1)   # [B]
        return logit


# ── Training / Evaluation ─────────────────────────────────────────────────────

def build_code_vocab(train_events: pd.DataFrame, train_task: pd.DataFrame, top_n: int = VOCAB_SIZE) -> dict:
    """Build code→index vocab from train events (top_n by frequency). 0=PAD, 1=UNK."""
    valid_ids = set(train_task["subject_id"].unique())
    train_ev = train_events[train_events["subject_id"].isin(valid_ids)]
    top_codes = train_ev["code"].value_counts().head(top_n).index.tolist()
    vocab = {code: idx + 2 for idx, code in enumerate(top_codes)}  # 0=PAD, 1=UNK
    log.info(f"  Code vocab: {len(vocab):,} codes (PAD=0, UNK=1, codes=2..{len(vocab)+1})")
    return vocab


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train(train)
    total_loss = 0.0
    all_probs, all_labels = [], []

    with torch.set_grad_enabled(train):
        for codes, numeric, deltas, lengths, labels in loader:
            codes   = codes.to(device)
            numeric = numeric.to(device)
            deltas  = deltas.to(device)
            lengths = lengths.to(device)
            labels  = labels.to(device)

            logits = model(codes, numeric, deltas, lengths)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    n = len(all_labels)
    avg_loss = total_loss / n
    auroc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan")
    auprc = average_precision_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan")
    return avg_loss, auroc, auprc, np.array(all_probs), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--max_seq_len", type=int,   default=MAX_SEQ_LEN)
    parser.add_argument("--output_dir",  type=str,   default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--window_days", type=int,   default=None,
                        help="Context window in days before OR entry (None=all)")
    parser.add_argument("--train_frac",  type=float, default=1.0,
                        help="Stratified subsample fraction of training set (1.0=all)")
    parser.add_argument("--unweighted",  action="store_true",
                        help="Disable pos_weight in BCE loss (uses standard unweighted BCE)")
    parser.add_argument("--suffix",      type=str,   default="",
                        help="Suffix for output filenames (e.g., _window_30d)")
    parser.add_argument("--seed",        type=int,   default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set random seed if specified
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        log.info(f"Random seed set to {args.seed}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Load data
    log.info("Loading task and events …")
    task   = load_task()
    events = load_events()
    splits = load_splits()

    # Add split column to task
    task = task.merge(splits[["subject_id", "split"]], on="subject_id", how="left")
    task["split"] = task["split"].fillna("train")

    train_task = task[task["split"] == "train"].reset_index(drop=True)
    val_task   = task[task["split"] == "val"].reset_index(drop=True)
    test_task  = task[task["split"] == "test"].reset_index(drop=True)

    # Stratified subsample of training set (data-efficiency ablation)
    if args.train_frac < 1.0:
        rng_sub = np.random.RandomState(args.seed if args.seed is not None else 42)
        pos = train_task[train_task["boolean_value"] == True]
        neg = train_task[train_task["boolean_value"] == False]
        n_pos_keep = max(1, int(round(len(pos) * args.train_frac)))
        n_neg_keep = max(1, int(round(len(neg) * args.train_frac)))
        pos_keep = pos.sample(n=n_pos_keep, random_state=rng_sub).reset_index(drop=True)
        neg_keep = neg.sample(n=n_neg_keep, random_state=rng_sub).reset_index(drop=True)
        train_task = pd.concat([pos_keep, neg_keep], ignore_index=True).sample(
            frac=1.0, random_state=rng_sub).reset_index(drop=True)
        log.info(f"  Subsampled train to {args.train_frac*100:.1f}%: "
                 f"{n_pos_keep} pos + {n_neg_keep} neg = {len(train_task)} total")

    log.info(f"  Train: {len(train_task):,} | Val: {len(val_task):,} | Test: {len(test_task):,}")
    log.info(f"  Train IoD+: {train_task['boolean_value'].mean()*100:.1f}%")

    # Build vocab from train
    log.info("Building code vocabulary …")
    code_vocab = build_code_vocab(events, train_task, top_n=VOCAB_SIZE)

    # Datasets
    log.info("Building datasets …")
    train_ds = IoDSequenceDataset(train_task, events, code_vocab, args.max_seq_len, window_days=args.window_days)
    val_ds   = IoDSequenceDataset(val_task,   events, code_vocab, args.max_seq_len, window_days=args.window_days)
    test_ds  = IoDSequenceDataset(test_task,  events, code_vocab, args.max_seq_len, window_days=args.window_days)

    # Class weight for imbalance
    pos_rate = train_task["boolean_value"].mean()
    if args.unweighted:
        pos_weight = torch.tensor([1.0], dtype=torch.float, device=device)
        log.info(f"  pos_weight: 1.0 (unweighted BCE)")
    else:
        pos_weight = torch.tensor([(1 - pos_rate) / pos_rate], dtype=torch.float, device=device)
        log.info(f"  pos_weight: {pos_weight.item():.1f}x")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # Model
    model = IoDLSTM(vocab_size=VOCAB_SIZE + 2).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Training loop
    best_val_auroc = 0.0
    best_model_path = output_dir / f"lstm_best{args.suffix}.pt"
    results = []

    log.info("=== Training LSTM ===")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_auroc, tr_auprc, _, _ = run_epoch(
            model, train_loader, optimizer, criterion, device, train=True
        )
        val_loss, val_auroc, val_auprc, val_probs, val_labels = run_epoch(
            model, val_loader, optimizer, criterion, device, train=False
        )
        scheduler.step(val_auroc)

        log.info(
            f"Epoch {epoch:02d}/{args.epochs}  "
            f"train_loss={tr_loss:.4f} auroc={tr_auroc:.4f}  |  "
            f"val_loss={val_loss:.4f} auroc={val_auroc:.4f} auprc={val_auprc:.4f}"
        )

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save(model.state_dict(), best_model_path)
            log.info(f"  → Best model saved (val AUROC={val_auroc:.4f})")

        results.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_auroc": tr_auroc, "train_auprc": tr_auprc,
            "val_loss":   val_loss, "val_auroc":   val_auroc, "val_auprc":   val_auprc,
        })

    # Save training history
    pd.DataFrame(results).to_csv(
        output_dir / f"lstm_training_history{args.suffix}.csv", index=False)

    # Evaluate best model on test set
    log.info("\n=== Test Evaluation (best checkpoint) ===")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    _, test_auroc, test_auprc, test_probs, test_labels = run_epoch(
        model, test_loader, optimizer, criterion, device, train=False
    )
    test_brier = brier_score_loss(test_labels, test_probs)
    log.info(
        f"  Test AUROC={test_auroc:.4f}  AUPRC={test_auprc:.4f}  Brier={test_brier:.4f}"
        f"  (n={len(test_labels):,}, pos={int(test_labels.sum()):,})"
    )

    # Save test predictions
    test_enc = test_task.reset_index(drop=True)
    pred_df = pd.DataFrame({
        "subject_id":    test_enc["subject_id"],
        "encounter_csn": test_enc["encounter_csn"],
        "y_true":        test_labels.astype(int),
        "y_prob":        test_probs,
    })
    out_name = f"lstm_test_predictions{args.suffix}.parquet"
    pred_df.to_parquet(output_dir / out_name, index=False)
    log.info(f"  Predictions saved → {output_dir}/{out_name}")

    # Save vocab
    with open(output_dir / "lstm_code_vocab.pkl", "wb") as f:
        pickle.dump(code_vocab, f)

    # Summary row
    summary = pd.DataFrame([{
        "model": "LSTM",
        "feature_set": f"seq(vocab={VOCAB_SIZE},len={args.max_seq_len})",
        "split": "test",
        "auroc": test_auroc,
        "auprc": test_auprc,
        "brier": test_brier,
        "n_total": len(test_labels),
        "n_pos": int(test_labels.sum()),
    }])
    summary.to_csv(output_dir / "lstm_results.csv", index=False)
    log.info(f"  Summary saved → {output_dir}/lstm_results.csv")


if __name__ == "__main__":
    main()
