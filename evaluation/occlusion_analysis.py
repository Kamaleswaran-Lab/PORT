"""
occlusion_analysis.py
---------------------
Category-level occlusion analysis for ETHOS fine-tuned model.
Masks entire token categories (LAB, MED, DIAGNOSIS, etc.) and measures AUROC/AUPRC drop.

Also generates confusion matrices at optimal threshold for all models.

Usage:
    conda activate ethos
    CUDA_VISIBLE_DEVICES=X python -m evaluation.occlusion_analysis
"""
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Paths
MODEL_FP_NFS = Path("/path/to/CHD_MEDS/tokenized/models/chd_layer6_do0.3/best_model.pt")
MODEL_FP_LOCAL = Path("/tmp/chd_best_model.pt")
MODEL_FP   = MODEL_FP_LOCAL if MODEL_FP_LOCAL.exists() else MODEL_FP_NFS
TEST_DIR_NFS = Path("/path/to/CHD_MEDS/tokenized/test")
TEST_DIR_LOCAL = Path("/tmp/chd_test")
TEST_DIR   = TEST_DIR_LOCAL if TEST_DIR_LOCAL.exists() else TEST_DIR_NFS
TASK_PATH  = Path("/path/to/CHD_MEDS/outcome/iod_task.parquet")
BASELINES  = Path("/path/to/CHD_MEDS/results/baselines")
OUTPUT_DIR = Path("/path/to/CHD_MEDS/results/evaluation")

# Token category prefixes for occlusion
# Display names are clinical-facing; internal prefixes map to MEDS token names
CATEGORIES = {
    "Surgical Context":  ["ENCOUNTER//"],   # procedure type, ASA, admission type, cardiac dx
    "Care Trajectory":   ["ADT//"],         # department transfers (ICU, OR, floor)
    "Surgical History":  ["PROCEDURE//"],   # prior operations
    "Medications":       ["MED//"],
    "Problem List":      ["PROBLEM//"],
    "Medical History":   ["DIAGNOSIS//"],
    "Laboratory":        ["LAB//"],
    "Lines/Drains":      ["LDA//"],
    "Anesthesia Events": ["AN_EVENT//"],
    "Transfusions":      ["TRANSFUSION//"],
    "Demographics":      ["DEMO//"],
    "Structured Data":   ["SDE//"],         # Smart Data Elements
    "Vital Signs":       ["VITAL//"],       # HR, SBP, DBP, O2, Temp, Weight, Height, RR
    "Time Gaps":         ["5m-", "15m-", "45m-", "1h", "2h", "3h", "5h", "8h", "12h", "18h", "1d", "2d", "1w", "2w", "1mt", "2mt", "6mt"],
}


def get_hidden_state(model, input_ids):
    """Extract hidden state at last position from frozen backbone."""
    _, t = input_ids.size()
    tok_emb = model.transformer.wte(input_ids)
    pos_emb = model.transformer.wpe(th.arange(t, device=input_ids.device))
    x = model.transformer.drop(tok_emb + pos_emb)
    for block in model.transformer.h:
        out = block(x)
        x = out[0] if isinstance(out, tuple) else out
    x = model.transformer.ln_f(x)
    return x[:, -1, :]


def run_occlusion(model, head, ds, vocab, device, batch_size=64):
    """Run occlusion analysis: mask each category and measure performance."""
    from torch.utils.data import DataLoader

    # Build category -> token ID sets
    cat_token_ids = {}
    for cat_name, prefixes in CATEGORIES.items():
        ids = set()
        for token_str, token_id in vocab.stoi.items():
            for prefix in prefixes:
                if str(token_str).startswith(prefix):
                    ids.add(token_id)
                    break
        cat_token_ids[cat_name] = ids
        log.info(f"  {cat_name}: {len(ids)} token IDs")

    # Get baseline (no occlusion) predictions
    log.info("Computing baseline predictions (no occlusion)...")
    all_labels, all_probs_baseline = [], []
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    head.eval()

    with th.no_grad():
        for batch in loader:
            input_ids, labels = batch[0].to(device), batch[1]
            h = get_hidden_state(model, input_ids)
            probs = th.sigmoid(head.net(h)).cpu().numpy().flatten()
            all_labels.extend(labels.numpy().flatten())
            all_probs_baseline.extend(probs)

    y_true = np.array(all_labels)
    y_prob_base = np.array(all_probs_baseline)
    base_auroc = roc_auc_score(y_true, y_prob_base)
    base_auprc = average_precision_score(y_true, y_prob_base)
    log.info(f"  Baseline: AUROC={base_auroc:.4f}, AUPRC={base_auprc:.4f}")

    # Occlusion per category
    results = [{"category": "None (baseline)", "auroc": base_auroc, "auprc": base_auprc,
                "auroc_drop": 0, "auprc_drop": 0}]

    for cat_name, token_ids in cat_token_ids.items():
        if not token_ids:
            continue
        log.info(f"Occluding {cat_name} ({len(token_ids)} tokens)...")

        all_probs_occ = []
        with th.no_grad():
            for batch in loader:
                input_ids = batch[0].to(device).clone()
                labels = batch[1]
                # Mask all tokens in this category to 0 (padding)
                mask = th.zeros_like(input_ids, dtype=th.bool)
                for tid in token_ids:
                    mask |= (input_ids == tid)
                input_ids[mask] = 0

                h = get_hidden_state(model, input_ids)
                probs = th.sigmoid(head.net(h)).cpu().numpy().flatten()
                all_probs_occ.extend(probs)

        y_prob_occ = np.array(all_probs_occ)
        occ_auroc = roc_auc_score(y_true, y_prob_occ)
        occ_auprc = average_precision_score(y_true, y_prob_occ)

        results.append({
            "category": cat_name,
            "auroc": occ_auroc,
            "auprc": occ_auprc,
            "auroc_drop": base_auroc - occ_auroc,
            "auprc_drop": base_auprc - occ_auprc,
        })
        log.info(f"  {cat_name}: AUROC={occ_auroc:.4f} (drop={base_auroc-occ_auroc:+.4f}), "
                 f"AUPRC={occ_auprc:.4f} (drop={base_auprc-occ_auprc:+.4f})")

    return pd.DataFrame(results), y_true, y_prob_base


def plot_occlusion(df, output_dir):
    """Bar chart of AUROC/AUPRC drop per category."""
    cats = df[df.category != "None (baseline)"].sort_values("auroc_drop", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.barh(cats.category, cats.auroc_drop, color="#4e79a7")
    ax.set_xlabel("AUROC drop when category is masked")
    ax.set_title("Occlusion Analysis: AUROC Impact", fontweight="bold")
    ax.invert_yaxis()

    ax = axes[1]
    ax.barh(cats.category, cats.auprc_drop, color="#e15759")
    ax.set_xlabel("AUPRC drop when category is masked")
    ax.set_title("Occlusion Analysis: AUPRC Impact", fontweight="bold")
    ax.invert_yaxis()

    plt.tight_layout()
    out = output_dir / "occlusion_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {out}")


def plot_confusion_matrices(output_dir):
    """Generate confusion matrices for all models at Youden's J optimal threshold."""
    models = {
        "ASA":              (BASELINES / "asa_test_predictions.parquet", "y_prob"),
        "LR (manual)":      (BASELINES / "test_preds_manual.parquet", "prob_lr_manual"),
        "XGB (manual)":     (BASELINES / "test_preds_manual.parquet", "prob_xgb_manual"),
        "LR (MEDS)":        (BASELINES / "test_preds_meds.parquet", "prob_lr_meds"),
        "XGB (MEDS)":       (BASELINES / "test_preds_meds.parquet", "prob_xgb_meds"),
        "LSTM":             (BASELINES / "lstm_test_predictions.parquet", "y_prob"),
        "ETHOS (zero-shot)":(BASELINES / "ethos_test_predictions.parquet", "y_prob"),
        "ETHOS (fine-tuned)":(BASELINES / "ethos_finetune_test_predictions.parquet", "y_prob"),
    }

    n_models = len(models)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    summary_rows = []

    for i, (name, (fpath, prob_col)) in enumerate(models.items()):
        if not fpath.exists():
            log.warning(f"  {name}: file not found")
            continue
        df = pd.read_parquet(fpath)
        yt = df["y_true"].values.astype(float)
        yp = df[prob_col].values.astype(float)
        valid = ~(np.isnan(yt) | np.isnan(yp))
        yt, yp = yt[valid], yp[valid]

        # Optimal threshold (Youden's J)
        fpr, tpr, thresholds = roc_curve(yt, yp)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thresh = thresholds[best_idx]

        y_pred = (yp >= best_thresh).astype(int)
        cm = confusion_matrix(yt, y_pred)
        tn, fp, fn, tp = cm.ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        summary_rows.append({
            "model": name, "threshold": best_thresh,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "sensitivity": sens, "specificity": spec,
            "PPV": ppv, "NPV": npv,
        })

        ax = axes[i]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Pred -", "Pred +"], yticklabels=["True -", "True +"])
        ax.set_title(f"{name}\n(thresh={best_thresh:.3f}, sens={sens:.2f}, spec={spec:.2f})", fontsize=10)
        ax.set_ylabel(""); ax.set_xlabel("")

    plt.suptitle("Confusion Matrices at Youden's J Optimal Threshold", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = output_dir / "confusion_matrices.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {out}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "confusion_matrix_summary.csv", index=False)
    log.info(f"Saved: {output_dir / 'confusion_matrix_summary.csv'}")
    log.info("\n" + summary_df.to_string(index=False))


def run_fpfn_analysis(output_dir):
    """Analyze top false positives and false negatives for ETHOS fine-tuned."""
    fpath = BASELINES / "ethos_finetune_test_predictions.parquet"
    if not fpath.exists():
        log.warning("ETHOS FT predictions not found")
        return

    df = pd.read_parquet(fpath)
    yt, yp = df["y_true"].values, df["y_prob"].values

    # Optimal threshold
    fpr, tpr, thresholds = roc_curve(yt, yp)
    best_thresh = thresholds[np.argmax(tpr - fpr)]

    # Top FP: highest y_prob among true negatives
    fp_mask = (yt == 0)
    fp_df = df[fp_mask].nlargest(20, "y_prob")[["subject_id", "y_prob"]]
    fp_df["type"] = "False Positive"

    # Top FN: lowest y_prob among true positives
    fn_mask = (yt == 1)
    fn_df = df[fn_mask].nsmallest(20, "y_prob")[["subject_id", "y_prob"]]
    fn_df["type"] = "False Negative"

    result = pd.concat([fp_df, fn_df])
    result.to_csv(output_dir / "fpfn_top20.csv", index=False)
    log.info(f"Saved: {output_dir / 'fpfn_top20.csv'}")
    log.info(f"  Top 20 FP: y_prob range [{fp_df.y_prob.min():.4f}, {fp_df.y_prob.max():.4f}]")
    log.info(f"  Top 20 FN: y_prob range [{fn_df.y_prob.min():.4f}, {fn_df.y_prob.max():.4f}]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--occlusion-only", action="store_true",
                        help="Skip confusion matrix / FP-FN (already done) and run occlusion only")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.occlusion_only:
        # 1. Confusion matrices for all models (no GPU needed)
        log.info("=== Confusion Matrices ===")
        plot_confusion_matrices(OUTPUT_DIR)

        # 2. FP/FN analysis
        log.info("\n=== FP/FN Analysis ===")
        run_fpfn_analysis(OUTPUT_DIR)

    # 3. Occlusion analysis (GPU needed)
    log.info("\n=== Occlusion Analysis ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ethos"))

    from ethos.utils import load_model_checkpoint
    from datasets.iod_dataset import IoDDataset

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Load model
    model, _ = load_model_checkpoint(MODEL_FP, map_location=device)
    model.to(device).eval()
    n_embd = model.config.n_embd
    n_positions = model.config.n_positions

    # Load head
    head_path = Path("/tmp/chd_head_best.pt")
    if not head_path.exists():
        head_path = Path("/path/to/CHD_MEDS/results/ethos/finetune/finetune_probe_head_best.pt")
    if not head_path.exists():
        head_path = Path("/path/to/CHD_MEDS/results/ethos/finetune_probe_head_best.pt")
    if head_path.exists():
        ckpt = th.load(head_path, map_location=device, weights_only=False)
        from torch import nn
        cfg = ckpt["head_config"]
        head = nn.Module()
        head.net = nn.Sequential(
            nn.Linear(cfg["n_embd"], cfg["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["hidden_dim"], 1),
        )
        head.load_state_dict(ckpt["head_state_dict"])
        head = head.to(device).eval()
        ts = ckpt.get("timestamp", "unknown")
        test_auroc = ckpt.get("test_auroc", "unknown")
        test_auprc = ckpt.get("test_auprc", "unknown")
        log.info(f"Loaded classification head from {head_path} (timestamp={ts}, test_auroc={test_auroc}, test_auprc={test_auprc})")
    else:
        log.warning(f"Head checkpoint not found at {head_path}. Skipping occlusion analysis.")
        log.info("\n=== Done (occlusion skipped) ===")
        exit(0)

    # Load dataset + labels
    task = pd.read_parquet(TASK_PATH)
    task["pid_int"] = task["patient_id"].str.replace("C", "").astype(int)
    task["pt_ns"] = task["prediction_time"].astype("int64")
    label_dict = dict(zip(
        zip(task.pid_int, task.pt_ns // 1000),
        task.boolean_value.astype(int),
    ))

    ds = IoDDataset(str(TEST_DIR), n_positions=n_positions)

    # Build simple dataset wrapper using task parquet for label matching
    # (avoids slow per-sample safetensors I/O during init)
    from torch.utils.data import Dataset as TorchDataset

    # Match task labels to dataset indices using the first sample to validate,
    # then use all samples (task already filters to test-split encounters)
    log.info("Building test dataset for occlusion (label matching)...")

    # Use task parquet directly: we know all ds samples correspond to OR_ENTRY encounters
    # Just iterate and get labels from the base dataset's __getitem__ y dict
    # But that's slow (37K safetensors reads). Instead, use iod_label from ds directly.
    # Since ds already has start_indices aligned with OR_ENTRY tokens, we can assign labels:
    # Option: just use ALL ds samples and get y_true from task parquet post-hoc
    class SimpleDS(TorchDataset):
        """Wraps IoDDataset, attaching ground truth labels from task parquet."""
        def __init__(self, base_ds, task_df):
            self.base = base_ds
            # Build lookup: (pid, prediction_time_ns) -> label
            # Use task_df which is already filtered to test split
            task_lookup = {}
            for _, row in task_df.iterrows():
                pid = int(str(row["patient_id"]).replace("C", ""))
                pt_ns = int(row["prediction_time"].value)  # datetime64 -> ns
                task_lookup[(pid, pt_ns)] = int(row["boolean_value"])

            self.indices, self.labels = [], []
            start_indices = base_ds.start_indices
            n = len(start_indices)

            # Batch-read times for all start_indices (SliceableData supports individual reads fast)
            log.info(f"  Reading times for {n} OR_ENTRY indices...")
            times_list = [base_ds.times[si.item()].item() for si in start_indices]

            # Batch-read patient_ids: build a full pid_at_idx array from shards
            log.info("  Building patient_id lookup array...")
            all_pids = th.cat([s["patient_ids"] for s in base_ds._data.shards])
            all_offsets = th.cat([s["patient_offsets"] + s["offset"] for s in base_ds._data.shards])
            total_len = sum(s["tokens"].get_shape()[0] for s in base_ds._data.shards)
            # Use searchsorted: for each or_idx, find which patient it belongs to
            # patient_offsets are sorted; patient i owns tokens [offset[i], offset[i+1])
            patient_indices = th.searchsorted(all_offsets, start_indices, right=True) - 1
            pids_list = all_pids[patient_indices].tolist()

            matched = 0
            for i in range(n):
                pid = pids_list[i]
                pt_ns = times_list[i]
                label = task_lookup.get((pid, pt_ns))
                if label is None:
                    label = task_lookup.get((pid, pt_ns * 1000))
                if label is not None:
                    self.indices.append(i)
                    self.labels.append(label)
                    matched += 1
            log.info(f"  Matched {matched}/{n} encounters to task labels")

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            x, _ = self.base[self.indices[idx]]
            return x, self.labels[idx]

    # Filter task to test split
    splits = pd.read_parquet("/path/to/CHD_MEDS/splits/splits.parquet")
    test_sids = set(splits[splits["split"] == "test"]["subject_id"].values)
    task["sid"] = task["patient_id"].str.replace("C", "").astype(int)
    task_test = task[task["sid"].isin(test_sids)]
    log.info(f"  Task test encounters: {len(task_test)}")

    simple_ds = SimpleDS(ds, task_test)
    log.info(f"  {len(simple_ds)} test samples")

    vocab = ds.vocab
    occ_df, y_true, y_prob = run_occlusion(model, head, simple_ds, vocab, device)
    occ_df.to_csv(OUTPUT_DIR / "occlusion_results.csv", index=False)
    plot_occlusion(occ_df, OUTPUT_DIR)

    log.info("\n=== Done ===")
