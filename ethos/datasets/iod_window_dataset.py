"""
iod_window_dataset.py
---------------------
Context window ablation dataset for ETHOS zero-shot IoD inference.

Same as IoDDataset but truncates the patient timeline to only include
events within a specified time window before OR entry.

Prediction point: OR_ENTRY (fixed)
Time limit: 24 hours (fixed)
Variable: how much past history the model sees

Usage:
    ds = IoDWindowDataset(input_dir, n_positions=2048, window_days=30)
    # Only events from (OR_entry - 30 days) to OR_entry are in the context
"""

from pathlib import Path

import torch as th

from datasets.iod_dataset import IoDDataset


class IoDWindowDataset(IoDDataset):
    """
    IoDDataset with context window filtering.

    If window_days is None, behaves identically to IoDDataset (all history).
    If window_days is set, tokens with time < (OR_entry - window_days) are
    zeroed out in the returned context.
    """

    def __init__(self, input_dir: str | Path, n_positions: int = 2048,
                 window_days: int | None = None, **kwargs):
        super().__init__(input_dir, n_positions, **kwargs)
        self.window_days = window_days

    def __getitem__(self, idx: int) -> tuple[th.Tensor, dict]:
        x, gt = super().__getitem__(idx)

        if self.window_days is None:
            return x, gt

        # Zero out tokens older than window_days before OR_ENTRY
        or_idx = gt["data_idx"]
        or_time = self.times[or_idx].item()
        window_ns = int(self.window_days * 24 * 3600 * 1e9)
        cutoff_time = or_time - window_ns

        seq_len = x.size(0)
        x_new = x.clone()
        for i in range(seq_len - 1, -1, -1):
            if x_new[i].item() == 0:
                break
            global_idx = or_idx - (seq_len - 1 - i)
            if global_idx < 0 or global_idx >= len(self.times):
                continue
            t = self.times[global_idx].item()
            if t != 0 and t < cutoff_time:
                x_new[i] = 0

        return x_new, gt
