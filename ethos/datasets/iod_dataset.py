"""
iod_dataset.py
--------------
IoDDataset for ETHOS zero-shot inference on CHD data.

Imports from the installed ethos package (ethos-ares) without modifying it.
ethos-ares is a read-only reference/installation; all CHD-specific code lives here.

Prediction time : ENCOUNTER//AN//OR_ENTRY  (patient enters OR)
Target events   : IoD-related tokens that appear after OR entry
Time limit      : 24 hours (covers all but the longest cardiac surgeries)

IoD target tokens (stop when any appears in generated trajectory):
  AN_EVENT//CPR                 (IoD1 — cardiac arrest)
  MED//EPINEPHRINE              (IoD3/4 — vasoactive)
  MED//DOPAMINE                 (IoD4)
  MED//MILRINONE                (IoD4)
  MED//NOREPINEPHRINE           (IoD4)
  MED//PHENYLEPHRINE            (IoD3/4)
  MED//VASOPRESSIN              (IoD3/4)
  MED//ADENOSINE                (IoD3)
  MED//ATROPINE                 (IoD3)
  MED//EPHEDRINE                (IoD3)
  LDA//ARTERIAL_LIN             (IoD5 — arterial line after incision)
"""

from datetime import timedelta
from pathlib import Path

import torch as th

from ethos.datasets.base import InferenceDataset

OR_ENTRY_STOKEN = "ENCOUNTER//AN//OR_ENTRY"
OR_EXIT_STOKEN  = "ENCOUNTER//AN//OR_EXIT"

IOD_STOP_STOKENS_V3 = [
    # IoD1 — cardiac arrest
    "AN_EVENT//CPR",
    # IoD3/4 — epinephrine (infusion + bolus)
    "MED//EPINEPHRINE_DRIP",
    "MED//EPINEPHRINE_DRIP_16_MCG/ML_30_ML_SYRINGE",
    "MED//EPINEPHRINE_01_MG/ML_INJECTION_SYRINGE",
    "MED//EPINEPHRINE_1_MG/ML_1_ML_INJECTION_SOLUTION",
    "MED//EPINEPHRINE_HCL_PF_1_MG/ML_1_ML_INJECTION_SOLUTION",
    "MED//EPINEPHRINE_1_MG/ML_INJECTION_SOLUTION",
    # IoD4 — dopamine infusion
    "MED//DOPAMINE_DRIP",
    "MED//ANE_DOPAMINE_INFUSION",
    # IoD4 — milrinone infusion
    "MED//MILRINONE_DRIP",
    "MED//MILRINONE_DRIP_LOADING_DOSE",
    # IoD4 — norepinephrine infusion
    "MED//NOREPINEPHRINE_DRIP",
    # IoD3/4 — phenylephrine
    "MED//PHENYLEPHRINE_DRIP",
    # IoD3/4 — vasopressin
    "MED//VASOPRESSIN_DRIP_FOR_HYPOTENSION",
    # IoD3 — adenosine bolus
    "MED//ADENOSINE_3_MG/ML_INTRAVENOUS_SOLUTION",
    # IoD3 — atropine IV
    "MED//ATROPINE_04_MG/ML_INJECTION_SOLUTION",
    # IoD3 — ephedrine IV
    "MED//EPHEDRINE_SULFATE_50_MG/ML_INJECTION_SOLUTION",
    # IoD5 — arterial line
    "LDA//ARTERIAL_LIN",
]

# ATC-hierarchical equivalents (leaf-level SFX tokens for vasoactive drugs)
IOD_STOP_STOKENS_V4_ATC = [
    # IoD1 — cardiac arrest (unchanged, not a MED)
    "AN_EVENT//CPR",
    # Epinephrine (C01CA24) — ATC leaf suffix
    "ATC//SFX//A24",   # epinephrine
    "ATC//C01",        # cardiac stimulants (parent class)
    # Dopamine (C01CA04)
    "ATC//SFX//A04",   # dopamine
    # Milrinone (C01CE02)
    "ATC//SFX//E02",   # milrinone
    # Norepinephrine (C01CA03)
    "ATC//SFX//A03",   # norepinephrine
    # Phenylephrine (C01CA06)
    "ATC//SFX//A06",   # phenylephrine
    # Vasopressin (H01BA01)
    "ATC//SFX//A01",   # vasopressin (H01 family)
    # Adenosine (C01EB10)
    "ATC//SFX//B10",   # adenosine
    # Atropine (A03BA01)
    "ATC//SFX//A01",   # atropine (A03 family) — note overlap with vasopressin SFX
    # Ephedrine (R03CA02 or C01CA26)
    "ATC//SFX//A26",   # ephedrine
    # IoD5 — arterial line (unchanged, not a MED)
    "LDA//ARTERIAL_LIN",
]

# Combined: both flat names and ATC equivalents; filtered at runtime by vocab
IOD_STOP_STOKENS = list(set(IOD_STOP_STOKENS_V3 + IOD_STOP_STOKENS_V4_ATC))


class IoDDataset(InferenceDataset):
    """
    Inference dataset for Intraoperative Deterioration (IoD) prediction.

    Each sample is one surgical encounter represented by all tokens up to and
    including ENCOUNTER//AN//OR_ENTRY (the "In OR" timestamp).

    Ground truth `expected` is the first IoD token (if IoD occurred) or
    OR_EXIT (if surgery ended without IoD).
    """

    time_limit: timedelta = timedelta(hours=24)

    def __init__(self, input_dir: str | Path, n_positions: int = 2048, **kwargs):
        super().__init__(input_dir, n_positions, **kwargs)

        # Prepend IoD tokens to default stop tokens (DEATH, TIMELINE_END)
        # Filter out any default stop tokens not present in this vocab (e.g. MEDS_DEATH)
        vocab_set = set(self.vocab.stoi.keys())
        default_stop = [s for s in self.stop_stokens if str(s) in vocab_set]
        iod_in_vocab = [s for s in IOD_STOP_STOKENS if s in vocab_set]
        self.stop_stokens = iod_in_vocab + default_stop

        or_entry_indices = self._get_indices_of_stokens(OR_ENTRY_STOKEN)

        if len(or_entry_indices) == 0:
            raise ValueError(
                f"Token '{OR_ENTRY_STOKEN}' not found in vocabulary. "
                "Ensure CHD data was tokenized with an_patients events included."
            )

        self.start_indices = or_entry_indices

        # Ground truth: first IoD token OR OR_EXIT after each OR_ENTRY
        # Filter to tokens actually in this version's vocab (version compat)
        iod_stop_filtered = [s for s in IOD_STOP_STOKENS if s in vocab_set]
        iod_indices     = self._get_indices_of_stokens(iod_stop_filtered)
        or_exit_indices = self._get_indices_of_stokens(OR_EXIT_STOKEN)

        outcome_candidates = th.cat([iod_indices, or_exit_indices]).sort().values

        self.outcome_indices = self._match(
            outcome_candidates,
            self.start_indices,
            fill_unmatched=len(self.tokens) - 1,
        )

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, idx: int) -> tuple[th.Tensor, dict]:
        or_idx      = self.start_indices[idx].item()
        outcome_idx = self.outcome_indices[idx]

        outcome_token  = self.tokens[outcome_idx]
        outcome_stoken = self.vocab.decode(outcome_token.item())

        y = {
            "expected":        outcome_stoken,
            "true_token_dist": (outcome_idx - or_idx),
            "true_token_time": (self.times[outcome_idx] - self.times[or_idx]).item(),
            "patient_id":      self.patient_id_at_idx[or_idx].item(),
            "prediction_time": self.times[or_idx].item(),
            "data_idx":        or_idx,
            "iod_label":       int(outcome_stoken in IOD_STOP_STOKENS),
        }

        # ── Build context: all patient tokens UP TO and INCLUDING OR_ENTRY ──
        # (NOT tokens after OR_ENTRY — that would be data leakage)
        pid = self.patient_id_at_idx[or_idx].item()

        # Find patient start by scanning backward from or_idx
        pat_start = or_idx
        while pat_start > 0 and self.patient_id_at_idx[pat_start - 1].item() == pid:
            pat_start -= 1

        # Patient context (demographics): static tokens at the start (time == 0)
        static_end = pat_start
        while static_end <= or_idx and self.times[static_end].item() == 0:
            static_end += 1
        pt_ctx = self.tokens[pat_start : static_end]

        # Pad/truncate pt_ctx to exactly context_size
        if len(pt_ctx) < self.context_size:
            pad = th.zeros(self.context_size - len(pt_ctx), dtype=pt_ctx.dtype)
            pt_ctx = th.cat([pt_ctx, pad])
        elif len(pt_ctx) > self.context_size:
            pt_ctx = pt_ctx[:self.context_size]

        # Timeline: non-static tokens from patient start up to OR_ENTRY (inclusive)
        timeline_start = max(static_end, or_idx - self.timeline_size + 1)
        timeline = self.tokens[timeline_start : or_idx + 1]  # ends with OR_ENTRY

        # Left-pad timeline if shorter than timeline_size (so OR_ENTRY is at position -1)
        if len(timeline) < self.timeline_size:
            pad = th.zeros(self.timeline_size - len(timeline), dtype=timeline.dtype)
            timeline = th.cat([pad, timeline])

        # x = [pt_ctx (context_size) | timeline (timeline_size)]
        # OR_ENTRY is always at position -1
        x = th.cat([pt_ctx, timeline])

        return x.clone(), y
