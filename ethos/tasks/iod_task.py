"""
iod_task.py
-----------
IoD (Intraoperative Deterioration) task definition for ETHOS zero-shot inference.

ETHOS zero-shot inference:
  1. For each patient encounter at prediction_time (= In OR),
     feed all pre-op tokens as context to the model
  2. Generate N=100 simulated future trajectories
  3. For each trajectory, check if any IoD-related token appears
  4. Risk score = fraction of trajectories containing IoD tokens

IoD tokens to look for in generated trajectories:
  - AN_EVENT//CPR               (IoD1)
  - MED//EPINEPHRINE             (IoD3/4)
  - MED//DOPAMINE                (IoD4)
  - MED//MILRINONE               (IoD4)
  - MED//NOREPINEPHRINE          (IoD4)
  - MED//PHENYLEPHRINE           (IoD3/4)
  - MED//VASOPRESSIN             (IoD3/4)
  - MED//ADENOSINE               (IoD3)
  - MED//ATROPINE                (IoD3)
  - MED//EPHEDRINE               (IoD3)
  - LDA//ARTERIAL_LIN            (IoD5)

This file defines the task config that ethos_infer loads.
It follows the ETHOS task definition pattern.

Usage (after training):
    conda activate ethos
    bash ethos/infer.sh
"""

# IoD-relevant token patterns to search for in generated trajectories
# ethos_infer uses these as the "target event" for zero-shot scoring

IOD_TARGET_TOKENS = [
    # CPR
    "AN_EVENT//CPR",
    # Vasoactive medications (IV bolus or infusion)
    "MED//EPINEPHRINE",
    "MED//DOPAMINE",
    "MED//MILRINONE",
    "MED//NOREPINEPHRINE",
    "MED//PHENYLEPHRINE",
    "MED//VASOPRESSIN",
    "MED//ADENOSINE",
    "MED//ATROPINE",
    "MED//EPHEDRINE",
    # Arterial line placement
    "LDA//ARTERIAL_LIN",
]

# Task metadata for ETHOS inference config
TASK_NAME = "iod"
TASK_DESCRIPTION = (
    "Intraoperative Deterioration (IoD) in pediatric cardiac surgery. "
    "Prediction time = In OR. Target = any IoD token in generated trajectory."
)
