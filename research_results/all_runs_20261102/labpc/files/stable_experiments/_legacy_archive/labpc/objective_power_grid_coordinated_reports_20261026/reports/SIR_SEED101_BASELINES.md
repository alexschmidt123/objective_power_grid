# LabPC corrected SIR seed101 baseline status

Final common evaluations complete: **0/6**. Valid training inputs: **2/6** (Random needs no training; Fixed reuses its verified LabPC1M checkpoint).

|Method|Status|Last recorded episodes|Final EIG ± MC SE|
|---|---|---:|---:|
|Random|VALID_NO_TRAINING|0|pending|
|Optimized Fixed|VALID_BUT_REEVALUATION_REQUIRED|1000000|pending|
|Corrected DAD|RUNNING_TRAINING|252000|pending|
|Corrected RL-sBOED|QUEUED|pending|pending|
|Corrected vsOED|QUEUED|pending|pending|
|Corrected ScoreBED|QUEUED|pending|pending|

Running-row counts are last recorded progress, not final budgets. Early curve EIG is not inserted as final EIG. Paired differences and95%CIs remain blank until complete matched scores exist. Fixed historical optimization wall is not asserted to be exclusive-device wall. All methods must reach1M primary episodes for a valid final trained result.

Live training source: `839bb7d7c232f555a4f2f34b86fd9a00cc0b0eea`. Common evaluation resolution is pending or explicitly provisional until its frozen calibration record exists. No training-seed variability estimate is implied by MCSE. No extra training seeds were launched.
