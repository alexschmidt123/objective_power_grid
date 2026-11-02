# LabPC corrected SIR seed101 baselines

Completed common evaluations: 0/6. Missing numerical cells are pending, not zero. Fixed reuses its verified LabPC1M checkpoint. Other-machine checkpoints are never primary LabPC results.

| Method | Status | Episodes | EIG ± MC SE | Evaluation |
|---|---|---:|---:|---|
|Random|VALID_NO_TRAINING|0|pending|PENDING|
|Optimized Fixed|VALID_BUT_REEVALUATION_REQUIRED|1000000|pending|PENDING|
|Corrected DAD|READINESS_FAILED_INCOMPLETE|pending|pending|PENDING|
|Corrected RL-sBOED|READINESS_FAILED_INCOMPLETE|pending|pending|PENDING|
|Corrected vsOED|READINESS_FAILED_INCOMPLETE|pending|pending|PENDING|
|Corrected ScoreBED|READINESS_FAILED_INCOMPLETE|pending|pending|PENDING|

95% normal trajectory-MC CIs and valid paired case-wise differences are in CSV. MCSE is conditional on checkpoint and contrast bank, not training-seed variation. Native objectives and compute differ;1Mepisodes is matched data only. Seed101 is development evidence. Confirmation/replications are separate. No posterior accuracy is claimed.
