# Seed101 convergence status

Final-budget selection is fixed; no curve selects a best model.

dad: 2 common validation points; last recorded 252000 episodes, EIG 3.953757 ± 0.214084. Current status `training`. This does not certify convergence at1M.

Native ScoreBED has only3 policy updates within its declared1M allocation; native vsOED has980 updates before its10000-update transition, and DAD has500updates before its1000-update LR decay. These are explicit optimization-budget limitations, not reasons to selectively extend a baseline. Fixed historical curves use a different resolution and are not silently pooled.
