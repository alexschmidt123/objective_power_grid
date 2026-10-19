# Frozen location E1 training choices

Use development1001 evidence and correctness only; never the consumed8831001 ranking. One coherent five-seed cohort per architecture. This is a complete-method comparison, not a switching-only or architecture-only causal contrast.

| Architecture | E1 training | Rationale and limitation |
|---|---|---|
| MoE-sBOED | Verified existing straight-through cohort101-105 | Improved development stability; all policies collapse to one active expert. No useful four-expert specialization claim. |
| ResMoE-sBOED | Established relaxed cohort101-105 | Stable original; small straight-through mean difference does not justify replacement. Reuse500k bases plus500k residual phases. |
| SwitchMoE-sBOED | Matched original relaxed30%/95% cohort101-105 | ST seed101 is poor. Current isolated manager passes hard-forward, carried-state, continuation, gate-gradient, reset and forced-reroute tests; do not automatically promote ST or launch its grid. |
| SwitchResMoE-sBOED | Established relaxed cohort101-105 | Stable original; revised behavior not established. Verify exact shared500k base hash per seed. |

43 CPU correctness tests pass; one optional historical fixture was skipped, so checkpoint-specific checks are separately required. The current LabPC ST code preserves StrategyManager and recomputes adaptive actions. Old GitHub's restricted ST branch differs: the active manager implementation has been preserved in isolated versioned source, not replaced by merely deleting its restriction. No changed relaxed computation is intended. Existing ST switching regressions remain negative development results, not removed. Warmup intervention is closed and will not be repeated.

Selected checkpoint paths, hashes, final budgets, source/config identities, complete evaluations and reuse status are recorded by the E1 verification registry. Any failing row remains blocked instead of substituted. SIR must use compatible SIR weights and bases; location cohorts do not supply SIR weights.
