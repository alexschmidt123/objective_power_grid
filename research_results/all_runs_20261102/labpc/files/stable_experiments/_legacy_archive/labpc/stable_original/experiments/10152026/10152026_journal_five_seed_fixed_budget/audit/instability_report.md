# Location-finding instability audit

Read-only analysis of completed development runs. Evaluation seed 1001, 2,000 trajectories, L=1,000,000. No checkpoint reselection or retraining. Training seeds are independent replicates; trajectory Monte Carlo errors are separate.

| Method | Seed 101 | Seed 102 | Seed 103 | Seed 104 |
|---|---:|---:|---:|---:|
|vsoed|8.58218|8.75028|7.96138|8.42605|
|scorebed|9.91568|7.18907|10.13136|9.79230|
|moe|9.82944|9.72441|6.25322|10.41524|
|switchmoe|10.10045|10.03829|6.50948|Pending|

## Findings

MoE and SwitchMoE: seed 103 exhibits late optimization breakdown, not merely evaluation noise. Both complete all 15,625 updates / 1,000,000 primary episodes. Mean raw gradient norm in the final fifth exceeds 10 million, compared with roughly 50–164 during preceding fifths. Component clipping limits recorded group norms but does not prevent the observed deterioration. This is evidence of instability coinciding with sharpening; it is not proof that the router alone causes it.

- moe seed 103: final-fifth router gradient p99 1.39e+08, max 1.12e+10; expert final-fifth nonzero update fractions E1=100.00%, E2=92.51%, E3=100.00%, E4=100.00%.
- switchmoe seed 103: final-fifth router gradient p99 1.68e+08, max 4.7e+09; expert final-fifth nonzero update fractions E1=100.00%, E2=31.58%, E3=100.00%, E4=31.62%.

The final training-score medians fall to 6.32827 (MoE) and 5.49302 (SwitchMoE); only 29.69% and 25% of final-batch trajectories are within 0.1 of log(1025). Therefore objective saturation alone cannot explain the failed endpoints. Good runs do saturate, so increasing contrastives is not justified as the first correction to these failures.

SwitchMoE: final training rerouting probabilities are approximately 0.993–0.997 for completed seeds 101–103. Hard threshold proxies request rerouting at every stage. Current evidence does not demonstrate learned temporal persistence. Seed 103 uses only E1/E3 during hard evaluation; starvation is present, but similarly concentrated routing can occur in successful runs.

ScoreBED: seed 102 has 28.29% of action coordinates with absolute value >3.8, versus zero for seeds 101,103,104. Its final EIG is 7.18907. This is evidence of an inferior boundary-concentrated policy, not sufficient evidence of the cause in score learning.

A concrete **diagnostic validation defect** was found: `journal_scorebed_bridge.maybe_validate` exports policy parameters without `common_design_bounds`, whereas final worker export includes bounds. The intermediate Torch validator therefore evaluates an unbounded policy. Final deployed checkpoints include bounds and the JAX-to-Torch action bridge was checked by training. The fixed-budget protocol selects final weights, not best diagnostic validation, so the defect does not itself explain final seed variation. All original ScoreBED validation curves must be marked invalid pending bounded replay.

- ScoreBED seed 102: final bounded validation EIG 7.18030; originally logged unbounded validation 6.21599.
- ScoreBED seed 103: final bounded validation EIG 10.39889; originally logged unbounded validation 6.42771.
- ScoreBED seed 104: final bounded validation EIG 9.72000; originally logged unbounded validation 5.97182.

vsOED: seeds 102 and 103 continue improving through the final checkpoint. Seed 104 has a small late decrease. No comparable catastrophic drop appears in these validation curves. Fixed-budget underoptimization and seed-dependent learning are supported possibilities; actor/critic/posterior contributions remain unresolved. No extra episodes are authorized for the primary comparison.

## Next controlled diagnostic

Keep the frozen campaign unchanged. For MoE, first quantify hard-versus-relaxed rollout EIG on the same failed and successful checkpoints to distinguish discretization mismatch from already-poor relaxed policies. Any sharpening correction must be a separately labeled development experiment with one changed factor. For ScoreBED, repair metadata propagation and replay diagnostic validation before interpreting its learning curves; then investigate score-gradient quality using saved score models. Do not discard failed seeds or select earlier checkpoints.

## Infrastructure

A previous disk-full MoE seed-103 attempt was archived and restarted from scratch; accepted fresh training completed the full budget. This incident is separate from the observed late degradation in both full-expert methods. External GPU contention affects current runtime comparisons and must be flagged, not treated as evidence of an EIG cause.

## Provenance

Sources: registry.json, per-method final training JSON, diagnostics JSONL, and eval/*_seed1001.json. Numerical extracts: audit/instability_snapshot.json and audit/instability_verification.json. Original source, checkpoints, configurations and final results were not changed.
