# High-variance methods: journal readiness report

This is a development troubleshooting report, not final confirmatory evidence. Training seeds 101–105 are running under the frozen fixed-budget protocol; only completed methods enter this snapshot. Evaluation uses seed 1001, 2,000 paired trajectories, L=1,000,000, and the final 1M-budget checkpoint. No retraining, checkpoint reselection, algorithm change, or reserved-seed evaluation was performed for this audit.

A nonzero gradient (>1e-8) is an exposure proxy, not proof of informative learning. Raw gradient norms are measured before component clipping; large raw values are not the applied parameter update. Correlation with sharpening does not establish the router as the sole cause.

## Evidence-based disposition

| Method | Confirmed finding | Still unresolved | Immediate disposition |
|---|---|---|---|
| MoE-sBOED | Seed-103 late training breakdown | Router versus shared pathwise amplification; hard/relaxed mismatch | Finish original seeds; separate controlled diagnostic afterward |
| SwitchMoE-sBOED | Same breakdown; no hard continuation demonstrated | Gate initialization effects and switching contribution | No temporal-persistence claim; diagnose shared failure first |
| ScoreBED | Validation exports omit bounds; poor final seed102 | Score-gradient quality and training-distribution mismatch | Quarantine original validation curves; final EIG retained |
| vsOED | Moderate seed spread, learning ongoing | Convergence and actor/critic/posterior contribution | Retain runs, label budget-limited/unverified convergence |

## Conditions for journal claims

1. Complete seeds101–105 without dropping failures. Publish per-seed outcomes, mean/SD and failure incidence; distinguish training uncertainty from trajectory MC error.
2. Resolve demonstrated implementation defects with regression tests. Preserve original and corrected evidence separately.
3. For any new MoE correction, predeclare one changed factor, equal total episode accounting and final-budget selection. Confirm across all declared seeds, not only a repaired failing seed.
4. Demonstrate whether the correction reduces collapse as well as improves EIG. Report expert starvation and hard/relaxed gaps; do not infer informative specialization from utilization alone.
5. Freeze settings before untouched final evaluation. Failed experiments remain part of development history. Five runs cannot establish that rare failures are impossible, even if a correction has no failures among five.
6. Instability can be reported scientifically; the present evidence does not support claiming these configurations are robust or uniformly superior. There is no universal journal-acceptable SD cutoff: uncertainty, failure severity, reproducibility and supported claims matter.

## Execution boundary

The active frozen seed101–105 campaign is unchanged. Corrections and additional training are deferred as requested. This report runs only CPU analysis of existing records. Seed105 and any incomplete methods are pending, not inferred. No final-confirmation results are claimed.

## Files

- moe_report.md
- switchmoe_report.md
- scorebed_report.md
- vsoed_report.md
- results.csv, training_curves.csv, evidence.json, source_hashes.json
- instability_diagnostics.png/.pdf
