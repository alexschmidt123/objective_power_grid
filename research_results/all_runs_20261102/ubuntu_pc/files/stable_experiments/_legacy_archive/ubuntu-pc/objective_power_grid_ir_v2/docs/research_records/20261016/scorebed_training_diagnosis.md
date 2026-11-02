# ScoreBED training troubleshooting report

## Outcome

The validation-bounds defect is confirmed and fixed in isolated code. The poor seed102 final policy remains a real result. A second definite implementation defect has not been found. There is evidence of boundary concentration and imperfect score/policy-gradient approximation, but no established sole cause of the across-seed EIG spread. Do not claim the high variance is solved.

## What was checked

- Original five seeds all completed1M total primary training episodes; original checkpoint hashes/tensor finiteness passed the earlier audit.
- Native score-matching algorithm, architecture and score-gradient policy optimization remain unchanged.
- Bounded-map Jacobian matches the exact8*sigmoid(u)*(1-sigmoid(u)) derivative; maximum discrepancy0 for models102–104. No missing chain-rule error was detected at this check.
- Standardization statistics are finite for each reconstructed native target. This does not by itself prove optimal scaling.
- Corrected validation actions match saved native JAX action references; complete validation/deployed Torch rollouts match. Final held-out EIG uses bounds correctly.
- Diagnostic reference is the Fisher identity (posterior-weighted conditional explicit-likelihood scores), using prefixes65,536 /262,144 /1,000,000 of the existing contrast bank. This is a score diagnostic, not a new policy objective or EIG estimator.
- Reference gradient finite-difference check passed9 coordinates; max absolute error4.66e-10.

## Quantitative evidence

| Quantity | Result |
|---|---:|
| ScoreBED seed102 held-out EIG |7.18907|
| Seed103 held-out EIG |10.13136|
| Seed104 held-out EIG |9.79230|
| Seed102 policy coordinates with abs(x)>3.8 |28.294%|
| Sampled score-training coordinates with abs(x)>3.8 |9.844%|
| Reference-qualified seed102 histories |7/8|
| Reference-qualified seed103 histories |1/8|
| Seed102 mean design-score relative error, qualified histories |0.496|
| Seed102 mean observation-score relative error, qualified histories |0.169|
| Seed102 mean native policy-gradient cosine versus reference |0.908|
| Seed102 mean native policy-gradient norm ratio |3.929|

The policy-gradient statistic compares averages over7 qualified histories. It is not an across-training-seed confidence interval. Reference qualification requires ESS>=32 and <=20% relative score change262k→1M; these checks are necessary screening, not proof of exact reference accuracy. JAX full rollouts must agree with saved histories to within0.001 before a case enters gradient analysis.

## Interpretation

1. **Policy-distribution drift is measured:** seed102 is more concentrated near boundaries than the random design histories used for score training. This is not proof that an observation is wholly outside training support.
2. **Score errors can affect policy gradients:** observation-score predictions are more accurate than design-score predictions in this small qualified sample. Native backpropagation through the simulator converts those errors into a distorted policy gradient.
3. **The score model is not uniquely identified as bad:** on the same seed102 histories, saved models103/104 also show nontrivial errors. Model103 is not uniformly more accurate than102. Therefore simply replacing the score seed or adding more pretraining is not yet supported as a fix.
4. **Magnitude inflation alone is insufficient:** native global clipping and Adam transform gradients. Reducing learning rate merely because the raw/reference norm ratio is3.9 would not establish a causal solution.
5. **Training-loss convergence is insufficient:** seed102 tail score loss18.89 is lower than successful seeds103(20.86) and104(20.42). Its poor EIG is not explained by an unusually large logged score loss. Logged final policy-gradient clipping occurs at every sampled tail update across all five seeds, not only102.
6. **Reference uncertainty limits successful-policy analysis:** most seed103 full histories have inadequate importance-sample resolution at1M. Negative cosine values on excluded cases must not be cited as proof of incorrect gradient direction. Only1 qualified seed103 case remains, too few for a reliable model-level comparison.

## Recommended next controlled diagnostic

Before selecting a training correction, isolate score-model versus policy-initialization/optimization effects with a crossed frozen-score-model ablation. Keep the original native policy architecture, optimizer, bounded mapping and objective; compare policy seed102 trained with its own score model against the same initialization/random stream with the saved103 score model. Preserve each score model's full native standardization/configuration; do not transplant parameters alone. Treat this as development mechanism analysis, not independent full-seed confirmation or an extra named algorithm. Count score pretraining plus policy episodes for each complete policy and physical shared computation separately.

A favorable crossed result would support further investigating score training/calibration; an unchanged failure would direct attention to policy initialization/optimization. It is not legitimate to select whichever frozen score model gives the best EIG and replace the original run.

No crossed training ablation or new ScoreBED full training has been launched. Allfive ScoreBED reruns, including102first, remain pending a justified training correction, consistent with the latest request that validation repair alone is insufficient. MoE and SwitchMoE full reruns continue independently. SIR is deferred.

## Artifacts

- mapping_and_coverage.json
- score_reference_records.json: particle-resolution/ESS and reference vectors
- score_quality_records.json: learned-versus-reference score errors
- policy_gradient_records.json: native policy VJP comparisons and rollout errors
- policy_gradient_summary.json
- reference_correctness.json
- summary.json
- scorebed_optimization_audit.json in parent campaign
- reproduce scripts copied into this folder

All diagnostics run on CPU with two-core affinity and no GPU training contention. New diagnostic-only simulations:30,768 for standardization and sampler checks, separate from training budgets. Reference calculations add likelihood work and no training episodes. Evaluation seed1001 is development;8831001 untouched. Historical results are preserved.
