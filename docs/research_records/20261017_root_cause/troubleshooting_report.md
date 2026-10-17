# BOED instability investigation: current evidence

Updated 2026-10-17 UTC. This is a development diagnosis, not a corrected-method
result. All original five-seed location results remain unchanged. No final
evaluation seed8831001 has been used.

## Executive finding

**The failures are real, but a single causal implementation defect has not been
isolated.** The original-setting MoE103 repeat reproduced both the complete
checkpoint SHA256 and held-out EIG exactly. The ten-checkpoint audit across
MoE/SwitchMoE seeds101--105 supports late relaxed-training deterioration and
unsettled routing/optimization in the failed seeds. It does not support simply
blaming a dead router, final selector saturation, or universal expert starvation.
ScoreBED's validation metadata defect does not explain the bad final policy.

## Rerun status after the user's stop instruction

- MoE103 completed1M episodes, EIG6.253217049663595, identical to the original;
  paired difference0 and identical checkpoint SHA
  `108adf1b63521c650b003dd96ce66e5049b7bb210776a400a20bfb419db8f1d2`.
  Repeat optimization1748.49s; validation72.01s; complete wall1858.88s.
- SwitchMoE103 terminated with155,968 primary episodes recorded. Its partial
  checkpoint and traces remain saved. This is not a completed1M result.
- ScoreBED102 was cancelled before starting.
- The LabPC rerun coordinator and training process stopped on user instruction.
  No replacement result or new training correction has been promoted. Ubuntu's
  independent on-the-fly SIR campaign is unaffected.

## Required diagnosis table

|Failed method/seed|Earliest observed deterioration|Evidence|Causal status|Recommended correction now|Retraining needed?|
|---|---|---|---|---|---|
|MoE103, EIG6.25322|Validation declined between750,336 and875,392 episodes; descriptive sustained training decline begins in the883,200-episode window|Training sPCE falls; very large raw router gradients; exactly reproducible final checkpoint|Optimization deterioration confirmed; initiating cause unresolved|None: measure the routing/encoder gradient path before changing settings|No justified corrected retraining yet|
|SwitchMoE103, EIG6.50948|Sustained training decline begins in the832,000-episode window; validation also falls by875,392|Shares unstable gradient pattern; final policy is poor under hard and relaxed routing; hard gate always reroutes|Deterioration and absent hard persistence confirmed; causal trigger unresolved|None: gamma and switch-gate changes are not justified by this evidence alone|No justified corrected retraining yet|
|ScoreBED102, EIG7.18907|Exact onset cannot be recovered: intermediate physical design/policy checkpoints were not retained sufficiently|Boundary concentration, imperfect learned score gradients, but no uniquely bad score loss or demonstrated extra training bug|Score approximation versus policy optimization remains unresolved|First isolate the frozen-score-model contribution with the paired diagnostic below|No broad retraining until the diagnostic identifies a cause|

The training onset criterion is descriptive: nonoverlapping200-update windows
(12,800 episodes), a mean drop of0.3 from the previous best, sustained for three
windows. It is not a statistical change-point confidence interval. Final-budget
checkpoint selection is unchanged.

## MoE and SwitchMoE: what the measurements show

The raw-gradient values below are before clipping. Component clipping remains5;
large raw gradients are not evidence of equally large applied parameter updates.
Norm-only logs do not reconstruct Adam parameter-step directions or sizes.

|Model/seed|Last5% router median raw norm|Last5% zero-router-gradient updates|Final held-out saturated selector bits|
|---|---:|---:|---:|
|MoE103|6515.96|0.00%|99.39%|
|SwitchMoE103|4735.03|0.13%|99.61%|
|MoE105, successful|0|99.36%|100.00%|
|SwitchMoE105, successful|0|95.14%|100.00%|

Zero means norm<=1e-12; negligible exposure is separately recorded at<=1e-8.
Final saturation means abs(logit)>=gamma/2, measured on128 held-out histories
over the29 post-initial decisions, gamma0.001. The JSON includes complete
stage-wise logit quantiles and selector derivative distributions, both on each
policy's own histories and common failed103 histories.

These comparisons contradict saturation alone as a sufficient explanation:
successful105 is more completely saturated. The failed models have persistent,
sometimes extreme gradients rather than a wholly frozen router. A plausible
mechanism is unstable joint routing/encoder/expert adaptation near sharpening,
but final-point measurements and correlations do not establish its causal path.

Expert starvation is not a sufficient shared explanation either. Every expert
in both failed runs receives a non-negligible gradient in all recorded updates
through80% of training. In the final5%, MoE103 exposure is100%/70.1%/100%/100%;
SwitchMoE103 is100%/0%/100%/0%. Successful105 models also leave experts largely
unused late. Nonzero gradient exposure is not proof of informative learning.

Successful checkpoints generally show increasing high-precision validation EIG.
For MoE103 it is7.373 at750,336,7.015 at875,392,6.355 at1M on the validation
population. For SwitchMoE103 it is5.934,5.247,6.610 at those same budgets:
its final hard-policy validation partially recovers even while relaxed training
has deteriorated. Do not describe its hard-validation trajectory as a monotonic
collapse. Validation and the2,000-case final development evaluator are distinct.

Existing paired512-case/L1M/seed1001 diagnostics show:

|Checkpoint|Hard EIG|Relaxed EIG|Relaxed-minus-hard|Paired MC SE|
|---|---:|---:|---:|---:|
|MoE103|6.27867|6.29222|0.01356|0.02060|
|SwitchMoE103|6.52223|6.57408|0.05185|0.03808|

Both deployed and relaxed policies are poor; a final hardening mismatch alone
cannot explain a several-nat deficit. These subset diagnostics do not replace
the original2,000-case EIG results.

Expert proposals differ on common histories: mean pairwise bounded-action
distance is3.08 for MoE103 and3.84 for SwitchMoE103. Different proposals do not
prove useful specialization or expert action quality. The saved trajectories,
stage-wise action disagreement and instantaneous observation-sensitivity
statistics are available in the audit. Controlled forced-expert quality/EIG
interventions have not been newly run and would require distribution-shift
caveats. No expert has been given a handcrafted behavioral role.

Historical raw training logits were not logged, and intermediate weights were
not retained at every diagnostic validation. Therefore the temporal fraction
of saturated selector logits cannot be reconstructed honestly from final
weights. The audit does not label final-checkpoint measurements as training-time
observations. This prevents confirming the precise initiating feedback path.

## ScoreBED: confirmed checks and unresolved mechanism

- Validation IPC omitted bounded-action metadata; that defect was fixed.
  The final evaluator already used correct bounds. The final EIG7.18907 remains
  a valid development observation, not a stale unbounded score.
- The bounded-action Jacobian check found no missing chain-rule factor.
- Final seed102 boundary-coordinate fraction(abs(x)>3.8) is28.294%, versus
  approximately9.844% in sampled score-training designs.
- Seed102's tail score loss18.89 is lower than successful103's20.86 and104's20.42.
  Training loss alone does not identify which frozen score model is useful.
- On seven reference-qualified seed102 histories, the mean native policy-gradient
  cosine is0.908 and norm ratio3.929 versus the explicit-likelihood diagnostic.
  Only1/8 successful103 histories passes the reference-quality filter; excluded
  cases cannot establish erroneous gradient direction.
- Native clipping and Adam alter raw gradients. A norm ratio of3.9 is not, by
  itself, a reason to lower learning rate. Other saved score models also have
  errors on the bad policy's histories;103 is not uniformly better.

## Exactly one next controlled experiment

**Cross frozen ScoreBED score models while holding policy initialization and
training randomness fixed.** This is a diagnostic ablation, not a new method
or proposed hyperparameter correction. It has not been trained.

Compatibility preflight passed:102 and103 have identical variable structure;
their configurations differ only in the model initialization seed. Each keeps
its own full normalization statistics, network configuration and saved variables.
Restoring the statistics reproduces prior predictions exactly. The paired native
policy initialization has identical parameter hash
`bde6cc1ad3e2083e80240bfece064e427d1f87cfbaf9508c6a184dee148962bd`
and identical RNG state. No historical initial-policy hash exists for independent
bitwise verification; this limitation remains explicit.

ArmA uses frozen score102; armB frozen score103. Both use the same native
policy-seed102 initialization, optimizer, observation simulator, bounded map,
score-gradient objective and policy random stream. Preserve the complete score
model bundle; do not insert103 variables into a102-normalized network.

Each complete model's budget is exactly1M:

|Component|Primary episodes|
|---|---:|
|Normalization|10000|
|Score initialization|4|
|Score training|692990|
|Policy initialization|10|
|Policy training|296996|
|Total|1000000|

Historical frozen-score pretraining is reused and charged to each model's total;
physical shared computation and new policy-training work must also be reported.
Preflight computation is diagnostic overhead, not hidden policy training:
30,038 simulated episodes across checks/failed preflight attempts, no optimizer
updates. Do not silently regenerate normalization during each arm and omit its
cost. Use the cached verified normalization bundle.

Evaluate both final policies using the existing paired2,000/L1M/seed1001 scorer.
A changed outcome isolates the frozen score-bundle contribution for this policy
seed; it does not prove general stabilization. An unchanged failure directs
attention toward policy initialization/optimization. Do not choose the better
arm to replace a historical seed. Any later optimization correction must be
tested on all five original training seeds before replacing study results.

## Files and limitations

On LabPC: `experiments/10162026/10162026_location_root_cause_v3/` contains
`moe_evidence.json`, `audit_complete.json`, `score_cross_preflight.json`, verified
normalization bundles and matched initial-policy parameter files. The prior
ScoreBED diagnosis and hard/relaxed diagnostic JSON remain in the preserved
`10162026_location_stability_correction` campaign. The selected-repeat stop
receipt records the exact completed and partial work. No original checkpoint,
scientific architecture, optimizer or final result was overwritten.

Current conclusion: reproducible failures and several failure mechanisms are
measured; a causal, validated correction has **not** yet been demonstrated.
