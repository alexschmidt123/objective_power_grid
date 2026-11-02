# Ubuntu SIR T=5 execution-efficiency development protocol

This is an implementation and runtime-calibration campaign. It is not a final
publication comparison. The existing location campaign and its selected reruns
are independent and are not modified.

## Scientific invariants

- Exactly ten main methods: Random, Optimized Fixed, DAD, Simplified RL-sBOED,
  vsOED, ScoreBED, MoE-sBOED, ResMoE-sBOED, SwitchMoE-sBOED,
  SwitchResMoE-sBOED. `sum_single` is an internal residual base, not a baseline.
- SIR T=5, population 500, initial infected 2, final time 100, minimum spacing
  0.1, the existing log-normal beta/gamma prior, Gaussian infected-count noise
  SD 1, and the unchanged adaptive RK4 rule with at least 512 steps per stage.
- Four independent 64/64 tanh experts, sum encoder, DSelect-1, existing switching
  mechanism, component-wise clipping, and existing native baseline algorithms.
- sPCE methods retain L=1024. Native vsOED and ScoreBED retain their own training
  objectives; they are not converted to sPCE trainers.
- Each complete trained model receives exactly 1,000,000 primary episodes.
  Residual models use one shared seed-specific 500,000-episode SIR base and
  500,000 additional episodes each. All auxiliary generation is counted.
- Final-budget checkpoints only. Diagnostic validation cannot select a
  checkpoint, stop for a plateau, change hyperparameters or extend the budget.
- Seed 101 is development. Reserved final evaluation seed 8831001 is forbidden.
  Final publication requires frozen settings and independent training seeds.

## Proposed execution-only changes and adoption gates

1. Compile blocks of the existing float64 RK4 arithmetic. Preserve rate-dependent
   integer subdivision, state carry, all padded zero steps, and autograd.
2. Optionally run diagnostic policy rollouts on GPU; retain the same CPU-generated
   truth, observation noise and contrast bank. The existing common likelihood
   scorer already supports GPU; it is not a new EIG estimator.
3. Use 64 systems and L=1024 for diagnostic validation every 125,000 episodes.
   These inexpensive curves may saturate and do not establish EIG convergence.
   This is a declared change from 128 systems/L=16384, applied to all ten methods.
4. Keep the final development evaluator at 2048 paired systems and L=16384,
   seed 1001. Test resolution at 4096/16384/65536 on a declared subset before
   freezing confirmatory evaluation settings. Do not claim resolution is adequate
   before this check. Resolution checks never use seed 8831001.

Adopt compiled execution only after eager-versus-compiled comparisons pass:
individual carried states and parameter/action gradients (rtol 1e-7, atol 1e-8),
complete paired T=5 actions, observations, sPCE and policy gradients (rtol 1e-6,
atol 1e-7), finite gradients at the real batch/L setting, and compatibility tests.
Compare GPU and CPU validation on the same fixed draws. Check native baseline
smoke runs and the JAX-to-Torch ScoreBED bridge separately.

## Cost and integrity

Measure cold compilation, warm forward/backward, peak memory, final evaluation
and method-specific pilot costs separately. Report compiler/environment versions.
Do not change batch size, optimizer, integration accuracy or contrasts to obtain
a speedup without a separately declared experiment. One full-size forward/backward
is a timing probe, not a measured full-training runtime.

Serialize GPU jobs. Record all development overhead separately from each final
model's training budget. Preserve failures and partial runs; no silent retry or
checkpoint replacement. Freeze source hashes and commands before full training.

Journal readiness requires independent replication, paired and seed-level
uncertainty, complete simulation and likelihood accounting, documented baseline
adaptations, negative results, and mechanism evidence. One accelerated seed-101
campaign is development evidence and cannot guarantee publication or superiority.
