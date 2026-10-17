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

1. Fuse the existing float64 RK4 arithmetic in a CUDA kernel. Preserve rate-dependent
   integer subdivision, state carry, all padded zero steps, and autograd.
   The custom backward differentiates the discrete RK4 operations using exact
   first-order sensitivities. It is not a continuous-adjoint approximation.
   Higher-order derivatives are explicitly unsupported; native policy training
   requires first-order derivatives. The original eager backend remains default.
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

The bounded Ubuntu probe measured 3.38 s eager versus 0.409 s warm fused per
32-episode T=5/L=1024 step (about 8.3x). This projects to 3.55 optimization hours
per million episodes for the probe policy. Plan provisionally 36--72 h for all
ten methods, including evaluation; native algorithm costs remain to be measured.
Use a 72-hour campaign safety cap and 24-hour optimization cap per method.
If a cap prevents completion, preserve the partial run and mark budget-limited;
do not present it as a completed 1M-episode result.

Full-run order: Random; internal base; ResMoE; SwitchResMoE; MoE; SwitchMoE;
Optimized Fixed; DAD; Simplified RL-sBOED; vsOED; ScoreBED. Each uses a fresh
process and its original optimizer/batch conventions. Exactly ten main methods
are reported; the shared base has its own computational-accounting row.

Serialize GPU jobs. Record all development overhead separately from each final
model's training budget. Preserve failures and partial runs; no silent retry or
checkpoint replacement. Freeze source hashes and commands before full training.

Journal readiness requires independent replication, paired and seed-level
uncertainty, complete simulation and likelihood accounting, documented baseline
adaptations, negative results, and mechanism evidence. One accelerated seed-101
campaign is development evidence and cannot guarantee publication or superiority.
