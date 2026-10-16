# Journal protocol: ten methods, location finding and SIR ODE

Exactly six baselines (Random, Fixed, DAD, simplified RL-sBOED, vsOED, ScoreBED) and four MoE variants (moe,resmoe,switchmoe,switchresmoe). No DAD-MLP baseline or tanh-architecture ablation. A compatible sum-encoder single policy is only an internal residual base.

## Budget, selection and provenance

Each trained method has exactly1000000 newly simulated primary training episodes per seed. Residual methods share an exact seed-specific500000-episode frozen base, then each trains500000 additional episodes. Count pretraining in each complete method; disclose shared physical computation. ScoreBED uses10014 initialization/standardization episodes,692990 score episodes and296996 policy episodes; one restart, batch64. vsOED location uses20000 replay initialization +980*1000 new episodes, native GMM-I discount0.9. Simplified RL uses1million fresh trajectories, L1024, gamma1, twoQ networks, two replay cycles; replay reuse is not a new trajectory. Fixed includes every training simulation. DAD/sPCE training L1024, native sum256/16 linear emitter. All published-method departures remain explicit. Baseline architectures/objectives/optimizers are not replaced by MoE conventions.

Final allocated-budget checkpoint only; no best-validation/test selection and no early plateau stop. Validation every125000 phase episodes is diagnostic and uses independent draws. Fixed-budget complete does not establish convergence. Safety-cap/failed/partial jobs are reported as incomplete. A completed phase may be reused after identity/budget checks; exact mid-phase resume is not claimed for native trainers lacking complete replay/auxiliary state.

## Location and matched routing

Two2D standard-normal sources, log(0.1+sum1/(0.0001+distance²)), Gaussian SD0.5, T30, bounds[-4,4]², unchanged simulator and action transformations. Four64/64tanh experts, sum encoder, component clipping, native regularizers. Matched MoE/SwitchMoE router clock1M; ResMoE/SwitchResMoE clock500k. Gamma1 through30%, cosine to0.001 at95%; gate sigmoid unchanged. Same optimizer settings within each switching pair. Historical SwitchMoE0/80 schedule is incompatible and must be replaced, never interpreted as switching-only evidence.

## Seeds, evaluation and statistics

Training seeds101–105. Seed101 compatibility is checked from saved artifacts, benchmark signatures, budget, native settings, final selection, bank identity and complete action-sequence replay for16 cases. No favorable-seed replacement. Random has no training replicates; one common development reference is not five independent random trainings. Held-out location2000 paired trajectories/L1000000, seed1001. Validation512/L1000000, seed771001. Reserved8831001 remains untouched through development; no final-test selection. The algorithms, source/config manifest and this analysis are frozen before remaining replication.

Principal contrasts: ResMoE-ScoreBED, ResMoE-DAD, ResMoE-MoE, SwitchMoE-MoE, SwitchResMoE-ResMoE, SwitchResMoE-SwitchMoE. Pair identical training-seed indices. Report individual checkpoint trajectoryMC uncertainty separately from SD/SE over independently trained policy means. Use paired Student t95% intervals across available complete seed differences and conservative Bonferroni simultaneous95% intervals over six contrasts; evidence labels use simultaneous intervals. Five runs are limited precision, not10000 independent training replicates. All1001 analyses are development-evaluation replication, not untouched confirmation.

## Computational accounting and mechanisms

Serialize training/evaluation on one device, sample GPU utilization/memory where available. Report primary episodes, contrast particles, stage likelihood work, gradient/replay updates, offline optimization/validation/evaluation seconds and online latency. Historical timings may have overlap and are flagged, not presented as controlled speed comparisons. Residual end-to-end curves include the base; reused computation is disclosed. Missing historical validation points remain missing. Mechanisms report E1–E4 utilization/stage, gradient exposure and final20% starvation, matched-history proposals/residual magnitudes and observation counterfactuals, requests versus actual changes/durations. Counterfactuals are not rollout EIG; no hand-assigned expert roles or guaranteed specialization.

## SIR T5 readiness, then development

Preserve config population500,I0=2,end100,gap0.1; beta=exp(log0.5+0.5Z1),gamma=exp(log0.1+0.5Z2), infected observations with Gaussian SD1, original carried adaptive-min512 RK4 and explicit likelihood. Parameters0.5/0.1 are log-normal medians. No location checkpoints. All four families/baselines require finite feasible rollout/likelihood/gradient checks. Native vsOED is an explicitODE adapter, not the paper stochasticSIR bank reproduction; initial replay320 (instead of unexecuted draft256) plus7810*128 fresh episodes totals exactly1M. ScoreBED pretraining is counted. Proposed common SIR scoring2048/L16384, chunk4096 requires nested-resolution checks; initial readiness checks on smoke/random policies cannot certify developed policies. SIR base500k and residual500k; same base within seed. One T5 development seed101 after location validation and readiness; T10 deferred. GPU batch64 pathwise timing predicts roughly15hours per million episodes, not convergence. Per-phase25hour execution safety cap: incomplete jobs remain explicit; no silent budget reduction. Full SIR five-seed replication is prepared only after development/readiness and benchmark-specific precision freeze.

## Reproduction

Rootrun.sh launches every training phase. tools/journal_campaign.py orchestrates sequential work with source guards and per-phase command JSON. CSV/JSON/figures are generated from artifacts by `python -m tools.journal_campaign report`. Source snapshots/commit hashes and atomic run registry are retained. No historical source manifest/result/checkpoint is overwritten. The source tree includes preexisting uncommitted work, whose frozen content hashes are authoritative in addition to the Git commit.
