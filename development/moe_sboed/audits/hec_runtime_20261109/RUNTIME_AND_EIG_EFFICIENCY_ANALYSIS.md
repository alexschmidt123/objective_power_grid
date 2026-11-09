# Runtime and EIG efficiency analysis

Audit date: 2026-11-09. Authoritative inputs are the saved LabPC configurations, JSONL training logs, per-arm RESULT/STARTED receipts, actor checkpoints, full-runtime hashes, frozen-policy score vectors and evaluation receipts. This audit performed **zero training updates, zero neural inference calls and zero new simulator calls**, and used CPU only. Actor tensor dictionaries were read on CPU with restricted PyTorch loading; full runtimes were hashed without deserializing training objects. All 141 read input files, including six terminal actors and six full runtimes, retained their SHA256 values after the audit.

Source roots, relative to LabPC `~/Documents/objective_power_grid_temp/scratch_runs/`:

- C: `moe_collaborative_soft_queue_20261108/`.
- S: `moe_specialization_guided_eig_20261108/`.
- H: `moe_hec_sboed_20261108/`.

Audit evidence is isolated in `scratch_runs/moe_hec_runtime_efficiency_audit_20261109/{AUDIT_SUMMARY.json,RAW_RECEIPTS_AND_LOGS.json,INPUT_PRESERVATION.json}`. Reproduce with `python tools/audit_hec_runtime.py --output ~/Documents/objective_power_grid_temp/scratch_runs/NEW_AUDIT_DIRECTORY`; the collector requires a new directory and refuses to overwrite an existing audit. No existing experiment, checkpoint, evaluation rule or model was changed. The current machine-access workflow supersedes old access restrictions, without editing historical protocol files.

## Main finding

The quoted 3h57m, 53m and 1h58m are **two-arm experiment durations**, not the training time of one candidate. HEC used five times fewer native updates per arm than fresh Collaborative, while doing about 2.5 times as much wall-time work per update. That explains its shorter total versus Collaborative. Relative to SG, HEC used the same 2,000-update continuation budget but substantially more model-query and gradient work; it was slower, not a demonstrated implementation speedup.

## Measured time decomposition

All three ran sequential arms on the same recorded GPU type, NVIDIA RTX 4090. The scientific-driver timer begins after acquisition of the GPU lease. The completion receipt is written before final report generation and preservation sealing; queue wait, implementation time, preflight checks and final sealing are outside these headline durations. Arm times include training/checkpoint serialization within their timer; terminal hash calculations and some loading are in the remaining driver time. These are wall-time allocations, not measured SM-busy GPU hours or energy.

| Experiment | Control | Candidate | Shared preparation (s) | Evaluation (s) | Other driver (s) | Whole pair |
|---|---|---|---|---|---|---|
| Collaborative | 1h 56m 48s | 1h 57m 44s | 0.000 | 156.676 | 1.972 | 3h 57m 11s |
| SG | 0h 23m 09s | 0h 23m 22s | 0.000 | 408.900 | 2.811 | 0h 53m 22s |
| HEC | 0h 51m 55s | 0h 58m 41s | 5.934 | 434.468 | 4.188 | 1h 58m 01s |

Sources: each root’s `arms/*/RESULT.json`, `preparation/RESULT.json` where present, `evaluation/COMPLETION.json`, `evidence/EXPERIMENT_COMPLETION.json` and `source/run_experiment.py`. HEC preparation’s 5.934s is physically performed once, copied to both arms and charged equally to them in conditional efficiency calculations. It is not 300 native updates or 300 new simulator batches.

## Training budgets and actual observation simulation

| Per arm | Collaborative | SG | HEC |
|---|---|---|---|
| Initialization | Fresh actor/critic/posteriors/Adam/replay | Trained Collaborative candidate | Trained SG candidate |
| Native update counter | 0 → 10,001 | 10,001 → 12,001 | 12,001 → 14,001 |
| New actor updates | 10,001 | 2,000 | 2,000 |
| Critic / posterior-net optimizer steps | 50,005 / 1,500,150 | 10,000 / 300,000 | 10,000 / 300,000 |
| New primary episodes incl. initial replay | 10,021,000 | 2,000,000 | 2,000,000 |
| New primary observation simulations (T=30) | 300,630,000 | 60,000,000 | 60,000,000 |
| Fresh initial replay episodes | 20,000 | 0; inherited 1M buffer | 0; inherited 1M buffer |
| Replay minibatch / capacity | 10,000 / 1M | 10,000 / 1M | 10,000 / 1M |
| Actor design rows per native update | 10,000 × 30 | 10,000 × 30 | 10,000 × 30 |
| Discount / final specialist loss | 0.9 / ongoing auxiliary | 1.0 / none | 1.0 / none in B/C |
| Expert-learning schedule | Joint plus task gradients throughout | Joint EIG for all 2,000 | 300 shared supervised steps; freeze experts in B=200; joint C=1,800 |
| Training prior contrasts | 0; variational AC | 0; variational AC | 0; variational AC |

The native profile uses five critic updates and five updates of each of 30 posterior nets per actor iteration. Its fixed replay batch is 10,000, not the EIG-evaluation batch of 16. HEC’s 4,096-row gradient-checkpoint chunks partition the 300,000-row actor calculation; they do **not** reduce its statistical minibatch. The fresh 20,000-episode replay accounts for 600,000 observations per Collaborative arm.

HEC additionally generated **27,648 synthetic one-step predictive observations** for shared specialist preparation from the saved approximate belief, not true-latent labels read from replay. These are separate from 60M native observations per arm and were physically generated once for the pair. Preparation had 4,096 training and 512 validation prefixes, 300 Adam steps × 256 samples = 76,800 reused training-prefix presentations, and no new real episodes. Direct teacher-provider accounting adds 26,682 SG-policy rows, 26,682 frozen-critic rows and 966 terminal-posterior rows to the policy-forward counters (`H/evidence/PREPARATION_DIRECT_PROVIDER_COUNTS.json`). Neither a quadrature signal evaluation nor a replay presentation is a fresh experimental episode.

The initial native learning rates were approximately 0.001, 0.000367824 and 0.000301146 respectively; exploration was 0.5, 0.183912 and 0.150573. Both continue their saved .9999 decays, and all use the same native Polyak rate 0.1. Collaborative had ongoing expert auxiliary gradients and gamma 0.9. SG explicitly changed both arms to gamma 1 and a normalized prior-mixture density safeguard; HEC inherits those choices. Therefore the three campaigns are not identical training-objective experiments.

Sources: locked configs, per-arm STARTED/RESULT receipts, JSONL optimizer counts, `C/source/train_soft_pair.py`, `S/source/train_sg_pair.py`, `H/source/{train_pair,migration,prepare_specialists}.py`, and the imported direct-head `frozen_protocol.json` native profile. Checkpoint metadata confirms terminal counters 10001,12001,14001 and their distinct candidate/control method flags.

## Throughput, phase timings and GPU memory

| Experiment | Arm | Wall s/update | Median logged interval s/update | Peak torch allocated GiB |
|---|---|---|---|---|
| Collaborative | control | 0.700774 | 0.705416 | 4.183 |
| Collaborative | candidate | 0.706313 | 0.708858 | 4.183 |
| SG | control | 0.694415 | 0.695393 | 5.045 |
| SG | candidate | 0.700843 | 0.703048 | 5.043 |
| HEC | control | 1.557529 | 1.564483 | 2.505 |
| HEC | candidate | 1.760637 | 1.766680 | 2.505 |

The mean divides the arm RESULT wall time by completed native updates; the interval median uses successive saved JSONL records (usually 50 updates), not unsaved per-step kernel measurements. The final logger-to-result gap is only about 0.5–1.0s per arm.

| HEC arm | B=200 time (s) | B s/update | C=1800 logged time (s) | C s/update |
|---|---|---|---|---|
| control | 298.990 | 1.494952 | 2815.178 | 1.563988 |
| candidate | 341.468 | 1.707338 | 3178.828 | 1.766015 |

Phase-C numbers end at the terminal log and exclude the sub-second terminal arm-serialization gap. Logs show zero expert/history gradient allocation during B by design, then nonzero gradients for all four experts/history during C; candidate commander/coordinator gradients are nonzero, while the control’s unused commander/coordinator gradients remain zero.

HEC candidate is **13.04% slower than its matched control** (3521.274 vs 3115.058s). Its forward primitive counters match, but the dynamic commander differentiates through commanded likelihood/sensitivity/critic-input queries and uses feature-aware coordination. These extra gradient paths are verified by code and logged gradients. They plausibly explain the added cost; no profiler isolates their exact share from synchronization, kernels or contention.

HEC candidate is **2.512× SG candidate’s s/update**. Pair training is 2.378× SG and whole driver time is 2.211× SG. SG candidate’s 0.701s/update is close to Collaborative’s 0.706s/update. Therefore SG’s much shorter completion is overwhelmingly a reduced continuation budget, not a measured faster kernel. Removing continuous auxiliary work is a real code difference, but Collaborative’s recorded auxiliary host section totals only 293.259s (4.15% of its candidate arm); it cannot explain the roughly 5× reduction in update count. This host timing is not a synchronized GPU-kernel attribution.

The logs record peak `torch.cuda.max_memory_allocated`, not total device reservation, context memory or average GPU utilization. No complete utilization/power trace or kernel profiler was saved; current idle GPU readings cannot reconstruct historical GPU occupancy. HEC’s lower 2.505GiB allocation is consistent with source-level gradient checkpointing and its different live/frozen graph; **no on/off ablation proves the exact memory saving or any throughput gain**. It trades recomputation for retained activation memory. There is no evidence HEC was faster because of higher GPU utilization, faster hardware, a smaller statistical batch or an undocumented budget change.

## HEC’s extra work and deployed model footprint

Each HEC arm’s actual instrumentation records 706,560,000,000 exact signal evaluations, 117,760,000,000 analytic parameter Jacobians, 1,218,000,000 frozen-belief rows, 1,240,000,000 frozen-SG provisional rows and 5,520,000,000 frozen-critic rows. Both arms have identical forward counts; these include actor recomputation. They are model-derived query work, not billions of independently observed experiments. Native actor/critic/posterior work is additional and is accounted by the optimizer/simulator budget above.

`H/source/hec_policy.py` uses 64 deterministic GMM sigma points, exact likelihood-derived features, analytic Fisher queries, three frozen-critic probes, stage-wise belief precomputation and checkpointed live computation. The analytic Fisher avoids unnecessary reverse-mode Jacobian calls, but this audit did not benchmark another implementation. It cannot claim any quantitative speedup from those engineering choices.

Checkpoint tensor inspection gives:

| Checkpoint state | Collaborative candidate | SG candidate | HEC candidate/control |
|---|---|---|---|
| All stored tensor elements | 166,308 | 330,280 | 15,368,770 |
| Allocated new/live parameter elements incl. readouts | 166,308 | 199,592 | 139,297 |
| Frozen providers / anchor state | 0 | 130,688 | 15,229,473 |
| Policy-function active live parameters (phase C) | 164,358 | 197,642 | 136,506 candidate / 101,542 control |

HEC buffers include the SG policy (330,280 elements), belief providers (14,736,632 including small constants) and critic (162,561). Stored elements are not all trainable weights. HEC’s 139,297 allocated parameters include 2,275 diagnostic readout parameters and unused alternative-routing modules. Its candidate has 34,964 more policy-function active parameters than its control (**34.43%**); identical tensor allocation is not identical effective capacity. The EIG receipt’s `network_parameters=0` for frozen SG reflects conversion to buffers, not a zero-cost/zero-size policy.

On the HEC evaluation’s same saved histories, inference timings are HEC 2.5655ms/design, control 2.5211ms and frozen SG 0.4521ms: HEC is approximately **5.675×** slower than SG in this single-case benchmark. Earlier recorded Collaborative and SG candidate timings were 0.2256ms and 0.4434ms/design, with different repetition schedules; compare the same-run HEC/SG timing first. HEC is not an established deployment-efficiency improvement.

## Evaluation workloads

All use 2,000 new full trajectories, T30, 1M negative prior contrasts, one positive included once (1,000,001 denominator terms), evaluation batch16/chunk16384 and the frozen scorer. Distinct banks are never paired across campaigns.

| Experiment | Main frozen policies | Main observation simulations | All full-rollout receipts: observations | All evaluation wall s |
|---|---|---|---|---|
| Collaborative | 4 | 240,000 | 301,440 | 156.676 |
| SG | 10 | 600,000 | 738,240 | 408.900 |
| HEC | 11 | 660,000 | 752,160 | 434.468 |

Collaborative evaluated four main policies, a 512-case diagnostic rollout and six 256-case intervention rollouts. SG evaluated ten main policies, the diagnostic base and sixteen 256-case interventions; it additionally generated 132,288 suffix observations for its separate conditional native-return panel. HEC evaluated eleven main policies, the diagnostic base and ten 256-case interventions; its competency teacher adds 9,216 synthetic one-step observations (1,536 prefixes ×3 queries ×2 replicas). These extra native-return/teacher observations are not contrast-scored full trajectories. Thus observation totals represented here are 301,440 (C), 870,528 (S including conditional panel) and 761,376 (H including teacher), under these explicit definitions.

The summed full-rollout receipts take 151.859, 396.260 and 426.187s. Remaining evaluation time also covers bank construction/loading, model construction, probes, teacher/conditional work and latency timing; no finer exact timing is available. HEC vsSG adds one main policy but removes six full-rollout interventions. Its full contrast workload rises only from 738.24B to752.16B negative observation-likelihood terms (1.89%); evaluation wall rises6.25%. HEC’s 25.6s extra evaluation explains little of the 3,878.7s whole-driver gap versus SG. It does explain why a comparison using only update counts misses some cost.

## Inherited costs and meaningful efficiency comparisons

The saved parent hashes confirm the lineage **fresh Collaborative candidate → SG candidate → HEC**. HEC inherits the trained SG provisional policy, live history initialization and compatible Adam moments, trained native critic/posteriors, schedules and replay. The four new heterogeneous experts and incompatible heads do not inherit old expert weights/Adam; they receive shared fresh preparation, then 1,800 joint EIG updates. HEC is a warm-started system, not a fresh four-expert training result.

| Candidate-producing lineage | Accumulated native counter | Accumulated new primary episodes | Training wall (excluding controls/evaluation) |
|---|---|---|---|
| Collaborative candidate | 10,001 | 10,021,000 | 1h 57m 44s |
| SG lineage | 12,001 | 12,021,000 | 2h 21m 06s |
| HEC lineage | 14,001 | 14,021,000 | 3h 19m 53s |

HEC’s candidate lineage is 14,021,000 real episodes /420,630,000 observations, plus its teacher preparation, and about 3h20m candidate-producing training. This does not mean every HEC parameter received14,001 updates. Funding all three two-arm driver campaigns used6h48m34s, excluding preflights/queue/seals. Counting only the latest1h58m as the full development cost would omit inherited training. Parent costs can be treated as sunk only for a clearly labeled incremental continuation decision.

For HEC **on its own common bank**, conditional gain over the same frozen SG parent can be divided by each observed training-plus-preparation time. These are descriptive endpoint amortization ratios, not a learning-curve derivative, runtime-confidence bound or matched-time experiment. Preparation is charged in full to each arm for standalone use; physically the pair prepared it once. Evaluation time and inherited costs are excluded consistently. Control-vsSG intervals below are audit-derived, unadjusted descriptive paired intervals, not a new predeclared primary family.

| HEC arm | Estimate (nats) | Gain vs frozen SG [95% CI] | Incremental GPU-wall h | Conditional gain nats/GPU-wall h [95% CI] |
|---|---|---|---|---|
| control | 12.88750 | +0.11345 [+0.05544, +0.17145] | 0.866942 | 0.13086 [0.06395, 0.19777] |
| candidate | 12.95961 | +0.18556 [+0.12445, +0.24666] | 0.979780 | 0.18938 [0.12702, 0.25175] |

The matched-update result is candidate−control **+0.07211 [0.01889,0.12532] nats**, p=.00794, with the same2M new episodes but13% different wall times. This supports an improvement per matched new-episode/update budget for the complete package in one seed. It does not show a superior equal-time learning curve; giving the control the candidate’s extra406.2s was not tested. Dividing that gain by the time difference would not establish the causal marginal value of computation.

The analogous SG-on-its-own-bank rates are0.15452 (control) and 0.34646 (candidate) gain nats/GPU-wall h relative to its frozen Collaborative parent; they are conditioned on another parent/bank and common gamma/density-floor corrections. **Do not rank HEC against SG from these different-stage rates.** We do not report absolute final EIG/hour as an across-campaign winner, because inherited policy quality, objectives, extra providers, training opportunities and estimator banks differ. No compute-matched superiority over native vsOED or fresh-training-efficiency claim is justified.

## Which components plausibly explain HEC’s EIG gain?

On the HEC bank: frozen SG12.77406±0.03566; fixed-command heterogeneous continuation12.88750±0.03378; HEC12.95961±0.03374 (nats±MCSE). The frozen-SG gap is+0.18556 [0.12445,0.24666], Holm4 p=3.06e−9. The fixed-command control already gains+0.11345 over frozen SG. Arithmetically that is61.1% of the HEC point gap; the remaining38.9% is the candidate-control difference. This is **not a causal component decomposition**.

Verified: both new arms use the same saved SG parent, prepared specialist tensors, replay and native update budget; initial state digests match. They differ in dynamic commands, feature-aware coordination and effective active capacity/gradient cost. Their package contrast is positive. Additional continuation and a learned residual correction around a strong SG policy are available to both arms and plausibly explain part of both gains over frozen SG.

Unestablished: whether distinct specialist semantics, command adaptation within a stage, feature-aware coordination alone, the extra34,964 active parameters, or optimization noise caused the remaining gain. A history-only SG residual control, ordinary SG continuation from the same parent, and a command×coordination factorial are missing. Inference ablations are informative but are not replacements for matched training controls. The specific ICML claims are assessed separately in ICML_SPECIALIZATION_AND_ROUTING_ASSESSMENT.md.

## Provenance and limits

Every actor/runtime SHA matches its RESULT receipt; parent hashes match the next config; native terminal counters agree with logs/checkpoints; both arms’ initial state digests match in every campaign. The audit recomputed same-bank paired contrasts from saved2000-case score arrays and matched the stored SG/HEC primary means to1e−12. No new trajectory, estimator, checkpoint selection or model fit was used. Full hashes are in INPUT_PRESERVATION.json; the six actor/runtime hashes are also in each original RESULT.json.

Primary information values are finite-contrast lower-bound estimates, not exact EIG. Their Monte Carlo intervals do not include training-seed variability, finite-contrast bias or systematic model error. Three differing campaign scores cannot be turned into a controlled architecture-efficiency comparison merely by normalizing wall time.
