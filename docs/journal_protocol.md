# Journal study: location finding and SIR ODE

Status: **development calibration; not frozen for confirmation**. Reference `123aed2`. Campaign `experiments/10152026/10152026_journal_location_sir/`. Historical final-checkpoint experiments remain separate and are never relabelled validation-selected.

## Scope and scientific questions

Exactly four MoE family methods: `moe`, `resmoe`, `switchmoe`, `switchresmoe`. Existing independent64/64 tanh experts, sum encoder, DSelect-1, sigmoid switching, full-horizon sPCE and feasibility maps are preserved. Single-policy controls are not extra family variants. Primary questions are achieved EIG, residual/routing/switching contributions, efficiency, and generalization between benchmarks. No architecture search, PK, power-grid experiments, MSE training, or auxiliary networks are introduced.

External algorithms keep their native architecture and learning mechanism. DAD's sum encoder and linear emitter remain distinct from the64/64 tanh experts. The native variational posterior/critic used by the **external vsOED baseline** and the score model used by **external ScoreBED** are retained; neither is added to MoE.

## Benchmark fairness and departures from published settings

All methods share the same parameter prior, physical initial state, observation model/noise, horizon, and feasible experimental-design space within each benchmark. This does not require identical neural weights or initialization across architectures. Paired evaluation uses common parameter/noise draws and one benchmark-specific EIG evaluator.

Method-specific training durations, optimizers, initialization and native losses are permitted. Resource-saving departures from published settings are permitted, but must be disclosed as adapted implementations rather than exact paper reproductions. Do not force external baselines to use MoE experts, sPCE training, or identical runtime. Record upstream revision, original setting, actual setting, reason, cost consequence and whether the learning mechanism changes for every departure. Distinguish paper-reported numbers from locally evaluated results.

The historical corrected location vsOED score8.58218 used980 updates, compared with10001 reference updates recorded in its saved profile; it is not established to be converged. Native actor/critic/posterior learning is retained. A fresh validation-controlled development rerun is justified after benchmark/action/likelihood and validation callback checks pass. Preserve historical results; do not infer that insufficient updates caused the entire gap or tune to reproduce a paper number. Differences in benchmark physics and scoring must be listed explicitly.

## Exact benchmark definitions

Location: two independent2D standard-normal sources; signal `log(0.1+sum_j 1/(1e-4+||theta_j-x||^2))`; independent Gaussian observation noise SD0.5; T30; common[-4,4]^2 physical designs. The original sigmoid feasibility transformation is applied once. Source prior and observation likelihood are not changed.

SIR: S(0)=498, I(0)=2, R(0)=0, N500. `dS/dt=-beta*S*I/N`, `dI/dt=beta*S*I/N-gamma*I`, `dR/dt=gamma*I`. Independently, `beta=exp(log(0.5)+0.5*Z1)` and `gamma=exp(log(0.1)+0.5*Z2)` for standard-normal Z. Despite their configuration names,0.5/0.1 are medians, not arithmetic means. Observe `I(t)+Normal(0,1)`. T5 first, T10 additional. Increasing observation times end by100 with minimum spacing0.1. The existing carried-state RK4 integrator uses at least512 steps per interval, with rate-dependent subdivision bounding `dt*(beta+gamma)<=0.2`; counts are detached for differentiation. Solver, initial state, prior, Gaussian likelihood and constraints are preserved.

## Policy selection and stopping

One shared **scoring and stopping controller**, with method-specific optimization and resource limits. It wraps existing trainers rather than replacing native losses. Validation uses independent fixed paired prior draws, observation noise and contrast bank; deployed hard policies; the existing streamed explicit-likelihood sPCE evaluator; no posterior MSE inference. Training RNG/state is preserved across validation. Validation seed771001 for location and772001 for SIR; reserved final seed8831001 is rejected before validation draws.

Draft location validation:512 trajectories, L1,000,000. SIR:128 trajectories, L16,384 initially. These settings are **not yet approved as adequate**. Nested contrast and chunk-resolution probes must verify numerical/estimator stability on representative checkpoints before freezing. Batch/chunk sizes may be calibrated without changing likelihoods or particle sets. Calibration should include later trained policies, not only initial or smoke models.

Draft stopping rule: best raw validation mean selects the checkpoint, with earliest checkpoint winning exact ties. A distinct progress anchor resets patience only if paired mean improvement exceeds `max(0.01,1.96*SE(paired difference))`. Stop after6 nonmeaningful improvements only after minimum experience and8 validation checkpoints. This is an operational plateau criterion, not a theorem of global convergence. Checkpoint selection is separate from progress-anchor significance.

Location: validate every125,000 episodes; minimum500,000 for single policies; minimum1,000,000 for MoE policies so the established routing sharpening has completed. Router schedule remains gamma1 through30% of its **fixed1million-episode schedule clock**, cosine to0.001 at95%; extending training does not stretch that clock. Safety caps10million episodes /7200 optimization seconds per run. SIR draft: validate every4096; minimum16,384 single/32,768 MoE; schedule clock32,768; safety1million episodes/3600 optimization seconds. SIR settings require cost/behavior calibration before becoming confirmatory choices.

A cap reached before the criterion is **budget-limited**, never called converged. Validation cost is reported separately and included in total offline cost. Total campaign resources must be declared from measured timings before multi-day jobs. No universal equal1million-episode or equal-time stopping rule is imposed. Any resource-limited omissions remain explicit in tables.

## Native learning and readiness gates

| Method/component | Current audit | Main-study requirement |
|---|---|---|
| DAD, location | Torch sum256→16 encoder/linear emitter adapter; pinned architecture provenance | Preserve emitter, sPCE, published/justified optimization; new validation selection |
| DAD, SIR | Existing `SIRSumDAD` was not selected by the legacy generic factory | Journal path explicitly uses normalized-time/count sum encoder and linear emitter; document benchmark adaptation |
| DAD-MLP | Native location772-input control preserved | New SIR-specific193-input sum/stage64/64tanh control; never substitute location weights |
| Four MoE variants | Existing implementations/checkpoints preserved | Four IDs only; shared encoder/action/objective conventions; new SIR base serialization and prior/device checks |
| RL-sBOED | Standalone Torch SAC/two-Q adapter, not established exact published equivalence | Verify against pinned source; record architecture, replay ratio, discount and contrast adaptations; do not silently call it exact published replication |
| vsOED, location | Pinned GMM-I native actor–critic code, corrected discount0.9 | Add common validation callback and selected actor/auxiliary state export |
| vsOED, SIR | Legacy generic BA actor–critic does not establish native equivalence; upstream SIR uses a stored SDE data bank | Port only with explicit ODE model/history/action adaptation and native algorithm retained, or report unsupported; generic replacement is excluded |
| ScoreBED | Pinned official score objective, policy gradient, JAX/Torch bridge; SIR ODE adapter exists | Native score/policy training validation/checkpoint callback and budget accounting must pass before journal launch |

Current journal entrypoint fails early for uncertified vsOED/ScoreBED validation adapters. This is a readiness gap, not evidence that their EIG is poor. Existing legacy entrypoints stay available. No confirmatory ranking may omit these methods silently or substitute unrelated learners.

## Residual base and provenance

Each full training seed has its own base, trained and validation-selected on the same benchmark. Within that seed ResMoE/SwitchResMoE share the same base. Keep base first design and action transformation, freeze base initially, preserve gradients through its inputs. Count base episodes, validation, simulations, likelihood work and time in each model's end-to-end cost; disclose actual sharing. Location's772-input head is unchanged; the SIR base uses normalized SIR histories and193 input. Disabled residuals must exactly reproduce the complete native base; switching-forced-reroute must match its nonswitching counterpart. Weights-only historical checkpoints may initialize explicitly labelled development continuation with fresh optimizer state, but cannot be called exact training resumes or converged bases without evidence.

## Test integrity and independent replication

Development training seed201; historical101 evidence remains separate. Proposed five complete confirmation seeds70101–70105; their nonuse must be verified against saved configs before freezing. Use independent seed-specific bases, not five residual heads sharing the historical base. Seed1001 is development. Seed8831001 remains untouched until algorithms, method profiles, validation precision, stopping logic, checkpoint selection and primary paired comparisons are sealed. Confirmation is currently locked.

Location final reference:2000 paired trajectories, L1million. SIR starting reference2048/L16384 requires contrast-resolution verification. Same test draws across methods, hard expert routing, explicit likelihood and common benchmark-specific scoring. No final-test checkpoint selection or tuning afterward.

Report per-seed trajectory mean/SE and paired trajectory differences separately from across-training-seed variability. For principal comparisons aggregate complete paired training-seed means with sample SD and a small-sample paired t interval (five runs means five independent trainings, not trajectories times seeds). Preserve every seed, including failures and budget-limited runs; no best-seed ranking. Report test contrast-bank sensitivity separately.

## Costs, mechanisms and figures

Save actual primary episodes, truth/contrast particle trajectories and stage likelihood work, native optimizer updates, gradient exposure, solver work, timing, validation costs, pretraining/auxiliary costs, hardware/utilization, total/active parameters and measured online latency. Record all stopping/selection decisions and both selected and final training-state artifacts. Legacy checkpoints contain weights only; new policy optimizer snapshots do not automatically imply complete replay/critic/score-state resumability.

Plot validation EIG versus episodes/time/likelihood work, selected-policy EIG with across-seed uncertainty, representative bounded location paths, SIR observation times and infected trajectories. Existing specialist diagnostics provide stage utilization, expert gradient exposure, matched-history disagreement, residual magnitudes, observation counterfactuals and forced-routing EIG. Forced modes are interventions with possible history distribution shift. Existing hard switching always reroutes; report that failure rather than claiming temporal persistence. Any single-residual ablation remains an ablation, not a fifth named family method.

## Reproduction

Use root launchers only. Workability examples (not performance experiments):

```bash
bash run.sh --config configs/sir_ode_research.yaml --journal --method sum_single --smoke --seed 201 --eval-seeds 1001
bash run.sh --config configs/location_finding_eig.yaml --journal --variant moe --smoke --seed 201 --eval-seeds 1001
```

Journal residual smoke runs require the complete same-benchmark base checkpoint with its run configuration/training metadata. The full-study commands use `--journal` without `--smoke`, individual methods/variants, seed-specific bases, fresh dated outputs and frozen protocol. They must not be launched until measured development/resource/precision gates pass. `--journal-phase confirmation` currently rejects execution. The older default/pilot paths retain their historical final-checkpoint budget conventions and are not the new main journal protocol.

Primary source references: [DAD](https://github.com/ae-foster/dad), [RL-BOED](https://github.com/csiro-mlai/RL-BOED), [vsOED](https://github.com/wgshen/vsOED), [ScoreBED](https://github.com/angusphillips/Score_Matching_BED). Pinned local manifests record exact upstream revisions and hashes. New results must come from saved artifacts rather than manually filled tables.
