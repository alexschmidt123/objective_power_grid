# ICML specialization and routing assessment

Audit date: 2026-11-09. Authoritative inputs are the saved LabPC configurations, JSONL training logs, per-arm RESULT/STARTED receipts, actor checkpoints, full-runtime hashes, frozen-policy score vectors and evaluation receipts. This audit performed **zero training updates, zero neural inference calls and zero new simulator calls**, and used CPU only. Actor tensor dictionaries were read on CPU with restricted PyTorch loading; full runtimes were hashed without deserializing training objects. All 141 read input files, including six terminal actors and six full runtimes, retained their SHA256 values after the audit.

Source roots, relative to LabPC `~/Documents/objective_power_grid_temp/scratch_runs/`:

- C: `moe_collaborative_soft_queue_20261108/`.
- S: `moe_specialization_guided_eig_20261108/`.
- H: `moe_hec_sboed_20261108/`.

Audit evidence is isolated in `scratch_runs/moe_hec_runtime_efficiency_audit_20261109/{AUDIT_SUMMARY.json,RAW_RECEIPTS_AND_LOGS.json,INPUT_PRESERVATION.json}`. Reproduce with `python tools/audit_hec_runtime.py --output ~/Documents/objective_power_grid_temp/scratch_runs/NEW_AUDIT_DIRECTORY`; the collector requires a new directory and refuses to overwrite an existing audit. No existing experiment, checkpoint, evaluation rule or model was changed. The current machine-access workflow supersedes old access restrictions, without editing historical protocol files.

## Verdict

**Not yet sufficiently established for a competitive ICML submission whose central claim is four complementary specialists intelligently commanded by history-dependent information needs.** This is a promising one-seed EIG improvement with a mathematically nontrivial query architecture. Its strongest mechanism claims remain unsupported by the current ablations. That is an evidence assessment, not a prediction of an acceptance decision or a demand for a theorem.

The package improves the matched-update heterogeneous control by +0.07211 nats, 95%CI[0.01889,0.12532],p=.00794. It also improves the frozen SG checkpoint by +0.18556[0.12445,0.24666],Holm4p=3.06e−9. These are valuable finite-contrast estimates. They do not by themselves establish all-four specialization, observation/uncertainty-dependent routing benefit, specialization×commanding synergy, or algorithmic novelty.

## What the implemented algorithm actually learns

The deployed policy has a frozen strong SG provisional action a0, frozen SG variational GMM belief providers and a frozen SG critic, plus trainable new features and a residual decision head. Native actor–critic/posterior training continues with gamma 1 and no specialist auxiliary loss in B/C. The frozen providers are approximations inherited from SG, not exact posterior inference or exact conditional-information oracles.

The commander receives a learned history representation, a0 and approximate-belief means, log variances and correlations. Its outputs alter mixture-region attention, hypothetical design directions, hypothesis weights, Fisher query locations/directions and opportunity probes **before** feature fusion. The exact likelihood and analytic sensitivities really are used. Source and no-update checks establish a differentiable computation path from the native action-value objective to commands and all four experts. Final logs show nonzero candidate commander/coordinator gradients.

Four specialist responsibilities and their different architectures/inputs are **manually specified**. Their representations, command values and integration are learned. This is not discovery of four unconstrained semantic roles, and the paper should not describe it that way. All feature pathways are available, but availability is not demonstrated necessity. A shared history-to-fusion path and the strong a0 can sustain good performance even when an expert is removed.

## Expert specialization: evidence and limits

Phase-A preparation was shared by both arms. On its 512 held-out replay prefixes, all four normalized task MSE point estimates improved versus initialization and the training-mean constant predictor:

| Expert / role | Initial MSE | Prepared MSE | Constant MSE |
|---|---|---|---|
| 1: Posterior geometry | 7.83324 | 0.71636 | 1.02933 |
| 2: Likelihood discrimination | 1.99001 | 0.54411 | 1.00993 |
| 3: Local sensitivity/identifiability | 2.84473 | 0.36596 | 0.98397 |
| 4: Sequential opportunities | 0.97401 | 0.72190 | 0.83222 |

These are point-estimate preparation checks, not a full claim of statistically proven downstream design competence. Geometry labels describe the existing approximate belief rather than calibrated true posterior regions. Likelihood/Fisher tasks are meaningful mathematical calculations, but learning to encode them does not establish that the downstream policy exploits their distinctions. Expert 4 has a two-replica **one-step variational-information plus SG-critic bootstrap** target; it does not demonstrate a learned distinct multi-step strategy or exact expected suffix EIG.

On the final diagnostic bank, calibration-only ridge probes were fitted separately to each frozen representation/task, rival selection used calibration histories, and designated-expert advantages were tested on 256 separate held-out histories with Holm4 correction. Positive values mean smaller normalized task loss than the calibration-selected rival:

| Expert | Designated task advantage [95% CI] | Holm4 p | Removal EIG penalty [95% CI] | Removal Holm4 p |
|---|---|---|---|---|
| 1 | +0.45381 [+0.41876, +0.48887] | 1.72e-71 | +0.03135 [-0.05727, +0.11998] | 1 |
| 2 | -0.00843 [-0.04050, +0.02364] | 0.605 | -0.01418 [-0.09459, +0.06624] | 1 |
| 3 | +0.25806 [+0.23216, +0.28396] | 2.13e-52 | +0.09115 [-0.02034, +0.20265] | 0.435 |
| 4 | +0.01383 [-0.00313, +0.03079] | 0.219 | +0.01153 [-0.12866, +0.15171] | 1 |

**Established:** experts 1 and 3 have distinct recoverable strengths on their designated mathematical tasks relative to other HEC representations. This goes beyond merely different output vectors. **Not established:** distinct designated advantages for experts 2/4, individually credible EIG usefulness of every expert, four-way complementarity, or conditional information-state advantages that can guide a deployable selector. No expert-removal penalty survives correction; their wide intervals do not prove the experts are useless or equal. Removal also renormalizes the remaining gates and shifts the joint feature distribution, so even positive ablations would need interpretation.

Fixed specialist readouts after joint EIG training are another caution:

| Expert | Constant loss minus frozen-readout loss [95% CI] | Holm4 p | Interpretation |
|---|---|---|---|
| 1 | -1.06353 [-1.12871, -0.99836] | 4.83e-91 | Readout worse than constant |
| 2 | +0.08325 [+0.05377, +0.11273] | 6.76e-08 | Readout beats constant |
| 3 | -0.25604 [-0.31293, -0.19915] | 2.73e-16 | Readout worse than constant |
| 4 | -1.96690 [-2.17591, -1.75788] | 1.14e-48 | Readout worse than constant |

Three fixed readouts no longer beat a constant predictor on the final histories. This is **not proof of expert collapse**: feature coordinates can rotate, features/fusion can coadapt, and final histories/commands differ from preparation data. Refitted probes are therefore essential. On the same final histories/commands, prepared-versus-live refitted probe differences show a modest disadvantage for live expert 1 (−0.01061 normalized loss advantage,Holm4p=1.72e−12); expert 3’s apparent disadvantage does not survive Holm4(p=.0953), and experts 2/4 differences are inconclusive. Thus there is no demonstrated across-expert competency improvement from final EIG adaptation, even though final EIG improves.

Task targets are heterogeneous by construction, not four independent discoveries. Correlated statistics can still make their decision contributions redundant. The saved tests do not measure coalition interaction/synergy, the conditional information value of one expert’s query on particular histories, or whether another expert/history-only head can replace it at matched capacity. Those are the missing links between representation specialization and BOED utility.

## Intelligent commanding and coordination

Observed within-stage command standard deviations range approximately 0.032–0.218; gate SDs0.023–0.047. The commander therefore varies with histories beyond stage in a descriptive sense. Architecture gives it approximate latent-uncertainty summaries, but nonzero variation and nonzero weights/gradients do not show that uncertainty causally drives useful command choices. No controlled same-stage uncertainty-summary ablation or conditional uncertainty-versus-command-utility analysis was performed.

Each following effect is **learned full HEC minus the indicated frozen-policy intervention**, with full trajectory resimulation, common latent/noise draws and 256 held-out histories. Commands and routing families have separate Holm3 corrections. Displayed95% intervals are unadjusted:

| Intervention | HEC EIG advantage [95% CI] | Holm3 p | Supported inference |
|---|---|---|---|
| Fixed zero commands | +0.30752 [+0.15112, +0.46392] | 0.0004114 | Positive; learned query settings matter in this coadapted policy |
| Stage-only commands | -0.01649 [-0.09520, +0.06222] | 0.6802 | No demonstrated history benefit beyond stage |
| Stage-matched shuffled commands | +0.07841 [-0.01428, +0.17111] | 0.1939 | Positive point estimate; inconclusive |
| Conventional linear soft-routing projection | +0.01948 [-0.03097, +0.06993] | 0.7547 | No demonstrated improvement over this projection |
| Uniform coordination | -0.07345 [-0.18891, +0.04201] | 0.6343 | No demonstrated benefit; point estimate favors uniform |
| Stage-only coordination | +0.02553 [-0.03132, +0.08239] | 0.7547 | No demonstrated history-aware coordination benefit |

The fixed-command intervention is **zero commands**, not the best learned constant command or best stage schedule. Even though zero occurred in preparation, switching a trained coadapted system to zero changes its feature inputs. Its positive effect does not establish that history-conditioned selection is required. Stage means preserve learned stage settings and are a stronger current challenge to the dynamic-history claim. Non-significance does not prove that stage-only and learned policies are equal; the current diagnostic uncertainty is substantial.

The “ordinary routing” intervention is a calibration-only ridge projection of learned log gate weights onto the candidate’s history encoding. It was not optimized for EIG. The trained matched heterogeneous control is the real ordinary-router comparison, but it also changes the commanding mechanism and effective active capacity. The positive candidate/control primary result supports the **whole changed package**, not isolated superiority of the commander or coordinator. Uniform tests replace coordination, not all command computations; there is no separately tested uniform-command mechanism to report.

Therefore the correct answer to “does it outperform fixed, uniform, stage-only and conventional routing?” is: it beats its tested zero-command intervention and its complete matched-update control; it has **not** demonstrated superiority to stage-only commands, shuffled commands, uniform/stage-only coordination or the linear ordinary-routing projection.

## Joint contribution to full-horizon EIG

All information comparisons use the same new2000-case bank and frozen finite-contrast estimator. On this bank HEC12.95961, heterogeneous control12.88750 and frozen SG12.77406. Candidate−control is positive, while the control already captures+0.11345 of the+0.18556 gap over frozen SG. Both new arms receive2,000 native updates and new residual-policy capacity; frozen SG receives none. Thus ordinary continuation/residual adaptation, extra data and additional model-derived features remain plausible explanations for much of the frozen-baseline gap.

Current evidence cannot isolate specialization×commanding interaction. Shared preparation is present in both arms, and there is no matched no-preparation or generic-feature control. Dynamic commands and feature-aware coordination change simultaneously; their presence/absence does not form a2×2 factorial. Candidate active capacity is34,964 parameters larger than control and training wall time13% longer. No matched ordinary SG continuation from this same SG checkpoint or history-only residual baseline is available. The earlier SG ordinary control starts from another parent and is not a replacement.

Useful joint contribution would require more than all three components being differentiable. For example, define fixed-commander/ordinary-coordinator value J00, dynamic-only J10, feature-coordinator-only J01 and full J11 under the same specialist preparation and budget. The interaction J11−J10−J01+J00 would test an explicit synergy claim; main effects may be useful even without positive synergy. A claim that specialist semantics cause that benefit additionally needs an information/capacity-matched representation/preparation control. None of these missing contrasts can be recovered by subtracting current frozen-policy ablations.

The matched HEC result is **sample/update matched, not exactly wall-time or effective-capacity matched**. Single-seed Monte Carlo CIs do not estimate training-seed variance. Higher frozen-vsOED EIG on this bank is observed(+0.51789,CI[0.42619,0.60959]), but extra inherited/continuation computation prevents a compute-matched superiority claim.

## Algorithmic novelty: substantive differences and inherited mechanisms

The prior-art comparison is targeted rather than exhaustive; it cannot establish novelty by absence of an exact name match.

| Prior art | Established contribution | HEC difference and remaining gap |
|---|---|---|
| [MMoE](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/) / [PLE](https://doi.org/10.1145/3383313.3412236) | Learned expert combinations and structured task/shared representations. | HEC uses four model-derived statistical query operators for a single sequential-information objective; heterogeneous tasks and soft fusion alone are inherited ideas. |
| [Expert Choice](https://arxiv.org/abs/2202.09368) / [DeepSeekMoE](https://arxiv.org/abs/2401.06066) | Expert-token allocation, fine-grained segmentation/shared-expert mechanisms. | HEC changes what dense experts compute rather than capacity-limited token selection. It needs decision-competence evidence, not a claim that different roles prove specialization. |
| [Guo et al.](https://arxiv.org/abs/2505.22323) | Orthogonality/routing-variance auxiliary objectives. | HEC uses pretraining then pure native information-value gradients, without final diversity/load-balancing rewards. Avoiding those auxiliaries is a distinction, not novelty proof. |
| [FiLM](https://arxiv.org/abs/1709.07871) / [HyperNetworks](https://arxiv.org/abs/1609.09106) | Conditioned feature modulation / generated network weights. | HEC commands query locations, belief attention and directions before feature encoding. It is not literally a hypernetwork, but learned conditional computation is established. |
| [Residual Policy Learning](https://arxiv.org/abs/1812.06298) | Learn corrections around an existing controller. | HEC corrects a frozen SG design using command-conditioned statistical features. Strong warm-start residual adaptation is prior art, not a new contribution by itself. |
| [DAD](https://proceedings.mlr.press/v139/foster21a.html) / [vsOED](https://arxiv.org/abs/2306.10430) | Amortized sequential design; variational actor–critic information optimization. | HEC inherits the sequential design framework and native gradient estimator. Its candidate novelty must be the query-learning/coordination mechanism, not EIG or AC. |
| [ScoreBED](https://arxiv.org/abs/2607.08335) | Separate policy-independent score learning from policy optimization. | HEC has no corresponding score-matching contribution and currently has higher per-update/inference costs than SG. |
| Project selection-credit/CIAS/Collaborative/SG | Prior competitive credit failures; successful feature collaboration and anchored adaptation. | HEC removes a responsibility teacher and selects model-derived queries, but current evidence does not show new complementary full-horizon competencies or observation-dependent commanding benefit. |

A defensible **candidate** methodological contribution is: native full-horizon information-value gradients learn the content of heterogeneous belief/likelihood/sensitivity/opportunity queries, then integrate their representations around a strong sequential design policy. This is more specific than a renamed softmax router. It is presently a new composition/application hypothesis, not an established new learning estimator, theory, general BOED specialization principle or ICML-level result. Exact likelihood access, Fisher formulas, quadrature, four experts, task labels, frozen providers and bounded residuals are individually established ingredients.

## Claim ledger

| Claim | Current status |
|---|---|
| Correct bounded sequential policy with model-derived heterogeneous inputs | Supported structurally and by saved no-update correctness checks. |
| One-seed improvement beyond fixed-command heterogeneous control at equal new episodes/native updates | Supported for the finite-contrast estimate; not equal wall time/active capacity. |
| Experts1 and 3 encode their designated statistics better than other experts | Supported by held-out corrected probe tests. |
| All four experts acquire distinct useful BOED competencies | Unproven: two designated-role tests inconclusive; no corrected positive individual-removal EIG effects. |
| Final joint EIG training preserves/improves every prepared competency | Unproven; fixed readouts degrade and no across-expert probe improvement is established. |
| Commands vary within stage and are mathematically more than routing weights | Supported; descriptive variation and changed model queries are real. |
| Latent uncertainty causes beneficial history-specific command allocation | Unproven: possible by inputs, not isolated empirically. |
| Dynamic commanding outperforms stage-only commands and ordinary/uniform coordination | Unproven in current diagnostic families. |
| Specialization and dynamic coordination jointly cause the EIG gain | Unproven: factorial, capacity and history-only residual controls missing. |
| Higher same-bank score than frozen SG/Option/vsOED | Supported as a frozen-checkpoint comparison, with unequal inherited/training compute. |
| Faster or more computationally efficient learning from scratch | Unsupported: warm starts, reduced update budgets,2.5× higher per-update cost and 5.7× higher inference latency. |
| Generalization and a competitive ICML mechanism claim | Not established: one task/seed, incomplete mechanism evidence and unverified algorithmic novelty. |

## Minimum additional evidence, ordered to avoid unnecessary training

These are **recommendations only**. This audit launches none of them and does not change historical evaluation rules or checkpoints.

1. **Strengthen frozen diagnostics before spending training compute.** Predeclare a fresh diagnostic bank and multiplicity families, separate calibration/validation, and resimulate all changed-policy observations. Test learned vsstage-only commands/coordinators as the key history-benefit contrast; compare optimized calibration-only constant commands as well as zero. Replace/mask each expert using stage-matched donor representations, evaluate leave-one-out/selected coalition interactions, and stratify by decision-time information summaries rather than true latent parameters. Use common randomness where valid. Quantify finite-contrast sensitivity at predeclared contrast counts instead of selecting favorable expert rankings.
2. **Establish semantic content and uncertainty sensitivity on common histories.** Compare prepared/live refitted probes, calibration-only transferred readouts and generic-feature controls using the same histories/commands/target generator. For expert 1 validate belief geometry against independently reliable model-specific references on a small prefix panel; for expert 2/3 distinguish nonlocal discrimination from local sensitivity; for expert 4 validate its bootstrap opportunity rankings against separately resimulated continuation returns on a small tractable panel. A critic/pretext score is not exact conditional EIG. Perturb or stage-match the commander’s uncertainty-summary input separately from its history and action inputs, with an explicit covariate-shift caveat, to test whether this information actually changes useful query selection.
3. **Only if frozen evidence is promising, request a small controlled training study.** From the same SG parent include ordinary SG continuation and an information/capacity-matched history-only residual control. Use the command×coordination2×2 with shared preparation to separate the two mechanisms; reuse the existing two cells only where the frozen protocol and parent match. Include an appropriate no-specialist-preparation or generic representation control to isolate semantic preparation rather than extra information/capacity. Match episodes/updates, active parameters and report all operator/frozen-provider cost; add an equal-wall-time control or fixed cost/return curve for an efficiency claim. Do not call differently effective modules “matched capacity” merely because both instantiate unused tensors.
4. **Replicate the predeclared central comparison across independent training seeds.** At least three independent runs per arm are an initial variability check; a power-planned five-seed comparison is preferable if variability requires it. Report run-level uncertainty and paired trajectory uncertainty separately, terminal-only selection, all negatives and inherited compute. Extra evaluation trajectories alone cannot replace independent training runs.
5. **For a broad ICML method, test portability after the mechanism survives controls.** One additional scientifically distinct problem is a minimum external check of the general query-learning claim; deterministic SIR ODE and dynamic swing-equation campaigns are planned but unexecuted. Reuse one algorithm with only justified likelihood/sensitivity/feasibility operators changed. A strong single-benchmark paper would require correspondingly deeper identification and mechanism evidence; multiple benchmarks alone cannot repair a missing causal mechanism.

Diagnostic sample sizes need a power plan. For illustration only, the current stage-command paired SE0.03997 with 256 cases implies a case-level SD about 0.6395 nats. A conventional two-sided80%-power test for a0.03-nat effect would need roughly3,600 independent cases before Holm/multiple-seed considerations, assuming that SD transfers. This is planning arithmetic, not a recommendation to optimize another reused bank or a promise that a larger test will turn positive. Avoid reading non-significance as equality.

A publishable claim should survive both plausible outcomes: if a stage schedule matches the commander, simplify the claim to learned stage-dependent query design; if only two specialists are necessary, do not maintain a four-essential-expert narrative. Preserve negative outcomes rather than enlarging budgets until they disappear.

## Research decision

Keep HEC as a reproducible **partial positive EIG pilot**. Do not yet claim four genuinely complementary experts, uncertainty-sensitive intelligent routing, jointly identified specialization/commanding synergy, compute-matched vsOED superiority or ICML-ready novelty. The most economical next authorized step would be sharper frozen-policy identification of what the useful query changes are; training a more complicated architecture is not justified by this audit alone.
