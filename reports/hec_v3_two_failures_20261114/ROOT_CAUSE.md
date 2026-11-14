# Why V3 fails in the two requested ways

**Finding 1: the V2-to-V3 expert decision conversion lost a successful policy function. Finding 2: the latest router fix used an unreliable value fit and a regularization scale that kept routing almost unchanged.** These are different failures; repairing the second alone has not repaired the first.

All performance comparisons use the exact original V2 bank and frozen scorer: 2,000 episodes, 1,000,000 shared contrasts, original routing uniforms and physics. The preceding canonical run reproduced all six historical reference vectors exactly. This diagnosis changed no model weights and performed no optimizer steps.

## 1. V3 is below V2: verified decision-interface loss

| Router function | V2 joint feature decoder | V3 standalone decoder + categorical selection |
|---|---:|---:|
| Inherited V2 router | 12.93775104 | 12.83470188 |
| Current V3 router | 12.71209458 | 12.80935983 |
| Uniform weights | 12.95392194 | 12.80527233 |

Each cell uses the same frozen final V3 query pathways and shared context. The inherited and current router functions are held fixed across the two decoder columns. The new bridges are diagnostic assemblies, not independently executable expert solutions.

With the inherited routing function fixed, the joint decoder gains +0.10304916 nats over V3 selected decisions (paired 95% CI [+0.05204128, +0.15405704], Holm p=0.000153843).
With uniform weights fixed, joint decoding gains +0.14864961 nats (95% CI [+0.09101964, +0.20627958], Holm p=3.68846e-06).

This localizes a real weakness in this conversion: V2 combines query contributions before nonlinear decoding; V3 decodes a single pathway and selects its action. The standalone heads were initialized to V2 one-hot pathways and then fitted to teacher actions; this does not preserve the complete V2 policy. Prior matched tests already found that low teacher-action MSE did not preserve full-episode EIG. It is not proof that decision-level MoE is inherently inferior, that all posterior information is irretrievably lost, or that query cores have collapsed.

The low current-router/joint-head result is important: the learned V3 gates cannot simply be transplanted into V2 fusion. Router preferences and decoder semantics interact. Restoring the shared head while retaining the current gates is not a demonstrated fix.

Code: `development/moe_sboed/hec_v3_v2_pathways_20261113/pathway_policy.py:41` and `development/moe_sboed/hec_v3_v2_pathways_20261113/pathway_policy.py:136`.

## 2. Learned V3 has not beaten uniform: the attempted fix barely changed allocation

The current matched difference is new-router minus uniform = +0.00408750 nats, 95% CI [-0.04546245, +0.05363745]. There is no demonstrated advantage.

### 2a. The fitted routing targets did not reliably learn advantages

Each round fitted three fresh 91,588-parameter value networks on only 64 distinct training histories, with 16 validation histories and 16 audit histories. There were only 192 fitting histories across all three rounds, and fitting reset each round rather than accumulating evidence. Millions of suffix observations therefore did not mean millions of distinct histories for value regression.

Every one of the nine counterfactual value fits selected the checkpoint after just one AdamW step. The frozen test reconstructed the original random initializations and their first updates analytically, matching saved weights within 7.46e-9. The selected ensemble did not reliably improve on its untrained initialization or a zero-advantage baseline.

| Round | Untrained audit MSE | Selected audit MSE | Zero-advantage audit MSE | Predicted advantage RMS | MC training-label RMS |
|---|---:|---:|---:|---:|---:|
| 1 | 0.01196489 | 0.01206285 | 0.01171791 | 0.007419 | 0.121831 |
| 2 | 0.00471846 | 0.00485916 | 0.00467180 | 0.009720 | 0.120123 |
| 3 | 0.01120986 | 0.01116683 | 0.01130751 | 0.013257 | 0.137497 |

The MC labels are noisy estimates, not true advantages. Their larger RMS alone is not proof of bias in the fitted values. The independent prediction errors and one-step selections establish that this fit did not supply a validated improvement in routing credit. The code still passed its predictions into router optimization without a gate requiring improvement over the zero baseline. This is a confirmed workflow/design weakness, not proof that useful conditional expert choices do not exist.

Code: `development/moe_sboed/hec_v3_counterfactual_router_20261114/experiment.py:26`, `development/moe_sboed/hec_v3_counterfactual_router_20261114/experiment.py:32`, `development/moe_sboed/hec_v3_counterfactual_router_20261114/experiment.py:36`, `development/moe_sboed/hec_v3_counterfactual_router_20261114/experiment.py:28`, `development/moe_sboed/hec_v3_counterfactual_router_20261114/experiment.py:206`.

### 2b. The unit KL penalty anchored the router to its old preferences

The implemented router loss was `-mean(sum(p*A_hat)) + KL(p_old || p)`, plus a separate KL<=0.01 step limit. The added penalty did not alter the simulator reward, but it did change the router optimization objective. For fixed history-wise values, its unconstrained optimum is `p_new(k)=p_old(k)/(tau-A_hat(k))`, with `tau>max(A_hat)` chosen so probabilities sum to one. Small fitted advantages therefore imply a small change even after further optimization.

| Round | Return/KL gradient cosine | Combined / return gradient norm | Mean probability-mass movement (TV) | Actual KL |
|---|---:|---:|---:|---:|
| 1 | -0.99743420 | 0.108695 | 0.1003% | 0.000008657 |
| 2 | -0.99799860 | 0.072181 | 0.1086% | 0.000009321 |
| 3 | -0.99981475 | 0.019267 | 0.1903% | 0.000034329 |

At the saved endpoints, the penalty reduced the return-gradient norm by about 89–98% through opposition. Mean probability-mass movement was only about 0.10–0.19% per round. Actual KL was far below the separate 0.01 cap. This is expected near a stationary point of the regularized objective; it is not a broken autograd path. It explains why increasing iteration count alone would not necessarily move this fixed-target update much farther. The penalty coefficient was not calibrated to the measured advantage scale. That was a flaw in the attempted routing fix.

Code: `development/moe_sboed/hec_v3_counterfactual_router_20261114/experiment.py:56`.

## Classification and limits

| Conclusion | Classification |
|---|---|
| Current V2-to-standalone conversion fails to preserve the successful decision function under inherited/uniform routing | Supported architectural/design weakness in this implementation |
| New value estimator supplies no reliable demonstrated prediction improvement | Supported empirical weakness; confirmed missing acceptance gate |
| Unit KL term mechanically opposes weak return targets and keeps updates tiny | Verified training-design limitation of the latest router update |
| Missing optimizer participation, disconnected router gradients, or zero expert execution | Not identified in the tested paths |
| No conditional expert advantage exists anywhere | Unresolved; not established by these tests |
| Original native critic misranking alone explains the original 250-update V3 router | Unresolved; latest-fix diagnosis does not establish the original critic cause |

## Targeted next correction, not another architecture replacement

For routing: test paired MC advantages directly rather than inserting this unvalidated regression model; remove the uncalibrated additive KL penalty while retaining a measured KL constraint. Validate return targets and an independent suffix improvement before scaling training. This is a proposed correction, not a claimed EIG gain.

For the V2 gap: require the standalone expert decision conversion to preserve full-episode V2 performance before another routing campaign. Pointwise teacher-action MSE and one-hot pathway parity are insufficient acceptance criteria. The query cores alone do not need to be discarded based on current evidence. Restoring V2 fusion is a diagnostic control, not a solution to the independently executable-expert requirement.

No fix or new model training was performed in this diagnostic turn.

## Canonical frozen reference results

| Policy | Prior-sPCE (nats) |
|---|---:|
| V2 | 12.92935261 |
| V2_uniform | 12.93794410 |
| vsoed | 12.44271971 |
| original_V3 | 12.80310089 |
| counterfactual | 12.80935983 |
| critic_control | 12.80030489 |
| uniform | 12.80527233 |
| expert1 | 12.73824955 |
| expert2 | 12.85301397 |
| expert3 | 12.86149725 |
| expert4 | 12.81226900 |
| dad | 10.31407383 |
| rl_sboed | 11.19164733 |
| scorebed | 10.99883133 |

All six saved historical reference score vectors reproduced exactly in the canonical evaluation. Training budgets are not matched, the 1M estimator is ceiling limited, and this is one training lineage. Value diagnostics use 16 audit histories per round; their uncertainty and approximate posterior labels prevent broad conditional-specialization conclusions.

Raw diagnostics: `/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/hec_v3_two_failures_20261114` and `/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/hec_v3_two_failures_20261114_factorial`.
