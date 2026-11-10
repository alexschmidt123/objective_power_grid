"""Write actual preflight evidence and explicitly incomplete v3 deliverables."""
from pathlib import Path
import json,hashlib
B=Path.home()/'Documents/objective_power_grid_temp';D=B/'development/hec_git_sync/development/moe_sboed/v3_competitive_20261110';O=B/'scratch_runs/moe_sboed_v3_competitive_20261110';R=B/'scratch_runs';E=O/'evidence'
def read(p):return json.loads(p.read_text())
def block(x):return '```json\n'+json.dumps(x,indent=2)+'\n```\n'
def write(n,s):
 for root in [D,O]:(root/n).write_text(s.strip()+'\n')
f=read(E/'FROZEN_HISTORY_AUDIT.json');g=read(E/'GRADIENT_FIXTURES.json');old=read(R/'moe_hec_v2_icml_20261109/evaluation/EIG_RESULTS.json')
status='INCOMPLETE V3 PILOT — STOPPED AT THE PRE-TRAINING METHODOLOGICAL GATE'
write('RUN_INDEX.md',f'''# MoE-sBOED v3 preflight

{status}

Read the requested task, prior code/results, and primary MoE/BOED sources. Implemented exact gradient/density fixtures and a frozen history-condition audit. No deployable v3 model, specialist preparation, neural training, new EIG bank, or new policy evaluation was run. This is a scientific no-go for the particular construction assessed, not a theorem ruling out v3 or proof that the proposed architecture would fail.

The task explicitly requires a substantive BOED-oriented responsibility mechanism beyond ordinary mixture actor-critic, and says to report the limitation before expensive training if one cannot be justified. I could justify a correct standard stochastic mixture and distinct preparation distributions, but not the additional joint-learning contribution. Multiplying native actor gradients by specialist scores without a derivation would not resolve this.

Start with V3_ALGORITHM_AND_NOVELTY.md, POLICY_GRADIENT_DERIVATION.md, IMPLEMENTATION_TESTS.md and NEXT_RESEARCH_DECISION.md. CONFIG_LOCKED.json explicitly records that the training protocol is NOT locked. Proposed 2000 updates per arm remain unexecuted. HEC-v2 remains the frozen reference.

Evidence directory contains actual frozen-history arrays, gradient receipts, and before/after hashes. Raw replay-derived arrays remain outside Git. Source is developed on LabPC; GitHub and Ubuntu synchronization have a separate receipt. No Ubuntu computation or unrelated job interruption.
''')
write('PRIOR_MOE_COMPARISON.md','''# Prior mechanisms and verified project evidence

## Primary literature

[Jacobs et al. 1991](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf): competing experts use a probabilistic mixture objective. Posterior responsibilities allocate learning according to how well a component explains an observed target; gate and component credit are coupled. Useful local competence is distinct from equal load. In BOED an action density explains a sampled action, not its information value. Reward-weighted policy credit is still needed; responsibility alone does not estimate competence.

[MMoE](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/): shared experts with task-specific learned gates model task relationships and differing feature requirements. It does not supply a conditional information-value signal for four complete design policies. Its task-specific prediction objectives are appropriate as declared preparation, not additional final rewards here.

[PLE](https://doi.org/10.1145/3383313.3412236) ([paper](https://datawhalechina.github.io/torch-rechub/file/PDF/2020%20%28Tencent%29%20%28Recsys%29%20%5BPLE%5D%20Progressive%20Layered%20.pdf)): explicitly separates shared and task-specific experts, then progressively routes representations across extraction layers to address negative transfer and competing task performance. Feature layers do not correspond to chronological experiments. Reusing this decomposition would be an adaptation, not a new stochastic BOED responsibility estimator.

[Switch Transformers](https://www.jmlr.org/papers/v23/21-0998.html): top-one token routing, capacity management, balancing and numerical techniques enable efficient sparse training. These address exposure and infrastructure. Capacity or uniform load is not evidence of conditional BOED competence; token dropping is inappropriate for required physical experiments.

[Expert Choice](https://arxiv.org/html/2202.09368v2): experts choose their high-affinity tokens within fixed capacity; tokens can receive different numbers of experts. This changes allocation and improves utilization, not conditional experimental information-value estimation. One online history cannot depend on a future batch of deployment histories.

[DeepSeekMoE](https://arxiv.org/html/2401.06066v1): fine-grained expert segmentation and shared-expert isolation aim to reduce redundancy, with load considerations and specialization ablations. Shared feature experts are not the same as an alternatively selected generalist action policy. A four-policy generalist/specialist design borrows an established principle but needs its own EIG competence evidence.

[DAD](https://proceedings.mlr.press/v139/foster21a.html): amortized adaptive policies and contrastive information bounds. Its training estimator is not adopted here. [vsOED](https://arxiv.org/html/2306.10430v1): variational posterior rewards, actor-critic policy optimization, and terminal/incremental information formulations are the inherited training foundation. Native approximate critics and replay occupancy do not become exact MI gradients merely by adding a mixture.

[RL-sBOED](https://proceedings.mlr.press/v162/blau22a.html): the sequential design problem becomes an MDP solved with reinforcement learning, including models lacking differentiable simulators. Routing via remaining return is therefore not a new BOED principle. [ScoreBED](https://arxiv.org/html/2607.08335v1): learns policy-independent score approximations and then optimizes policy gradients with them. This is a distinct source of approximation; copying its objective would violate the current task. None of these methods establishes novelty for the assessed v3 construction.

## Actual project source and results

Sources inspected under scratch_runs, with original bytes preserved:

|Method|Actual mechanism|Limit relevant to v3|
|---|---|---|
|Direct-head|Four logit heads mixed before feasible sigmoid action transform; shared encoder/native Q|Not a mixture of complete action densities; similar shared-target gradients possible|
|Top-1 competitive pilot 20261107|Categorical selection each observation, selected-expert path gradient, exact four-way gate Q expectation, matching categorical Bellman expectation|Already implemented the standard competitive native actor-critic core; candidate-control -0.024792, CI [-0.080816,+0.031232]|
|Option-MoE|Three-decision options, option-conditioned critic, four action experts plus frozen fallback|Temporal commitment and fallback are established options principles; not identical to per-step v3 routing|
|CIAS / information-specialized option|Joint-generated prefixes, complete resimulated remaining native returns, cross-fitted conditional-mean teacher, uncertainty-shrunk gate CE and counterfactual replay|Teacher/model drift, limited labels, and auxiliary gate updates; no useful specialization established; unsuccessful classification preserved|
|Collaborative Soft|Four feature slots and separately supervised capabilities, soft feature coordination, one design head|Feature tasks need not become useful design competencies|
|SG-MoE|Frozen competency anchors plus EIG-adaptive feature deltas and learned retention|Representation retention does not establish four alternative policy competencies|
|HEC-v1|Command-conditioned model queries, geometry/likelihood/Fisher/opportunity features, collaborative action head|All-four usefulness and benefit over stage-only commands unestablished|
|HEC-v2|Hypothesis-pair discrimination and predictive opportunity responses, anchors and joint native EIG|Task strengths E2/E3, only E3 corrected donor contribution; dynamic-stage difference +0.005817, CI [-0.045043,+0.056678]|

The earlier Top-1 and CIAS formulas used discounted native returns (documented gamma .9), whereas HEC-v2 uses gamma 1. They also used different parent lineages and policies. Their negative results do NOT prove a stochastic v3 with gamma 1 would fail. They do establish that categorical Q competition and remaining-return responsibility teachers have already been tried. No implementation failure has been established as the cause of their results; critic, teacher, occupancy and exposure explanations remain hypotheses.
''')
write('V3_ALGORITHM_AND_NOVELTY.md','''# One assessed v3 construction and its unresolved contribution

STATUS: mathematical construction assessed; not selected for expensive training; no deployable network implemented.

One proper construction is four complete bounded action distributions sharing a frozen approximate-belief/likelihood provider and a history encoder, with independent action-head parameters. E0 is a trainable generalist alternative, not a mandatory preceding controller. E1-E3 receive coverage, likelihood-discrimination and accumulated-Fisher features. Every expert is available at every step, and one expert/action is sampled before each new observation. No physical coordinate averaging and no permanent stage roles.

Preparation could optimize each expert's expected remaining native information return on d_k(h) proportional to s_k(h)d(h), with s_0=1. This is a distinct data curriculum; it is NOT enough to call the eventual design competence specialized. A frozen Q preparation objective is only a critic surrogate and needs resimulated outcome validation. No such preparation was run.

For the final policy, a possible score-conditioned gate is g_k proportional to exp(l_k(h))s_k(h), with all scores positive. The same policy must generate data, define actor gradients, and appear in Bellman expectations. This preserves a valid mixture objective, but is an established input-dependent gating prior. It does not supply a new competence-learning signal beyond native Q/return. If used unchanged in the stage-only arm, history-dependent s_k would also leak history into the supposedly stage-only router; those factors must be removed/projected there. This is not a fair route to claiming a dynamic responsibility discovery.

Alternatively, externally multiply selected-head updates by s_k(h) while leaving the executed mixture unchanged. That is not, in general, the unchanged mixture policy gradient. It can be investigated as a biased optimization/preconditioning rule only with an explicit derivation and disclosure; it cannot be described as automatically unbiased pure-EIG policy credit. Exact history importance correction restores the unweighted objective, leaving a sampling/variance mechanism rather than a new objective-level specialization signal.

Generalist background updates on all histories must similarly be accounted for. At preparation time they are a disclosed curriculum. During final mixture training, adding unweighted counterfactual generalist updates is an extra surrogate term unless they are a valid estimator of its gate-weighted contribution or justified optimizer. Calling it shared knowledge does not justify the missing factor.

A BOED-specific advance might be an information-state-conditioned, occupancy-correct credit estimator with demonstrably useful variance or conditional-competence properties under the native return. I did not derive and validate such an additional mechanism in this audit. No claim that it cannot exist, no theorem prerequisite, and no requirement to prove publication novelty before all future experiments. The present stop is specifically because the task disallows launching ordinary mixture actor-critic as the main new mechanism.

Differences from Top-1 that ARE justified: proper non-degenerate continuous action densities, gamma-1 full-horizon native reward, legitimate belief/likelihood/Fisher features, and separate condition-weighted preparation. These are concrete changes and a reasonable application hypothesis; their composition is not yet the requested substantive new joint responsibility algorithm. Reusing CIAS's teacher would also repeat its remaining-return/auxiliary-gate mechanism; relabeling it would not fix its limitations. HEC-v2 instead coordinates collaborative features, not alternative action policies.
''')
rows=['|Stage|Coverage group|Discrimination group|Weak-identifiability group|','|---|---:|---:|---:|']
for t,v in f['summary'].items():rows.append('|'+t+'|'+'|'.join(f'{100*x:.1f}%' for x in v['validation_upper_group_fraction'])+'|')
write('EXPERT_SPECIALIZATION_DEFINITION.md','''# Candidate non-chronological history conditions

FROZEN PREFLIGHT COMPLETE; design competence and full 30-stage support remain INCOMPLETE.

Let q_h be the inherited SG amortized GMM used by frozen HEC-v2, approximated by its existing 64 weighted quadrature nodes. This is not a certified conditional posterior. Let C_h be its covariance and a_ref(h) the frozen HEC-v2 action.

Coverage score: c1(h)=tr(C_h). Broad coordinate covariance can conflate exchangeable-source label ambiguity with spatial uncertainty. It is a candidate diagnostic, not a validated explanation of where a coverage policy is superior.

Discrimination score: c2(h)=E_{theta,theta'~q_h} KL[N(m(theta,a_ref),.25)||N(m(theta',a_ref),.25)]=4 Var_{q_h}m(theta,a_ref). This uses the exact Gaussian likelihood and approximate-belief quadrature. A high score means hypotheses are distinguishable at the reference action, not that the reference action is poor or a separate expert improves remaining EIG. It is not a direct multimodality test.

Weak-identifiability score: c3(h)=-log(1+lambda_min(D_h F_hist D_h)), where D_h=diag(sqrt(diag C_h)) and F_hist=4 sum_{i<t} E_{q_h}[J(theta,xi_i)J(theta,xi_i)^T]. This is a current approximate-posterior-averaged accumulated Fisher diagnostic with diagonal scaling; it is not the posterior precision or exact observed information. J is independently autograd-checked. Weak directions can reflect label symmetries, numerical scale, or genuine ambiguity; no policy benefit follows automatically.

Predeclared preflight scores use calibration upper-third thresholds at each sampled stage, then s_k=.1+1[c_k>threshold_t], s0=1, normalized for d_k proportional to s_k d. Groups may overlap. The positive floor keeps every expert eligible; this is a diagnostic preparation weighting, not an equal-utilization objective. Strict inequality gives no arbitrary split when the information state is identical at t=0.

128 calibration and 128 held-out episodes per stage; all 1280 replay episode indices are distinct. No theta field or future observations are used. Prefix mutation checks confirm future saved entries have no influence. Thresholds were not selected from expert wins or EIG scores. The parent replay distribution is mixed historical training occupancy, not fresh v3 occupancy. No training distribution or threshold is finally locked.

Held-out upper-group prevalence:

'''+ '\n'.join(rows)+'''

At t=0 the information histories are identical; all three scores tie, and no condition-specific preference can be inferred. All experts could still be sampled there. Later groups vary within stage, with pairwise Jaccard overlaps and weighted effective sample sizes in evidence/FROZEN_HISTORY_AUDIT.json. This validates distinct measurable conditions only. It does not establish useful design competencies, reliable posterior calibration, or conditional comparative advantages. Additional stages must be audited before any training lock.
''')
write('ROUTER_RESPONSIBILITY_RULE.md','''# Router and responsibility rule: valid foundation, missing additional mechanism

INCOMPLETE: no scientifically selected v3 training rule.

For bounded pi_k(a|h), g_k(h)>0 and sum g=1, sample K~g, a~pi_K, then simulate a new observation and reroute. Posterior component responsibility is r_k=g_k pi_k / sum_j g_j pi_j. It measures how well expert k explains the sampled action under the mixture. It is not P(expert k is information-optimal|h).

Joint latent-label score credit is A(h,a)[grad log g_K+grad log pi_K]. Marginal credit uses A sum_k r_k[grad log g_k+grad log pi_k]. With a K-independent environment return and baseline, the latter is the conditional expectation of the former given h,a. They have the same expectation; one assigns sampled-label credit, the other can spread action credit to overlapping policies. This is standard mixture policy learning, not a new responsibility target.

The sequential Q must integrate all subsequent routing and experimental observations. Critic-at-mean-action, one-step posterior spread, and one realized noisy suffix are not interchangeable with this value. Additional powers of s_k or winner labels need a derivation, not an appeal to probabilistic responsibility.

An early exploration mixture b=(1-epsilon)g+epsilon/4 is valid if log b, derivatives, data collection and target expectations use b consistently. A time-varying epsilon schedule changes the policy being optimized; gradient/evaluation policy definitions must state it. Using log g on b samples is not automatically on-policy. No schedule was selected or run.

For a genuinely stage-only control, g_B(h)=softmax(l_B(t)), and no belief, condition scores, Q table, shared history feature, cached conditional prior or mixture responsibility may feed its forward router logits. The experts still see full history. Native policy gradients may naturally give stage parameters different training signals across histories; that does not make their deployed gate history-dependent. The two arms must match all other learning opportunities.
''')
write('POLICY_GRADIENT_DERIVATION.md','''# Full-horizon native variational mixture gradient

For frozen native posterior weights w and zero extra physical rewards in location finding, r_t^w=log q_{w,t}(theta|h_{t+1})-log q_{w,t-1}(theta|h_t), q_{w,-1}=prior. Gamma=1 makes the sum telescope to the terminal native variational information score. Its population expectation is a variational information bound, not exact MI. Native q training and critic approximation remain separate approximations; replacing them by sPCE is excluded.

Define J_w(pi)=E_pi sum_{t=0}^{29}r_t^w, with the native terminal convention retained. For a legitimate stochastic mixture and on-policy histories,

    grad J_w = E sum_t A_t [grad log g_{K_t}(h_t)+grad log pi_{K_t}(a_t|h_t)].

A_t can use an action-independent history baseline. A_t=Q_t^pi(h_t,a_t)-V_t^pi(h_t) is the remaining variational return, not a greedy information score. Replacing Q by a learned critic introduces critic bias. Replay histories from another policy add an occupancy approximation; action importance ratios alone do not correct history occupancy.

Conditional posterior mixture responsibility r_k gives

    grad log pi(a|h)=sum_k r_k(h,a)[grad log g_k(h)+grad log pi_k(a|h)].

The fixture enumerates all 64 joint trajectories of a non-myopic two-decision toy problem with four experts and two actions, and checks the full time-summed score gradient against exact differentiation. This is an independent algebra fixture, not a full location-finding training integration.

Bounded continuous policies can use z~N(mu_k(h),diag sigma_k(h)^2), a=4 tanh z. For interior actions,

    log pi_k(a|h)=log Normal(atanh(a/4);mu_k,sigma_k)
                 -sum_i[log4+log(1-tanh(z_i)^2)].

Stable log Jacobian is log4+2(log2-z-softplus(-2z)). No clipping density is substituted. Boundary atoms introduced by the historical Gaussian-noise-plus-clipping collection policy are not represented by this continuous density. Actor, data collection, stochastic Bellman target and saved RNG all need coordinated changes; deterministic native actor code cannot simply receive a log-probability term.

A compatible fixed-history critic surrogate is F=E_{h~replay}sum_k g_k E_epsilon Q(h,4tanh(mu_k+sigma_k epsilon)). Expert path derivatives and a categorical score or exact finite gate expectation differentiate F. Target values must integrate expert AND continuous-action randomness. A stored action plus log-probability from an old behavior policy does not by itself validate an on-policy score gradient. This is mathematically feasible and is not an assertion of implementation impossibility.

For G_k(h) the appropriate native gradient contribution, resampling q_k(h)=s_k(h)d(h)/Z_k with d/q_k yields the same E_d G_k. Leaving the correction out yields E_d[s_k G_k]/Z_k, generally a different vector. A positive score is not enough to guarantee it is the original gradient. Parameter-only positive preconditioning can be an optimizer for the same objective, but history-dependent selection weights are not automatically such a preconditioner. The toy counterexample explicitly detects the gradient change. Correction does not preclude variance reduction; proving or measuring a useful estimator is a possible research contribution, not accomplished here.

Native off-policy deterministic training already has a documented surrogate. A new biased responsibility update could be studied honestly, but its relation to the original objective and difference from CIAS would have to be specified before training. No universal unbiased-MI claim is made.
'''+ '\nActual fixture receipt:\n\n'+block(g))
write('PROTOCOL_LOCK.md',f'''# Protocol state

{status}

Preflight definition saved before frozen feature computation: seed 202611103001; stages0,3,12,24,29;256 distinct episodes/stage split128/128;upper-third calibration thresholds;positive weighting floor .1. No wins or policy outcomes tune these definitions. Source/inputs/evidence are hashed in PREFLIGHT_MANIFEST.json.

Training is NOT locked. CONFIG_LOCKED.json keeps its requested filename but explicitly says PREFLIGHT_ONLY_NOT_A_TRAINING_LOCK and training_launch_enabled=false. No PRETRAIN_MANIFEST, training launcher, background queue, optimizer preparation, final-test access, simulator training episode, or v3 checkpoint exists.

Unexecuted pilot outline only: dynamic-v3 versus same-architecture stage-only-v3;2000 native updates/arm as a starting reference;gamma1;original posterior/critic/Adam protocol where compatible;independent evaluation2000 trajectories/1M prior contrasts. No final preparation budget, learning rates, continuous action scales, exploration schedule, stochastic target MC count or runtime budget was selected. Inventing these as a locked protocol after a scientific no-go would be misleading.

The exact unresolved gate is a substantive, mathematically specified BOED-oriented joint responsibility mechanism distinct from the established stochastic mixture/native Q gradient and prior CIAS teacher. A theorem is not required, but an actual new learning rule and falsifiable rationale are. The inspected construction did not meet it.
''')
write('IMPLEMENTATION_TESTS.md','''# Implementation and validation status

COMPLETE: component-level preflight. INCOMPLETE: actual v3 neural architecture and full native training integration.

The two source scripts run without optimizer steps. gradient_preflight.py independently verifies a bounded mixture density, stable transform, latent/marginal responsibility identity, full-horizon score gradient, exploration derivative, and the effect of unjustified specialist weights. frozen_history_audit.py uses existing frozen HEC-v2 and parent replay; no new v3 model.

Do not interpret these fixtures as a passed full native batch. A real stochastic native actor still needs full-batch finite-gradient checks, stochastic Bellman expectations, checkpoint/RNG round trips, matched stage-only invariance, no hidden-theta/future leakage, and exact behavior-policy density tests. No assertion was weakened or skipped and described as passed.

Gradient receipt:

'''+block(g)+'\nFrozen audit receipt summary:\n\n'+block({k:v for k,v in f.items() if k!='summary'}))
write('TRAINING_RESULTS.md','''# Training results

INCOMPLETE — NOT RUN.

No specialist preparation or v3 network training was launched. Optimizer updates:0. New training simulator episodes:0. No two-arm training claim, no loss curves, no checkpoint and no extrapolated EIG. The conditional methodological gate in the task was not met. Original successful HEC-v2 and previous models remain intact; no retraining/extension was attempted.
''')
write('COMMON_BANK_EIG_COMPARISON.md','''# V3 common-bank EIG comparison

INCOMPLETE — NO V3 MODEL OR NEW COMMON BANK.

No v3 result, uncertainty interval, dynamic-stage comparison or new baseline evaluation exists. These must not be replaced by historical values. Existing HEC-v2 reference results below are copied from its completed report for provenance only and do not constitute a v3 comparison.

'''+block({'source_run':'moe_hec_v2_icml_20261109','source_bank':'historical v2 bank,not a new v3 bank','candidate':old['methods']['candidate'],'stage':old['methods']['stage'],'continuation':old['methods']['continuation'],'vsoed':old['methods']['vsoed'],'paired_v2_tests':old['primary']})+'''
The historical bank used2000 trajectories/1M negatives. MCSE conditions on one trained lineage. Cross-bank pairing and compute-matched superiority over frozen vsOED are not claimed. No final-test data were accessed.
''')
write('EXPERT_AND_ROUTING_ANALYSIS.md','''# V3 expert and routing analysis

INCOMPLETE — no trained v3 experts or router.

Observed: coverage, likelihood-discrimination and weak-Fisher conditions vary within sampled post-observation stages, and different groups overlap partially. This makes measurable non-chronological preparation possible. It does not show that a specialist learns a better design policy there. The full condition receipts are preserved; no true theta or future observations enter the condition inputs.

Not observed: comparative expert EIG, history-predictable advantage, learned routing allocation, expert starvation, all-four useful contribution, policy diversity after training, and v3 benefit over frozen HEC-v2. Nothing here resolves HEC-v2's missing dynamic-versus-stage advantage.

If the methodological gate is eventually satisfied, evaluate a predeclared independent calibration/validation panel using the same native reward conventions and full policy resimulation after interventions. A same-stage signal must predict useful held-out return differences, not just router weights. Compare learned,stage-only,fixed,uniform and shuffled-history policies. Evaluate all fixed experts and donor/removal interventions with multiplicity correction and distribution-shift caveats. Grouped retained-joint-history returns are not certified individual conditional MI.

A later homogeneous-expert control should match capacity, starting stochastic policy, provider inputs, preparation compute, native updates, episodes and exploration;homogeneous versus specialized crossed with dynamic versus stage routing identifies an interaction more clearly than either pair alone. This is a plan for a subsequent authorized factorial;no extra arm has been launched.
''')
write('GPU_CPU_COMPUTE_LEDGER.md','''# Actual compute ledger

No training resource claim is made. CPU/GPU utilization and energy time series were not sampled;do not infer full utilization from wall time. ExistingGPUjobs were checked before the frozen audit;RTX4090 idle except display processes. The inherited nonblockingGPUleases were acquired;no jobs were interrupted.

'''+block({'gradient_fixture_seconds':g['seconds'],'frozen_audit_wall_seconds':f['seconds'],'GPU':f['GPU'],'peak_GPU_allocated_bytes':f['peak_GPU_allocated_bytes'],'training_updates':0,'preparation_updates':0,'new_simulator_episodes':0,'new_simulated_observations':0,'model_query_counts':f['likelihood_and_model_queries'],'CPU_utilization':'NOT_SAMPLED','CPU_RSS':'NOT_SAMPLED','GPU_utilization_time_series':'NOT_SAMPLED','new_EIG_evaluation_seconds':0,'v3_inference_latency':'NOT_AVAILABLE_NO_MODEL'})+'''
The frozen wall time includes loading,feature computation and hashing of protected inputs. These timings are not EIG per unit of training compute. Network/research/documentation time is not a training measurement. A future pilot would need explicitGPU,CPU,RAM telemetry, simulator and exact-likelihood counts, parameter counts,policy latency and separate preparation/training/evaluation clocks.
''')
write('ICML_ASSESSMENT.md','''# ICML assessment

The current v3 work is a mathematical/frozen preflight, not an algorithm-performance result. It does not establish an ICML contribution and does not prove one impossible.

Supported: standard stochastic mixture gradients can preserve the native variational information objective;valid bounded densities are feasible;non-chronological history conditions are measurable in saved post-observation histories;previous Top-1/CIAS mechanisms and HEC-v2's limitations are documented. The original models and inspected source/report bytes were unchanged.

Unproven: a new BOED-specific joint responsibility mechanism;actual specialized complete design policies;dynamic router usefulness beyond stage;EIG above frozen HEC-v2;all-four complementarity;homogeneous-control interaction;multiple-seed robustness;generalization;practical stochastic-native optimizer behavior.

Minimum next contribution is a precise additional learning/credit rule with a clear advantage over the already implemented Top-1 and CIAS rules, compatible with full-horizon native return and stage-only identification. It may be a variance-reduction or occupancy-aware specialization mechanism;no theorem is required. It must not add unexplained history weights and call the result the same gradient. Once justified,implementation validation precedes a locked two-arm pilot. Standard MoE with belief features could be a useful application experiment,but should be labeled as such and does not meet the present central requirement.

If later evidence warrants it: independent training seeds;specialization-by-routing factorial;equal-capacity and compute accounting;deterministic SIR ODE and power-grid swing-equation benchmarks with legitimate model-specific observation/likelihood features. These campaigns remain plans only. No SIR or power-grid training was launched.
''')
write('NEXT_RESEARCH_DECISION.md',f'''# Decision

{status}

No v3 training is running or queued. The current proposal reduces to established mixture actor-critic plus specialist preparation unless an additional history-weighted update is inserted;that insertion has not been justified as the stated objective's gradient or a distinct validated optimizer. A score-conditioned gate is mathematically valid but an established mechanism. Importance-corrected data allocation can be valid,but no BOED-specific useful estimator has been established here.

This invokes the explicit task instruction to report a missing substantive mechanism before expensiveGPUtraining. It is an implementation/research limitation of this attempt,not an impossibility claim or an empirical v3 failure. No theorem or publication-priority proof is being demanded as a new requirement.

Keep HEC-v2 as the frozen reference. Preserve all historical and new preflight artifacts. Do not automatically train a standard mixture under a novelty label,extend prior training,launch additional architectures,seeds,homogeneous controls,SIR or grid campaigns. The minimum unresolved item is a specified,falsifiable,objective-compatible BOED responsibility rule materially different from prior Top-1/CIAS. There is no v3 EIG result to promote.
''')
write('GITHUB_UBUNTUPC_SYNC_STATUS.md','''# Synchronization

The source,configuration and documentation in development/moe_sboed/v3_competitive_20261110 are the authoritative LabPC versions. The configured workflow will commit/push them through GitHub,fast-forward the clean Ubuntu worker,and verify equal hashes. The machine-readable receipt is evidence/GITHUB_SYNC_RECEIPT.json;until it exists,no success is claimed here. Checkpoints,replay arrays,raw private histories and runtime outputs are excluded from Git. Synchronization does not launch computation.
''')
write('README.md','''# V3 preflight only

No training launcher is provided. Read RUN_INDEX.md. Scripts use explicit existing frozen sources and independently tested CPU fixtures;they never train a model. Outputs belong in the isolated scratch run,not theGitdirectory. Never infer completed v3 training from existing HECEIG tables.

Run gradient_preflight.py --output TEMP/scratch_runs/moe_sboed_v3_competitive_20261110/evidence using the existing dellma environment. frozen_history_audit.py refuses to overwrite its completed receipt and respects the existingGPUlease. write_dossier.py renders the actual receipts;it performs no scientific computation.
''')
(D/'.gitignore').write_text('__pycache__/\n*.pyc\nevidence/\n*.npz\n*.npy\n*.pt\n*.pkl\n*.log\n')
print('DOSSIER_WRITTEN',status)
