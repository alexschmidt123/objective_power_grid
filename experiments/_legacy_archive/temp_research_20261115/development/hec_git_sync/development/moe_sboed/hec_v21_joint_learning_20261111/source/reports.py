"""Explicit protocol and actual results;never substitute historical EIG."""
from pathlib import Path
import json,numpy as np

def read(p):return json.loads(p.read_text()) if p.exists() else None
def block(x):return '```json\n'+json.dumps(x,indent=2)+'\n```\n' if x is not None else 'INCOMPLETE:not available yet.\n'
def save(r,n,s):(r/n).write_text(s.strip()+'\n')

def static(r,c,diagnosis,validation):
 save(r,'HEC_V2_ROUTING_DIAGNOSIS.md','''# Verified HEC-v2 routing diagnosis

Four heterogeneous expert pathways produce128-coordinate packets(first64 plus response64),not complete actions. E1 retains a frozen geometry anchor plus live differences;E2 combines likelihood representations with exact-hypothesis semantics and52 learned responses;E3 retains a frozen sensitivity anchor plus live differences;E4 combines opportunity representations with predictive continuation semantics and52 learned responses. A shared history path and frozen SG reference remain active in every policy. These inherited frozen weights are preserved exactly;live specialists remain EIG trainable. No new hand-assigned experimental stages.

The20-dimensional commander affects real model queries. Old coordination is a nonlinear network on history/legacy representations plus a linear response gate,followed by bounded logits and softmax. Its native gradients are NONZERO. The audit finds mean logit-squash derivatives about.99,so saturation/dead gradients are not supported explanations. Within-stage gate variability is also nonzero. Raw gradient norms of different-sized parameter blocks are not directly comparable learning rates.

In192 saved histories at stages3/12/24,shared-history hidden contributions exceed the weighted expert hidden contribution by roughly2.15/3.37/4.15 in norm. The gate-to-action tangent RMS is about.036-.039 physical units per logit. Expert contribution scales differ after the actual fusion matrices. Cancellation ratios are about.65-.71;this is not proof of harmful redundancy. Actual aggregate EIG evidence shows full versus uniform unresolved,while one-hot expert pathways are substantially worse. It does not prove each expert indispensable or dynamically allocated by comparative information value.

Feature/head scale compensation is architecturally possible and plausible,not causally established. A fixed positive gate can be absorbed into expert fusion-column scales;history-dependent gates do not in general admit one exact fixed rescaling. The head can use history directly. Current raw-representation coordination is not explicitly informed of these consumed contribution scales or their physical action/likelihood effects. This motivates one candidate;it is not a proven cause or guaranteed cure.

Actual frozen audit(critic gradients at saved histories are NOT policy EIG):

'''+block(diagnosis))
 save(r,'JOINT_LEARNING_ALGORITHM.md','''# One joint mechanism:routing from decision contributions

Let s(h) be shared history,first_k and resp_k the existing specialist channels,and c_k=W_k first_k+V_k resp_k their128-dimensional contributions to the actual shared decision preactivation. Let b_s include shared history and bias. The existing coordinator supplies logits ell0=10tanh(raw/10) and g0=softmax(ell0). The reference collaborative preactivation is p0=b_s+sum g0_k c_k. No standalone physical expert actions are formed or averaged.

For the existing decision transform a=H(p,a_SG),compute analytically J_a(p0),then v_k=J_a(p0)c_k and t_k=g0_k(v_k-sum_j g0_j v_j). t_k is the exact local derivative of the inherited collaborative physical design with respect to its routing logit. These local features describe effects around g0,not a fixed-point solution under the corrected gate. Native output is still recomputed exactly using the original fusion operations.

Use the exact location-finding likelihood derivative J_xi(theta,a_ref)=-sum_source J_theta,with sigma=.5. A per-expert model-response descriptor is4 E_q[(J_xi dot t_k)^2]. It is a sensitivity descriptor,NOT greedy EIG or an expert-quality reward. Reuse the original approximate GMM quadrature,posterior mean/variance,existing likelihood-discrimination and prospective-opportunity semantics. No additional posterior inference,critic query or deployment rollout is introduced.

A trainable16-dimensional projection of sign(c)log(1+abs(c)),the two physical tangent coordinates,three scale/sensitivity statistics and four expert identity coordinates join160 shared context coordinates. One shared185->64->1 SiLU network returns correction rho_k. Final g=softmax(ell0+rho). It is not merely a new name for the old gate:its inputs include actual consumed expert channels,the shared head's local action geometry and exact likelihood responses. Softmax,attention-like scoring,analytic sensitivity and residual transfer are established mechanisms;novelty is unproven.

The correction output layer starts at zero. Candidate and ordinary continuation therefore start with bitwise-identical physical actions,unchanged inherited gradients,and the same checkpoint/Adam/replay/RNG lineage. New tensors have fresh Adam moments at the inherited native learning rate. Existing expert,commander,head,critic and posterior objects/moments are retained. The new projection/hidden layer first receive gradients after the zero output layer moves;this is validated with a disposable nonzero output layer,not mistaken for starvation. No auxiliary preparation or new objective.

Joint update:use the unmodified native actor loss L=-mean_{h in replay}Q_psi(h,a_theta(h)) over all30 stages,with the same native critic/posterior updates. For p=b_s+sum g_k c_k,the direct logit derivative is

    dQ/dell_k = g_k grad_p Q dot(c_k-sum_j g_j c_j).

Autodiff also includes expert-to-c,head-to-c,commander-to-features,and all contribution/likelihood-feature-to-rho paths. Tokens are not detached. Thus specialists and router learn together from the actual collaborative design's remaining-value native signal. No standalone-head Q auxiliary,teacher CE,diversity/load balance,cost penalty or separate reward. Full native discount1 variational information rewards and optimizer schedules remain unchanged. Learned Q/off-policy replay and variational approximation retain their original bias;this is not an unbiased exact-MI theorem.

The finite-horizon reward telescopes from prior to terminal variational posterior at gamma1. It is NOT sPCE training. Main evaluation alone uses the fixed full-horizon prior-contrast lower bound. Compared with ordinary soft MoE,the changed component is consumed-contribution and model-informed coordination with full gradient chains. Compared with CIAS,there is no noisy counterfactual teacher or additional routing objective. Shared-target convergence can still occur;the mechanism must earn useful EIG and competence evidence.

The existing frozen E1/E3 anchors and model-semantic functions preserve their exact functions. This does not guarantee retained utility of every adaptive pathway;retention probes and resimulated interventions are required. No claim of useful specialization follows from representation differences or variable weights alone.
''')
 save(r,'GRADIENT_VALIDATION.md','# Correctness evidence\n\n'+block(validation)+'\nIncludes initial action and inherited gradient identity,exact gate-action/likelihood derivatives,all trainable pathway reachability,command/weight action effects,no future-prefix leakage,unchanged frozen checkpoint bytes,and full native10000-trajectory/300000-history batch with all32 fixture optimizer calls intercepted. No scientific optimizer steps. Evaluator/literal contrast,conditional resimulation,and external frozen loader receipts are separately required before training. No assertion is weakened.\n')
 save(r,'PROTOCOL_LOCK.md','''# Fixed protocol before training

One candidate,one matched ordinary HEC-v2 continuation;sequential continuation then candidate. Same verified successful HEC-v2 iteration16001 checkpoint;2000 native updates each,target18001. No new preparation,architecture sweep,extra seeds or training extension. Preserve original files. Configuration/source/evidence protected by PRETRAIN_MANIFEST before updates.

Same native variational actor-critic gamma1 objective,Adam state and learning-rate/exploration schedules,1M replay capacity,10000 replay episodes/batch,1000 new trajectories/update,5 critic updates and5 posterior updates per stage. Each arm adds2M simulator trajectories/60M observations,2000 actor/10000 critic/300000 posterior optimizer steps. New candidate router tensors alone have fresh Adam moments. Both restore the identical parent simulator RNG(seed101 lineage);new router initialization uses a forked fixed seed without advancing simulator randomness. Parameter count/compute overhead is disclosed,not claimed matched. All existing trainable/frozen flags and inference anchors remain unchanged.

New independent evaluation bank:2000 trajectories,T30,1M negative prior contrasts plus positive once,exact same frozen scorer,16-case batches,16384 contrast chunks. New full,its own uniform andfour one-hot feature ablations,matched continuation,original HEC-v2,conventional MoE;also frozen vsOED,DAD,RL-sBOED,ScoreBED context. Full changed-policy resimulation;shared head/history/frozenSG reference retained. No physical action averaging. No final-test data or selection against an already reused bank.

Six elementary primary tests(candidate-continuation,candidate-uniform,andcandidate-each of4 individual experts) use Holm6. This provides valid corrected inference for the empirically best individual without pretending a selected maximum was predeclared. Report all individual scores,the best-observed label,and its already-corrected contrast. Ordinary paired95%tCIs are labeled unadjusted. Candidate-original andcandidate-conventional secondary Holm2;four external baselines separate Holm4,not compute matched. No stage-only primary criterion.

Separate512 diagnostic trajectories split256 calibration/256 heldout at stages3/12/24. Calibrated ridge probes of learned representations exclude exact analytic bypass coordinates;fixed parent-native labels and common commands permit competency/retention comparison with original and continuation. These tasks are diagnostics,not added rewards. Four resimulated expert removals Holm4;new correction-off intervention is a single descriptive frozen-effect test. No training-level causal interaction claim.

Conditional panel:128 heldout joint-generated histories with their retained latent draw,2 fresh future-noise replicas,stages3/12/24. Replace one expert packet for the next decision by an independent calibration-history donor,then follow the same newly simulated candidate suffix. Compare remaining normalized parent-native variational scores;not exact individual conditional MI. Group by stage-specific calibration median posterior variance trace;24 grouped tests Holm24. Minimum16 independent histories/group;noise replicas are clustered,not independent posterior samples. Never replay saved future observations or substitute an independent prior theta.

Record actual simulator batch/observation/episode calls,model likelihood/Jacobian queries,per-arm times,10second processCPU/RAM/GPU utilization/memory/power samples,and per-design latency. GPU lease and duplicate-safe launcher;queue without interrupting jobs if occupied. Training grace32400seconds,evaluation at most3600seconds within that grace,TERM33000,hard33600. On correctness failure,STOP_REQUESTED or cap,preserve incomplete/negative results. No automatic restart,more training,or Ubuntu computation. All seed values in CONFIG_LOCKED.json are predeclared before training.
''')
 save(r,'ICML_NOVELTY_ASSESSMENT.md','''# Novelty and evidence

This is one model-informed coordination adaptation of established MoE/native actor-critic principles. Contribution decomposition,Jacobian sensitivity,residual initialization and softmax are not new individually. Their BOED-specific combination is a research hypothesis,not an established ICML contribution. Novelty uncertainty does not block this correct pilot.

Required evidence:improvement over ordinary continuation,learned-over-uniform EIG benefit,useful expert contributions,and retention/distinct conditional competencies. Full-versus-individual frozen ablations alone do not prove training-level specialization/routing interaction. A future matched homogeneous/expertise feature factorial and multiple seeds would be needed,plus cross-benchmark SIR ODE/power-grid evidence. None is authorized as an automatic extra campaign here.

The parent GMM is an inherited approximate belief,not an exact posterior;head tangent descriptors measure model response,not true conditional information value. Critic/replay approximation and one training lineage limit conclusions. Extra router capacity and compute are not perfectly matched by ordinary continuation. These limitations must remain even with positive EIG. No compute-matched superiority over frozen external baselines or theoretical novelty claim without evidence.
''')

def write(r):
 done=read(r/'evidence/EXPERIMENT_COMPLETION.json');ev=read(r/'evaluation/COMPLETION.json');eig=read(r/'evaluation/EIG_RESULTS.json');diag=read(r/'evaluation/EXPERT_AND_ROUTING_RESULTS.json');cond=read(r/'evaluation/CONDITIONAL_COMPETENCE.json');latency=read(r/'evaluation/LATENCY.json');arms={k:read(r/f'arms/{k}/RESULT.json') for k in ['continuation','candidate']};decision='INCOMPLETE:awaiting training/evaluation'
 if eig and ev and ev['status']=='COMPLETE':
  positive=lambda v:v['mean']>0 and v['Holm6_p']<.05
  goals=dict(beats_matched_continuation=positive(eig['primary']['continuation']),learned_beats_uniform=positive(eig['primary']['uniform']),beats_all_individuals=all(positive(eig['primary'][f'expert{k}']) for k in range(1,5)));decision='ALL THREE PRIMARY EIG CRITERIA SUPPORTED;specialization/causal novelty still require separate evidence' if all(goals.values()) else 'PRIMARY SUCCESS CRITERIA NOT ALL SUPPORTED'
 else:goals=None
 save(r,'RUN_INDEX.md','# HEC-v2.1 joint learning\n\n'+decision+'\n\n'+block(done)+'\nOne candidate and ordinary continuation. Native full-horizon variational EIG. All source/configuration/inputs are locked before training. See EIG_COMPARISON.md,EXPERT_AND_ROUTING_ANALYSIS.md,GPU_CPU_COMPUTE_LEDGER.md,andNEXT_RESEARCH_DECISION.md. No extra seeds or architectures.\n')
 save(r,'TRAINING_RESULTS.md','# Actual native training\n\n'+block(dict(arms=arms,completion=done))+'\nNo auxiliary objective or preparation. Original checkpoints,optimizer/replay lineage and all frozen anchors are preserved. Per-arm counts are measured from actual simulator and optimizer calls. Partial runs remain incomplete;no budget extension.\n')
 text='# Independent common-bank EIG\n\n'
 if eig:
  text+='2000 trajectories,T30,1M negative contrasts plus positive once. Finite-contrast lower bound,not exact EIG. MCSE conditional on one trained lineage.\n\n|Policy|EIG nats|MCSE|\n|---|---:|---:|\n'
  for name,v in eig['methods'].items():text+=f"|{name}|{v['mean']:.6f}|{v['MCSE']:.6f}|\n"
  text+='\nPrimary Holm6 comparisons;ordinary95%CIs are unadjusted:\n\n'+block(eig['primary'])+'\nBest observed individual and valid corrected contrast:\n\n'+block(dict(expert=eig['best_observed_expert'],comparison=eig['full_minus_best']))+'\nSecondary Holm2:\n\n'+block(eig['secondary'])+'\nFrozen external context Holm4:\n\n'+block(eig['external_baselines'])
 else:text+='INCOMPLETE:no new EIG result;historical values are not substituted.\n'
 save(r,'EIG_COMPARISON.md',text+'\nOriginal/external checkpoints have different accumulated training/provider budgets. No cross-bank pairing,final-test use or compute-matched superiority claim.\n')
 save(r,'EXPERT_AND_ROUTING_ANALYSIS.md','# Expert and routing evidence\n\n'+block(diag)+'\nConditional resimulated native-return diagnostics:\n\n'+block(cond)+'\nFrozen weights,output diversity or gate variation are not evidence of useful competence. Learned-representation tasks exclude exact analytic semantic bypasses;calibration chooses regressors/rivals without heldout outcomes. Retention contrasts are not equivalence proofs. Frozen anchors stay byte-identical,buttheir eventual usefulness is measured separately. Conditional donor effects estimate grouped expected native return,not exact per-history EIG. Distribution shift can affect single/removal/correction-off interventions. No causal training interaction claim from frozen ablations alone.\n')
 ledger={};p=r/'TRAINING_LOGS/resources.jsonl'
 if p.exists():
  rows=[json.loads(v) for v in p.read_text().splitlines()]
  if rows:
   for k in ['CPU_percent_one_core','RSS_bytes','GPU_percent','GPU_memory_MiB','GPU_power_W']:ledger[k]=dict(sample_mean=float(np.mean([v[k] for v in rows])),sample_max=float(max(v[k] for v in rows)))
   ledger['samples']=len(rows);ledger['interval_seconds']=10
 save(r,'GPU_CPU_COMPUTE_LEDGER.md','# Actual compute ledger\n\n'+block(dict(arms=arms,evaluation=ev,latency=latency,sampled_resources=ledger,total_driver_seconds=(done or {}).get('seconds')))+'\nCPU percentages are relative to one core;child cumulative CPU is separately logged. RAM samples and process high-water RSS,deviceGPU memory samples and Torch allocated peaks are different measurements. Resource samples span training/evaluation;per-arm times separate training. Queries include analytic feature computations and checkpoint recomputation,not only physical simulation. No cost enters the objective and no unmatched EIG-per-compute superiority is inferred.\n')
 save(r,'NEXT_RESEARCH_DECISION.md','# Research decision\n\n'+decision+'\n\n'+block(dict(primary_goals=goals,completion=done,evaluation=ev,primary=(eig or {}).get('primary')))+'\nPositive EIG requires separate competence/retention interpretation. Nonsignificance is not equality. One lineage cannot establish ICML novelty or robust external superiority. Preserve negative/incomplete outcomes;do not extend training or launch a new architecture,seed,SIR/grid or factorial campaign without a new decision.\n')
if __name__=='__main__':
 import sys
 write(Path(sys.argv[1]))
