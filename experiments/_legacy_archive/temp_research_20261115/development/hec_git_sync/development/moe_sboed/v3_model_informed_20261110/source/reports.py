"""Write the fixed protocol and actual outcomes, including incomplete runs."""
from pathlib import Path
import json,numpy as np

def read(p):return json.loads(p.read_text()) if p.exists() else None
def block(x):return '```json\n'+json.dumps(x,indent=2)+'\n```\n' if x is not None else 'INCOMPLETE: not available yet.\n'
def save(r,n,s):(r/n).write_text(s.strip()+'\n')

def static(r,c,checks):
 docs={
 'PRIOR_MOE_COMPARISON.md':'''# Prior methods

The revised authorization removes the novelty-only stop. The previous frozen preflight remains intact. Its literature review covers [Jacobs](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf),[MMoE](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/),[PLE](https://doi.org/10.1145/3383313.3412236),[Switch](https://www.jmlr.org/papers/v23/21-0998.html),[Expert Choice](https://arxiv.org/abs/2202.09368),[DeepSeekMoE](https://arxiv.org/abs/2401.06066),[DAD](https://proceedings.mlr.press/v139/foster21a.html),[vsOED](https://arxiv.org/abs/2306.10430),[RL-sBOED](https://proceedings.mlr.press/v162/blau22a.html),[ScoreBED](https://arxiv.org/abs/2607.08335). Conditional allocation,data curricula,shared knowledge and mixture actor-critic are established ideas,not new inventions.

Direct-head mixes logits before a bounded action. Previous Top-1 already sampled complete expert actions and used comparative native-Q gate credit and categorical Bellman targets;its EIG difference was inconclusive. Option-MoE used3-step expert options plus a frozen fallback. CIAS added resimulated remaining-return labels,a cross-fitted conditional teacher and auxiliary gate CE;it did not establish useful specialization. Their gamma.9/parents differ from gamma1 HEC/V3,so their negative results do not prove this candidate fails.

Collaborative Soft,SG and HEC-v1/v2 coordinate features toward one action. HEC-v2 improved over continuation but not stage-only control;all-four usefulness was unestablished. V3 instead executes complete stochastic action experts,prepares them on different history conditions with native remaining-value Q,then coordinates their actual proposals through likelihood/Fisher queries. It adds no CIAS teacher,CE,output-diversity or cost objective. These project differences are a testable application hypothesis,not established ICML novelty.
''',
 'V3_ALGORITHM_AND_NOVELTY.md':'''# Model-informed complete-policy competition

One frozen HEC-v2 reference/belief provider supplies a reference action and the inherited approximate SG GMM quadrature. Four independent64-hidden-unit action heads share a trainable128-dimensional history encoder. Head0 is a generalist;heads1-3 receive coverage,discrimination and identifiability preparation. Each emits a complete next-design distribution at every t;none is an ordered experimental stage.

b(h)=atanh(clip(a_ref(h)/4,-.999,.999));mu_k=b+.5tanh(E_k(h));z~N(mu_k,.025^2 I);a=4tanh(z). The reference is shared computation,not a fifth selected expert or an executed prior experiment. A bounded residual class protects a strong parent but may limit specialization. Initial stochastic V3 differs from deterministic HEC-v2;initial-policy EIG is reported separately.

At each mean proposal compute9 model features:predictive mean,expected pairwise Gaussian KL,four diagonals of posterior-scaled expected Fisher,logdet(I+F_scaled),and proposal coordinates. The router uses stage,history,belief mean/variance,and these proposal-dependent features. g=.95softmax(router)+.05/4 is used in collection,targets,gradients AND deployment. Sample K,then a from that expert,simulate y,and reroute. No physical action averaging.

Model-feature gradients through the actual proposals remain active. Selected-action credit reaches the selected expert;allocation derivatives can also reach other heads through their own proposal features. This is the composite policy gradient,not a separate reward. The approximate belief and quadrature are not exact posterior inference. All four dense proposals are computed;no sparse-compute claim. This composition may or may not improve EIG/competence;novelty is unproven.
''',
 'EXPERT_SPECIALIZATION_DEFINITION.md':'''# Operational specialist preparation

4800 distinct parent replay episodes:128 training and32 heldout prefixes per stage for all30 stages. Only history-prefix data enter the policy. Generalist weight1;specialist weight .1+1[score exceeds stage-specific training upper-third quantile],normalized by training mean. Conditions overlap and do not split tied prior histories arbitrarily.

Scores:posterior covariance trace;expected pairwise Gaussian likelihood KL at the reference design;and -log(1+lambda_min(D F_past D)),where D is diagonal posterior standard deviation and F_past=4 sum_{i<t} E_q[J(theta,xi_i)J(theta,xi_i)^T]. These are approximate-belief diagnostics,not known expert-optimal regimes. Label symmetries and posterior miscalibration can confound them.

Prepare action heads only for200 Adam steps,lr.001,batch256,using weighted sampled-design Q from the frozen trained parent native critic. No new simulator trajectories. Encoder,router,critic,posterior and reference stay fixed. This is a remaining-value design surrogate,not observed EIG or a reconstruction objective. Preparation is physically shared once and charged equally to both arms.

Afterward the preparation optimizer is discarded. New expert/router main Adam moments are fresh;compatible encoder Adam moments are inherited. Main native EIG updates have NO curriculum multiplier or auxiliary. Roles need not persist or prove useful;heldout intervention returns must establish actual competence. All experts remain interchangeable and available at every t.
''',
 'ROUTER_RESPONSIBILITY_RULE.md':'''# Proposal-informed responsibility

pi(a|h)=sum_k g_k(h,mu_0:3(h))pi_k(a|h). Histories and proposal likelihood/Fisher summaries inform allocation. The matched stage control has the same202-to64-to4 router,but its172 non-stage inputs are exactly zero. Its experts see full histories. Preparation,parameters,starting states,simulation,optimizer opportunities and exploration match. No condition prior or hidden history feature bypasses this restriction.

The last router layer starts at zero. Both arms therefore start with uniform probabilities and identical stochastic action policies after identical preparation. Initial tensor digests and a shared-policy check are recorded.

r_k=g_k pi_k(a|h)/sum_j g_j pi_j(a|h) is reported as posterior action-density responsibility,not probability an expert has highest EIG. Main credit uses selected-expert pathwise gradients plus the exact finite categorical native-Q expectation. No auxiliary gate CE,load balancing or arbitrary diversity target. Model-informed routing must demonstrate usefulness beyond stage/uniform interventions,not merely variable weights.
''',
 'POLICY_GRADIENT_DERIVATION.md':'''# Native full-horizon mixture gradient

Native incremental variational rewards r_t=log q_w(theta|h_{t+1})-log q_w(theta|h_t),with prior at empty history and the exact terminal convention,are unchanged. Gamma1 telescopes to the terminal variational information score. This remains a variational approximation,not exact MI;no sPCE training or new reward.

For fixed replay occupancy and critic Q,F=E_{h,epsilon}sum_k g_k(h,mu(h))Q(h,a_k(h,epsilon_k)). Let q_k=Q(h,a_k),b=sum stop(g_k)stop(q_k),K~g. Minimize

    L=-Q(h,a_K)-sum_k g_k stop(q_k-b).

The expected selected-action derivative gives sum g_k grad Q_k;the finite gate term gives sum(q_k-b)grad g_k,including proposal-feature paths. Since sum grad g=0,the baseline cancels. Explicit independent Gaussian draws for each expert and categorical uniforms are checkpoint inputs,so recomputation does not redraw them. A finite-sum gradient identity tests the full proposal-dependent gate chain.

The on-policy score equivalent sums A_t[grad log g_K+grad log pi_K] over all30 decisions;posterior component responsibilities yield the marginal-score identity. These are established identities. Native averaging over B*T preserves its scale convention. Learned Q and off-policy replay history occupancy retain approximation bias;no exact/unbiased MI claim.

Bellman targets integrate all four target experts,with one reparameterized continuous sample per expert and target probabilities. Q at a mean action is NOT substituted for the stochastic expectation. Native critic/posterior Adam,schedules,replay and Polyak.1 are retained. Compatible encoder parameter objects/moments survive;new expert/router moments are fresh. Retired incompatible HEC actor state is not falsely copied. Target encoder EMA is preserved;new target heads match preparation;both use the same immutable reference.

For a=4tanh z,log pi_k=log Normal(atanh(a/4);mu_k,.025)-sum[log4+2(log2-z-softplus(-2z))]. Clipping cannot be substituted for a density. External Gaussian exploration is zero for both arms because stochastic sampling is now inside the policy;the .05 categorical exploration is included consistently everywhere. There is no ongoing preparation objective or additional loss.
''',
 'PROTOCOL_LOCK.md':'''# Fixed protocol

Revised authorization supersedes the earlier novelty-only no-go. PRETRAIN_MANIFEST locks source,configuration,inputs and evidence before ANY preparation/main training. One candidate,two sequential arms(stage then candidate),2000 updates each from verified HEC-v2 iteration16001 to18001. Shared preparation200 steps. No extension,sweep or extra seeds.

Both arms:1000 new episodes/update,10000 replay trajectories,1M replay capacity,5 critic updates,5 posterior updates per stage,gamma1,original optimizer schedules where compatible. Each arm adds2M trajectories/60M observations,2000 actor/10000 critic/300000 posterior steps. Preparation uses no new episodes. The two changed native policy expectations and consistent in-policy exploration are documented separately from the unchanged reward.

One new independent evaluation bank:2000 trajectories,T30,1M prior negatives plus each positive once,fixed exact log-sum-exp scorer. Policies:candidate,stage,frozen HEC-v2/v1,SG,Option,Collaborative,ordinary,vsOED,DAD,RL-sBOED,ScoreBED,and initial stochastic policy. Candidate-stage and candidate-HECv2 primary Holm2;secondary V1/SG/Option/vsOED Holm4. Display ordinary paired95%CIs as unadjusted. One lineage;frozen baselines not compute matched. No final-test access.

Separate512 diagnostic cases:256 calibration/256 heldout. Four routing interventions(uniform,stage mean,shuffled gates,shuffled proposal features) Holm4;four fixed-expert comparisons Holm4;four removals Holm4. Every policy alteration resimulates observations and updates history. Calibration never uses validation outcomes. Distribution-shift caveats apply.

Conditional panel:128 heldout joint-generated prefixes,stages3/12/24,two future-noise replicas per retained theta/history pair. One-step expert intervention then common mixture suffix;compare fixed parent-native terminal variational scores,NOT individual conditional MI. Stage-specific upper-third condition thresholds use separate diagnostic calibration histories.18 high/low group advantages Holm18 and9 high-minus-low Welch interactions Holm9;minimum16 independent histories/group. No prior restart or saved future replay.

Seeds fixed in CONFIG_LOCKED;native simulator RNG resumes parent seed101 lineage. GPU lease and duplicate-safe launcher preserve other jobs. Training grace32400seconds including preparation;evaluation at most3600seconds within grace;TERM33000,hard33600. Correctness failures,STOP_REQUESTED or cap retain incomplete/negative results;no automatic restart or extra training. Ubuntu only receives source synchronization.
''',
 'ICML_ASSESSMENT.md':'''# Scientific assessment

No pre-result claim of higher EIG,useful four-way specialization,dynamic-routing advantage or ICML novelty. The method combines established stochastic MoE/native actor-critic with model-informed proposal coordination and a disclosed specialist curriculum.

Required evidence:paired primary EIG gains;conditional/intervention-supported useful competencies;proposal-information and dynamic-routing effects beyond stage/uniform;replicated performance and a specialization-by-routing factorial. Output diversity,density responsibility or preparation Q alone are insufficient. Frozen baselines are not compute matched;one lineage cannot establish robust superiority.

Later plans only:matched homogeneous experts with equal capacity/provider/preparation/episodes,independent seeds,interaction ablations,deterministic SIR ODE and swing-equation power-grid evaluation. None launches automatically. A negative pilot is preserved and frozen HEC-v2 remains reference.
''',
 'GITHUB_UBUNTUPC_SYNC_STATUS.md':'''# Synchronization

LabPC authoritative source:development/moe_sboed/v3_model_informed_20261110. Repository alexschmidt123/objective_power_grid,branch research/hec-moe-sboed-20261109. Before launch,commit/push validated source,fast-forward the clean Ubuntu worker and verify equal hashes. Actual receipt:evidence/GITHUB_SYNC_RECEIPT.json. No datasets,checkpoints or private histories in Git. Frozen dependencies are separately hashed;source synchronization is not compute/deployment readiness. No Ubuntu training.
'''
 }
 for n,s in docs.items():save(r,n,s)
 save(r,'IMPLEMENTATION_TESTS.md','# Correctness evidence\n\n'+block(checks)+'\nAdditional evidence:CONDITION_PREFLIGHT.json,EVALUATION_CHECKS.json,CONDITIONAL_CHECKS.json and three historical-baseline preflights. The first dynamic-variation fixture compared two identical prior histories;it was corrected to distinct same-stage observed histories without weakening the assertion. Failed logs are preserved. No optimizer stepped in correctness fixtures.\n')

def write(r):
 done=read(r/'evidence/EXPERIMENT_COMPLETION.json');eig=read(r/'evaluation/EIG_RESULTS.json');ev=read(r/'evaluation/COMPLETION.json');diag=read(r/'evaluation/ROUTING_AND_EXPERT_RESULTS.json');cond=read(r/'evaluation/CONDITIONAL_COMPETENCE.json');prep=read(r/'preparation/RESULT.json');arms={k:read(r/f'arms/{k}/RESULT.json') for k in ['stage','candidate']};lat=read(r/'evaluation/LATENCY.json');decision='INCOMPLETE: awaiting training and evaluation'
 if eig and ev and ev['status']=='COMPLETE':decision='PRIMARY EIG COMPARISONS POSITIVE;mechanism evidence requires separate assessment' if all(v['mean']>0 and v['Holm_p']<.05 for v in eig['primary'].values()) else 'NO DEMONSTRATED EIG IMPROVEMENT OVER BOTH PRIMARY REFERENCES'
 save(r,'RUN_INDEX.md','# V3 model-informed competitive pilot\n\n'+decision+'\n\n'+block(done)+'\nOne authorized candidate and matched stage-only control. Original full-horizon variational EIG. Source/configuration/evidence and protected inputs are locked before preparation. See EIG,expert/routing and compute reports. No extra architectures,seeds or Ubuntu computation. Earlier preflight and all experiments remain preserved.\n')
 save(r,'TRAINING_RESULTS.md','# Actual training\n\n'+block(dict(preparation=prep,arms=arms,completion=done))+'\nPreparation Q is a frozen native-critic surrogate,not measured EIG. Main training has no auxiliary loss. No budget extension.\n')
 s='# Common-bank full-horizon EIG\n\n'
 if eig:
  s+='2000 full trajectories,1M negatives plus positive once;finite-contrast lower bound,not exact EIG. MCSE conditional on one training lineage.\n\n|Policy|EIG nats|MCSE|\n|---|---:|---:|\n'
  for k,v in eig['methods'].items():s+=f"|{k}|{v['mean']:.6f}|{v['MCSE']:.6f}|\n"
  s+='\nPrimary Holm2;ordinary95%CIs are unadjusted:\n\n'+block(eig['primary'])+'\nSecondary Holm4:\n\n'+block(eig['secondary'])+'\nInitial stochastic-policy change versus frozen parent:\n\n'+block(eig['initial_change_vs_frozen'])
 else:s+='INCOMPLETE:no completed new common-bank comparison;no historical scores substituted.\n'
 save(r,'COMMON_BANK_EIG_COMPARISON.md',s+'\nFrozen baselines are not compute matched. No cross-bank pairing or compute-matched superiority over vsOED.\n')
 save(r,'EXPERT_AND_ROUTING_ANALYSIS.md','# Expert competence and coordination\n\n'+block(diag)+'\nConditional grouped native-return diagnostics:\n\n'+block(cond)+'\nDifferent outputs/weights alone are not competence. Interventions resimulate all changed observations and can shift distributions. Conditional scores are grouped variational-return contrasts with retained joint theta/history and clustered noise replicas,not exact per-history conditional MI. Homogeneous-factorial evidence remains a subsequent experiment;no extra training arm ran.\n')
 ledger={};p=r/'TRAINING_LOGS/resources.jsonl'
 if p.exists():
  rows=[json.loads(x) for x in p.read_text().splitlines()]
  if rows:
   for k in ['CPU_percent_one_core','RSS_bytes','GPU_percent','GPU_memory_MiB','GPU_power_W']:ledger[k]=dict(sample_mean=float(np.mean([x[k] for x in rows])),sample_max=float(max(x[k] for x in rows)))
   ledger['samples']=len(rows);ledger['cadence_seconds']=10
 save(r,'GPU_CPU_COMPUTE_LEDGER.md','# Actual compute\n\n'+block(dict(preparation=prep,arms=arms,evaluation=ev,latency=lat,resources=ledger,total_seconds=(done or {}).get('seconds')))+'\nRaw10second CPU,RAM,GPU utilization/memory/power samples:TRAINING_LOGS/resources.jsonl. CPU percent is relative to one core;child cumulative CPU is separately logged. Device samples are not exact per-kernel accounting or energy integration. Preparation is physically shared once and charged equally for training opportunities. Frozen provider costs/buffers are included;no sparse-FLOP claim or EIG-per-compute comparison across unmatched frozen baselines.\n')
 save(r,'NEXT_RESEARCH_DECISION.md','# Research decision\n\n'+decision+'\n\n'+block(dict(completion=done,evaluation=ev,primary=(eig or {}).get('primary')))+'\nEIG gains alone cannot establish useful specialization or intelligent routing. Non-significance is not equality. One lineage cannot establish ICML novelty or robust superiority. Preserve negative/incomplete results;no additional seeds,architectures,homogeneous control,SIR or grid campaigns automatically run.\n')
if __name__=='__main__':
 import sys
 write(Path(sys.argv[1]))
