"""Evidence-derived V4 scientific reports; no training or result selection."""
import json,math
from pathlib import Path
import numpy as np
NAMES=['INFORMATION_DECOMPOSITION','HYPOTHESIS_PARTITION','EXPERT_ARCHITECTURE','ROLE_TRAINING','MATHEMATICAL_TESTS','PILOT_RESULTS','EXPERT_SPECIALIZATION','V1_COMPARISON']
def initial_reports(C,cfg):
 partition=json.loads((C/'provenance/partition_validation.json').read_text());alpha=json.loads((C/'readiness/alpha_calibration.json').read_text());gpu=json.loads((C/'readiness/gpu_smoke.json').read_text())
 reports={
 'INFORMATION_DECOMPOSITION':r'''# V4 information decomposition

DEVELOPMENT_UNVALIDATED. C=f(theta) is a fixed prior-predictive partition frozen before policy training. For a realized observed history h_t, let Y=y_(t+1), Z=Y_(t+2:T) under the actual continuation policy. Because C is a deterministic function of theta:

I(theta;Y,Z|h)=I(C;Y|h)+I(theta;Y|C,h)+I(C;Z|h,Y)+I(theta;Z|C,h,Y).

Here I(theta;Y|C,h) integrates over C~p(C|h); future coarse information integrates over Y~p(Y|h,pi); future fine information integrates jointly over(C,Y)|h. The fixed continuation policy is included in every predictive distribution. Conditional information given random Y is an expectation, not a value evaluated at one unintegrated future observation. At the final stage Z is empty and both future terms are exactly zero. No four independent MI objectives are invented.

Finite particles: theta0 is the generating draw, theta1:L are iid prior contrasts. A fresh bank is shared across outer cases; it remains independent of each outer truth/noise. Let M=L+1, ell_j(k)=sum_(s<=k)log p(y_s|theta_j,d_s), logq_j=ell_j-logsumexp ell. Let n_c count the AUGMENTED prior bank including truth and pi_c=n_c/M. Then Jtotal(k)=logM+logq0(k), Jcoarse(k)=logsumexp_(j:Cj=C0)ell_j-logsumexp ell-log pi_C0, Jfine=Jtotal-Jcoarse. Using the same augmented bank makes Jtotal(0)=Jcoarse(0)=Jfine(0)=0 and avoids empty true groups.

E1=Jcoarse(t+1)-Jcoarse(t); E2=Jfine(t+1)-Jfine(t); E3=Jcoarse(T)-Jcoarse(t+1); E4=Jfine(T)-Jfine(t+1). Pointwise E1+E2+E3+E4=Jtotal(T)-Jtotal(t). Negative realized values are NEVER clipped. All reductions stay in log space. This algebraic identity is exact for the implemented densities; the densities are finite self-normalized MC proxies, not proven unbiased continuous-model MI estimators. Prior-group coverage, posterior ESS, ceiling saturation and independent contrast-sensitivity audits are retained.
''',
 'HYPOTHESIS_PARTITION':'''# Frozen prior-predictive partition

Domain adapter src/research/moe_v4_partition.py: sample8192 prior parameters; predict noise-free observation means on a fixed5x5 grid over the existing feasible location box; standardize by sqrt(prior predictive variance+observation noise variance); kmeans++ initialization and deterministic Lloyd updates into4groups. No held-out EIG chooses the grid, partition, or thresholds. A separate4096-prior validation set checks every group>=5%,between variance>=5% and within-group variance>=1% of total. Source-label exchange leaves signatures and labels unchanged. Groups are domain-dependent predictive hypotheses, not physical parameter-coordinate rules or named strategies.

Exactly the SAME four information roles apply to every partition. E1 does not belong to group0; it predicts discrimination among ALL groups. SIR references are16 fixed increasing feasible times; grid references use the canonical configured feasible sequence with the existing physical backend and required configuration metadata. Distinct domain partitions and weights are expected, not zero-shot transfer.

Actual independent validation:\n\n'''+json.dumps(partition,indent=2)+'\n',
 'EXPERT_ARCHITECTURE':'''# Four-expert architecture

src/research/moe_sboed_v4.py reuses the pinned v1 compact SwiGLU feature expert and raw-observation256ReLU->16 sum-history principle, followed by16-dimensional LayerNorm and known stage embedding. Domain scales/dimensions are adapters; no named physical parameters or true theta enter the general expert classes.

Exactly FOUR independently parameterized width16 feature experts, each with its own linear role-prediction head. NO always-active shared FFN expert. History encoding is shared, not a fifth expert. Router(history)->top2distinct experts; selected-softmax weights renormalized to one. Only selected rows execute in the policy forward pass. Unselected identity slots are zero; history plus four16-dimensional slots enter a nonlinear joint LayerNorm/32SiLU/action head, then the unchanged benchmark feasible map. No averaged physical action proposals, online EIG search, teacher, posterior numerical optimization, or deployment gradient update.

Offline auxiliary training evaluates all four role predictors on detached history embeddings. It is intentionally not sparse auxiliary computation and is charged. Deployment never executes prediction heads. Four role targets, not interchangeable names or weight diversity, define responsibilities. Predictions alone still do not establish causal competence.

Uniform control samples a uniform subset of two experts; BOTH learned and uniform-subset policies use the SAME selected-softmax normalization. This tests selection while preserving K and gate-weight semantics. Two independent fixed routing realizations are used for frozen interventions. Sparse membership is discrete; derivatives are exact conditional on the selected set, with no derivative for membership changes. This is a piecewise pathwise approximation, not an unbiased categorical routing estimator.
''',
 'ROLE_TRAINING':'''# Proper role-specific learning

Main loss is -mean(terminal full-horizon sequential sPCE). All30 measurements and complete simulator/history gradients contribute. Each expert has its own realized information-density target and proper squared prediction loss. Targets and the entire shared history embedding supplied to auxiliary predictors are stop-gradient. Thus role loss e updates ONLY expert e's private feature parameters and prediction head; encoder/router/design head and other experts receive no role-e auxiliary gradient. The main sPCE gradient still updates all selected experts, router weights, shared encoder and joint action head through the complete sequential simulator path.

Prediction targets condition on observed history under the current on-policy action and continuation policy. They may use hidden parameters and future observations offline ONLY. Targets move as the policy learns; this is a conditional value-prediction mechanism, not a proof that an expert causally chooses better experiments. Future prediction outputs have an analytic zero boundary at the final stage; expert routing itself is not manually assigned to early/late stages.

Per-role mean/RMS normalization is frozen from a pretraining-only diagnostic (scale floor.1nats); no realized target is clipped. Alpha is min(1,.1*primary-expert-gradient-norm/auxiliary-expert-gradient-norm), measured once before any held-out performance evaluation. This is an explicitly heuristic gradient-scale choice, not an MI identity or a guarantee of long-run gradient balance. The no-role-loss control tests whether it helps EIG. No router diversity/usage regularizer is imposed.

Measured calibration:\n\n'''+json.dumps(alpha,indent=2)+'\n',
 'MATHEMATICAL_TESTS':'''# Mathematical and numerical validation

24CPU/reference tests passed;2GPU-dependent grid checks skipped in the deliberately CPU-only suite. They were then run under the exclusive GPU lock using conda's nvcc. IEEE9/14 physical predictions and action-gradient finite differences pass after supplying their canonical _config_path metadata; the initial missing-metadata failures are preserved, not hidden. No grid or SIR simulator was altered. SIR T5 forward/backward/chronology tests pass. No SIR/grid V4 training is launched.

Tests cover exact finite-bank four-term identity, negative realized targets, zero final future, log-shift stability, brute-force exact MI in an adaptive finite discrete model, frozen partition variation/source-exchangeability, v1 history encoder parity, unchanged DAD-rollout objective AND gradient parity, complete early-action gradients, expert-only auxiliary gradients, sparse top2execution, router finite differences, deployment/checkpoint reproducibility and dense parameter matching. The finite discrete enumeration validates the estimator only for its exhaustive finite prior; it does not prove unbiased continuous-MI estimation.

Full GPU shape B32/L131072 passed for all matched arms and contextual gradient protection; B2000/L2000 passed separately for the native reference. Streamed role-target/prefix intervention integration is recorded in readiness/evaluation_integration.json. Auxiliary setup/smoke costs are conservatively charged to the corresponding pilot arms; source/partition/config/checkpoint hashes and hardware provenance accompany them.

GPU measurements:\n\n'''+json.dumps(gpu,indent=2)+'\n',
 'PILOT_RESULTS':'''# V4 bounded pilot

PENDING actual pilot results. No EIG gain or useful specialization claimed.

Initial target8192 optimizer updates. Each seed/policy arm has7200charged GPU-training seconds including initialization, readiness training/preparation, all trainable/auxiliary phases and checkpoint serialization. Seed101:V4,contextual-v1,no-role,V4-uniform-top2,parameter-matched dense. Seed102:V4/contextual-v1 primary replication. Supplemental seed101 contextual-v1 uses originalB2000/L2000 recipe; never pooled as training-matched. All primary arms use B32/L131072,Adam5e-5 betas(.8,.998),clip5,decay.98 after1,1001,... . These batch/contrast changes are explicit pilot adaptations, not the historical native training exposure. Training ceiling log131073≈11.78351nats versus log2001≈7.60140. Same expected shared-bank finite-L objective, different outer-case gradient covariance relative to native independent banks.

Checkpoint selection is highest common budget-eligible saved update, not largest EIG. Time-budget comparisons use latest eligible checkpoints not exceeding each shared wall ceiling; actual timestamps/update counts are reported. Completed native pilot schedules need not spend the full7200seconds. Source/config are frozen before launch. Only detached guarded processes execute; no automatic failure restart or deletion of partial/negative evidence.
''',
 'EXPERT_SPECIALIZATION':'''# Expert specialization evidence

PENDING trained-policy diagnostics. Four training targets are mathematically distinct components, not yet four proven useful information-acquisition experts.

Held-out diagnostics: per-role squared error/R2/calibration bins, realized-target correlations, negative-value frequencies, full-history expert selection, physical next-action change under individual masks, and paired full-horizon EIG under masks. Learned versus uniform top2 preserves selected-softmax gate semantics and active count, with two random routing realizations. Controlled expert-function SWAPS at stages5,15,25 preserve two distinct expert functions, gate weights and the entire common prefix; the original policy resumes afterward.

Observable context definitions are predicted coarse-share>=.1 and predicted immediate-share>=1/(remaining decisions), conditional on a positive predicted total. They are fixed information-based definitions, not median-feature responsibilities or forced stage assignments. Each swap/context interaction needs>=16cases per cohort; missing coverage is reported rather than forced. Holm correction covers eligible tests in the prespecified36-test interaction family. Effects include nonlinear fusion/slot interactions, not additive expert EIG contributions. Replication across two independently trained seeds is required for a stable attribution pattern. Correlated targets, poor prediction calibration or unreplicated utility mean the four intended roles are not established even if EIG improves.
''',
 'V1_COMPARISON':'''# V1 comparison and claim boundaries

Pinned contextual-gradient foundation042915bfe8312faa333b1b825b94766c00c68dbf; original v1 foundation13681d7091189a4e68056e6a2864dc58f9d5700b. Recovered helper/trainer/evaluator source is byte-identical, and the active deepseek_sboed.py equals that pinned policy. Reference training retains its median-cohort gradient PROTECTION every10updates at stages1,10,20,29, native balance coefficient.001, encoder, optimizer, decay and clipping. V4 DOES NOT use the median split to define expertise; only the reference retains its original engineering mechanism.

Primary comparison uses the SAME B32,L131072,seed pairing,physical model and evaluation arrays. This is a matched adapted-recipe comparison, not proof of superiority over the originalB2000/L2000 training protocol. A fresh native-recipe contextual reference is evaluated separately under the same compute ceiling. Historical11.04404166±.04281140(seed101),10.83415593±.04301131(seed102) remain references with different training/evaluation exposure, not evidence for expert specialization or pooled test samples.

Development milestones use fresh256-case contrasts262144. Final frozen checkpoints use1024paired cases/1Mcontrasts; independent training-seed102 has its own paired evaluation randomness. Official seed1001 and previous official confirmation sets are unused. Role diagnostics and contrast sensitivity use separate fresh cases. Primary performance success needs positive paired95% lower bounds for both matched seed comparisons; native-recipe supplemental outcomes and all limitations remain explicit. No STABLE promotion, GitHub push, automatic merge or new MoE variant follows from this pilot.
'''}
 for name,text in reports.items():(C/'reports'/f'V4_{name}.md').write_text(text)

def final_reports(C,cfg,results,compute,decision):
 text='# V4 pilot results\n\nDEVELOPMENT_UNVALIDATED. Decision: '+decision+'. One frozen partition and two training seeds; MC intervals condition on training runs and contrast banks.\n\n|Seed|Policy|Common updates|EIG nats|MCSE|\n|---|---|---:|---:|---:|\n'
 for seed,data in results.items():
  for arm,r in data['arms'].items():text+=f"|{seed}|{arm}|{data['common_update']}|{r['mean']:.6f}|{r['MC_SE']:.6f}|\n"
  d=data['paired_v4_minus_v1'];text+=f"\nSeed{seed} pairedV4−V1: {d['mean']:.6f}±{d['MC_SE']:.6f};95%CI{d['CI95']}.\n"
 text+='\nB32/L131072 adapted-recipe results are separate from supplemental nativeB2000/L2000. No best-EIG checkpoint selection. See final_comparison.json, matched_compute.json, contrast_sensitivity.json and per-seed role/utility diagnostics. No official promotion or causal four-role claim is automatic.\n';(C/'reports/V4_PILOT_RESULTS.md').write_text(text)
 (C/'reports/V4_V1_COMPARISON.md').write_text((C/'reports/V4_V1_COMPARISON.md').read_text()+'\n\nFinal measured comparison:\n\n'+text)
 calibration={};utility={}
 for seed in ['101','102']:
  p=C/f'evaluations/roles_seed{seed}/role_calibration.json';q=C/f'evaluations/roles_seed{seed}/utility_diagnostics.json'
  if p.exists():calibration[seed]=json.loads(p.read_text())
  if q.exists():utility[seed]=json.loads(q.read_text())
 (C/'reports/V4_EXPERT_SPECIALIZATION.md').write_text((C/'reports/V4_EXPERT_SPECIALIZATION.md').read_text()+'\n\nMeasured role calibration:\n\n'+json.dumps(calibration,indent=2)+'\n\nMeasured causal interventions:\n\n'+json.dumps(utility,indent=2)+'\nUseful E1–E4 specialization requires scientific inspection/replication; prediction accuracy or unequal routing alone is insufficient.\n')
 (C/'reports/V4_ROLE_TRAINING.md').write_text((C/'reports/V4_ROLE_TRAINING.md').read_text()+'\n\nHeld-out role prediction results:\n\n'+json.dumps(calibration,indent=2))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(8,4));names=[];means=[];errors=[]
 for seed,data in results.items():
  for arm,r in data['arms'].items():names.append(arm);means.append(r['mean']);errors.append(1.96*r['MC_SE'])
 ax.bar(names,means,yerr=errors);ax.set_ylabel('Full-horizon EIG,nats;95%MC interval');ax.tick_params(axis='x',rotation=35);fig.tight_layout();fig.savefig(C/'reports/V4_EIG_comparison.png',dpi=180);fig.savefig(C/'reports/V4_EIG_comparison.svg');plt.close(fig)
