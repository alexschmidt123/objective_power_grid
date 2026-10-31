
"""Evidence-only per-seed reports; publishing is performed through LabPC."""
import json,csv,hashlib,math
from pathlib import Path
import numpy as np
from src.research.deepseek_sboed_experiment import write
from src.research.v1_specialization_experiment import paired

def write_results(C,cfg):
 seed=cfg['training_seed'];R=C.parents[2]
 official=json.loads((C/'reports/official_seed1001_result.json').read_text())
 confirmation=json.loads((C/'reports'/f"confirmation_seed{cfg['confirmation_seed']}_result.json").read_text())
 comparisons=json.loads((C/'reports/official_paired_comparisons.json').read_text())
 development=json.loads((C/'reports/development_comparisons.json').read_text())
 costs=json.loads((C/'costs/full_final.json').read_text());latency=json.loads((C/'reports/inference_latency.json').read_text())
 features=json.loads((C/'reports/expert_features.json').read_text())['stages']
 counts=costs['clock']['counts'];context=json.loads((C/'reports/context_competence.json').read_text()) if (C/'reports/context_competence.json').exists() else {}
 full=np.load(C/'evaluations/development_full.npy')
 routing=paired(full,np.load(C/'evaluations/development_uniform_subset.npy'))
 contribution=paired(full,np.load(C/'evaluations/development_mask_routed.npy'))
 # Multiple-comparison-controlled context reversals, using identical fixed-prefix worlds.
 reversals=[]
 for row in context.get('records',[]):
  stage=row['stage'];p=C/'evaluations'/f'context_competence_stage{stage}.npz'
  if not p.exists():continue
  data=np.load(p);scores=data['scores'];groups=data['group'];sets=row.get('tested_sets',[])
  if len(sets)!=35 or any((groups==c).sum()<32 for c in [0,1]):continue
  z=next(x['familywise_z'] for x in row['contexts'] if 'familywise_z' in x)
  for a in range(35):
   for b in range(a+1,35):
    intervals=[]
    for c in [0,1]:
     d=scores[a,groups==c]-scores[b,groups==c];mean=float(d.mean());se=float(d.std(ddof=1)/len(d)**.5);intervals.append([mean-z*se,mean+z*se])
    if (intervals[0][0]>0 and intervals[1][1]<0) or (intervals[1][0]>0 and intervals[0][1]<0):
     reversals.append(dict(stage=stage,sets=[sets[a],sets[b]],intervals=intervals))
 feature_changed=bool(np.mean([r['expert_function_change_L2'] for r in features])>1e-8)
 route_positive=routing['CI95'][0]>0
 paired_v1_positive=comparisons['v1']['CI95'][0]>0
 dense_positive='dense' in comparisons and comparisons['dense']['CI95'][0]>0
 shared_positive='shared_only' in comparisons and comparisons['shared_only']['CI95'][0]>0
 routed_positive=contribution['CI95'][0]>0
 supported=bool(feature_changed and reversals and route_positive and paired_v1_positive and dense_positive and shared_positive and routed_positive)
 summary=dict(method=cfg['method'],training_seed=seed,EIG=official['EIG'],MC_SE=official['MC_SE'],confirmation_EIG=confirmation['EIG'],confirmation_MC_SE=confirmation['MC_SE'],updates=costs['executed_updates'],selected_update=costs['selected_update'],training_seconds=costs['clock']['charged_seconds'],specialization_seconds_within_training=counts.get('specialization_step_microseconds',0)/1e6,specialization_adjustments=counts.get('specialization_expert_adjustments',0),expert_effective_updates=costs.get('expert_effective_updates'),learned_minus_uniform=routing,full_minus_masked_routed=contribution,context_reversals=reversals,shared_only_control_advantage=shared_positive,routed_contribution_advantage=routed_positive,meaningful_specialization_supported_on_audited_data=supported,online_latency=latency,checkpoint_sha256=costs['checkpoint_sha256'],single_seed=True,scope='Two seeds on different GPUs are not a pure seed-only replication; trajectory MC SE is not across-training-seed uncertainty')
 write(C/'reports/final_summary.json',summary)
 evidence=C/'reports/evidence';evidence.mkdir(exist_ok=True)
 for name in ['official_seed1001','confirmation_seed'+str(cfg['confirmation_seed'])]:
  values=np.load(C/'evaluations'/f'{name}_scores.npy')
  with (evidence/(name+'_trajectory_scores.csv')).open('w',newline='') as f:
   w=csv.writer(f,lineterminator='\n');w.writerow(['trajectory_id','information_nats']);w.writerows(enumerate(values))
 table="| Reference | New minus reference | Paired MC SE | Approx95% CI |\n|---|---:|---:|---|\n"
 for name,x in comparisons.items():table+=f"|{name}|{x['difference']:+.8f}|{x['paired_MC_SE']:.8f}|{x['CI95']}|\n"
 text=f"""# DeepSeekMoE-v1 contextual expert learning: seed{seed}

Official EIG **{official['EIG']:.8f} +/- {official['MC_SE']:.8f} nats**, seed1001,N2000,L1000000,H30.
Independent confirmation seed{cfg['confirmation_seed']}: **{confirmation['EIG']:.8f} +/- {confirmation['MC_SE']:.8f}**.
All uncertainties are trajectory MC SE conditional on fixed contrast banks and this trained checkpoint.

{table}
Architecture is exactly the original v1 DeepSeekPolicy:8434 parameters,1shared+7routed SwiGLU experts,top3,unchanged encoder/head. The only learning change is context-protected,norm-preserving routed-expert gradients. Main initialization was fresh seed{seed}. No pilot weights,teachers,IR/RC/SA/PIMoE architecture/training code or competitor policies were used.

Main updates{costs['executed_updates']}; selected{costs['selected_update']}; charged seconds{costs['clock']['charged_seconds']:.6f} of14400.
Measured specialization subset{summary['specialization_seconds_within_training']:.6f}s INSIDE that allowance; no additional simulator forward calls are introduced by the two extra context backward passes.
Training trajectories/counts:{json.dumps(counts)}. Expert effective update counters:{summary['expert_effective_updates']}.
No raw-SGD descent alignment check guarantees a finite Adam step improves EIG. No statistical guarantee of superiority is made.

Learned minus uniform-subset development routing:{json.dumps(routing)}.
Full minus routed-masked:{json.dumps(contribution)}.
Development comparisons:{json.dumps(development,indent=2)}.
Frozen interventions retain trained weights and are not retrained controls. Dense/shared-only reference controls come from previous teacher-free training and are loaded through the original v1 architecture variants; their recorded training/throughput differences limit strict causal claims.

Useful specialization supported by all audited criteria:{supported}.
Function changes:{feature_changed}; context reversals after correction:{len(reversals)}; learned routing advantage:{route_positive}; paired improvement over v1:{paired_v1_positive}; improvement over dense:{dense_positive}; improvement over shared-only:{shared_positive}; routed contribution:{routed_positive}.
Different expert weights/features/utilization alone are insufficient. Full-horizon context comparisons use shared-prefix policies and independent joint-distribution histories. A generating theta is NOT treated as a fixed-history posterior. Context classifiers come from training only; exploratory contextual set selection is confirmed on disjoint development cases. Results conditional on small cohorts or incomplete combination coverage remain limited.

Online latency:{json.dumps(latency)}. Deployment is unchanged frozen v1 inference with no online learning,MC search,teacher or numerical design optimization.
Checkpoint:{C}/models/full/selected.pt
SHA256:{costs['checkpoint_sha256']}
Foundation:13681d7091189a4e68056e6a2864dc58f9d5700b.
Source revision:{json.loads((C/'reports/checkpoint_freeze.json').read_text())['source_commit']}
"""
 (C/'reports'/f'DEEPSEEK_V1_SPECIALIZATION_SEED{seed}_RESULTS.md').write_text(text)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 freq=np.array([r['utilization'] for r in features]).T
 im=axes[0].imshow(freq,aspect='auto',origin='lower',vmin=0,vmax=1);axes[0].set(xlabel='Stage',ylabel='Routed expert',title='Supporting utilization diagnostic');fig.colorbar(im,ax=axes[0])
 cos=np.mean([r['cosine'] for r in features],0)
 im=axes[1].imshow(cos,vmin=-1,vmax=1,cmap='coolwarm');axes[1].set(xlabel='Expert',ylabel='Expert',title='Learned feature similarity');fig.colorbar(im,ax=axes[1]);fig.tight_layout();fig.savefig(evidence/'expert_functions.png',dpi=170);plt.close(fig)
 rows=[];labels=[]
 for row in context.get('records',[]):
  for c in row.get('contexts',[]):
   if c.get('status')=='FINITE_JOINT_GROUP_AVERAGE' and len(c['advantage_over_learned'])==35:
    rows.append(c['advantage_over_learned']);labels.append(f"t{row['stage']}/context{c['context']}")
 if rows:
  matrix=np.array(rows);limit=max(.001,float(np.max(abs(matrix))))
  fig,ax=plt.subplots(figsize=(15,4));im=ax.imshow(matrix,aspect='auto',cmap='coolwarm',vmin=-limit,vmax=limit);ax.set_yticks(range(len(labels)));ax.set_yticklabels(labels);ax.set(xlabel='Expert triple (lexicographic index)',title='Paired full-horizon information advantage within observed-history groups');fig.colorbar(im,ax=ax,label='Nats');fig.tight_layout();fig.savefig(evidence/'context_competence.png',dpi=170);plt.close(fig)
 write(evidence/'manifest.json',{str(p.relative_to(evidence)):hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence.iterdir() if p.is_file() and p.name!='manifest.json'})
 print('SEED_COMPLETE',json.dumps(summary),flush=True)
