"""Manuscript tables and figures from completed checkpoints only;no training calls."""
import csv,json,pathlib,math
import numpy as np
from scipy.stats import t
M=pathlib.Path.home()/'Documents/objective_power_grid';I=M/'experiments/location_finding/hec_v3_study_20261115';X=M/'experiments/location_finding/moe_sboed_location_benchmark_20261115';R=M/'reports/moe_sboed_location_benchmark'
METHODS=['MoE-sBOED','MoE-sBOED-uniform',*[f'MoE-sBOED-E{k}' for k in range(1,5)],'DAD','vsOED','RL-sBOED','ScoreBED']
SEEDS=[101,102,103,104,105]
def put(p,v):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');q.replace(p)
def distribution(v):
 n=len(v);a=np.asarray(v,float);mu=float(a.mean()) if n else None;sd=float(a.std(ddof=1)) if n>1 else None
 w=float(t.ppf(.975,n-1))*sd/math.sqrt(n) if n>1 else None
 return dict(n=n,mean=mu,SD=sd,mean_CI95=[mu-w,mu+w] if w is not None else None,complete_five=n==5)
def main():
 R.mkdir(exist_ok=True);records={};arrays={};inventory=[];costs=[]
 def add(method,seed,d,values,path,checkpoint,training=None):
  assert len(values)==2000 and np.isfinite(values).all();s=dict(method=method,seed=seed,EIG=float(np.mean(values)),MCSE=float(np.std(values,ddof=1)/math.sqrt(2000)),source=str(path),extracted=method.startswith('MoE-sBOED-E'))
  records[method,seed]=s;arrays[method,seed]=values;inventory.append(dict(method=method,seed=seed,result=str(path),checkpoint=checkpoint,checkpoint_exists=pathlib.Path(checkpoint).exists() if checkpoint else False,
   logs=[str(p) for p in path.parent.glob('*.log')],scope='Extracted inherited expert' if s['extracted'] else ('Shared-pretraining continuation' if method.startswith('MoE-sBOED') else 'Native external training')))
 for seed in SEEDS:
  for arm,method in [('learned','MoE-sBOED'),('uniform','MoE-sBOED-uniform')]:
   root=I/f'seed{seed}_{arm}';p=root/'RESULTS.json'
   if not p.exists():continue
   d=json.loads(p.read_text())
   if d['status']!='COMPLETE':continue
   add(method,seed,d,np.load(root/(arm+'_scores.npy')),p,str(root/'joint_actor.pt'))
   res=json.loads((root/'RESOURCES.json').read_text()) if (root/'RESOURCES.json').exists() else {}
   costs.append(dict(method=method,seed=seed,training_wall_seconds=d['training_seconds'],new_training_episodes=d['new_training_episodes'],simulator_observations=d['new_training_observations'],resources=res,
    inherited_parent='16001 native actor updates,16,021,000 known episodes;shared across these continuation seeds;provider/preparation costs remain disclosed separately.'))
   q=root/'standalone_experts/RESULTS.json'
   if arm=='learned' and q.exists():
    e=json.loads(q.read_text())
    if e['status']=='COMPLETE':
     for name,v in e['experts'].items():add(name,seed,v,np.load(q.parent/(name+'_scores.npy')),q,str(q.parent/(name+'.pt')))
 for impl,name in [('dad','DAD'),('vsoed','vsOED'),('rl_sboed','RL-sBOED'),('scorebed','ScoreBED')]:
  for seed in SEEDS:
   root=X/f'external_evaluations/{impl}/seed{seed}';p=root/'RESULTS.json'
   if not p.exists():continue
   d=json.loads(p.read_text())
   if d['status']!='COMPLETE':continue
   add(name,seed,d,np.load(root/'scores.npy'),p,d['checkpoint']['checkpoint'])
   costs.append(dict(method=name,seed=seed,training_wall_seconds=d['checkpoint']['offline_seconds'],evaluation_seconds=d['evaluation_seconds'],online_amortized_seconds_per_episode=d['online_amortized_seconds_per_episode'],
     simulator_observations_evaluation=d['simulator_observations'],cpu_evaluation_user_seconds=d['cpu_user_seconds'],torch_peak_allocated_bytes=d['torch_peak_allocated_bytes'],training_receipt=d['checkpoint']['training_receipt'],
     missing_metrics='Historical CPU/GPU training utilization is unavailable unless provided by original receipt;never imputed as zero.'))
 summaries={m:distribution([records[m,s]['EIG'] for s in SEEDS if (m,s) in records]) for m in METHODS};comparisons={};within={}
 for other in METHODS[1:]:
  pairs=[];cases=[]
  for seed in SEEDS:
   if ('MoE-sBOED',seed) not in arrays or (other,seed) not in arrays:continue
   delta=arrays['MoE-sBOED',seed]-arrays[other,seed];v=distribution(delta);pairs.append(float(delta.mean()));cases.append(dict(seed=seed,mean=float(delta.mean()),case_MCSE=float(delta.std(ddof=1)/math.sqrt(2000)),case_CI95=v['mean_CI95']))
  comparisons[other]=dict(**distribution(pairs),positive_seeds=sum(v>0 for v in pairs),status='COMPLETE' if len(pairs)==5 else 'PENDING')
  if len(pairs)==5:
   v=np.asarray(pairs);se=float(v.std(ddof=1)/math.sqrt(5));mu=float(v.mean());comparisons[other]['p_two_sided']=float(2*t.sf(abs(mu/se),4)) if se else float(mu==0)
   w=float(t.ppf(1-.05/(2*9),4))*se;comparisons[other]['simultaneous_Bonferroni_CI95']=[mu-w,mu+w]
  within[other]=cases
 if all(v['status']=='COMPLETE' for v in comparisons.values()):
  largest=0
  for j,name in enumerate(sorted(comparisons,key=lambda k:comparisons[k]['p_two_sided'])):
   largest=max(largest,min(1,(9-j)*comparisons[name]['p_two_sided']));comparisons[name]['p_Holm']=largest
 best=None
 if all(summaries[f'MoE-sBOED-E{k}']['complete_five'] for k in range(1,5)):
  best=max([f'MoE-sBOED-E{k}' for k in range(1,5)],key=lambda m:summaries[m]['mean']);best=dict(method=best,comparison=comparisons[best],selection='Best five-seed mean is descriptive;inference uses simultaneous family protection across all expert comparisons,not an unadjusted selected-expert test.')
 status='COMPLETE' if len(records)==50 else 'PENDING_EXPERIMENTS'
 put(R/'BENCHMARK_RESULTS.json',dict(status=status,completed_method_seed_rows=len(records),required_rows=50,training_families=6,extracted_expert_families=4,records=list(records.values()),summaries=summaries,
  primary_comparisons=comparisons,within_seed_case_comparisons=within,strongest_expert=best,evaluation_seed=202611099206,
  caveats=['MoE runs share pretrained initialization;these are continuation-seed robustness,not independent from-scratch training.','MCSE conditions on the shared1M contrast bank;training-seed uncertainty is separate.','Original bank informed historical development;not an untouched final test.','Extracted experts inherit learned training;zero extra training does not mean zero inherited cost.','Native external objectives/schedules/compute differ;no perfect compute-matching claim.']))
 put(R/'CHECKPOINT_INVENTORY.json',inventory);put(R/'COMPUTE_FAIRNESS.json',costs)
 put(R/'CHECKPOINT_STATUS.json',[dict(method=method,seed=seed,status='COMPLETE' if (method,seed) in records else 'PENDING',role='Extracted inherited expert' if '-E' in method else 'Trained policy') for method in METHODS for seed in SEEDS])
 put(R/'PROTOCOL_STATUS.json',dict(status=json.loads((R/'BENCHMARK_RESULTS.json').read_text())['status'],internal=json.loads((M/'configs/moe_sboed_resume_20261115.json').read_text()),evaluation_seed=202611099206,primary_method_seed_rows=50,training_families=6,extracted_expert_families=4))
 with (R/'MAIN_RESULTS.csv').open('w',newline='') as f:
  w=csv.writer(f,lineterminator='\n');w.writerow(['Method',*SEEDS,'Mean ± SD','n','Role'])
  for m in METHODS:
   s=summaries[m];mu=f"{s['mean']:.5f} ± {s['SD']:.5f}" if s['complete_five'] else 'PENDING'
   w.writerow([m,*[f"{records[m,k]['EIG']:.5f}" if (m,k) in records else 'PENDING' for k in SEEDS],mu,s['n'],'Extracted inherited expert' if '-E' in m else 'Trained policy'])
 text='# MoE-sBOED location-finding benchmark\n\nStatus: '+status+f'; {len(records)}/50 method-seed evaluations complete.\n\n'
 text+='| Method | 101 | 102 | 103 | 104 | 105 | Mean ± SD |\n|---|---:|---:|---:|---:|---:|---:|\n'
 for m in METHODS:
  s=summaries[m];cells=[f"{records[m,k]['EIG']:.5f}" if (m,k) in records else 'PENDING' for k in SEEDS];mu=f"{s['mean']:.5f} ± {s['SD']:.5f}" if s['complete_five'] else 'PENDING';text+='| '+m+' | '+' | '.join(cells)+' | '+mu+' |\n'
 text+='\nE1–E4 are extracted standalone policies from the matching learned seed,not independently trained experts. MoE seeds are conditional on common pretraining. No superiority claim follows from a higher point estimate. Primary paired intervals and Holm-adjusted tests require all five seed pairs. Case MCSE and across-seed SD are separate in BENCHMARK_RESULTS.json.\n'
 text+='\nThe controller preserves uniform seed101 and excludes all15 development-ablation training jobs. New internal training uses only native variational EIG;native external methods retain their own legitimate objectives. Four-hour total cap includes each training run and its required evaluations.\n'
 text+='\nReproduce evaluation with the exact original bank seed202611099206,2000 truth/noise cases,1M contrasts,T30 and frozen scorer. Never substitute1001 or choose checkpoints with this bank. Source/config/checkpoint paths are in CHECKPOINT_INVENTORY.json;compute records are in COMPUTE_FAIRNESS.json.\n'
 (R/'RESULTS.md').write_text(text)
 try:
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  for filename,title,names in [('five_seed_comparison','Five-seed EIG comparison',METHODS),('learned_vs_uniform','Learned versus uniform routing',METHODS[:2]),('expert_comparison','Collaboration versus extracted experts',[METHODS[0],*METHODS[2:6]])]:
   fig,ax=plt.subplots(figsize=(max(7,len(names)*1.05),4.8));xs=np.arange(len(names));values=[summaries[n]['mean'] or 0 for n in names];ax.bar(xs,values,color=['#2463a6' if n=='MoE-sBOED' else '#81a7bc' for n in names])
   for x,n,y in zip(xs,names,values):
    s=summaries[n]
    if s['complete_five']:
     ci=s['mean_CI95'];ax.errorbar(x,y,yerr=[[y-ci[0]],[ci[1]-y]],fmt='none',color='black',capsize=4)
    ax.text(x,max(y,0)+.25,f"n={s['n']}/5" if s['n'] else 'PENDING',ha='center',fontsize=8)
   ax.set_xticks(xs,names,rotation=35,ha='right');ax.set_ylabel('Full-horizon EIG (nats)');ax.set_ylim(min(0,min(values)-.5),15.5);ax.set_title(title+'\n95% seed-mean intervals only when all five seeds are complete',fontsize=10);fig.tight_layout();fig.savefig(R/(filename+'.png'),dpi=220);plt.close(fig)
 except ImportError as e:put(R/'FIGURE_DEPENDENCY.json',dict(status='PENDING',error=str(e)))
 print(json.dumps(dict(status=status,completed_rows=len(records))))
if __name__=='__main__':main()
