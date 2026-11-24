"""Predeclared seed-level inference; partial results never imply campaign completion."""
import csv,json,pathlib,subprocess
import numpy as np
from scipy.stats import t
MAIN=pathlib.Path.home()/'Documents/objective_power_grid';OUT=MAIN/'experiments/location_finding/hec_v3_study_20261115';REPORT=MAIN/'reports/hec_v3_study_20261115';HERE=pathlib.Path(__file__).resolve().parent
P=json.loads((HERE/'protocol.json').read_text())
def interval(v):
 v=np.asarray(v,float);n=len(v);mean=float(v.mean());sd=float(v.std(ddof=1)) if n>1 else None;se=sd/np.sqrt(n) if sd is not None else None
 return dict(n=n,mean=mean,seed_sd=sd,seed_SE=se,CI95=[mean-float(t.ppf(.975,n-1))*se,mean+float(t.ppf(.975,n-1))*se] if n>1 else None,
  p_two_sided=float(2*t.sf(abs(mean/se),n-1)) if se else (None if n<2 else float(mean==0)))
def main(publish=False):
 rows=[];available={};stages=[]
 for seed in P['seeds']:
  for arm in P['arms']:
   run=OUT/f'seed{seed}_{arm}';path=run/'RESULTS.json'
   if not path.exists():continue
   result=json.loads(path.read_text())
   if result['status']!='COMPLETE':continue
   e=result['evaluations'][arm];resource_path=run/'RESOURCES.json';resources=json.loads(resource_path.read_text()) if resource_path.exists() else None
   rows.append(dict(seed=seed,arm=arm,mean=e['mean'],MCSE=e['MCSE'],CI95=e['CI95'],training_seconds=result['training_seconds'],total_seconds=result['total_seconds'],resources=resources,
     new_training_episodes=result['new_training_episodes'],new_training_observations=result['new_training_observations'],trainable_parameters=result['phases'][-1]['trainable_parameters'],
     online_amortized_seconds_per_episode=e['online_amortized_seconds_per_episode']))
   available[seed,arm]=np.load(run/(arm+'_scores.npy'))
   for stage,(means,sds) in enumerate(zip(e.get('stage_mean_weights',[]),e.get('stage_std_weights',[])),1):
    for k,(mu,sd) in enumerate(zip(means,sds),1):stages.append(dict(seed=seed,arm=arm,stage=stage,expert=k,mean_weight=mu,case_sd=sd))
 contrasts={};paired_cases={}
 for label,control in [('learned_minus_trained_uniform','uniform'),('learned_minus_v2_continuation','v2_control')]:
  differences=[];seed_records=[]
  for seed in P['seeds']:
   if (seed,'learned') in available and (seed,control) in available:
    v=available[seed,'learned']-available[seed,control];mean=float(v.mean());se=float(v.std(ddof=1)/np.sqrt(len(v)));critical=float(t.ppf(.975,len(v)-1))
    seed_records.append(dict(seed=seed,mean=mean,MCSE=se,CI95=[mean-critical*se,mean+critical*se]));differences.append(mean)
  if differences:contrasts[label]=dict(**interval(differences),all_five_seeds=len(differences)==5,positive_seeds=sum(v>0 for v in differences));paired_cases[label]=seed_records
 # Correct only the two locked primary hypotheses;incomplete collections are descriptive.
 if all(k in contrasts and contrasts[k]['all_five_seeds'] for k in ['learned_minus_trained_uniform','learned_minus_v2_continuation']):
  largest=0
  for j,k in enumerate(sorted(contrasts,key=lambda k:contrasts[k]['p_two_sided'])):
   largest=max(largest,min(1.,(2-j)*contrasts[k]['p_two_sided']));contrasts[k]['p_Holm']=largest
 interactions=[]
 for seed in P['seeds']:
  if all((seed,k) in available for k in ['learned','router_only','encoders_only','v2_control']):interactions.append(float((available[seed,'learned']-available[seed,'router_only']-available[seed,'encoders_only']+available[seed,'v2_control']).mean()))
 status='COMPLETE' if len(rows)==25 else 'PENDING_EXPERIMENTS'
 summary=dict(status=status,completed_runs=len(rows),required_runs=25,seed_scope=P['seed_scope'],rows=rows,primary_seed_contrasts=contrasts,within_seed_case_contrasts=paired_cases,
   secondary_factorial_interaction=interval(interactions) if interactions else None,scientific_limits=['Five-seed t inference has small-sample assumptions.','Shared pretrained V2/providers/replay;not independently pretrained lineages.',
   'Canonical benchmark has been observed during development;not untouched final test.','Case MCSE is conditional on fixed1M contrasts;finite-contrast bias remains.','Incomplete collections are descriptive and do not establish success.'])
 REPORT.mkdir(exist_ok=True);(REPORT/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
 if stages:
  with (REPORT/'ROUTING_STAGE_WEIGHTS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(stages[0]));w.writeheader();w.writerows(stages)
 text='# V3 routing validation\n\nStatus: '+status+f'. Completed {len(rows)}/25 sequential runs.\n\n'
 text+='Native full-horizon variational EIG only. Four-hour total cap per run. Five continuation seeds conditional on a common pretrained V2 parent. Test-bank selection is prohibited.\n\n'
 text+='| Seed | Arm | EIG | MCSE | Training minutes |\n|---|---|---:|---:|---:|\n'
 for r in rows:text+=f"| {r['seed']} | {r['arm']} | {r['mean']:.8f} | {r['MCSE']:.5f} | {r['training_seconds']/60:.1f} |\n"
 text+='\nPrimary across-seed differences: `'+json.dumps(contrasts)+'`.\n\n'
 text+='No campaign-level superiority claim is allowed until all five paired seeds are complete. Across-seed intervals and case-level intervals answer different questions. See SUMMARY.json for measured resources and uncertainty.\n'
 text+='\nFrozen-checkpoint diagnosis and original experiment verification are separate reports. Attribution factorial results are secondary;feature differences or varying weights alone do not establish useful specialization.\n'
 (REPORT/'RESULTS.md').write_text(text)
 if publish:publish_scoped()
 return summary

def publish_scoped():
 branch=subprocess.check_output(['git','-C',str(MAIN),'branch','--show-current'],text=True).strip();assert branch=='research/hec-v3-routing-validation-20261115',branch
 staged=subprocess.check_output(['git','-C',str(MAIN),'diff','--cached','--name-only'],text=True).strip();assert not staged,'Preserve unrelated staged work'
 dirty=subprocess.check_output(['git','-C',str(MAIN),'status','--porcelain','--','src/research/hec_v3_study'],text=True).strip();assert not dirty,'Never publish changed running source'
 for path in REPORT.iterdir():assert path.suffix in ['.json','.md','.csv'] and path.stat().st_size<1000000,path
 subprocess.run(['git','-C',str(MAIN),'add','--','reports/hec_v3_study_20261115'],check=True)
 patch=subprocess.check_output(['git','-C',str(MAIN),'diff','--cached','--','reports/hec_v3_study_20261115'],text=True)
 (OUT/'LATEST_REPORT_REVIEW.diff').write_text(patch)
 subprocess.run(['git','-C',str(MAIN),'diff','--cached','--check'],check=True)
 if patch:
  subprocess.run(['git','-C',str(MAIN),'commit','-m','Record V3 routing validation progress and measured results','--','reports/hec_v3_study_20261115'],check=True)
  subprocess.run(['git','-C',str(MAIN),'push','origin',branch],check=True)
 (OUT/'PUBLISHED_COMMIT.json').write_text(json.dumps(dict(commit=subprocess.check_output(['git','-C',str(MAIN),'rev-parse','HEAD'],text=True).strip(),source='LabPC main project'),indent=2)+'\n')
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--publish',action='store_true');args=p.parse_args();main(args.publish)
