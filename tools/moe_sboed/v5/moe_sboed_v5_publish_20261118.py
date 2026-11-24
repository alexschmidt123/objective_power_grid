"""CPU-only terminal verification and scoped publication for the one V5 pilot."""
from pathlib import Path
import hashlib,json,subprocess,sys
import numpy as np
M=Path.home()/'Documents/objective_power_grid';O=M/'experiments/location_finding/moe_sboed_v5_20261118';R=M/'reports/moe_sboed_v5_20261118';B='research/hec-v3-routing-validation-20261115'
def read(p):return json.loads(Path(p).read_text())
def put(name,data):(R/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
 return h.hexdigest()
def main():
 state=read(O/'STATUS.json');assert state['status'] in ['COMPLETE','FAILED_OR_CAPPED']
 subprocess.run([sys.executable,'-B',str(M/'tools/moe_sboed_v5_20261118.py'),'--task','report'],cwd=M,check=True)
 for path,value in read(O/'SOURCE_MANIFEST.json').items():assert sha(M/path)==value,path
 d=read(R/'RESULTS.json');manifest=[];compute={};inits={};training_counts=dict(actor=0,critic=0,posterior=0,new_episodes=0)
 for arm in ['learned','uniform']:
  p=O/arm/'RESULTS.json'
  if not p.exists():continue
  row=read(p);assert row['status']=='COMPLETE' and row['updates']==2000 and row['new_training_episodes']==2000000
  counts={k:sum(v['counts'][k] for v in row['phases']) for k in ['actor','critic','posterior']}
  assert counts==dict(actor=2000,critic=10000,posterior=300000)
  assert row['total_seconds']<=14400 and sha(O/arm/'joint_actor.pt')==row['final_actor_sha256']
  values=np.load(O/arm/'evaluation/scores.npy');assert len(values)==2000 and np.isfinite(values).all() and abs(values.mean()-row['evaluation']['mean'])<1e-12
  inits[arm]=read(O/arm/'INITIALIZATION.json')
  manifest.append(dict(method=row['method'],checkpoint=str(O/arm/'joint_actor.pt'),checkpoint_sha256=row['final_actor_sha256'],runtime=str(O/arm/'runtime.pkl'),runtime_sha256=sha(O/arm/'runtime.pkl'),initialization=inits[arm],configuration=str(O/arm/'CONFIG.json'),scores=str(O/arm/'evaluation/scores.npy')))
  compute[arm]=dict(training_seconds=row['training_seconds'],total_run_seconds=row['total_seconds'],evaluation_seconds=row['evaluation']['seconds'],phases=row['phases'],counts=counts,
   new_episodes=row['new_training_episodes'],observations=row['new_training_observations'],model_queries=row['training_model_derived_calls'],resources=read(O/arm/'RESOURCES.json'),hardware=read(O/arm/'HARDWARE.json'),
   synchronized_batch16_actor_seconds=row['evaluation']['actor_batch16_seconds'],amortized_design_seconds=row['evaluation']['actor_amortized_seconds_per_design'],scorer_seconds=row['evaluation']['scorer_seconds'])
  for key in counts:training_counts[key]+=counts[key]
  training_counts['new_episodes']+=row['new_training_episodes']
 if len(inits)==2:
  for key in ['actor_digest','actor_target_digest','critic_digest','critic_target_digest','posterior_digest','optimizer_digest','replay_digest','training_rng_digest']:assert inits['learned'][key]==inits['uniform'][key],key
 for arm in ['previous_learned','previous_uniform','vsoed']:
  p=O/arm/'RESULTS.json'
  if p.exists():
   row=read(p);assert sha(row['checkpoint'])==row['checkpoint_sha256'];manifest.append(dict(method=row['method'],checkpoint=row['checkpoint'],checkpoint_sha256=row['checkpoint_sha256'],retrained=False,scores=str(O/arm/'evaluation/scores.npy')))
   compute[arm]=dict(resources=read(O/arm/'RESOURCES.json'),evaluation=row['evaluation'],additional_training_episodes=0)
 sensitivity=read(O/'sensitivity/RESULTS.json') if (O/'sensitivity/RESULTS.json').exists() else dict(status='PENDING_OR_FAILED_PRESERVED')
 if state['status']=='COMPLETE':
  assert d['status']=='COMPLETE' and len(manifest)==5 and training_counts==dict(actor=4000,critic=20000,posterior=600000,new_episodes=4000000)
 if sensitivity['status']=='COMPLETE':assert all(compute[a]['total_run_seconds']+sensitivity['seconds']<=14400 for a in ['learned','uniform'])
 put('SATURATION_SENSITIVITY.json',sensitivity);put('CHECKPOINT_MANIFEST.json',manifest)
 put('COMPUTE.json',dict(methods=compute,physical_unique_training=training_counts,sensitivity=sensitivity,
  preparation=dict(disposable_native_audit_episodes=8,CPU_forward_check=read(O/'COMMON_MODULE_AND_COST_AUDIT.json'),complete_CPU_audit_wall_seconds='Not independently measured;do not infer zero'),
  limitations=['Frozen V2 inference/query/opportunity pretraining and inherited1M replay are common dependencies;historical total training/selection/restart costs unavailable here,not zero.',
  'Same updates/episodes/time ceiling do not imply equal FLOPs;V5 has larger head than V4.',
  'GPU busy seconds are sampled estimates;GPU lease wall includes CPU/serialization gaps.',
  'CUDA-synchronized batch16 timing includes inference/model queries,excludes simulator/scorer and intervention diagnostics;not single-case latency.',
  'Publication hashing/report generation is CPU postprocessing outside training-run timing.']))
 text='# MoE-sBOED V5 scientific conclusion\n\n'
 if d['status']=='COMPLETE':
  for name,row in d['comparisons'].items():text+=f"- {name}: {row['mean']:+.5f} nats;95% paired evaluation CI [{row['CI95'][0]:+.5f},{row['CI95'][1]:+.5f}].\n"
  decision=d['decision'];text+='\n'+('Primary absolute-EIG improvement has a positive paired interval. Additional seeds may be recommended subject to uniform performance and separate authorization.' if decision['primary_credible'] else 'The primary absolute-EIG improvement was not established. Stop this architectural pilot;no expansion or additional tuning.')+'\n'
  text+='\nRecommendation: '+('Consider separately authorized continuation seeds102-105.' if decision['recommend_five_seeds'] else 'Do not expand to five seeds on this pilot evidence.')+'\n'
 else:text+='Incomplete pilot. Preserve outputs and exact error;no success claim.\n'
 text+='\nController state: '+json.dumps(state)+'.\n\n'
 text+='All primary estimates are finite1M-contrast EIG lower bounds. Known estimator saturation/contrast bias is not represented by the MCSE. Optional sensitivity status: '+sensitivity['status']+'. This is one continuation seed from shared V2 pretraining,not seed robustness or independent from-scratch training.\n\n'
 text+='V5 changes both weighted expert-slot preservation and head capacity. An improvement cannot identify their individual causal effects. Learned routing must be assessed through paired EIG,not weight variation or gradient norms. No canonical-bank checkpoint/hyperparameter selection,no teacher,no new rewards,no V4-Spec masking,and no automatic extra seeds.\n'
 (R/'SCIENTIFIC_CONCLUSION.md').write_text(text)
 put('COMPLETION_VERIFICATION.json',dict(controller=state,primary_evaluation_complete=d['status']=='COMPLETE',paired_initialization_verified=len(inits)==2,checkpoint_count=len(manifest),source_hashes_verified=True,bank_sha256=read(O/'BANK_IDENTITY.json')['sha256'],automatic_expansion=False))
 if d['status']=='COMPLETE':
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  rows=d['rows'];fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained');y=np.arange(len(rows));ax.errorbar([x['EIG'] for x in rows],y,xerr=[1.96*x['MCSE'] for x in rows],fmt='o',capsize=4);ax.set_yticks(y,[x['method'] for x in rows]);ax.invert_yaxis();ax.set_xlabel('EIG lower bound (nats);95% case MC intervals');ax.grid(axis='x',alpha=.25);fig.savefig(R/'EIG_COMPARISON.png',dpi=180);plt.close(fig)
 assert subprocess.check_output(['git','branch','--show-current'],cwd=M,text=True).strip()==B
 files=[str(p.relative_to(M)) for p in R.iterdir() if p.suffix in ['.json','.md','.csv','.png']]
 staged=set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=M,text=True).splitlines());assert staged<=set(files),'Unrelated staged work preserved'
 subprocess.run(['git','diff','--check','--',*files],cwd=M,check=True)
 normal=[x for x in files if not x.endswith('.png')];images=[x for x in files if x.endswith('.png')]
 subprocess.run(['git','add','--',*normal],cwd=M,check=True)
 if images:subprocess.run(['git','add','--force','--',*images],cwd=M,check=True)
 if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=M,text=True).strip():subprocess.run(['git','commit','-m','Report matched V5 expert-slot pilot and paired comparisons'],cwd=M,check=True)
 subprocess.run(['git','push','origin',B],cwd=M,check=True);commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=M,text=True).strip()
 (O/'PUBLISHED.json').write_text(json.dumps(dict(commit=commit,status=d['status']),indent=2)+'\n');print(commit,flush=True)
if __name__=='__main__':main()
