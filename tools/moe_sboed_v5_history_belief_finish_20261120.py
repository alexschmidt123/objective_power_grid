"""CPU-only final verification/publication for one belief-aware-critic V5 pilot."""
from pathlib import Path
import ast,fcntl,hashlib,importlib.util,json,os,subprocess,sys,time
import numpy as np
M=Path.home()/'Documents/objective_power_grid';O=M/'experiments/location_finding/moe_sboed_v5_history_belief_20261120';R=M/'reports/moe_sboed_v5_history_belief_20261120';BRANCH='research/hec-v3-routing-validation-20261115'
def read(p):return json.loads(Path(p).read_text())
def put(name,value):(R/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def publish(status):
 spec=importlib.util.spec_from_file_location('critic_v5_completion',M/'src/research/moe_sboed_v5_history_belief_20261120/run.py');s=importlib.util.module_from_spec(spec);sys.modules[spec.name]=s;spec.loader.exec_module(s)
 s.report();assert s.source_identity()==read(O/'SOURCE_MANIFEST.json')
 d=read(R/'RESULTS.json');config=s.cfg();inits={};manifest=[];compute={};audits={}
 import torch
 guard_spec=importlib.util.spec_from_file_location('critic_terminal_deployment',M/'tools/moe_sboed_v5_belief_diagnostics_20261119.py');g=importlib.util.module_from_spec(guard_spec);sys.modules[guard_spec.name]=g;guard_spec.loader.exec_module(g)
 parent=None
 for arm in ['learned','uniform']:
  path=O/arm/'RESULTS.json'
  if not path.exists():continue
  r=read(path);counts={k:sum(x['counts'][k] for x in r['phases']) for k in ['actor','critic','posterior']}
  assert r['status']=='COMPLETE' and counts==dict(actor=2000,critic=10000,posterior=300000) and r['new_training_episodes']==2000000
  total=r['total_seconds']+config['preparation_credit_seconds'];assert total<=14400
  cp=O/arm/'joint_actor.pt';assert sha(cp)==r['final_actor_sha256']
  values=np.load(O/arm/'evaluation/scores.npy');assert len(values)==2000 and np.isfinite(values).all() and abs(values.mean()-r['evaluation']['mean'])<1e-12
  inits[arm]=read(O/arm/'INITIALIZATION.json');assert inits[arm]['actor_digest']==config['expected_initial_actor_digest']
  assert inits[arm]['critic_digest']==config['expected_initial_critic_digest']
  if parent is None:parent=s.ORIGINAL_LOAD('cpu')['algorithm'].actor_net
  model=s.new_actor(parent,arm);model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['actor_state']);model.configure('frozen');model.eval()
  x=torch.zeros(6,117)
  for j,t in enumerate([0,0,15,15,29,29]):x[j,t]=1
  audits[arm]=g.deployment_audit(model,x)
  assert sum(v.numel() for v in model.parameters())==371136
  assert not any(type(m).__name__=='BeliefCritic' for m in model.modules())
  if arm=='uniform':
   with torch.no_grad():_,info=model(x,collect=True)
   assert torch.equal(info['gates'],torch.full_like(info['gates'],.25))
  manifest.append(dict(method=r['method'],checkpoint=str(cp),checkpoint_sha256=r['final_actor_sha256'],runtime=str(O/arm/'runtime.pkl'),runtime_sha256=sha(O/arm/'runtime.pkl'),initialization=inits[arm],training_config=str(O/arm/'CONFIG.json')))
  compute[arm]=dict(training_seconds=r['training_seconds'],logical_total_seconds=total,preparation_credit=config['preparation_credit_seconds'],counts=counts,new_training_episodes=2000000,new_training_observations=60000000,
   resources=read(O/arm/'RESOURCES.json'),evaluation=r['evaluation'],model_queries=r['training_model_derived_calls'])
 if status['status']=='COMPLETE':
  assert len(manifest)==2 and d['status']=='COMPLETE'
  for key in ['actor_digest','critic_digest','posterior_digest','actor_target_digest','critic_target_digest','optimizer_digest','replay_digest','training_rng_digest']:assert inits['learned'][key]==inits['uniform'][key],key
 put('CHECKPOINT_MANIFEST.json',manifest);put('TERMINAL_DEPLOYMENT_AUDITS.json',audits);put('COMPUTE.json',dict(methods=compute,preparation=read(R/'NATIVE_PREFLIGHT.json'),notes=['Discarded preparation used2000 additional episodes and60000 observations;charged to both logical time caps but executed once.','Frozen inherited V2/provider/replay costs are shared dependencies,not zero.','Additional deterministic role-evidence computation costs more despite identical parameter count;report actual compute.','Native critic/posterior remain unchanged;no online optimization or posterior refresh.','Publication/verification CPU overhead is separate from train/evaluation run timing.']))
 text='# Primary conclusion:policy-only history–belief V5,seed101\n\n'
 if d['status']=='COMPLETE':
  for key,r in d['comparisons'].items():text+=f"- {key}: {r['mean']:+.5f} nats;95% paired case CI [{r['CI95'][0]:+.5f},{r['CI95'][1]:+.5f}].\n"
  primary=d['comparisons']['newV5_minus_reference_learned'];routing=d['comparisons']['newV5_minus_uniform'];uniform=d['comparisons']['uniform_minus_previous_uniform']
  put('DECISION.json',dict(credible_improvement_over_previous_V5=primary['CI95'][0]>0,routing_credible=routing['CI95'][0]>0,uniform_change=uniform,automatic_expansion=False,recommend_discussing_more_seeds=primary['CI95'][0]>0 and uniform['mean']>=0))
  text+='\n'+('Absolute-EIG potential over previous V5 is supported by the paired case interval;additional seeds require authorization.' if primary['CI95'][0]>0 else 'Absolute-EIG improvement over previous V5 was not established. Stop this pilot;no expansion or tuning.')+'\n'
 else:text+='Incomplete/capped pilot;preserve partial artifacts,no success claim.\n'
 text+='\nOne continuation seed,not across-seed robustness. Original canonical1M-contrast EIG lower bound;MCSE excludes finite-contrast bias/saturation. Only expert history–belief evidence changes;critic and posterior training are unchanged,no extra networks or parameters. No guarantee of13.15–13.30 EIG. No model was selected using canonical scores. No automated additional seeds.\n';(R/'SCIENTIFIC_CONCLUSION.md').write_text(text)
 put('COMPLETION_VERIFICATION.json',dict(controller=status,result_status=d['status'],matched_initializations_verified=len(inits)==2,deployment_audits=audits,source_manifest_verified=True,automatic_expansion=False))
 assert subprocess.check_output(['git','branch','--show-current'],cwd=M,text=True).strip()==BRANCH
 files=[str(p.relative_to(M)) for p in R.iterdir() if p.suffix in ['.json','.md','.csv']]
 staged=set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=M,text=True).splitlines());assert staged<=set(files)
 subprocess.run(['git','diff','--check','--',*files],cwd=M,check=True);subprocess.run(['git','add','--',*files],cwd=M,check=True)
 if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=M,text=True).strip():subprocess.run(['git','commit','-m','Report matched policy-only V5 history-belief canonical pilot'],cwd=M,check=True)
 subprocess.run(['git','push','origin',BRANCH],cwd=M,check=True);commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=M,text=True).strip();(O/'PUBLISHED.json').write_text(json.dumps(dict(commit=commit,status=d['status']),indent=2)+'\n')
def main():
 handle=(O/'.finish.lock').open('a+');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def status(name,**extra):(O/'PUBLICATION_STATUS.json').write_text(json.dumps(dict(status=name,pid=os.getpid(),**extra),indent=2)+'\n')
 status('WAITING')
 try:
  for _ in range(960):
   path=O/'STATUS.json'
   if path.exists():
    state=read(path)
    if state['status'] in ['COMPLETE','FAILED_PRESERVED']:status('VERIFYING');publish(state);status('PUBLISHED');return
   time.sleep(45)
  status('TIME_LIMIT_NO_JOB_CHANGE')
 except Exception as e:status('FAILED_PRESERVED',error=repr(e));raise
if __name__=='__main__':main()
