"""CPU-only durable terminal verification and scoped publication;never starts training."""
from pathlib import Path
import fcntl,hashlib,json,os,subprocess,sys,time
import numpy as np
M=Path.home()/'Documents/objective_power_grid';O=M/'experiments/location_finding/moe_sboed_v5_belief_20261119';R=M/'reports/moe_sboed_v5_belief_20261119';BRANCH='research/hec-v3-routing-validation-20261115'
def read(p):return json.loads(Path(p).read_text())
def put(name,value):(R/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(8388608),b''):h.update(c)
 return h.hexdigest()
def publish(state):
 subprocess.run([sys.executable,'-B','tools/moe_sboed_v5_belief_20261119.py','--task','report'],cwd=M,check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
 for path,digest in read(O/'NEW_SOURCE_MANIFEST.json').items():assert sha(M/path)==digest,path
 d=read(R/'RESULTS.json');manifest=[];compute={};audits={};inits={}
 for arm in ['learned','uniform']:
  path=O/arm/'RESULTS.json'
  if not path.exists():continue
  row=read(path);assert row['status']=='COMPLETE' and row['counts']==dict(actor=2000,critic=10000,posterior=300000) and row['new_episodes']==2000000
  assert row['logical_total_seconds']<=14400 and sha(row['checkpoint'])==row['checkpoint_sha256']
  assert [x['continuation_update'] for x in row['refreshes']]==list(range(200,2000,200))
  assert all(v['optimizer_parameter_identities_preserved'] and v['target_belief_identical'] and v['new_simulator_episodes']==0 for v in row['refreshes'])
  audits[arm]=read(O/arm/'NO_ONLINE_OPTIMIZATION_AUDIT.json');assert audits[arm]['status']=='PASS'
  inits[arm]=read(O/arm/'INITIALIZATION.json')
  manifest.append(dict(method=row['method'],checkpoint=row['checkpoint'],checkpoint_sha256=row['checkpoint_sha256'],runtime=str(O/arm/'runtime.pkl'),runtime_sha256=sha(O/arm/'runtime.pkl'),initialization=inits[arm],source_manifest=read(O/arm/'CONFIG.json')['source_manifest']))
  compute[arm]=dict(resources=read(O/arm/'RESOURCES.json'),hardware=read(O/arm/'HARDWARE.json'),training_seconds=row['training_seconds'],logical_total_seconds=row['logical_total_seconds'],preparation_credit_seconds=row['preparation_credit_seconds'],counts=row['counts'],new_episodes=row['new_episodes'],new_observations=row['new_observations'],evaluation=row['evaluation'],training_model_queries=row['training_model_queries'])
 if state['status']=='COMPLETE':assert len(manifest)==2 and inits['learned']==inits['uniform'] and d['status']=='COMPLETE'
 put('CHECKPOINT_MANIFEST.json',manifest);put('COMPUTE.json',dict(methods=compute,preparation=read(R/'PREPARATION_ACCOUNTING.json'),preflight=read(R/'PREFLIGHT_RESULTS.json'),notes=['Matched updates/episodes are not identical FLOPs. Offline native belief transfer uses copies of existing native posterior;no additional optimizer update or rollout.','Preparation credit includes failure handling and is conservatively charged to both caps. Actual historical inherited pretraining costs are dependencies,not zero.','Batch16 CUDA-synchronized timings are amortized,not single-case latency. GPU busy estimate is sampled,not exact kernel time.']))
 put('TERMINAL_NO_ONLINE_OPTIMIZATION_AUDITS.json',audits)
 text='# New V5 primary canonical conclusion\n\n'
 if d['status']=='COMPLETE':
  for key,row in d['comparisons'].items():text+=f"- {key}: {row['mean']:+.5f} nats;95% paired case CI [{row['CI95'][0]:+.5f},{row['CI95'][1]:+.5f}].\n"
  primary=d['comparisons']['newV5_minus_previous_learned'];routing=d['comparisons']['newV5_minus_uniform'];uniform=d['comparisons']['uniform_minus_V4uniform']
  credible=primary['CI95'][0]>0
  text+='\n'+('Primary absolute-EIG potential is supported in this one-seed evaluation. Further seeds require separate authorization.' if credible else 'Primary absolute-EIG improvement was not established. Stop this pilot;do not expand or tune against the canonical bank.')+'\n'
  put('DECISION.json',dict(primary_credible=credible,routing_credible=routing['CI95'][0]>0,uniform_change=uniform,automatic_expansion=False,recommend_discussing_additional_seeds=credible and uniform['mean']>=0))
 else:text+='Training was incomplete or capped. Preserve all partial outputs. No success claim.\n'
 text+='\nAll numbers use the original canonical bank and frozen1M-contrast scorer. These are finite-contrast lower bounds with saturation;MCSE excludes contrast bias. This is one continuation seed,not end-to-end independent training or across-seed robustness.\n\n'
 text+='Belief proper-score improvement is not EIG improvement. Importance-reference and critic-ranking diagnostics are finite/noisy;the critic objective was not changed. Actor/target belief refresh happens only between offline training blocks. Final deployment freezes all weights/buffers and passes the executable no-online-optimization audit. No test-time fitting,action search,MCMC or gradient optimization is used.\n'
 (R/'SCIENTIFIC_CONCLUSION.md').write_text(text)
 put('COMPLETION_VERIFICATION.json',dict(status=d['status'],controller=state,matched_initializations_verified=len(inits)==2,terminal_deployment_audits=audits,source_hashes_verified=True,automatic_expansion=False))
 assert subprocess.check_output(['git','branch','--show-current'],cwd=M,text=True).strip()==BRANCH
 files=[str(f.relative_to(M)) for f in R.iterdir() if f.suffix in ['.json','.md','.csv']]
 staged=set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=M,text=True).splitlines());assert staged<=set(files),'Unrelated staged work preserved'
 subprocess.run(['git','diff','--check','--',*files],cwd=M,check=True);subprocess.run(['git','add','--',*files],cwd=M,check=True)
 if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=M,text=True).strip():subprocess.run(['git','commit','-m','Report new V5 native-belief refresh matched pilot and deployment audits'],cwd=M,check=True)
 subprocess.run(['git','push','origin',BRANCH],cwd=M,check=True);commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=M,text=True).strip();(O/'PUBLISHED.json').write_text(json.dumps(dict(commit=commit,status=d['status']),indent=2)+'\n')
def main():
 lock=(O/'.finish.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 status=O/'PUBLICATION_STATUS.json'
 def record(s,**kwargs):status.write_text(json.dumps(dict(status=s,pid=os.getpid(),**kwargs),indent=2)+'\n')
 record('WAITING')
 try:
  for _ in range(960):
   path=O/'NEW_V5_STATUS.json'
   if path.exists():
    state=read(path)
    if state['status'] in ['COMPLETE','FAILED_PRESERVED']:
     record('VERIFYING');publish(state);record('PUBLISHED');return
   time.sleep(45)
  record('MONITOR_TIME_LIMIT_NO_TRAINING_CHANGE')
 except Exception as e:record('FAILED_PRESERVED',error=repr(e));raise
if __name__=='__main__':main()
