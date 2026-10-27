"""Seed101 only: serialized strict native baseline recovery through run.sh.
No location/FH training, no extra seeds and no confirmation-bank consumption.
"""
import argparse,copy,fcntl,hashlib,json,os,signal,subprocess,time
from pathlib import Path
import yaml
R=Path(__file__).resolve().parents[1];C=R/'experiments/10262026/10262026_sir_strict_seed101';METHODS=['vsoed','scorebed','dad','rl_sboed']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def read(p):return json.loads(Path(p).read_text())
def prepare():
 assert not C.exists(),'Preserve existing campaign'
 for name in ['configs','logs','readiness','metadata','runs','reports']:(C/name).mkdir(parents=True)
 old=Path.home()/'Documents/objective_power_grid_e1_original_protocol_20261019/experiments/10192026/10192026_e1_original_baseline_restoration/configs/sir_1000000.yaml';raw=yaml.safe_load(old.read_text());(C/'configs/sir_1000000.yaml').write_bytes(old.read_bytes())
 # Original established native-size probe settings. Discard all probe weights.
 from src.research.original_protocol import score_allocation
 for m,cap in [('dad',2000),('rl_sboed',100),('vsoed',21000),('scorebed',102190)]:
  cfg=copy.deepcopy(raw);p=cfg['research']['journal_protocol'];p['stopping_rule'].update(minimum_episodes=cap,maximum_episodes=cap,maximum_training_seconds=600);p['validation'].update(systems=4,contrasts=8,batch_size=2,chunk=4);cfg['research']['settings'].update(eval_systems=2,eval_contrasts=8,evaluation_population_systems=2,evaluation_batch_size=2,evaluation_contrast_chunk=4);cfg['research']['paper_protocol']['nominal_training_trajectories']=cap
  if m=='scorebed':p['method_settings'][m]['scorebed_pretraining_episodes']=score_allocation(cap,'sir_ode')['score']
  (C/'configs'/f'probe_{m}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 files=[*R.glob('src/**/*.py'),*R.glob('tools/**/*.py'),R/'run.sh',R/'sweep_run.sh',*C.glob('configs/*.yaml')]
 write(C/'source_manifest.json',dict(foundation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),sha256={str(p.relative_to(R)):sha(p) for p in files}))
 bank=Path.home()/'Documents/objective_power_grid_e1_sir_replication_20261019/experiments/10192026/10192026_sir_e1_five_seed_verified/evaluations/development_bank'
 # Actual location of canonical files may be flat in evaluations.
 if not (bank/'cases.npz').exists():bank=bank.parent
 cases=bank/'cases.npz';contrasts=bank/'contrasts_1m.npy'
 assert sha(cases)=='4b0a17909a46726b794316dd5ffc979b2c77b1bb3b1f2fab4845ddaa46703d25'
 assert sha(contrasts)=='20fcc4b2cc0e4a6ac071fabc74c56e498dce0c6397e780001ea39488b40a7a48'
 write(C/'plan.json',dict(methods=METHODS,seed=101,primary_episodes=1000000,config_sha256=sha(old),bank_cases=str(cases),bank_contrasts=str(contrasts),bank_sha256=dict(cases=sha(cases),contrasts=sha(contrasts)),native_training_safety_seconds=172800,per_full_phase_wall_safety_seconds=259200,readiness_seconds=600,reused_methods=['random','fixed','moe','resmoe','switchmoe','switchresmoe'],reserved_seed8831001_used=False,no_FH_training=True,policy_selection='final exact1Mbudget',source_revision='Final promoted commit recorded separately; no source edits during execution'))
 write(C/'registry.json',[]);write(C/'status.json',dict(state='PREPARED_NOT_RUNNING',pid=None,methods=METHODS))
 print(C)
def device_lock():
 lockpath=Path.home()/'.cache/objective_power_grid/device_0.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
 while True:
  lock=lockpath.open('a+')
  try:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
   if not active:return lock
  except BlockingIOError:pass
  lock.close();write(C/'status.json',dict(state='WAITING_LOCAL_DEVICE_LOCK',pid=os.getpid()));time.sleep(30)
def guard():
 for rel,digest in read(C/'source_manifest.json')['sha256'].items():assert sha(R/rel)==digest,rel
 plan=read(C/'plan.json');assert sha(plan['bank_cases'])==plan['bank_sha256']['cases'];assert sha(plan['bank_contrasts'])==plan['bank_sha256']['contrasts']
def execute(method,probe):
 guard();plan=read(C/'plan.json');label=('probe_' if probe else '')+method;out=C/('readiness' if probe else 'runs')/method;cfg=C/'configs'/(f'probe_{method}.yaml' if probe else 'sir_1000000.yaml')
 assert not out.exists(),'Never overwrite partial scientific run'
 lock=device_lock();started=time.perf_counter();cmd=['bash','run.sh','--config',str(cfg),'--journal','--seed','101','--eval-seeds','1001','--eig-only','--methods',method,'--source-check-manifest',str(C/'source_manifest.json'),'--output',str(out)]
 if not probe:cmd+=['--evaluation-cases',plan['bank_cases'],'--evaluation-bank',plan['bank_contrasts']]
 write(C/'metadata'/(label+'_command.json'),dict(command=cmd));write(C/'status.json',dict(state='RUNNING_READINESS' if probe else 'RUNNING_STRICT_FULL_BUDGET',method=method,pid=os.getpid(),output=str(out)))
 rc=None;failure=None
 try:
  with (C/'logs'/(label+'.log')).open('x') as log:
   p=subprocess.Popen(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   write(C/'metadata'/(label+'_pid.json'),dict(pid=p.pid,coordinator=os.getpid()))
   try:rc=p.wait(timeout=600 if probe else 259200)
   except subprocess.TimeoutExpired:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=20)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    failure='Established per-phase resource safety cap reached';rc=124
  if rc==0:
   t=read(out/'models'/f'{method}_training.json')
   if not probe:
    assert t['generated_training_trajectories']==1000000 and t['selected_checkpoint']['episodes']==1000000
    assert t['checkpoint_selection']=='final_predeclared_fixed_budget'
   assert t.get('checkpoint_sha256')==sha(out/'models'/f'{method}.pt')
  elif (out/'failure.json').exists():failure=read(out/'failure.json')
 except Exception as e:failure=repr(e);rc=-1
 finally:lock.close()
 result=dict(method=method,probe=probe,status='COMPLETE' if rc==0 else 'INCOMPLETE',returncode=rc,failure=failure,wall_seconds=time.perf_counter()-started,run=str(out),checkpoint_sha256=sha(out/'models'/f'{method}.pt') if rc==0 else None)
 write(C/'metadata'/(label+'_completion.json'),result);registry=read(C/'registry.json');registry.append(result);write(C/'registry.json',registry);print(label,result['status'],flush=True);return rc==0

def run():
 process=(C/'coordinator.lock').open('a+');fcntl.flock(process,fcntl.LOCK_EX|fcntl.LOCK_NB);guard();write(C/'metadata/coordinator.json',dict(pid=os.getpid(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()))
 for m in METHODS:
  registry=read(C/'registry.json')
  if any(x['method']==m and not x['probe'] for x in registry):continue
  probed=next((x for x in registry if x['method']==m and x['probe']),None)
  ok=probed['status']=='COMPLETE' if probed else execute(m,True)
  if ok:execute(m,False)
  else:write(C/'metadata'/(m+'_blocked.json'),dict(status='READINESS_FAILED_NO_FULL_RUN',method=m))
 registry=read(C/'registry.json');complete=[x['method'] for x in registry if not x['probe'] and x['status']=='COMPLETE'];write(C/'status.json',dict(state='QUEUE_FINISHED',strict_completed=complete,strict_completed_count=len(complete),missing=[m for m in METHODS if m not in complete],no_extra_seeds=True));print('STRICT_QUEUE_FINISHED',complete,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args();prepare() if a.action=='prepare' else run()
