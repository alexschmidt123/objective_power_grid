"""One bounded readiness diagnostic of complete native SIR units; no EIG selection.

Run only under the campaign's aggregate GPU readiness meter and device lock.
All outputs are development evidence and must live under TEMP/scratch_runs/.
"""
import time
PROCESS_STARTED=time.perf_counter()
import argparse,json,math,os,subprocess,sys,hashlib
from pathlib import Path

def write(path,value):
 path=Path(path);q=path.with_suffix(path.suffix+'.partial');q.write_text(json.dumps(value,indent=2,allow_nan=False));q.replace(path)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True);parser.add_argument('--methods',default='dad,fixed,rl_sboed,vsoed,scorebed');args=parser.parse_args()
 root=Path(__file__).resolve().parents[1];out=Path(args.output).resolve()
 if root.name!='objective_power_grid_temp' or (root/'scratch_runs').resolve() not in out.parents:raise ValueError('Native readiness outputs require TEMP/scratch_runs/')
 out.mkdir(parents=True,exist_ok=False)
 import torch,yaml,numpy as np
 from src.research.benchmarks import make_benchmark
 from src.research.runner import DEFAULTS
 from src.research.original_protocol import dad_settings
 raw=yaml.safe_load(Path(args.config).read_text());h=raw['experiment']['step_number'];seed=101
 if raw['system']['name']!='sir_ode' or h!=5:raise ValueError('Readiness must use the declared physical SIR horizon5')
 if not torch.cuda.is_available():raise ValueError('Native runtime measurement requires the allocated CUDA GPU')
 torch.set_num_threads(2)
 data={'status':'RUNNING','hardware':torch.cuda.get_device_name(0),'native_reference_source':'3940753 plus focused charged-time adapter','methods':{},'EIG_selection_used':False,'primary_training':False}
 write(out/'status.json',data)
 def sync():torch.cuda.synchronize()
 class Observer:
  compute_clock=None
  def __init__(self):self.previous=time.perf_counter();self.times=[];self.records=[]
  def allow_next_unit(self,*a,**kw):return True
  def check(self,p,episodes,update,*a,**kw):
   sync();now=time.perf_counter();self.times.append(now-self.previous);self.previous=now
   if not all(torch.isfinite(v).all() for v in p.state_dict().values()):raise FloatingPointError('Nonfinite native readiness parameters')
   self.records.append({'update':int(update),'episodes':int(episodes),'eig':None})
   return bool(kw.get('force',False))
  def finish(self,p):return {'selected_checkpoint':self.records[-1],'checkpoint_selection':'LAST_NATIVE_TIMING_ITERATE_NOT_SCIENTIFIC_RESULT','convergence_status':'NATIVE_TIMING_ONLY','validation_seconds':0.}
 for method in args.methods.split(','):
  data['active_method']=method;write(out/'status.json',data);start=time.perf_counter();directory=out/method;directory.mkdir()
  b=make_benchmark(raw);s=DEFAULTS|raw['research'].get('settings',{})
  if method in ('dad','fixed'):
   from src.research.training import train_spce
   from src.research.sir_dad import SIRSumDAD
   s.update(dad_settings());s.update(updates=50000,validate_every=1,reference_optimizer_updates=50000,spce_trajectory_budget_unit='primary_episodes')
   if method=='fixed':s['learning_rate']=1e-4
   observer=Observer()
   with torch.device('cuda:0'):
    policy=SIRSumDAD(h) if method=='dad' else None
    p,record=train_spce(b,method,h,s,seed,directory,True,2*s['batch_size'],policy=policy,observer=observer)
   sync();result={'native_unit_seconds':observer.times,'conservative_unit_seconds':max(observer.times),'actual_updates':record['actual_updates'],'new_training_trajectories':record['generated_training_trajectories'],'batch_size':s['batch_size'],'contrasts':s['contrasts'],'optimizer':'Adam','native_reference_updates':50000,'checkpoint_sha256':hashlib.sha256((directory/(method+'.pt')).read_bytes()).hexdigest()}
   del p,policy
  elif method=='rl_sboed':
   from src.research.sir_native_rl import NativeSIRRLPolicy,algorithm,Replay
   from src.research.engine import rollout
   with torch.device('cuda:0'):
    torch.manual_seed(seed);np.random.seed(seed)
    p=NativeSIRRLPolicy(h,b);algo=algorithm(p,h);replay=Replay(10000000,1+3*h)
    sync();initialized=time.perf_counter();n=100;L=100000
    with torch.no_grad():r=rollout(b,p,h,n,seed+104729,L)
    info=r['info'];reward=info-torch.cat((torch.zeros_like(info[:,:1]),info[:,:-1]),1)
    inputs=torch.stack(r['features'],1).reshape(n*h,-1).float();nxt=torch.stack(r['features'][1:]+[r['final_features']],1).reshape_as(inputs).float()
    actions=torch.stack(r['actions'],1)[...,0];previous=torch.cat((actions.new_zeros(n,1),actions[:,:-1]),1);lo=previous+b.gap;hi=b.end-(h-torch.arange(h,device=actions.device)-1)*b.gap
    unit=(2*(actions-lo)/(hi-lo)-1).reshape(-1,1).float();done=torch.zeros(n,h);done[:,-1]=1
    batch=torch.cat((inputs,unit,reward.reshape(-1,1).float(),done.reshape(-1,1),nxt),-1)
    replay.add(batch);sync();collected=time.perf_counter()
    # Timing only: repeat the same real native transitions to profile the exact
    # post-warmup optimization kernel. These are not new training trajectories.
    while replay.count<100000:replay.add(batch)
    sync();gradstart=time.perf_counter()
    for _ in range(64):
     aloss,qloss,entropy=algo.optimize_policy(replay.sample(4096,h));algo._update_targets()
     if not all(torch.isfinite(x).all() for x in [aloss,*qloss,entropy,algo._log_alpha]):raise FloatingPointError('Nonfinite native readiness replay update')
    sync();gradend=time.perf_counter()
    torch.save({'kind':'readiness_only_native_rl','state_dict':p.state_dict()},directory/'timing_policy.pt')
   result={'initialization_seconds':initialized-start,'collection_seconds':collected-initialized,'native_gradient_cycle_seconds':gradend-gradstart,'conservative_unit_seconds':collected-initialized+gradend-gradstart,'gradient_updates':64,'critic_updates':128,'batch_size':100,'replay_batch':4096,'replay_capacity':10000000,'warmup_transitions':100000,'contrasts':L,'native_reference_updates':20001,'new_training_trajectories':100,'replay_timing_note':'Repeated500 real collected transitions to100000 rows solely to time native postwarmup64-update kernel; no new scientific result'}
   del p,algo,replay,r
  elif method=='vsoed':
   from src.research.upstream_vsoed import train_upstream_vsoed
   profile=dict(raw['research'].get('journal_protocol',{}).get('native_vsoed_profile',{}))
   if not profile:raise ValueError('Native vsOED profile must be frozen before measuring it')
   profile.update(updates=2,_trajectory_budget=22000)
   observer=Observer();p,record=train_upstream_vsoed(b,h,s,seed,directory,profile,observer=observer);sync()
   result={'native_unit_seconds':observer.times,'conservative_unit_seconds':max(observer.times),'steady_cycle_seconds':observer.times[-1],'initialization_and_first_cycle_seconds':observer.times[0],'native_updates':record['native_optimizer_updates'],'new_training_trajectories':record['generated_training_trajectories'],'profile':profile}
   del p
  elif method=='scorebed':
   envroot=root/'.venv_scorebed';stable=root.with_name('objective_power_grid')
   if not envroot.exists() and not envroot.is_symlink():envroot.symlink_to(stable/'.venv_scorebed',target_is_directory=True)
   if not (envroot/'bin/python').is_file():raise ValueError('Verified isolated ScoreBED environment unavailable')
   sir=raw['sir_ode'];configuration=dict(population=sir['population'],i0=sir['i0'],t_end=sir['t_end'],min_spacing=sir['min_spacing'],integration_steps_per_stage=sir['integration_steps_per_stage'],beta_mean=sir['beta_mean'],gamma_mean=sir['gamma_mean'],log_std=sir['log_std'],sigma=raw['observation']['noise_sigma'])
   write(directory/'sir_configuration.json',configuration)
   command=[str(envroot/'bin/python'),str(root/'src/research/sir_scorebed_worker.py'),'--output',str(directory),'--native-timing','--horizon',str(h),'--seed',str(seed),'--batch-size','1024','--sir-config',str(directory/'sir_configuration.json')]
   torch.cuda.empty_cache()
   with (directory/'worker.log').open('x') as log:subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,XLA_PYTHON_CLIENT_PREALLOCATE='false'))
   record=json.loads((directory/'scorebed_official_training.json').read_text());timing=record['native_unit_timing_seconds'];wall=time.perf_counter()-start
   score=max(timing['score_updates'][1:]);policy=max(timing['policy_updates'][1:])
   # Include imports, native initialization and first-compilation excess once.
   preparation=max(0.,wall-len(timing['score_updates'])*score-len(timing['policy_updates'])*policy)
   result={'native_unit_timing_seconds':timing,'initialization_and_compilation_seconds':preparation,'initialization_seconds':preparation,'score_update_seconds':score,'policy_update_seconds':policy,'steady_score_update_seconds':score,'steady_policy_update_seconds':policy,'full_phase_wall_seconds':wall,'new_training_trajectories':record['generated_training_trajectories'],'score_batch':1024,'policy_batch':1024,'score_reference_updates':50000,'policy_reference_updates':500}
  else:raise ValueError('Unknown native readiness method '+method)
  result['phase_wall_seconds']=time.perf_counter()-start
  if method in ('dad','fixed','rl_sboed'):result['unit_seconds']=result['conservative_unit_seconds'];result.setdefault('initialization_seconds',max(0.,result['phase_wall_seconds']-sum(result.get('native_unit_seconds',[]))))
  elif method=='vsoed':result['unit_seconds']=result['steady_cycle_seconds'];result['initialization_seconds']=max(0.,result['phase_wall_seconds']-len(result['native_unit_seconds'])*result['unit_seconds'])
  data['methods'][method]=result
  write(out/'timings.json',data);write(out/'native_timing.json',data|data['methods']);write(out/'status.json',data);torch.cuda.empty_cache()
 data.update(status='COMPLETE',active_method=None,total_process_wall_seconds=time.perf_counter()-PROCESS_STARTED)
 write(out/'timings.json',data);write(out/'native_timing.json',data|data['methods']);write(out/'status.json',data);print(json.dumps(data,indent=2),flush=True)

if __name__=='__main__':main()
