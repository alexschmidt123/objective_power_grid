"""Pinned native RL-sBOED;fixed wall stopping only,no EIG-based selection."""
import argparse,hashlib,json,os,pathlib,subprocess,sys,time,resource
M=pathlib.Path.home()/'Documents/objective_power_grid';O=M/'experiments/location_finding/moe_sboed_location_benchmark_20261115';C=M/'configs/moe_sboed_rl_reproduction_20261115.json';PY='/home/grads/g/g.lin/miniconda3/envs/dellma/bin/python'
sys.path.insert(0,str(M/'tools'))
import moe_sboed_external_evaluation as external

def put(p,d):external.put(p,d)
def train(seed,smoke=False):
 external.prepare_imports()
 import numpy as np,torch
 from src.research.benchmarks import make_benchmark
 from src.research.native_rl_reference import train_native_rl
 cfg=json.loads(C.read_text());root=O/('rl_smoke_cpu' if smoke else 'rl_training')/('seed'+str(seed));assert not root.exists();modeldir=root/'models';modeldir.mkdir(parents=True)
 torch.set_num_threads(2 if smoke else 4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 physics=json.loads((external.B/'inputs/HISTORICAL_BASELINES.json').read_text())['physics'];b=make_benchmark(physics);settings=dict(rl_native_contrasts=3 if smoke else 100000,rl_native_replay_capacity=1000 if smoke else 1000000,updates=1 if smoke else 2000)
 put(root/'run_config.json',dict(benchmark=physics,method='RL-sBOED',seed=seed,settings=settings,protocol=cfg));start=time.monotonic();limit=60 if smoke else 12600;cpu0=resource.getrusage(resource.RUSAGE_SELF)
 class Observer:
  def __init__(self):self.episodes=0;self.updates=0
  def allow_next_unit(self,*unused):return time.monotonic()-start<limit-1 if smoke else time.monotonic()-start<limit-90
  def check(self,p,episodes,updates,opt,force=False):
   self.episodes=episodes;self.updates=updates
   if not smoke and (updates==1 or updates%25==0):
    sample=dict(seconds=time.monotonic()-start,method='RL-sBOED',seed=seed,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader'],text=True).strip())
    with (root/'RESOURCE_SAMPLES.jsonl').open('a') as f:f.write(json.dumps(sample)+'\n')
   if updates==1 or updates%25==0 or force:put(root/'STATUS.json',dict(status='TRAINING',method='RL-sBOED',seed=seed,primary_episodes=episodes,collection_updates=updates,seconds=time.monotonic()-start))
   return time.monotonic()-start>=limit-(1 if smoke else 90)
  def finish(self,p):return dict(checkpoint_selection='Terminal native state at fixed schedule or fixed wall boundary;no test EIG selection',selected_update=self.updates,stop_reason='reference_schedule_completion' if self.episodes>=2000000 else 'predeclared_four_hour_run_reserve')
 observer=Observer();held=[]
 try:
  if not smoke:
   import fcntl
   for p in [M/'.gpu0.lock',pathlib.Path.home()/'.cache/objective_power_grid/device_0.lock',pathlib.Path.home()/'Documents/objective_power_grid_temp/.gpu0.lock']:
    p.parent.mkdir(exist_ok=True,parents=True);f=p.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);held.append(f)
   assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  device='cpu' if smoke else 'cuda'
  with torch.device(device):policy,result=train_native_rl(b,30,settings,seed,modeldir,2 if smoke else 2000000,observer)
  cpu=resource.getrusage(resource.RUSAGE_SELF);result.update(status='COMPLETE',method='RL-sBOED',seed=seed,total_offline_seconds=time.monotonic()-start,cpu_user_seconds=cpu.ru_utime-cpu0.ru_utime,cpu_system_seconds=cpu.ru_stime-cpu0.ru_stime,
    checkpoint_sha256=external.sha(modeldir/'rl_sboed.pt'),wall_cap_seconds=14400,training_soft_seconds=12600,training_objective='Unmodified pinned native SAC/REDQ information-gain rewards and entropy regularization;no new balancing/diversity/cost reward.',scope='Fixed-time budget;2M native target may be incomplete;not a convergence claim.')
  if not smoke:result['torch_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
  put(modeldir/'rl_sboed_training.json',result);put(root/'RESULTS.json',result);put(root/'STATUS.json',dict(status='COMPLETE',method='RL-sBOED',seed=seed,primary_episodes=result['generated_training_trajectories'],seconds=time.monotonic()-start))
 finally:
  for f in held:f.close()

def evaluate(seed):
 root=O/'rl_training'/('seed'+str(seed));receipt=root/'models/rl_sboed_training.json';d=json.loads(receipt.read_text())
 job=dict(method='rl_sboed',display_name='RL-sBOED',seed=seed,checkpoint=str(root/'models/rl_sboed.pt'),checkpoint_sha256=d['checkpoint_sha256'],training_receipt=str(receipt),training_receipt_sha256=external.sha(receipt),offline_seconds=d['total_offline_seconds'],compare_saved_reference=False)
 external.worker(job)

def queue():
 import fcntl
 lock=(O/'.rl_queue.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);failed=[]
 for seed in [101,102,103,104,105]:
  root=O/'rl_training'/('seed'+str(seed));evaluation=O/'external_evaluations/rl_sboed'/('seed'+str(seed));start=time.monotonic()
  while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(30)
  started_gpu=time.monotonic();start=started_gpu
  put(O/'RL_QUEUE_STATUS.json',dict(status='RUNNING',method='RL-sBOED',seed=seed,failed=failed))
  if not (root/'RESULTS.json').exists():
   if root.exists():failed.append(dict(seed=seed,status='PRESERVED_INCOMPLETE_NO_DUPLICATE'));continue
   with (O/f'RL-sBOED_seed{seed}.log').open('x') as f:rc=subprocess.call(['timeout','--signal=TERM','--kill-after=30','14370',PY,'-u',str(pathlib.Path(__file__).resolve()),'--train','--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
   if rc:failed.append(dict(seed=seed,status='FAILED_TRAINING',returncode=rc));continue
  remaining=14400-(time.monotonic()-start)
  if remaining<60:failed.append(dict(seed=seed,status='EVALUATION_CAP'));continue
  if not (evaluation/'RESULTS.json').exists():
   if evaluation.exists():failed.append(dict(seed=seed,status='PRESERVED_INCOMPLETE_EVALUATION'));continue
   with (O/f'RL-sBOED_evaluation_seed{seed}.log').open('x') as f:rc=subprocess.call(['timeout','--signal=TERM','--kill-after=30',str(int(remaining-30)),PY,'-u',str(pathlib.Path(__file__).resolve()),'--evaluate','--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
   if rc:failed.append(dict(seed=seed,status='FAILED_EVALUATION',returncode=rc))
 put(O/'RL_QUEUE_STATUS.json',dict(status='COMPLETE' if not failed else 'FINISHED_WITH_FAILURES',failed=failed))

def main():
 p=argparse.ArgumentParser();p.add_argument('--train',action='store_true');p.add_argument('--evaluate',action='store_true');p.add_argument('--queue',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--launch',action='store_true');p.add_argument('--seed',type=int,default=101);args=p.parse_args()
 if args.launch:
  receipt=O/'RL_QUEUE_LAUNCH.json';assert not receipt.exists()
  with (O/'rl_queue.log').open('x') as f:child=subprocess.Popen([PY,'-u',str(pathlib.Path(__file__).resolve()),'--queue'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
  put(receipt,dict(pid=child.pid,seeds=[101,102,103,104,105],method='RL-sBOED'))
 elif args.queue:queue()
 elif args.train or args.smoke:train(args.seed,args.smoke)
 elif args.evaluate:evaluate(args.seed)
 else:p.error('Explicit action required')
if __name__=='__main__':main()
