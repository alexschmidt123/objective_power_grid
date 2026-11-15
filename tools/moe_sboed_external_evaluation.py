"""Reuse verified native baseline training;evaluate only on the locked MoE benchmark."""
import argparse,fcntl,hashlib,importlib.machinery,io,json,os,pathlib,resource,subprocess,sys,time
sys.dont_write_bytecode=True
M=pathlib.Path.home()/'Documents/objective_power_grid';A=M/'experiments/_legacy_archive/temp_research_20261115';B=A/'scratch_runs/moe_hec_v2_icml_20261109';C=A/'scratch_runs/journal_LF_T30_native_final_20261102T052000Z'
O=M/'experiments/location_finding/moe_sboed_location_benchmark_20261115';R=M/'reports/moe_sboed_location_benchmark';CONFIG=M/'configs/moe_sboed_external_evaluation.json';PY='/home/grads/g/g.lin/miniconda3/envs/dellma/bin/python'
OLD='Documents/objective_power_grid_temp';NEW='Documents/objective_power_grid/experiments/_legacy_archive/temp_research_20261115'
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8388608),b''):h.update(block)
 return h.hexdigest()
def put(p,d):
 p=pathlib.Path(p);p.parent.mkdir(exist_ok=True,parents=True);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');temp.replace(p)
def manifest():
 jobs=[]
 inv=json.loads((B/'inputs/HISTORICAL_BASELINES.json').read_text());dad101=pathlib.Path(next(v['checkpoint'] for v in inv['rows'] if v['method']=='dad'))
 for method in ['dad','scorebed','vsoed']:
  for seed in [101,102,103,104,105]:
   path=C/f'runs/seed{seed}/{method}/models/{method}.pt';receipt=path.with_name(method+'_training.json')
   if seed==101 and method=='dad':path=dad101;receipt=path.with_name('dad_training.json')
   if seed==101 and method=='vsoed':path=B/'inputs/vsoed.pt';receipt=A/'scratch_runs/moe_development_migration_20261105/imported_ubuntu_tree/scratch_runs/20261104T161647Z_direct_head_moe_vsoed_seed101/reference/vsoed_training.json'
   d=json.loads(receipt.read_text());seconds=float(d.get('total_offline_seconds',d.get('training_seconds')));assert seconds<14400,(method,seed,seconds)
   expected=d.get('checkpoint_sha256')
   actual=sha(path)
   if expected:assert expected==actual,(method,seed)
   if method=='vsoed' and seed==101:assert actual=='68035f5da5b4bef7597d7434ed2db9ce95b56d4f6f39371f4acd23f3752b05bb'
   if method=='dad':assert d['actual_updates']==50000
   if method=='vsoed':assert d.get('selected_update',d.get('actor_updates'))==10001
   if method=='scorebed':assert d['actual_score_updates']==50000 and d['actual_policy_updates']==100000
   job=dict(method=method,seed=seed,checkpoint=str(path),checkpoint_sha256=actual,training_receipt=str(receipt),training_receipt_sha256=sha(receipt),offline_seconds=seconds,
    action='EVALUATE_EXISTING_COMPLETED_NATIVE_RUN;NO_RETRAINING',source=str(C/'source'))
   if method=='scorebed':job['arrays_sha256']=sha(path.with_name('scorebed_official_policy.npz'))
   jobs.append(job)
 return dict(status='LOCKED',jobs=jobs,hard_seconds_per_evaluation_run=14400,evaluation_seed=202611099206,trajectories=2000,contrasts=1000000,T=30,
  bank_sha256='52c0c614c4ee3b63d58ccda60714b38b00a2c0ca35564a428fe8cc2ce8d727c5',rule='No test-bank selection;seed101 must reproduce full saved vectors within1e-9;no SIR/power-grid or development-ablation training.')

def prepare_imports():
 old=importlib.machinery.SourceFileLoader.get_code
 def code(loader,name):
  path=pathlib.Path(loader.path)
  if path.is_relative_to(A):return compile(path.read_text().replace(OLD,NEW),str(path),'exec',dont_inherit=True)
  return old(loader,name)
 importlib.machinery.SourceFileLoader.get_code=code
 sys.path.insert(0,str(C/'source'));sys.path.insert(0,str(A/'development/hec_git_sync/development/moe_sboed/hec_canonical_evaluation_20261114'))

def worker(job):
 prepare_imports()
 import numpy as np,torch
 from protocol import load,LOCK
 from src.research.journal_checkpoints import load_checkpoint
 from src.research.benchmarks import make_benchmark
 from src.research.engine import rollout
 from scipy.stats import t
 out=O/'external_evaluations'/job['method']/('seed'+str(job['seed']));assert not out.exists();out.mkdir(parents=True)
 start=time.monotonic();deadline=start+14100;held=[]
 try:
  for p in [M/'.gpu0.lock',pathlib.Path.home()/'.cache/objective_power_grid/device_0.lock',pathlib.Path.home()/'Documents/objective_power_grid_temp/.gpu0.lock']:
   p.parent.mkdir(exist_ok=True,parents=True);f=p.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);held.append(f)
  active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert not active,active
  assert sha(job['checkpoint'])==job['checkpoint_sha256'] and sha(job['training_receipt'])==job['training_receipt_sha256']
  torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  bank,u,scorer=load();inv=json.loads((B/'inputs/HISTORICAL_BASELINES.json').read_text());b=make_benchmark(inv['physics'])
  if job['method']=='vsoed' and job['seed']==101:
   from src.research.upstream_vsoed import UpstreamActor
   from src.research.vendor.vsoed.utils import Net
   saved=torch.load(job['checkpoint'],map_location='cpu',weights_only=False);actor=Net([117,256,256,256,2],torch.nn.ReLU,torch.tensor([[-4.,4.]]*2,dtype=torch.float32),'actor').float();actor.load_state_dict(saved['actor_state'],strict=True);policy=UpstreamActor(actor,30,b.action_scale,b.obs_scale)
  else:policy=load_checkpoint(job['checkpoint'],b)
  buf=io.BytesIO();torch.save(policy,buf);buf.seek(0);policy=torch.load(buf,map_location='cuda',weights_only=False).eval()
  if hasattr(policy,'stochastic'):policy.stochastic=False
  scores=[];curves=[];online=0.;scoring=0.;cpu0=resource.getrusage(resource.RUSAGE_SELF);samples=[]
  with torch.no_grad(),torch.device('cuda'):
   negatives=torch.tensor(bank['contrasts'],device='cuda',dtype=torch.float64)
   for i in range(0,2000,16):
    if time.monotonic()>deadline:raise TimeoutError('Predeclared four-hour evaluation cap')
    torch.cuda.synchronize();tick=time.monotonic()
    data=rollout(b,policy,30,min(16,2000-i),2026110813902+i*97451,0,truth=torch.tensor(bank['truth'][i:i+16],device='cuda',dtype=torch.float64),noise=torch.tensor(bank['noise'][i:i+16,:,None],device='cuda',dtype=torch.float64))
    torch.cuda.synchronize();online+=time.monotonic()-tick;tick=time.monotonic();values,_=scorer(b,data,negatives,16384,'cuda');torch.cuda.synchronize();scoring+=time.monotonic()-tick
    scores.extend(values[:,-1].numpy());curves.append(values.numpy())
    if i%256==0:samples.append(dict(seconds=time.monotonic()-start,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader'],text=True).strip()))
    put(out/'STATUS.json',dict(status='EVALUATING',method=job['method'],seed=job['seed'],completed=min(i+16,2000),total=2000))
  scores=np.asarray(scores);np.save(out/'scores.npy',scores);np.save(out/'curves.npy',np.concatenate(curves));error=None
  if job['seed']==101:
   original=np.load(B/f'evaluation/{job["method"]}_scores.npy');error=float(np.abs(scores-original).max());assert error<1e-9,(job['method'],error)
  assert sha(job['checkpoint'])==job['checkpoint_sha256'];se=float(scores.std(ddof=1)/np.sqrt(2000));width=float(t.ppf(.975,1999))*se;cpu=resource.getrusage(resource.RUSAGE_SELF)
  result=dict(status='COMPLETE',method=job['method'],seed=job['seed'],mean=float(scores.mean()),MCSE=se,CI95=[float(scores.mean()-width),float(scores.mean()+width)],protocol=LOCK,checkpoint=job,
   evaluation_seconds=time.monotonic()-start,online_rollout_total_seconds=online,online_amortized_seconds_per_episode=online/2000,scorer_seconds=scoring,
   cpu_user_seconds=cpu.ru_utime-cpu0.ru_utime,cpu_system_seconds=cpu.ru_stime-cpu0.ru_stime,torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_samples=samples,
   simulator_observations=60000,scorer_density_calls=int(b.calls),seed101_saved_vector_max_error=error,training_updates_this_job=0,
   limitations='Historical training native schedules differ;pretraining/CPU/GPU details remain explicit. Single-bank case MCSE is separate from across-training-seed variability.')
  put(out/'RESULTS.json',result);put(out/'STATUS.json',dict(status='COMPLETE',method=job['method'],seed=job['seed'],mean=result['mean']))
 except Exception as e:put(out/'STATUS.json',dict(status='FAILED',error=repr(e)));raise
 finally:
  for f in held:f.close()

def queue():
 lock=(O/'.external_eval_queue.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=json.loads(CONFIG.read_text())
 for job in p['jobs']:
  out=O/'external_evaluations'/job['method']/('seed'+str(job['seed']))
  if out.exists():
   assert (out/'RESULTS.json').exists() and json.loads((out/'RESULTS.json').read_text())['status']=='COMPLETE','Preserve incomplete output;no retry'
   continue
  while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():put(O/'STATUS.json',dict(status='WAITING_FOR_UNRELATED_GPU'));time.sleep(30)
  put(O/'STATUS.json',dict(status='RUNNING_EXTERNAL_EVALUATION',method=job['method'],seed=job['seed'],training=False))
  with (O/(job['method']+'_seed'+str(job['seed'])+'.log')).open('x') as f:
   result=subprocess.run(['timeout','--signal=TERM','--kill-after=60','14340',PY,'-u',str(pathlib.Path(__file__).resolve()),'--worker',job['method'],'--seed',str(job['seed'])],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  if result.returncode:put(O/'STATUS.json',dict(status='BLOCKED',method=job['method'],seed=job['seed'],returncode=result.returncode,no_retry=True));return
  summary()
 put(O/'STATUS.json',dict(status='EXTERNAL_EVALUATIONS_COMPLETE',completed=15,remaining='MoE checkpoint audit and four-hour RL reproductions'))

def summary():
 import numpy as np,csv
 p=json.loads(CONFIG.read_text());rows=[]
 for j in p['jobs']:
  f=O/'external_evaluations'/j['method']/('seed'+str(j['seed']))/'RESULTS.json'
  if f.exists():
   d=json.loads(f.read_text());rows.append(dict(method=d['method'],seed=d['seed'],EIG=d['mean'],MCSE=d['MCSE'],CI95=d['CI95'],checkpoint_sha256=j['checkpoint_sha256']))
 put(R/'EXTERNAL_RESULTS.json',dict(status='COMPLETE' if len(rows)==15 else 'PENDING',completed=len(rows),required=15,rows=rows))
 with (R/'EXTERNAL_RESULTS.csv').open('w',newline='') as f:
  w=csv.writer(f,lineterminator='\n');w.writerow(['Method','101','102','103','104','105','Mean','SD'])
  for m in ['vsoed','scorebed','dad']:
   known={d['seed']:d['EIG'] for d in rows if d['method']==m};v=list(known.values());w.writerow([m,*[known.get(s,'PENDING') for s in [101,102,103,104,105]],float(np.mean(v)) if len(v)==5 else 'PENDING',float(np.std(v,ddof=1)) if len(v)==5 else 'PENDING'])

def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--launch',action='store_true');p.add_argument('--queue',action='store_true');p.add_argument('--worker');p.add_argument('--seed',type=int);args=p.parse_args()
 if args.prepare:
  assert not CONFIG.exists();O.mkdir(exist_ok=True,parents=True);put(CONFIG,manifest());print(CONFIG)
 elif args.launch:
  assert not (O/'EVALUATION_LAUNCH.json').exists()
  with (O/'evaluation_controller.log').open('x') as f:child=subprocess.Popen([PY,'-u',str(pathlib.Path(__file__).resolve()),'--queue'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
  put(O/'EVALUATION_LAUNCH.json',dict(pid=child.pid,training=False,sequential_jobs=15,started=time.time()));print(child.pid)
 elif args.queue:queue()
 elif args.worker:worker(next(j for j in json.loads(CONFIG.read_text())['jobs'] if j['method']==args.worker and j['seed']==args.seed))
 else:p.error('Explicit action required')
if __name__=='__main__':main()
