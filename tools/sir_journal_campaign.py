"""Guarded LabPC-only six-baseline journal campaign; native training via run.sh."""
import argparse,copy,fcntl,hashlib,json,os,signal,subprocess,threading,time
from pathlib import Path
import yaml,numpy as np
R=Path(__file__).resolve().parents[1];C=R/'experiments/10262026/10262026_sir_journal_corrected_seed101_r2'
METHODS=['random','fixed','dad','rl_sboed','vsoed','scorebed'];ORDER=['dad','vsoed','scorebed','rl_sboed']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');q.replace(p)
def prepare():
 assert not C.exists(),'Never overwrite an existing campaign'
 for d in ['configs','metadata','runs','readiness','logs','reports','evaluations']:(C/d).mkdir(parents=True)
 raw=yaml.safe_load((R/'configs/sir_journal_corrected.yaml').read_text());assert raw['experiment']['methods']==METHODS
 (C/'configs/full.yaml').write_text(yaml.safe_dump(raw,sort_keys=False))
 from src.research.original_protocol import score_allocation
 for m,episodes in [('dad',2000),('vsoed',21000),('scorebed',102190),('rl_sboed',100)]:
  x=copy.deepcopy(raw);p=x['research']['journal_protocol'];p['stopping_rule'].update(minimum_episodes=episodes,maximum_episodes=episodes,maximum_training_seconds=900);p['validation'].update(systems=4,contrasts=8,batch_size=2,chunk=4);p['preserve_progress_checkpoints']=False
  if m=='scorebed':p['method_settings'][m]['scorebed_pretraining_episodes']=score_allocation(episodes,'sir_ode')['score']
  (C/f'configs/probe_{m}.yaml').write_text(yaml.safe_dump(x,sort_keys=False))
 old=Path.home()/'Documents/objective_power_grid_e1_sir_native_20261018/experiments/10182026/10182026_e1_sir_t5_seed101/sir/seed101/fixed'
 t=read(old/'models/fixed_training.json');cfg=read(old/'run_config.json')
 assert cfg['hardware']['execution_site']=='labpc' and '4090' in cfg['hardware']['gpu_model'];assert cfg['train_seed']==101
 assert t['generated_training_trajectories']==t['selected_checkpoint']['episodes']==1000000 and t['checkpoint_selection']=='final_predeclared_fixed_budget'
 assert sha(old/'models/fixed.pt')=='61cfe6a8fdb5c4aaa23abba99332353671ed3f5e6933cc127cb1459a65945982'
 from src.research.journal_checkpoints import load_checkpoint
 from src.research.benchmarks import make_benchmark
 load_checkpoint(old/'models/fixed.pt',make_benchmark(raw))
 reuse=dict(run=str(old),checkpoint=str(old/'models/fixed.pt'),checkpoint_sha256=sha(old/'models/fixed.pt'),training_json=str(old/'models/fixed_training.json'),status='VALID_BUT_REEVALUATION_REQUIRED',training_host=cfg['hardware'],source_manifest=str(old/'source_manifest.json'),source_manifest_sha256=sha(old/'source_manifest.json'),config_sha256=sha(old/'run_config.json'))
 write(C/'metadata/fixed_reuse.json',reuse)
 old_csv=Path.home()/'Documents/objective_power_grid_sir_seed101_recovery/reports/SIR_SEED101_ALL_BASELINES.csv'
 import csv
 audits=[]
 for row in csv.DictReader(old_csv.open()):
  if row['method'] not in METHODS:continue
  audits.append(dict(method=row['method'],classification='VALID_LABPC_COMPLETE' if row['method']=='random' else 'VALID_BUT_REEVALUATION_REQUIRED' if row['method']=='fixed' else 'PRE_CORRECTION_INVALID',reason='No training; reuse sampling protocol' if row['method']=='random' else 'Verified LabPC1M final fixed checkpoint; common evaluation to rerun' if row['method']=='fixed' else 'Historical resource-adapted hyperparameters do not meet strict journal profile; retain as historical, not primary',source_commit=row['source_commit'],checkpoint_source=row['checkpoint_source'],checkpoint_sha256=row['checkpoint_sha256']))
 write(C/'metadata/historical_audit.json',audits)
 tracked=subprocess.check_output(['git','ls-files'],cwd=R,text=True).splitlines();files=[p for p in tracked if not p.startswith('historical/') and (p.endswith('.py') or p.endswith('.sh') or p=='configs/sir_journal_corrected.yaml')]+[str(p.relative_to(R)) for p in (C/'configs').glob('*.yaml')]
 write(C/'source_manifest.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),sha256={p:sha(R/p) for p in files}))
 write(C/'plan.json',dict(methods=METHODS,training_order=ORDER,training_seed=101,analysis=raw['study_analysis'],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),config_sha256=sha(R/'configs/sir_journal_corrected.yaml'),hostname=subprocess.check_output(['hostname'],text=True).strip(),confirmation_seed=98502691,reserved_seed8831001_used=False))
 write(C/'registry.json',[]);write(C/'status.json',dict(state='PREPARED',completed=0,methods=METHODS));print(C)
def guard():
 for rel,v in read(C/'source_manifest.json')['sha256'].items():assert sha(R/rel)==v,rel
 assert 'ecen-80r5x04' in subprocess.check_output(['hostname'],text=True),'LabPC only'
def acquire(label):
 p=Path.home()/'.cache/objective_power_grid/device_0.lock';p.parent.mkdir(parents=True,exist_ok=True)
 while True:
  lock=p.open('a+')
  try:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   if not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():return lock
  except BlockingIOError:pass
  lock.close();write(C/'status.json',dict(state='WAITING_LOCAL_DEVICE_LOCK',phase=label,pid=os.getpid()));time.sleep(15)
def execute(m,probe=False):
 guard();label=('probe_' if probe else '')+m;out=C/('readiness' if probe else 'runs')/m
 assert not out.exists(),'Preserve partial run; no implicit restart'
 cfg=C/'configs'/(f'probe_{m}.yaml' if probe else 'full.yaml');lock=acquire(label);start=time.perf_counter();peak=[];stop=threading.Event()
 def monitor():
  while not stop.wait(1):
   try:peak.append(float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0]))
   except Exception:pass
 threading.Thread(target=monitor,daemon=True).start()
 cmd=['bash','run.sh','--config',str(cfg),'--journal','--train-only','--seed','101','--eval-seeds','1001','--eig-only','--methods',m,'--source-check-manifest',str(C/'source_manifest.json'),'--output',str(out)]
 write(C/f'metadata/{label}_command.json',dict(command=cmd));write(C/'status.json',dict(state='RUNNING_NATIVE_READINESS' if probe else 'TRAINING_LABPC_SEED101',method=m,pid=os.getpid(),output=str(out)))
 rc=-1;failure=None
 try:
  with (C/f'logs/{label}.log').open('x') as log:
   child=subprocess.Popen(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);write(C/f'metadata/{label}_pid.json',dict(pid=child.pid,coordinator=os.getpid()))
   try:rc=child.wait(timeout=900 if probe else 259200)
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=20)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    rc=124;failure='Prespecified resource cap; no reduced-budget completion'
  if rc==0:
   t=read(out/f'models/{m}_training.json');assert t['checkpoint_sha256']==sha(out/f'models/{m}.pt')
   if not probe:assert t['generated_training_trajectories']==t['selected_checkpoint']['episodes']==1000000 and t['checkpoint_selection']=='final_predeclared_fixed_budget'
  elif (out/'failure.json').exists():failure=read(out/'failure.json')
 except Exception as e:rc=-1;failure=repr(e)
 finally:stop.set();lock.close()
 result=dict(method=m,probe=probe,status='COMPLETE' if rc==0 else 'INCOMPLETE_BUDGET_LIMITED' if rc==124 else 'INCOMPLETE',returncode=rc,failure=failure,run=str(out),GPU_locked_wall_seconds=time.perf_counter()-start,peak_device_memory_MiB=max(peak) if peak else None,checkpoint_sha256=sha(out/f'models/{m}.pt') if rc==0 else None)
 write(C/f'metadata/{label}_completion.json',result);reg=read(C/'registry.json');reg.append(result);write(C/'registry.json',reg);return rc==0

def policy(method):
 from src.research.benchmarks import make_benchmark
 from src.research.journal_checkpoints import load_checkpoint
 raw=yaml.safe_load((C/'configs/full.yaml').read_text())
 if method=='random':
  from src.research.deepseek_sboed_experiment import RandomControl
  return RandomControl(raw,97402641)
 cp=Path(read(C/'metadata/fixed_reuse.json')['checkpoint']) if method=='fixed' else C/f'runs/{method}/models/{method}.pt'
 return load_checkpoint(cp,make_benchmark(raw))

def calibrate(available):
 import torch
 from src.research.journal_validation import CommonValidation
 cfg=yaml.safe_load((C/'configs/full.yaml').read_text());a=cfg['study_analysis'];lock=acquire('EIG_RESOLUTION');start=time.perf_counter();data={};complete=[];counts=a['calibration_contrasts'];last=None
 try:
  for L in counts:
   level={}
   for seed in [a['calibration_seed'],a['calibration_replicate_seed']]:
    # Bank prefixes are nested within each fixed truth/noise split.
    val=CommonValidation(cfg,5,seed,a['calibration_systems'],max(counts),batch=16,chunk=4096,device='cuda:0')
    for m in available:
     if time.perf_counter()-start>a['calibration_gpu_seconds']:break
     values,meta=val(policy(m),contrasts=L);level[f'{seed}:{m}']=values
     np.save(C/f'evaluations/calibration_{seed}_{m}_{L}.npy',values)
    if len(level)<(1 if seed==a['calibration_seed'] else 2)*len(available):break
   if len(level)!=2*len(available):break
   complete.append(L);data[L]=level;last=L
   if time.perf_counter()-start>a['calibration_gpu_seconds']:break
  chosen=complete[-1] if complete else 1000000;resolved=False;metrics=[]
  for lower,higher in [(lower,complete[-1]) for lower in complete[:-1]]:
   absolute=[];paired=[];se_values=[]
   for seed in [a['calibration_seed'],a['calibration_replicate_seed']]:
    shifts={m:data[higher][f'{seed}:{m}']-data[lower][f'{seed}:{m}'] for m in available}
    for m,d in shifts.items():
     se=float(d.std(ddof=1)/len(d)**.5);absolute.append(abs(float(d.mean()))+1.96*se);se_values.append(se)
    for i,m in enumerate(available):
     for n in available[i+1:]:
      d=shifts[m]-shifts[n];se=float(d.std(ddof=1)/len(d)**.5);paired.append(abs(float(d.mean()))+1.96*se);se_values.append(se)
   ok=max(absolute,default=1)<a['absolute_sensitivity'] and max(paired,default=1)<a['paired_sensitivity'] and max(se_values,default=1)<a['shift_MC_SE_target']
   metrics.append(dict(lower=lower,higher=higher,max_absolute_95_bound=max(absolute),max_paired_95_bound=max(paired,default=0),max_shift_SE=max(se_values),PASS=ok))
   if ok and len(available)==6:chosen=lower;resolved=True;break
  result=dict(status='CALIBRATED_AT_DECLARED_SENSITIVITY' if resolved else 'PROVISIONAL_FINITE_CONTRAST',contrasts=chosen,completed_counts=complete,available_methods=available,metrics=metrics,GPU_locked_seconds=time.perf_counter()-start,remaining_tail_note='Finite sensitivity checks do not prove unbiased true EIG; common-bank MCSE excludes contrast-bank uncertainty')
  write(C/'evaluations/evaluator_freeze.json',result);return result
 finally:lock.close()

def evaluate_all(available,freeze,confirmation=False):
 import torch
 from src.research.benchmarks import make_benchmark
 from src.research.journal_validation import CommonValidation
 from src.research.engine import rollout
 from src.research.sir_evaluation import contrast_scores
 cfg=yaml.safe_load((C/'configs/full.yaml').read_text());L=freeze['contrasts'];N=2048;seed=98502691 if confirmation else 1001;lock=acquire('CONFIRMATION' if confirmation else 'COMMON_EVALUATION');start=time.perf_counter();label='confirmation' if confirmation else 'development'
 try:
  val=CommonValidation(cfg,5,seed,N,L,batch=16,chunk=4096,device='cuda:0')
  if not confirmation:
   bankdir=Path.home()/'Documents/objective_power_grid_e1_sir_replication_20261019/experiments/10192026/10192026_sir_e1_five_seed_verified/evaluations'
   if not (bankdir/'cases.npz').exists():bankdir=bankdir/'development_bank'
   assert sha(bankdir/'cases.npz')=='4b0a17909a46726b794316dd5ffc979b2c77b1bb3b1f2fab4845ddaa46703d25'
   cases=np.load(bankdir/'cases.npz');val.truth=torch.from_numpy(cases['truth']).double();val.noise=torch.from_numpy(cases['noise']).double()
   existing=np.load(bankdir/'contrasts_1m.npy');n=min(L,len(existing));val.bank[:n]=torch.from_numpy(existing[:n]).double()
  identity=hashlib.sha256(val.truth.numpy().tobytes()+val.noise.numpy().tobytes()+val.bank.numpy().tobytes()).hexdigest();val.identity=identity
  write(C/f'evaluations/{label}_bank_identity.json',dict(identity=identity,seed=seed,systems=N,contrasts=L,paired_exogenous_draws=True,not_training_seed_variability=True))
  np.savez(C/f'evaluations/{label}_cases.npz',truth=val.truth.numpy(),noise=val.noise.numpy());np.save(C/f'evaluations/{label}_bank.npy',val.bank.numpy())
  for m in available:
   target=C/f'evaluations/{label}_{m}.npz'
   if target.exists():continue
   p=policy(m);scores,meta=val(p);se=float(scores.std(ddof=1)/N**.5)
   np.savez(target,scores=scores,identity=identity,seed=seed,contrasts=L)
   # Pure single-history frozen decision latency; simulator work excluded.
   q=p.to('cuda:0').eval();b=make_benchmark(cfg)
   with torch.device('cuda:0'),torch.inference_mode():
    trace=rollout(b,q,5,1,97402651,0);timings=[]
    for rep in range(220):
     stage=rep%5;f=trace['features'][stage];torch.cuda.synchronize();t=time.perf_counter();z=q(f,stage)
     if not getattr(q,'outputs_design',False):b.feasible(z,trace['actions'][:stage],5)
     torch.cuda.synchronize()
     if rep>=20:timings.append(time.perf_counter()-t)
   write(C/f'evaluations/{label}_{m}.json',dict(method=m,EIG=float(scores.mean()),MC_SE=se,CI95=[float(scores.mean()-1.96*se),float(scores.mean()+1.96*se)],metadata=meta,identity=identity,latency_seconds=dict(mean=float(np.mean(timings)),median=float(np.median(timings)),p95=float(np.quantile(timings,.95))),precision_status=freeze['status']))
  write(C/f'evaluations/{label}_cost.json',dict(GPU_locked_seconds=time.perf_counter()-start,methods=available))
 finally:lock.close()

def report():
 from tools.sir_journal_report import generate
 generate(C)
def run():
 with (C/'coordinator.lock').open('a+') as coordinator:
  fcntl.flock(coordinator,fcntl.LOCK_EX|fcntl.LOCK_NB);guard();write(C/'metadata/coordinator.json',dict(pid=os.getpid(),source_commit=read(C/'source_manifest.json')['commit']))
  for m in ORDER:
   reg=read(C/'registry.json')
   if any(x['method']==m and not x['probe'] for x in reg):continue
   probe=next((x for x in reg if x['method']==m and x['probe']),None)
   if (probe['status']=='COMPLETE' if probe else execute(m,True)):execute(m)
   report()
  available=['random','fixed']+[m for m in ORDER if any(x['method']==m and not x['probe'] and x['status']=='COMPLETE' for x in read(C/'registry.json'))]
  freeze=calibrate(available);evaluate_all(available,freeze);report()
  if len(available)==6 and freeze['status']=='CALIBRATED_AT_DECLARED_SENSITIVITY':evaluate_all(available,freeze,True);report()
  write(C/'status.json',dict(state='FINISHED' if len(available)==6 else 'FINISHED_WITH_INCOMPLETE_METHODS',completed_methods=available,completed_count=len(available),evaluator=freeze,confirmation_performed=len(available)==6 and freeze['status']=='CALIBRATED_AT_DECLARED_SENSITIVITY',no_replication_seeds_launched=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','report']);a=p.parse_args();globals()[a.action]()
