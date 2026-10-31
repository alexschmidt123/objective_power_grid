"""Full six-baseline testing extension; active parent source is never modified."""
import time
ENTRY=time.monotonic()
import argparse,json,os,subprocess,hashlib,tarfile,traceback,csv,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def root_command(module,args,jax=False,cpu=False):
 # Use the documented sourced-root setup without changing its frozen dispatch.
 interpreter=str(ROOT.parent/'objective_power_grid/.venv_scorebed/bin/python') if jax else 'python3'
 cmd=['bash','-c','source "$1/run.sh"; shift; exec "$1" -m "$2" "${@:3}"','sir-t3-full-testing',str(ROOT),interpreter,module,*args]
 env=dict(os.environ)
 if cpu:env.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 elif jax:env.update(JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false')
 return cmd,env

def fixed_continue(C,cfg):
 import numpy as np,torch
 from src.research.sir_t3_campaign import BudgetObserver,lock_gpu
 from src.research.compute_budget import ComputeClock
 from src.research.sir_t3_common import T3SIR,fixed_shared_spce
 P=Path(cfg['full_testing']['parent_campaign']);original=json.loads((P/'manifests/fixed_complete.json').read_text());credit=original['charged_training']['charged_seconds'];old=original['checkpoint'];assert sha(old['path'])==old['sha256'];locks=lock_gpu();torch.set_num_threads(4)
 saved=torch.load(old['path'],map_location='cpu',weights_only=False);p=saved['policy'].cuda();opt=torch.optim.Adam(p.parameters(),lr=5e-5,betas=(.8,.998));opt.load_state_dict(saved['optimizer']);scheduler=torch.optim.lr_scheduler.ExponentialLR(opt,.98);scheduler.load_state_dict(saved['scheduler']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);np.random.set_state(saved['numpy_rng'])
 class ContinuedObserver(BudgetObserver):
  def __init__(self):
   self.C,self.method,self.cfg=C,'fixed',cfg;self.directory=C/'checkpoints/fixed';self.directory.mkdir(exist_ok=False);self.compute_clock=ComputeClock(self.directory/'compute_clock.json',14400,credit_seconds=credit,hardware=torch.cuda.get_device_name(0),start=ENTRY);self.bound=1.;self.last=0.;self.next_checkpoint=7200.;self.next_recovery=credit+300;self.rows=json.loads((P/'logs/fixed_learning_curve.json').read_text());self.selected=None;self.runtime=None;self.update=saved['update'];self.episodes=saved['episodes']
 observer=ContinuedObserver();observer.set_runtime(dict(policy=p,optimizer=opt,scheduler=scheduler));observer.save(p,saved['episodes'],saved['update'],opt,scheduler,'continuation_start');observer.state('CONTINUING_WITH_CREDIT',previous_charged_seconds=credit)
 count=saved['episodes'];update=saved['update'];history=[]
 try:
  with torch.device('cuda:0'):
   b=T3SIR(cfg['benchmark'])
   while observer.allow_next_unit(p,count,update,opt,scheduler):
    update+=1;opt.zero_grad();loss=-fixed_shared_spce(b,p,2000,2000,101+update*104729).mean();loss.backward()
    if not torch.isfinite(loss) or not all(torch.isfinite(x.grad).all() for x in p.parameters() if x.grad is not None):raise FloatingPointError('Nonfinite continued Fixed objective/gradient')
    opt.step()
    if (update-1)%1000==0:scheduler.step()
    count+=2000
    if update%100==0:history.append(dict(update=update,loss=float(loss.detach()),lr=opt.param_groups[0]['lr'],charged_seconds=observer.compute_clock.spent()));write(C/'logs/fixed_optimization.json',history)
    if observer.check(p,count,update,opt,scheduler):break
   observer.finish(p)
  result=dict(method='fixed',training_seed=101,horizon=3,optimizer_updates=observer.update,generated_training_trajectories=observer.episodes,contrasts=2000,native_schedule_complete=False,scientific_status='COMPUTE_LIMITED',checkpoint=observer.selected,charged_training=observer.compute_clock.stop(),previous_training_credit_seconds=credit,previous_checkpoint=old,adaptation='Same optimizer/scheduler/RNG continued; old50k schedule insufficient to infer convergence; total capped at4h including old work')
  write(C/'manifests/fixed_complete.json',result);observer.state('TRAINING_COMPLETE',checkpoint=observer.selected)
 except Exception as e:write(C/'manifests/fixed_failure.json',dict(error=str(e),traceback=traceback.format_exc(),clock=observer.compute_clock.stop(),last_checkpoint=observer.selected));raise

def bridge(C,cfg,smoke=False):
 import numpy as np,torch
 from src.research.sir_scorebed_gru import SIRScoreGRUPolicy
 from src.research.sir_scorebed import verify_bridge
 from src.research.sir_t3_common import T3SIR
 folder=C/('cpu_smoke/scorebed' if smoke else 'checkpoints/scorebed');result=json.loads((folder/'complete.json').read_text());assert result['budget_eligible'];policies=[];receipts=[]
 with torch.device('cpu'):
  b=T3SIR(cfg['benchmark'])
  for r in range(result['policy_restarts']):
   path=folder/f'restart{r}_ema_policy.npz';a=np.load(path);p=SIRScoreGRUPolicy(a,3,b).eval();error=verify_bridge(p,a);target=folder/f'restart{r}_policy.pt';torch.save(dict(policy=p,method='scorebed',horizon=3,training_seed=101,restart=r,score_updates=result['score_updates'],policy_updates=result['policy_updates_per_restart'],source_manifest_sha256=result['source_manifest_sha256']),target);receipts.append(dict(restart=r,path=str(target),sha256=sha(target),bridge_max_absolute_error=error));policies.append(target)
 intermediate=[]
 source=folder/'intermediate_exports.json'
 if source.exists():
  for row in json.loads(source.read_text()):
   arrays=np.load(row['path']);p=SIRScoreGRUPolicy(arrays,3,b).eval();target=Path(row['path']).with_suffix('.pt');torch.save(dict(policy=p,method='scorebed',restart=0,horizon=3,update=row['round']),target);intermediate.append(dict(path=str(target),update=row['round'],charged_seconds=row['charged_seconds'],sha256=sha(target)))
  write(folder/'intermediate_torch_policies.json',intermediate)
 intermediate=list({row['update']:row for row in sorted(intermediate,key=lambda x:x['charged_seconds'])}.values())
 write(folder/'intermediate_torch_policies.json',intermediate)
 write(folder/'bridge_verification.json',dict(status='PASS',policies=receipts,intermediate_policies=intermediate))
 if not smoke:
  primary=receipts[0];data=dict(method='scorebed',training_seed=101,horizon=3,optimizer_updates=result['policy_updates_per_restart']*result['policy_restarts'],score_optimizer_updates=result['score_updates'],generated_training_trajectories=result['generated_training_trajectories'],contrasts='native score gradient M1024',native_schedule_complete=result['native_schedule_complete'],scientific_status='COMPUTE_LIMITED',checkpoint=primary,charged_training=dict(charged_seconds=result['charged_training_seconds']+time.monotonic()-ENTRY,budget_seconds=14400,native_training_seconds=result['charged_training_seconds'],CPU_adapter_checkpoint_seconds=time.monotonic()-ENTRY),all_restarts=receipts,primary_restart=0,checkpoint_selection='predeclared restart0; all ten EMA finals evaluated, no test-based selection',policy_updates_per_restart=result['policy_updates_per_restart'])
  if data['charged_training']['charged_seconds']>14400:raise RuntimeError('ScoreBED checkpoint adapter exceeded full-method4h cap; do not report eligible result')
  write(C/'manifests/scorebed_complete.json',data)
 print(json.dumps(dict(status='PASS',policies=receipts),indent=2))

def full_evaluation(C,cfg):
 import numpy as np,torch
 from src.research.sir_t3_campaign import lock_gpu
 from src.research.sir_t3_evaluation import evaluate,stats
 locks=lock_gpu();torch.set_num_threads(4);P=Path(cfg['full_testing']['parent_campaign']);models={'random':None};records={};other_score={}
 for m in ['fixed','dad','rl_sboed','vsoed','scorebed']:
  owner=C if m in ['fixed','dad','scorebed'] else P;record=owner/'manifests'/f'{m}_complete.json'
  if not record.exists():raise RuntimeError('Full testing incomplete: '+m+' has no trained budget-eligible checkpoint')
  x=json.loads(record.read_text());cp=x['checkpoint'];assert sha(cp['path'])==cp['sha256'];models[m]=torch.load(cp['path'],map_location='cpu',weights_only=False)['policy'];records[m]=x
 if records['scorebed'].get('all_restarts'):
  for r in records['scorebed']['all_restarts']:
   if r['restart']>0:other_score[f'scorebed_restart{r["restart"]}']=torch.load(r['path'],map_location='cpu',weights_only=False)['policy']
 # All resolution decisions are based solely on development resolution cases.
 candidates=[16384,65536,262144,1000000];resolution={};audits=[]
 for m,p in {**models,**other_score}.items():x,meta=evaluate(cfg['benchmark'],p,2026102852,256,candidates,C/'evaluations/resolution',m,method=m);resolution[m]=x
 chosen=1000000
 for L in candidates:
  checks={};passed=True
  for m,x in resolution.items():
   change=stats(x[L]-x[1000000]);limit=min(.02,.25*max(stats(x[1000000])['MC_SE'],1e-12));ok=abs(change['mean'])+1.96*change['MC_SE']<=limit;checks[m]=dict(change=change,threshold=limit,passes=ok);passed&=ok
  for a in resolution:
   for b in resolution:
    if a<=b:continue
    d=stats((resolution[a][L]-resolution[b][L])-(resolution[a][1000000]-resolution[b][1000000]));ok=abs(d['mean'])+1.96*d['MC_SE']<=.02;passed&=ok;checks[a+'-'+b]=dict(change=d,passes=ok)
  audits.append(dict(contrasts=L,passes=passed,checks=checks))
  if passed:chosen=L;break
 write(C/'manifests/frozen_resolution.json',dict(contrasts=chosen,development_seed=2026102852,systems=256,audits=audits,confirmation_not_used_for_selection=True))
 final={};scores={}
 for m,p in {**models,**other_score}.items():
  x,meta=evaluate(cfg['benchmark'],p,cfg['seeds']['confirmation'],2048,[chosen],C/'evaluations/confirmation',m,method='scorebed' if m.startswith('scorebed_restart') else m);scores[m]=x[chosen];final[m]=dict(**stats(x[chosen]),metadata=meta);write(C/'evaluations/final_results.json',final)
 paired={a+'-'+b:stats(scores[a]-scores[b]) for a in models for b in models if a>b};restart_names=['scorebed']+list(other_score);restart_mean=np.stack([scores[m] for m in restart_names]).mean(0);write(C/'evaluations/scorebed_restart_summary.json',dict(primary_restart=0,all_restarts=restart_names,mean_across_restarts=stats(restart_mean),between_restart_SD=float(np.std([final[m]['mean'] for m in restart_names],ddof=1)),claim='Secondary restart summary, not a selected best policy'))
 write(C/'evaluations/paired_differences.json',paired)
 # Full held-out assessment of every available canonical timepoint, no tuning.
 curves={};convergence={}
 for m in ['fixed','dad','rl_sboed','vsoed']:
  owner=C if m in ['fixed','dad'] else P;path=owner/'logs'/f'{m}_learning_curve.json';curve=json.loads(path.read_text()) if path.exists() else [];assessed=[]
  # Choose by proximity to requested training times, never by EIG.
  selected=[]
  for target in [1800,3600,7200,14400]:
   options=[r for r in curve if abs(r['charged_seconds']-target)<180]
   if options:
    row=min(options,key=lambda r:abs(r['charged_seconds']-target))
    if row not in selected:selected.append(row)
  if curve and curve[-1] not in selected:selected.append(curve[-1])
  for row in selected:
   cp=row['checkpoint'];p=torch.load(cp['path'],map_location='cpu',weights_only=False)['policy'];x,meta=evaluate(cfg['benchmark'],p,2026102851,2048,[chosen],C/'evaluations/learning',m+f'_update{row["update"]}');assessed.append(dict(update=row['update'],charged_seconds=row['charged_seconds'],scores=stats(x[chosen]),path=str(C/'evaluations/learning'/f'{m}_update{row["update"]}_L{chosen}.npy')))
  assessed.sort(key=lambda r:r['charged_seconds']);changes=[]
  for a,b in zip(assessed[-3:-1],assessed[-2:]):
   if b['charged_seconds']-a['charged_seconds']<1500:continue
   d=stats(np.load(b['path'])-np.load(a['path']));changes.append(dict(**d,upper_absolute_change=abs(d['mean'])+1.96*d['MC_SE']))
  supported=len(changes)==2 and all(x['upper_absolute_change']<=.05 for x in changes);convergence[m]=dict(status='CONVERGED_OR_PLATEAU_SUPPORTED' if supported else 'COMPUTE_LIMITED',paired_changes=changes,selection='last canonical timepoints, never maximum EIG');curves[m]=assessed
 score_curve=[]
 for row in json.loads((C/'checkpoints/scorebed/intermediate_torch_policies.json').read_text()):
  p=torch.load(row['path'],map_location='cpu',weights_only=False)['policy'];x,meta=evaluate(cfg['benchmark'],p,2026102851,2048,[chosen],C/'evaluations/learning',f'scorebed_update{row["update"]}');score_curve.append(dict(update=row['update'],charged_seconds=row['charged_seconds'],scores=stats(x[chosen]),metadata=meta))
 curves['scorebed']=score_curve
 convergence['scorebed']=dict(status='COMPUTE_LIMITED',reason='Score phase truncated; all ten policy restarts tested. Native full score schedule incomplete; no full-method convergence claim.')
 write(C/'evaluations/full_learning_curves.json',curves);write(C/'evaluations/convergence.json',convergence)
 fields=['method','training_seed','horizon','checkpoint','optimizer_updates','training_trajectories','training_contrasts','charged_training_seconds','native_schedule_complete','eig_mean','eig_mcse','eig_ci_low','eig_ci_high','status']
 with (C/'sir_t3_baseline_results.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for m in models:
   r=records.get(m,{});s=final[m];w.writerow(dict(method=m,training_seed=101,horizon=3,checkpoint=r.get('checkpoint',{}).get('path',''),optimizer_updates=r.get('optimizer_updates',0),training_trajectories=r.get('generated_training_trajectories',0),training_contrasts=r.get('contrasts','N/A'),charged_training_seconds=r.get('charged_training',{}).get('charged_seconds',0),native_schedule_complete=r.get('native_schedule_complete',m=='random'),eig_mean=s['mean'],eig_mcse=s['MC_SE'],eig_ci_low=s['CI95_low'],eig_ci_high=s['CI95_high'],status=convergence.get(m,{}).get('status','NO_TRAINING_REQUIRED')))
 report='# Full capped six-baseline assessment\n\n2048 paired confirmation trajectories; common contrasts='+str(chosen)+'. Primary ScoreBED restart0 was predeclared; all10restart results and secondary mean/spread retained. No statistical superiority from one training seed.\n\n|Method|EIG nats|MCSE|95%CI|Status|\n|---|---:|---:|---|---|\n'
 for m in models:
  s=final[m];report+=f"|{m}|{s['mean']:.6f}|{s['MC_SE']:.6f}|[{s['CI95_low']:.6f},{s['CI95_high']:.6f}]|{convergence.get(m,{}).get('status','No training')}|\n"
 (C/'reports/SIR_T3_FULL_TESTING_RESULTS.md').write_text(report)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(7,4))
 for method,curve in curves.items():
  if curve:ax.errorbar([x['charged_seconds']/3600 for x in curve],[x['scores']['mean'] for x in curve],yerr=[1.96*x['scores']['MC_SE'] for x in curve],marker='o',label=method)
 ax.set(xlabel='Total charged training hours',ylabel='Held-out terminal EIG, nats');ax.legend();fig.tight_layout();fig.savefig(C/'reports/full_learning_curves.png',dpi=180);fig.savefig(C/'reports/full_learning_curves.svg');plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,4));names=list(models);ax.bar(names,[final[m]['mean'] for m in names],yerr=[1.96*final[m]['MC_SE'] for m in names]);ax.set_ylabel('Terminal EIG, nats;95%MC interval');fig.tight_layout();fig.savefig(C/'reports/full_eig_comparison.png',dpi=180);plt.close(fig)


def queue(C,cfg):
 P=Path(cfg['full_testing']['parent_campaign']);queue_lock=(C/'manifests/controller.lock').open('a+')
 import fcntl
 fcntl.flock(queue_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def status(state,**kw):write(C/'status.json',dict(state=state,PID=os.getpid(),updated_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw))
 while True:
  parent=json.loads((P/'status.json').read_text())
  if parent['state'] in ['COMPLETE','COMPLETE_WITH_READINESS_BLOCKERS']:break
  if not Path('/proc/'+str(parent['PID'])).exists():status('BLOCKED_PARENT_EXITED',parent_status=parent);return
  status('QUEUED_AFTER_ACTIVE_CAMPAIGN',parent_status=parent);time.sleep(30)
 # Wait for normal parent exit, avoiding its final CUDA-context teardown.
 while Path('/proc/'+str(parent['PID'])).exists():
  proc=Path('/proc/'+str(parent['PID']))
  if proc.joinpath('stat').read_text().split()[2]=='Z':break
  if b'src.research.sir_t3_campaign' not in proc.joinpath('cmdline').read_bytes():break
  status('WAITING_PARENT_PROCESS_EXIT',parent_PID=parent['PID']);time.sleep(10)
 jobs=[('fixed',None),('dad',None),('scorebed',None),('export',None),('bridge',None),('final',None)]
 for phase,_ in jobs:
  done=C/'manifests'/f'{phase}_exit.json'
  if done.exists():
   if json.loads(done.read_text())['returncode']!=0:status('FAILED_REQUIRES_REVIEW',phase=phase);return
   continue
  for rel,digest in json.loads((C/'manifests/source_sha256.json').read_text()).items():
   if sha(ROOT/rel)!=digest:raise RuntimeError('Frozen full-testing source changed: '+rel)
  if phase=='dad':cmd=['bash','run.sh','--sir-t3-campaign','--campaign',str(C),'--phase','train','--method','dad'];env=dict(os.environ)
  elif phase=='scorebed':cmd,env=root_command('src.research.sir_t3_full_score',['--campaign',str(C)],jax=True)
  elif phase=='export':cmd,env=root_command('src.research.sir_t3_full_score',['--campaign',str(C),'--export-intermediates'],jax=True,cpu=True)
  else:cmd,env=root_command('src.research.sir_t3_fulltest',['--campaign',str(C),'--phase',phase],cpu=phase=='bridge')
  with (C/'logs'/f'{phase}.log').open('x') as f:
   p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);status('RUNNING',phase=phase,child_PID=p.pid)
   try:rc=p.wait(timeout=14520 if phase=='scorebed' else 18000 if phase in ['fixed','dad'] else 86400)
   except subprocess.TimeoutExpired:
    import signal
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    rc=124
  write(done,dict(returncode=rc))
  if rc:status('FAILED_REQUIRES_REVIEW',phase=phase,returncode=rc);return
 status('FULL_TESTING_COMPLETE')

def main():
 import yaml
 ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);ap.add_argument('--phase',choices=['queue','fixed','bridge','final','cpu-bridge'],default='queue');args=ap.parse_args();C=Path(args.campaign).resolve()
 if ROOT.name!='objective_power_grid_temp' or ROOT/'scratch_runs/sir_ode_t3_baselines' not in C.parents:raise ValueError('TEMP T3 root only')
 cfg=yaml.safe_load((C/'configs/sir_ode_t3_continuous_v1.yaml').read_text())
 if args.phase=='queue':queue(C,cfg)
 elif args.phase=='fixed':fixed_continue(C,cfg)
 elif args.phase in ['bridge','cpu-bridge']:bridge(C,cfg,args.phase=='cpu-bridge')
 elif args.phase=='final':full_evaluation(C,cfg)
if __name__=='__main__':main()
