"""Independent Ubuntu T3 campaign. Profiles gate training; interrupted jobs never restart automatically."""
import time
ENTRY=time.monotonic()
import argparse,copy,csv,fcntl,hashlib,json,math,os,signal,subprocess,sys,tarfile,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,torch,yaml,cloudpickle
from src.research.sir_t3_common import T3SIR,fixed_shared_spce
from src.research.sir_t3_evaluation import evaluate,stats
from src.research.compute_budget import ComputeClock
ROOT=Path(__file__).resolve().parents[2];METHODS=['random','fixed','dad','rl_sboed','vsoed','scorebed']

def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source_manifest():
 paths=[ROOT/'run.sh',ROOT/'sweep_run.sh']
 for d in ['src','tools','configs','scripts']:paths.extend(p for p in (ROOT/d).rglob('*') if p.is_file() and p.suffix in ['.py','.yaml','.yml','.sh','.json'] and '__pycache__' not in p.parts)
 return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
def lock_gpu():
 handles=[]
 for p in [Path.home()/'.cache/objective_power_grid/device_0.lock',ROOT/'.gpu0.lock',ROOT.parent/'objective_power_grid/.gpu0.lock']:
  p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(f)
 busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 if busy:raise RuntimeError('Foreign GPU jobs present: '+busy)
 return handles

def readiness(C,cfg):
 result={};budget=cfg['training_budget_seconds'];reserve=120
 def rows(method):return json.loads((C/'profiling'/method/'result.json').read_text())['records']
 if not (C/'manifests/integration_tests.json').exists():raise RuntimeError('Missing integration receipt')
 for name in ['numerical_fma','gpu_tests_retry','vsoed_fma']:
  if not (C/'profiling'/name/'result.json').exists():raise RuntimeError('Missing passing readiness: '+name)
 for method,probe in [('fixed','fixed_shared'),('dad','dad_fma')]:
  records=[r for r in rows(probe) if r['phase']=='native_update'];unit=records[-1]['seconds'];projected=int((budget-reserve)/unit);gate=cfg['feasibility_gate'][method+'_minimum_projected_updates']
  result[method]=dict(eligible=projected>=gate,unit_seconds=unit,projected_updates_4h=projected,minimum_projected_updates=gate,reason='Native full-batch throughput gate; not convergence evidence',scientifically_useful_1000_update_hours=(unit*1000+reserve)/3600,native_reference_updates=50000)
 rr=rows('rl_sboed_fma');collect=next(x['seconds'] for x in rr if x['phase']=='native_collection');optim=next(x['seconds'] for x in rr if x['phase']=='critic_actor_64_updates_cost_only');warm=math.ceil(100000/300);warm_cost=warm*collect;post=int((budget-reserve-warm_cost)/(collect+optim));result['rl_sboed']=dict(eligible=post>=cfg['feasibility_gate']['rl_minimum_projected_postwarm_collections'],collection_seconds=collect,optimization64_seconds=optim,warmup_collections=warm,projected_warmup_seconds=warm_cost,projected_postwarm_collections_4h=max(0,post),projected_native_actor_updates=max(0,post)*64,unit_seconds=collect+optim,reason='Native warm-up preserved; optimization cost measured separately with diagnostic replay, no warmed pilot checkpoint claimed')
 vr=rows('vsoed_fast');times=[r['seconds'] for r in vr if r['phase']=='native_actor_critic_posterior_cycle'];unit=max(.001,times[-1]-times[-2]);projected=int((budget-reserve-times[0])/unit);result['vsoed']=dict(eligible=projected>=1000,unit_seconds=unit,projected_actor_updates_4h=projected,reason='Full native initial replay and actor/5critic/5posterior cycles exercised; projection omits future buffer-growth uncertainty')
 sp=C/'profiling/scorebed/result.json'
 if sp.exists():
  sr=json.loads(sp.read_text());score=[r['seconds'] for r in sr['records'] if r['phase']=='native_score_update'];policy=[r['seconds'] for r in sr['records'] if r['phase']=='native_policy_update_cost_only'];sunit=max(score[-2:]);punit=max(policy[-2:]);init=sum(r['seconds'] for r in sr['records'] if r['phase'] not in ['native_score_update','native_policy_update_cost_only']);required=init+sunit*50000+punit*500*10
  result['scorebed']=dict(eligible=False,unit_seconds=sunit,score_update_seconds=sunit,policy_update_seconds=punit,initialization_seconds=init,projected_full_native_seconds=required,reason='Required full50000-step score pretraining alone exceeds4h at measured throughput; policy stages and all10restarts add further cost. No shortened or one-restart primary substituted.')
 else:
  failure=C/'profiling/scorebed/failure.json';result['scorebed']=dict(eligible=False,reason='Native ScoreBED phase profile failed or exceeded20min; no primary run authorized by readiness',failure=json.loads(failure.read_text()) if failure.exists() else {'timeout_or_missing_result':True})
 write(C/'manifests/readiness.json',result);return result

class BudgetObserver:
 def __init__(self,C,method,cfg,unit):
  self.C,self.method,self.cfg=C,method,cfg;self.directory=C/'checkpoints'/method;self.directory.mkdir(exist_ok=False);self.compute_clock=ComputeClock(self.directory/'compute_clock.json',cfg['training_budget_seconds'],hardware=torch.cuda.get_device_name(0),start=ENTRY);self.bound=max(1.,unit);self.last=0.;self.next_checkpoint=1800.;self.next_recovery=300.;self.rows=[];self.selected=None;self.runtime=None;self.update=0;self.episodes=0
 def set_runtime(self,runtime):self.runtime=runtime
 def state(self,phase,**kw):
  record=dict(state=phase,method=self.method,PID=os.getpid(),updated_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),updates=self.update,training_trajectories=self.episodes,charged_seconds=self.compute_clock.spent(),**kw);write(self.C/'manifests'/f'{self.method}_status.json',record);print(record,flush=True)
 def save(self,p,episodes,update,opt=None,scheduler=None,label='recovery'):
  target=self.directory/f'update{update}_{label}.pt';tmp=target.with_suffix('.partial');state=dict(policy=copy.deepcopy(p).cpu(),optimizer=None if opt is None else opt.state_dict(),scheduler=None if scheduler is None else scheduler.state_dict(),update=update,episodes=episodes,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),configuration=self.cfg,source_manifest_sha256=sha(self.C/'manifests/source_sha256.json'))
  torch.save(state,tmp);tmp.replace(target)
  runtime=None
  if self.runtime is not None:
   runtime=target.with_suffix('.native.pkl');temp=runtime.with_suffix('.partial')
   with temp.open('wb') as f:cloudpickle.dump(self.runtime,f)
   temp.replace(runtime)
  digest=sha(target);mark=self.compute_clock.checkpoint(str(target));row=dict(path=str(target),sha256=digest,native_state=None if runtime is None else str(runtime),native_state_sha256=None if runtime is None else sha(runtime),**mark,update=update,episodes=episodes,label=label)
  # Hashing native state is charged too. Final eligibility is rechecked.
  row['charged_seconds']=self.compute_clock.spent();row['eligible']=row['charged_seconds']<=self.cfg['training_budget_seconds'];write(target.with_suffix('.json'),row)
  if row['eligible']:self.selected=row;write(self.directory/'latest_eligible.json',row)
  return row
 def check(self,p,episodes,update,opt=None,scheduler=None,force=False):
  torch.cuda.synchronize();spent=self.compute_clock.spent();self.update=int(update);self.episodes=int(episodes);stop=not self.compute_clock.can_start(max(30.,1.5*self.bound));due=spent>=self.next_checkpoint;recovery=update==1 or spent>=self.next_recovery
  if force or stop or due or recovery:
   mark=self.save(p,episodes,update,opt,scheduler,'timepoint' if due or force or stop else 'recovery');self.next_recovery=self.compute_clock.spent()+300
   if due or force or stop:
    with self.compute_clock.exclude('validation'):
     values,meta=evaluate(self.cfg['benchmark'],p,self.cfg['seeds']['development'],128,[65536],self.C/'evaluations/development',self.method+f'_update{update}')
    row=dict(update=update,episodes=episodes,charged_seconds=mark['charged_seconds'],checkpoint=mark,development=stats(values[65536]),evaluation=meta);self.rows.append(row);write(self.C/'logs'/f'{self.method}_learning_curve.json',self.rows);self.next_checkpoint=next((t for t in [1800,3600,7200,14400] if t>self.compute_clock.spent()),float('inf'))
    if hasattr(p,'train'):p.train()
  if update==1 or update%25==0 or stop or due:self.state('TRAINING',latest_checkpoint=self.selected)
  return stop or force
 def allow_next_unit(self,p,episodes,update,opt=None,scheduler=None):
  if self.compute_clock.can_start(max(30.,1.5*self.bound)):return True
  self.check(p,episodes,update,opt,scheduler,force=True);return False
 def finish(self,p):
  if self.selected is None:self.save(p,self.episodes,self.update,label='final')
  if self.selected is None:raise RuntimeError('No budget eligible checkpoint')
  saved=torch.load(self.selected['path'],map_location=next(p.parameters()).device,weights_only=False);p.load_state_dict(saved['policy'].state_dict());return dict(selected_checkpoint=dict(update=self.selected['update'],path=self.selected['path']),checkpoint_selection='latest serialized budget eligible',journal_validation_history=self.rows)

def train_method(C,cfg,method,ready):
 locks=lock_gpu();torch.set_num_threads(4);torch.cuda.init();torch.manual_seed(101);np.random.seed(101);b=None;observer=BudgetObserver(C,method,cfg,ready[method]['unit_seconds']);observer.state('INITIALIZING')
 try:
  with torch.device('cuda:0'):
   b=T3SIR(cfg['benchmark'])
   if method in ['dad','fixed']:
    from src.research.original_protocol import dad_settings
    from src.research.sir_dad import SIRSumDAD
    from src.research.policies import Policy
    if method=='fixed':
     p=Policy(10,1,3,'fixed');opt=torch.optim.Adam(p.parameters(),lr=5e-5,betas=(.8,.998));scheduler=torch.optim.lr_scheduler.ExponentialLR(opt,.98);observer.set_runtime(dict(policy=p,optimizer=opt,scheduler=scheduler));history=[];count=0
     # One nonadaptive schedule, no restart or favorable validation selection.
     with torch.no_grad():p.sequence.copy_(b.random_latent(3,torch.Generator(device='cuda').manual_seed(918)))
     for update in range(1,50001):
      if not observer.allow_next_unit(p,count,update-1,opt,scheduler):break
      t=time.monotonic();opt.zero_grad();value=fixed_shared_spce(b,p,2000,2000,101+update*104729);loss=-value.mean();loss.backward()
      if not all(torch.isfinite(x.grad).all() for x in p.parameters() if x.grad is not None):raise FloatingPointError('Nonfinite Fixed gradient')
      opt.step()
      if (update-1)%1000==0:scheduler.step()
      count+=2000
      if update==1 or update%100==0:
       row=dict(update=update,loss=float(loss.detach()),lr=opt.param_groups[0]['lr'],charged_seconds=observer.compute_clock.spent());history.append(row);write(C/'logs/fixed_optimization.json',history)
      if observer.check(p,count,update,opt,scheduler,force=update==50000):break
     observer.finish(p);data=dict(generated_training_trajectories=count,optimizer_updates=observer.update,contrasts=2000,history=history,native_schedule_complete=observer.update==50000,adaptation='Shared iid prior contrast bank across outer cases; exact same expected finite-L objective, different within-batch gradient covariance; exact deterministic trajectory reuse')
    else:
     from src.research.training import train_spce
     s=dict(dad_settings(),updates=50000,hidden=64,validation_systems=128,validation_contrasts=65536,spce_trajectory_budget_unit='primary_episodes');p,data=train_spce(b,'dad',3,s,101,observer.directory,True,100000000,SIRSumDAD(3),observer);data['optimizer_updates']=observer.update;data['native_schedule_complete']=observer.update>=50000
   elif method=='rl_sboed':
    from src.research.sir_t3_native_rl import train_sir_native_rl
    p,data=train_sir_native_rl(b,3,dict(updates=100000000,rl_native_contrasts=100000,rl_native_replay_capacity=1000000),101,observer.directory,100000000,observer);data['optimizer_updates']=data['gradient_updates'];data['native_schedule_complete']=False
   elif method=='vsoed':
    from src.research.sir_t3_native_vsoed import train_upstream_vsoed
    p,data=train_upstream_vsoed(b,3,{},101,observer.directory,dict(cfg['methods']['vsoed'],updates=1000000000),observer);data['optimizer_updates']=data['actor_optimizer_updates'];data['native_schedule_complete']=False
   else:raise ValueError('No eligible native primary worker for '+method)
  final=observer.selected
  if sha(final['path'])!=final['sha256']:raise RuntimeError('Checkpoint hash mismatch')
  data.update(method=method,horizon=3,training_seed=101,checkpoint=final,charged_training=observer.compute_clock.stop(),actual_ODE_particle_stages=b.calls,actual_RK4_steps=b.ode_forward_steps,scientific_status='COMPUTE_LIMITED',source_manifest_sha256=sha(C/'manifests/source_sha256.json'))
  write(C/'manifests'/f'{method}_complete.json',data);observer.state('TRAINING_COMPLETE',checkpoint=final)
 except Exception as e:
  write(C/'manifests'/f'{method}_failure.json',dict(error=str(e),traceback=traceback.format_exc(),charged_training=observer.compute_clock.stop(),last_eligible_checkpoint=observer.selected,updates=observer.update,episodes=observer.episodes,status='INVALID_OR_INCOMPLETE'));observer.state('FAILED_NO_AUTOMATIC_RESTART');raise

FIELDS=['method','training_seed','horizon','checkpoint','optimizer_updates','training_trajectories','training_contrasts','charged_training_seconds','native_schedule_complete','eig_mean','eig_mcse','eig_ci_low','eig_ci_high','status']
def table(C,ready,final=None):
 rows=[]
 for m in METHODS:
  row=dict.fromkeys(FIELDS,'');row.update(method=m,training_seed=101,horizon=3,status='PENDING');p=C/'manifests'/f'{m}_complete.json'
  if m=='random':row.update(optimizer_updates=0,training_trajectories=0,training_contrasts=0,charged_training_seconds=0,native_schedule_complete=True,status='NO_TRAINING_REQUIRED')
  elif not ready[m]['eligible']:row['status']='BLOCKED_READINESS'
  if p.exists():
   x=json.loads(p.read_text());row.update(checkpoint=x['checkpoint']['path'],optimizer_updates=x['optimizer_updates'],training_trajectories=x['generated_training_trajectories'],training_contrasts=x.get('contrasts','N/A'),charged_training_seconds=x['charged_training']['charged_seconds'],native_schedule_complete=x.get('native_schedule_complete',False),status=x['scientific_status'])
  if final and m in final:
   x=final[m];row.update(eig_mean=x['mean'],eig_mcse=x['MC_SE'],eig_ci_low=x['CI95_low'],eig_ci_high=x['CI95_high'])
  rows.append(row)
 with (C/'sir_t3_baseline_results.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
 return rows

def evaluate_completed(C,cfg,ready):
 locks=lock_gpu();torch.set_num_threads(4);models={'random':None};updates={}
 for m in METHODS[1:]:
  path=C/'manifests'/f'{m}_complete.json'
  if path.exists():
   x=json.loads(path.read_text());cp=x['checkpoint'];assert sha(cp['path'])==cp['sha256'];models[m]=torch.load(cp['path'],map_location='cpu',weights_only=False)['policy'];updates[m]=x['optimizer_updates']
 # Resolution selection is entirely disjoint from final confirmation draws.
 resolution={};arrays={}
 for method,p in models.items():
  x,meta=evaluate(cfg['benchmark'],p,cfg['seeds']['resolution'],64,cfg['contrast_candidates'],C/'evaluations/resolution',method,method=method);arrays[method]=x;resolution[method]=meta
 selected=1000000;threshold=cfg['resolution_thresholds'];audits=[]
 for L in cfg['contrast_candidates']:
  passes=True;checks={}
  for m,x in arrays.items():
   delta=x[L]-x[1000000];s=stats(delta);limit=min(threshold['mean_absolute_change_nats'],threshold['maximum_MCSE_multiple']*max(stats(x[1000000])['MC_SE'],1e-12));ok=abs(s['mean'])+1.96*s['MC_SE']<=limit;passes&=ok;checks[m]=dict(change=s,threshold=limit,passes=ok)
  for a in models:
   for b in models:
    if a<=b:continue
    delta=(arrays[a][L]-arrays[b][L])-(arrays[a][1000000]-arrays[b][1000000]);s=stats(delta);ok=abs(s['mean'])+1.96*s['MC_SE']<=threshold['paired_change_nats'];passes&=ok;checks[a+'-'+b]=dict(change=s,passes=ok)
  audits.append(dict(contrasts=L,passes=passes,checks=checks))
  if passes:selected=L;break
 write(C/'manifests/frozen_evaluation_resolution.json',dict(contrasts=selected,audits=audits,resolution_cases=resolution,selection_uses_confirmation=False))
 final={};rawscores={}
 for m,p in models.items():
  x,meta=evaluate(cfg['benchmark'],p,cfg['seeds']['confirmation'],2048,[selected],C/'evaluations/confirmation',m,method=m);rawscores[m]=x[selected];final[m]=dict(**stats(x[selected]),metadata=meta);write(C/'evaluations/final_results.json',final);table(C,ready,final)
 paired={a+'-'+b:stats(rawscores[a]-rawscores[b]) for a in models for b in models if a>b};write(C/'evaluations/paired_differences.json',paired)
 text='# SIR T3 EIG comparison\n\nTerminal full-horizon explicit-likelihood sPCE,2048 paired confirmation trajectories; common L='+str(selected)+'. One training seed, finite contrast bias/training-seed variance excluded from MC uncertainty.\n\n| Method | EIG nats | MC SE |95%CI|\n|---|---:|---:|---|\n'
 for m,v in final.items():text+=f"|{m}|{v['mean']:.6f}|{v['MC_SE']:.6f}|[{v['CI95_low']:.6f},{v['CI95_high']:.6f}]|\n"
 text+='\nReadiness-blocked methods are absent, not assigned invented EIG. See paired_differences.json for all paired differences.\n';(C/'reports/SIR_T3_EIG_COMPARISON.md').write_text(text)
 return final

def campaign(C,cfg,continuing=False):
 if (C/'manifests/campaign_started.json').exists() and not continuing:raise RuntimeError('Existing campaign: explicit --continue required; no automatic restart')
 ready=json.loads((C/'manifests/readiness.json').read_text()) if (C/'manifests/readiness.json').exists() else readiness(C,cfg)
 if not (C/'manifests/source_sha256.json').exists():
  manifest=source_manifest();manifest[str((C/'configs/sir_ode_t3_continuous_v1.yaml').relative_to(ROOT))]=sha(C/'configs/sir_ode_t3_continuous_v1.yaml');write(C/'manifests/source_sha256.json',manifest)
  with tarfile.open(C/'manifests/frozen_source.tar.gz','w:gz') as a:
   for rel in manifest:a.add(ROOT/rel,arcname=rel,recursive=False)
  write(C/'manifests/source_revision.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),dirty=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),source_archive_sha256=sha(C/'manifests/frozen_source.tar.gz'),source_manifest_sha256=sha(C/'manifests/source_sha256.json')))
 from src.research.sir_t3_reports import update_reports
 update_reports(C,ready)
 write(C/'manifests/campaign_started.json',dict(PID=os.getpid(),UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),automatic_zero_restart=False));table(C,ready)
 def status(state,method=None,child=None):write(C/'status.json',dict(state=state,method=method,PID=os.getpid(),child_PID=child,updated_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),readiness=ready))
 # Random's development-only estimate is separate from final confirmation.
 randomdone=C/'manifests/random_development.json'
 if not randomdone.exists():
  status('RANDOM_DEVELOPMENT_EVALUATION','random')
  with (C/'logs/random_development.log').open('x') as log:
   rc=subprocess.call(['bash','run.sh','--sir-t3-campaign','--campaign',str(C),'--phase','random'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  if rc:raise RuntimeError('Random development evaluation failed; see log')
 for m in METHODS[1:]:
  if not ready[m]['eligible'] or (C/'manifests'/f'{m}_complete.json').exists():continue
  if (C/'manifests'/f'{m}_status.json').exists() or (C/'manifests'/f'{m}_failure.json').exists():
   status('INTERRUPTED_METHOD_REQUIRES_EXPLICIT_NATIVE_RECOVERY',m);continue
  for rel,digest in json.loads((C/'manifests/source_sha256.json').read_text()).items():
   if sha(ROOT/rel)!=digest:raise RuntimeError('Frozen source changed: '+rel)
  log=C/'logs'/f'{m}_primary.log'
  with log.open('x') as f:
   p=subprocess.Popen(['bash','run.sh','--sir-t3-campaign','--campaign',str(C),'--phase','train','--method',m],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);status('TRAINING',m,p.pid)
   # Independent wall watchdog includes ample evaluation time; it never
   # grants training budget and cannot kill any unrelated process.
   try:rc=p.wait(timeout=14400+6*3600)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=30);rc=124
  write(C/'manifests'/f'{m}_exit.json',dict(returncode=rc));update_reports(C,ready)
 status('CALIBRATING_AND_FINAL_EVALUATING');final=evaluate_completed(C,cfg,ready);table(C,ready,final);status('COMPLETE_WITH_READINESS_BLOCKERS' if any(not x['eligible'] for x in ready.values()) else 'COMPLETE')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--sir-t3-campaign',action='store_true');ap.add_argument('--campaign',required=True);ap.add_argument('--phase',choices=['campaign','train','random','integration'],default='campaign');ap.add_argument('--method',choices=METHODS);ap.add_argument('--continue',dest='continuing',action='store_true');args=ap.parse_args();C=Path(args.campaign).resolve()
 if ROOT.name!='objective_power_grid_temp' or ROOT/'scratch_runs/sir_ode_t3_baselines' not in C.parents:raise ValueError('T3 development must remain in TEMP scratch root')
 cfg=yaml.safe_load((C/'configs/sir_ode_t3_continuous_v1.yaml').read_text())
 if cfg['benchmark']['experiment']['step_number']!=3 or cfg['training_seed']!=101:raise ValueError('Frozen T3 protocol mismatch')
 if args.phase=='train':train_method(C,cfg,args.method,json.loads((C/'manifests/readiness.json').read_text()))
 elif args.phase=='random':
  locks=lock_gpu();torch.set_num_threads(4);x,meta=evaluate(cfg['benchmark'],None,cfg['seeds']['development'],128,[65536],C/'evaluations/development','random',method='random');write(C/'manifests/random_development.json',dict(**stats(x[65536]),metadata=meta,not_final_confirmation=True))
 elif args.phase=='integration':
  from src.research.sir_t3_regression import integration
  integration(C,cfg)
 else:campaign(C,cfg,args.continuing)
if __name__=='__main__':main()
