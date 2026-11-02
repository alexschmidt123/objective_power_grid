"""Reviewed recovery in the existing T3 campaign; completed methods never train."""
import time
ENTRY=time.monotonic()
import os,json,hashlib,subprocess,sys,traceback,csv,copy,math,shutil
from pathlib import Path
import numpy as np,torch,yaml
from src.research.sir_t3_campaign import BudgetObserver,lock_gpu,write,sha
from src.research.compute_budget import ComputeClock
from src.research.sir_t3_common import T3SIR
from src.research.sir_dad import SIRSumDAD
from src.research.original_protocol import dad_settings
from src.research.training import train_spce
from src.research.sir_t3_evaluation import evaluate,stats
R=Path(__file__).resolve().parents[2]
P=R/'scratch_runs/sir_ode_t3_baselines/sir_ode_t3_corrected_baselines_seed101_20261028T155016Z';C=P/'full_testing_4h_v2';A=C/'manifests/recovery_20261029'
cfg=yaml.safe_load((C/'configs/sir_ode_t3_continuous_v1.yaml').read_text())
def status(state,**kw):
 x=dict(state=state,PID=os.getpid(),updated_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),recovery_directory=str(A),**kw);write(C/'status.json',x);write(P/'status.json',dict(x,extension=str(C)));write(R/'reports/SIR_T3_FULL_TESTING.json',dict(campaign=str(C),**x))
def unchanged():
 for p,h in json.loads((A/'preserved_sha256.json').read_text()).items():
  if sha(p)!=h:raise RuntimeError('Preserved artifact changed: '+p)
 for p,h in json.loads((A/'source_sha256.json').read_text()).items():
  if sha(R/p)!=h:raise RuntimeError('Recovery source changed: '+p)

class SmokeObserver:
 def __init__(self,d):
  self.directory=d;d.mkdir(exist_ok=False,parents=True);self.compute_clock=ComputeClock(d/'compute_clock.json',180,hardware=torch.cuda.get_device_name(0));self.n=0;self.rows=[];self.path=None;self.last_time=time.monotonic()
 def allow_next_unit(self,p,episodes,u,opt,sch):return u<2
 def check(self,p,episodes,u,opt,sch,force=False):
  torch.cuda.synchronize();now=time.monotonic();self.rows.append(dict(update=u,seconds=now-self.last_time,lr=opt.param_groups[0]['lr'],optimizer_steps=sorted(set(float(v['step']) for v in opt.state.values()))));self.n=u
  self.path=self.directory/f'update{u}.pt';torch.save(dict(policy=copy.deepcopy(p).cpu(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),update=u,episodes=episodes),self.path);assert self.compute_clock.checkpoint(str(self.path))['eligible'];self.last_time=time.monotonic();return u==2
 def finish(self,p):return dict(selected_checkpoint=dict(update=self.n,path=str(self.path)),checkpoint_selection='latest serialized budget eligible')
def smoke():
 locks=lock_gpu();torch.set_num_threads(4);torch.cuda.init();torch.manual_seed(101);d=C/'recovery_checks/dad_native_gpu_smoke';obs=SmokeObserver(d)
 with torch.device('cuda:0'):
  b=T3SIR(cfg['benchmark']);p=SIRSumDAD(3);s=dict(dad_settings(),updates=50000,hidden=64,validation_systems=128,validation_contrasts=65536,spce_trajectory_budget_unit='primary_episodes')
  before={k:v.clone() for k,v in p.state_dict().items()};p,data=train_spce(b,'dad',3,s,101,d,True,100000000,p,obs)
  saved=torch.load(obs.path,map_location='cuda:0',weights_only=False);q=SIRSumDAD(3);q.load_state_dict(saved['policy'].state_dict(),strict=True);assert all(torch.equal(x,y) for x,y in zip(q.state_dict().values(),p.state_dict().values()));assert any(not torch.equal(before[k],v) for k,v in p.state_dict().items());assert data['actual_updates']==2 and data['selected_update']==2
  from src.research.engine import rollout,replay_likelihood
  with torch.no_grad():
   rr=rollout(b,p,3,4,2026102836,16);a=torch.stack(rr['actions'],1)[:,:,0];assert a.shape==(4,3) and (a[:,1:]-a[:,:-1]>=.1-1e-9).all() and a.min()>=0 and a.max()<=100
   th=torch.cat((rr['truth'][:,None],rr['particles']),1);ll,_,_=replay_likelihood(b,th,rr['actions'],rr['observations']);target=ll[:,0]-torch.logsumexp(ll,-1)+math.log(17);torch.testing.assert_close(target,rr['info'][:,-1],rtol=1e-10,atol=1e-9)
  assert obs.rows[0]['lr']==5e-5*.98 and obs.rows[1]['lr']==5e-5*.98
  failed=json.loads((C/'manifests/dad_failure.json').read_text())['charged_training']['charged_seconds'];unit=max(x['seconds'] for x in obs.rows);out=dict(status='PASS',batch=2000,contrasts=2000,horizon=3,discarded_smoke_updates=2,rows=obs.rows,conservative_seconds_per_update=unit,failed_compute_seconds=failed,remaining_primary_allowance_seconds=14400-failed,projected_recovery_updates=max(0,int((14400-failed-120)/unit)),reference_native_updates=50000,projected_native_schedule_hours=unit*50000/3600,checkpoint_roundtrip=True,full_horizon_sPCE_verified=True,chronological_constraints_verified=True,native_lr_schedule_verified=True,checkpoint_selection='latest budget eligible; no EIG selection',clock=obs.compute_clock.stop())
  write(A/'gpu_smoke.json',out);print(json.dumps(out,indent=2),flush=True)

class RecoveryObserver(BudgetObserver):
 def __init__(self,unit):
  self.C,self.method,self.cfg=C,'dad',cfg;self.directory=C/'checkpoints/dad/recovery_20261029';self.directory.mkdir(exist_ok=False);failed=json.loads((C/'manifests/dad_failure.json').read_text())['charged_training']['charged_seconds'];self.compute_clock=ComputeClock(self.directory/'compute_clock.json',14400,credit_seconds=failed,hardware=torch.cuda.get_device_name(0),start=ENTRY);self.bound=unit;self.last=0.;self.next_checkpoint=1800.;self.next_recovery=self.compute_clock.spent()+300;self.rows=[];self.selected=None;self.runtime=None;self.update=0;self.episodes=0

def train():
 unchanged();locks=lock_gpu();torch.set_num_threads(4);torch.cuda.init();torch.manual_seed(101);np.random.seed(101);unit=json.loads((A/'gpu_smoke.json').read_text())['conservative_seconds_per_update'];obs=RecoveryObserver(unit);obs.state('RECOVERING_FRESH_NO_VALID_CHECKPOINT',prior_failed_compute_seconds=obs.compute_clock.report()['credit_seconds'],prior_valid_updates=0);start=time.monotonic()
 try:
  with torch.device('cuda:0'):
   b=T3SIR(cfg['benchmark']);s=dict(dad_settings(),updates=50000,hidden=64,validation_systems=128,validation_contrasts=65536,spce_trajectory_budget_unit='primary_episodes');p,data=train_spce(b,'dad',3,s,101,obs.directory,True,100000000,SIRSumDAD(3),obs)
  assert obs.selected and sha(obs.selected['path'])==obs.selected['sha256'];clock=obs.compute_clock.stop();data.update(method='dad',horizon=3,training_seed=101,optimizer_updates=obs.update,native_schedule_complete=obs.update>=50000,contrasts=2000,checkpoint=obs.selected,charged_training=clock,prior_failed_compute_seconds=clock['credit_seconds'],productive_recovery_charged_seconds=clock['charged_seconds']-clock['credit_seconds'],scientific_status='COMPUTE_LIMITED',source_amendment_sha256=sha(A/'source_sha256.json'),resume_lineage='Fresh seed101; no valid checkpoint survived failed call; prior failed compute credited to same4h cap')
  write(C/'manifests/dad_complete.json',data);obs.state('TRAINING_COMPLETE',checkpoint=obs.selected)
 except Exception as e:
  write(A/'dad_recovery_failure.json',dict(error=str(e),traceback=traceback.format_exc(),last_valid_checkpoint=obs.selected,actual_valid_updates=obs.update,clock=obs.compute_clock.stop()));raise

def confirm(method):
 unchanged();locks=lock_gpu();torch.set_num_threads(4);out=C/'evaluations/recovery_confirmation';out.mkdir(exist_ok=True);old=P/'evaluations/confirmation/cases_seed2026102833_N2048_L1000000.npz';new=out/old.name
 if not new.exists():shutil.copy2(old,new)
 assert sha(old)==sha(new)
 row=json.loads((C/'manifests'/f'{method}_complete.json').read_text());cp=row['checkpoint'];assert sha(cp['path'])==cp['sha256'];p=torch.load(cp['path'],map_location='cpu',weights_only=False)['policy'];tag=method+'_final_recovered'
 existing=out/f'{tag}_metadata.json'
 if existing.exists():return
 vals,meta=evaluate(cfg['benchmark'],p,2026102833,2048,[1000000],out,tag,method=method);assert meta['paired_cases_sha256']==json.loads((P/'evaluations/confirmation/random_metadata.json').read_text())['paired_cases_sha256'];write(A/f'{method}_final_confirmation.json',dict(**stats(vals[1000000]),metadata=meta,checkpoint=cp,cases_file_sha256=sha(new)));unchanged()

def report():
 out=C/'evaluations/recovery_confirmation';scores={m:np.load(P/f'evaluations/confirmation/{m}_L1000000.npy') for m in ['random','rl_sboed','vsoed']};rows=[]
 for m in ['fixed','dad']:
  p=out/f'{m}_final_recovered_L1000000.npy'
  if p.exists():scores[m]=np.load(p)
 paired={a+'-minus-'+b:stats(scores[a]-scores[b]) for a in scores for b in scores if a!=b};write(A/'paired_final_comparisons.json',paired)
 convergence={}
 cp=P/'reports/SIR_T3_CONVERGENCE_ASSESSMENT.md'
 if cp.exists():
  text=cp.read_text();convergence=json.JSONDecoder().raw_decode(text[text.index('{'):])[0]
 fields=['method','training_status','actual_optimizer_updates','charged_training_seconds','native_schedule_completion','evaluation_status','EIG','MC_SE','paired_comparison','convergence_status']
 for m in ['random','fixed','dad','rl_sboed','vsoed','scorebed']:
  path=(C if m in ['fixed','dad'] else P)/'manifests'/f'{m}_complete.json';rec=json.loads(path.read_text()) if path.exists() else {};done=bool(rec) or m=='random';sm=stats(scores[m]) if m in scores else {};running=m=='dad' and not done;row=dict(method=m,training_status='COMPUTE_LIMITED_NOT_TRAINED' if m=='scorebed' else 'NO_TRAINING_REQUIRED' if m=='random' else 'COMPLETE_WITHIN_CAP' if done else 'RUNNING',actual_optimizer_updates=rec.get('optimizer_updates',0),charged_training_seconds=rec.get('charged_training',{}).get('charged_seconds',0),native_schedule_completion=rec.get('native_schedule_complete',m=='random'),evaluation_status='FINAL_CONFIRMATION_2048_L1000000' if m in scores else 'UNAVAILABLE' if m=='scorebed' else 'PENDING',EIG=sm.get('mean',''),MC_SE=sm.get('MC_SE',''),paired_comparison=paired.get(m+'-minus-random',{}),convergence_status='INVALID_OR_INCOMPLETE' if m=='scorebed' else 'NO_TRAINING_REQUIRED' if m=='random' else 'COMPUTE_LIMITED' if m=='dad' and done else 'PENDING' if not done else convergence.get(m,{}).get('status','COMPUTE_LIMITED'))
  ds=C/'manifests/dad_status.json'
  if running and ds.exists():x=json.loads(ds.read_text());row.update(actual_optimizer_updates=x.get('updates',0),charged_training_seconds=x.get('charged_seconds',0))
  rows.append(row)
 for p in [C/'sir_t3_recovery_results.csv',P/'sir_t3_recovery_results.csv']:
  temp=p.with_suffix('.tmp')
  with temp.open('w') as f:
   w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{**r,'paired_comparison':json.dumps(r['paired_comparison'])} for r in rows])
  temp.replace(p)
 text='# SIR T=3 reviewed recovery\n\nFinal EIG uses2048paired trajectories and1,000,000contrasts, exact copied frozen prior/noise arrays. Development scores excluded. One training seed; MCSE excludes contrast bias and training-seed variance.\n\n|Method|Training|Updates|Charged seconds|EIG|MCSE|Evaluation|\n|---|---|---:|---:|---:|---:|---|\n'
 for r in rows:text+='|'+ '|'.join(str(r[k]) for k in ['method','training_status','actual_optimizer_updates','charged_training_seconds','EIG','MC_SE','evaluation_status'])+'|\n'
 text+='\nScoreBED remains COMPUTE_LIMITED_NOT_TRAINED: full50000score steps plus500policy steps for each of10restarts require approximately37993.69seconds(10.55h), beyond14400seconds. No shortened pretraining/restarts or longer run launched. DAD uses full nativebatch2000/contrasts2000 and latest serialized eligible checkpoint. Fixed/RL/vsOED records are preserved.\n'
 (C/'reports/SIR_T3_RECOVERY_RESULTS.md').write_text(text);write(A/'six_method_table.json',rows)

def run():
 if not json.loads((A/'gpu_smoke.json').read_text())['status']=='PASS':raise RuntimeError('No GPU smoke PASS')
 if (A/'controller_started.json').exists():raise RuntimeError('No automatic restart')
 import fcntl
 f=(A/'controller.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);write(A/'controller_started.json',dict(PID=os.getpid(),source_manifest_sha256=sha(A/'source_sha256.json')))
 try:
  for phase in ['fixed_final','dad_train','dad_final']:
   if phase=='dad_train' and (C/'manifests/dad_complete.json').exists():continue
   status('RECOVERY_RUNNING',phase=phase);report();cmd=['bash','-c','source ./run.sh; exec python3 -m src.research.sir_t3_recovery "$1"','sir-recovery',phase]
   env=dict(os.environ);env.pop('CUDA_VISIBLE_DEVICES',None)
   with (C/'logs'/f'recovery_{phase}.log').open('x') as log:
    child=subprocess.Popen(cmd,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);status('RECOVERY_RUNNING',phase=phase,child_PID=child.pid);rc=child.wait()
   write(A/f'{phase}_exit.json',dict(returncode=rc))
   if rc:raise RuntimeError(f'{phase} failed with returncode{rc}; no automatic restart')
   report()
  unchanged();status('COMPLETE_WITH_SCOREBED_COMPUTE_LIMIT');report()
 except Exception as e:status('RECOVERY_FAILED_REQUIRES_REVIEW',error=str(e));write(A/'controller_failure.json',dict(error=str(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':
 phase=sys.argv[1]
 if phase=='smoke':smoke()
 elif phase=='dad_train':train()
 elif phase=='fixed_final':confirm('fixed')
 elif phase=='dad_final':confirm('dad')
 elif phase=='report':report()
 elif phase=='run':run()
 else:raise ValueError(phase)
