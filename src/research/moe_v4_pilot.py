"""Persistent controlled V4 pilot. CPU preparation, GPU smoke and bounded training."""
import time
ENTRY=time.monotonic()
import argparse,copy,fcntl,hashlib,json,math,os,signal,subprocess,sys,tarfile,traceback
from pathlib import Path
import numpy as np,torch,yaml
from src.research.moe_v4_partition import PredictiveAdapter,fit_partition
from src.research.moe_sboed_v4 import RoleMoE,dense_control
from src.research.moe_v4_information import role_rollout,paired_training_draws
from src.research.deepseek_sboed import DeepSeekPolicy
from src.research.context_expert_gradients import capture_context,partition_context,collect_context_gradients,apply_context_gradients
from src.research.engine import rollout
ROOT=Path(__file__).resolve().parents[2]
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cfgread(C):
 cfg=yaml.safe_load((ROOT/'configs/moe_sboed_v4_development.yaml').read_text());cfg['benchmark']=yaml.safe_load((ROOT/cfg['benchmark_recipe']).read_text())['benchmark'];cfg['training_information_ceiling_nats']=math.log(cfg['contrasts']+1);cfg['legacy_nested_benchmark_budget_fields']='Historical descriptive metadata only, not read by this worker; actual pilot settings/counters are top-level';return cfg

def gpu_lock():
 handles=[]
 for p in [ROOT.parent/'objective_power_grid/.gpu0.lock',Path.home()/'.cache/objective_power_grid/device_0.lock',ROOT/'.gpu0.lock']:
  p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(f)
 busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 if busy:raise RuntimeError('Foreign GPU processes present: '+busy)
 return handles

def models_for_seed(seed):
 models={}
 with torch.device('cpu'):
  for arm in ['v1_context','v4','no_role','dense','uniform_top2']:
   torch.manual_seed(seed)
   if arm=='v1_context':p=DeepSeekPolicy()
   elif arm=='dense':p=dense_control(models['v4'].counts()['total_parameters'])
   else:p=RoleMoE(routing='uniform_top2' if arm=='uniform_top2' else 'learned')
   models[arm]=p
 return models

def arm_loss(b,model,arm,seed,update,cfg,partition,adapter,alpha,allow_context=True):
 B,L=cfg['batch_size'],cfg['contrasts'];sample_seed=seed+update*104729;special=arm in ['v1_context','native_v1'] and allow_context and update%10==0;stage=[1,10,20,29][(update//10-1)%4] if special else 1
 snapshot={}
 from contextlib import nullcontext
 with capture_context(model,stage) if special else nullcontext(snapshot) as observed:
  if arm=='native_v1':r=rollout(b,model,30,2000,sample_seed,2000)
  else:
   truth,bank,noise=paired_training_draws(b,30,B,L,sample_seed)
   if arm in ['v4','uniform_top2']:r=role_rollout(b,model,30,truth,bank,noise,partition,adapter)
   else:r=rollout(b,model,30,B,sample_seed,L,truth=truth,noise=noise,prior_particles=bank[None].expand(B,-1,-1))
 information=r['info'][:,-1];balance=model.balance_loss() if arm in ['v1_context','native_v1'] else information.mean()*0;loss=-information.mean()+balance;roles=None;context=None;cohort=None
 if arm in ['v4','uniform_top2']:
  roles,pred=model.role_losses(r['features'],r['role_targets']);loss=loss+alpha*roles.sum()
 if special:
  context=partition_context(observed)
  if context['valid']:cohort=collect_context_gradients(model,information,balance,context)
 return loss,r,roles,context,cohort

def prep(C,cfg):
 start=time.monotonic();torch.set_num_threads(4);adapter=PredictiveAdapter(cfg['benchmark']);partition,report=fit_partition(adapter,cfg['partition']['fit_samples'],cfg['partition']['validation_samples'],cfg['partition']['fit_seed']);torch.save(dict(state_dict=partition.state_dict(),raw=cfg['benchmark'],report=report),C/'provenance/partition.pt');write(C/'provenance/partition_validation.json',dict(**report,CPU_preparation_seconds=time.monotonic()-start,partition_sha256=sha(C/'provenance/partition.pt')))
 (C/'configs/effective_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False));write(C/'provenance/effective_config.json',cfg);print(json.dumps(report,indent=2),flush=True)

def load_partition(C,device):
 from src.research.moe_v4_partition import FrozenPartition
 data=torch.load(C/'provenance/partition.pt',map_location='cpu',weights_only=False);s=data['state_dict'];p=FrozenPartition(s['mean'],s['scale'],s['centers']);return p.to(device)

def smoke(C,cfg):
 locks=gpu_lock();smoke_start=time.monotonic();torch.set_num_threads(4);torch.cuda.init();partition=load_partition(C,'cuda:0');adapter=PredictiveAdapter(cfg['benchmark']);cost={};records=[]
 with torch.device('cuda:0'):
  b=adapter.b
  with torch.device('cpu'):torch.manual_seed(101);p=RoleMoE()
  p=p.cuda();start=time.monotonic();truth,bank,noise=paired_training_draws(b,30,cfg['batch_size'],cfg['contrasts'],cfg['alpha']['diagnostic_seed']);r=role_rollout(b,p,30,truth,bank,noise,partition,adapter);flat=r['role_targets'].flatten(0,1);mean=flat.mean(0);scale=flat.std(0,unbiased=False).clamp_min(.1);p.role_mean.copy_(mean);p.role_scale.copy_(scale);losses,_=p.role_losses(r['features'],r['role_targets']);parameters=[q for e in p.experts for q in e.features.parameters()];primary=torch.autograd.grad(-r['info'][:,-1].mean(),parameters,retain_graph=True,allow_unused=True);aux=torch.autograd.grad(losses.sum(),parameters,retain_graph=True,allow_unused=True)
  def norm(gs):return float(torch.stack([g.norm() for g in gs if g is not None]).norm())
  gp,ga=norm(primary),norm(aux)
  if not math.isfinite(gp+ga) or gp<=1e-12 or ga<=1e-12:raise RuntimeError('Degenerate alpha scale/gradient diagnostic')
  alpha=min(1.,.1*gp/ga);torch.cuda.synchronize();cost['v4']=time.monotonic()-start;corr=np.corrcoef(flat.cpu().numpy().T);calib=dict(alpha=alpha,primary_expert_gradient_norm=gp,raw_auxiliary_expert_gradient_norm=ga,weighted_aux_to_primary_gradient_ratio=alpha*ga/gp,role_mean=mean.cpu().tolist(),role_scale=scale.cpu().tolist(),initial_target_correlation=corr.tolist(),training_diagnostic_seed=cfg['alpha']['diagnostic_seed'],no_EIG_tuning=True);write(C/'readiness/alpha_calibration.json',calib)
  del r,losses,primary,aux,p;torch.cuda.empty_cache()
  probes=models_for_seed(101)
  with torch.device('cpu'):torch.manual_seed(101);probes['native_v1']=DeepSeekPolicy()
  for arm,p in probes.items():
   start=time.monotonic();p=p.cuda()
   if isinstance(p,RoleMoE):p.role_mean.copy_(mean);p.role_scale.copy_(scale)
   opt=torch.optim.Adam(p.parameters(),lr=5e-5,betas=(.8,.998));opt.zero_grad();loss,r,roles,context,cg=arm_loss(b,p,arm,101,10,cfg,partition,adapter,alpha);loss.backward()
   if cg is not None:apply_context_gradients(p,cg,context)
   gn=torch.nn.utils.clip_grad_norm_(p.parameters(),5,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();dt=time.monotonic()-start;cost[arm]=cost.get(arm,0)+dt
   row=dict(arm=arm,seconds=dt,batch=2000 if arm=='native_v1' else cfg['batch_size'],contrasts=2000 if arm=='native_v1' else cfg['contrasts'],train_sPCE=float(r['info'][:,-1].mean().detach()),loss=float(loss.detach()),gradient_norm=float(gn),peak_memory_bytes=torch.cuda.max_memory_allocated(),counts=p.counts(),context_gradient_applied=cg is not None)
   if arm in ['v4','uniform_top2']:row.update(role_losses=roles.detach().cpu().tolist(),minimum_prior_group_particles=int(r['group_prior_counts'].min()),identity_max_error=float((r['role_targets'].sum(-1)-(r['total_density'][:,-1:]-r['total_density'][:,:-1])).abs().max()))
   records.append(row);write(C/'readiness/gpu_progress.json',records);print(row,flush=True);del loss,r,roles,cg,opt,p;torch.cuda.empty_cache()
 unallocated=max(0.,time.monotonic()-smoke_start-sum(cost.values()))
 cost={arm:value+unallocated for arm,value in cost.items()}
 write(C/'readiness/gpu_smoke.json',dict(status='PASS',shared_GPU_preparation_seconds_conservatively_charged_to_each_arm=unallocated,hardware=torch.cuda.get_device_name(0),records=records,charged_preparation_seconds_by_arm=cost,weights_discarded=True,training_ceiling_nats=math.log(cfg['contrasts']+1)))

class BudgetEnd(Exception):pass

def pilot(C,cfg):
 if (C/'provenance/pilot_started.json').exists():raise RuntimeError('Pilot exists; no automatic restart or duplicate budget')
 if not (C/'readiness/gpu_smoke.json').exists():raise RuntimeError('Missing GPU readiness receipt')
 locks=gpu_lock();torch.set_num_threads(4);torch.cuda.init();startup=time.monotonic()-ENTRY;partition=load_partition(C,'cuda:0');adapter=PredictiveAdapter(cfg['benchmark']);calib=json.loads((C/'readiness/alpha_calibration.json').read_text());alpha=calib['alpha'];pre=json.loads((C/'readiness/gpu_smoke.json').read_text())['charged_preparation_seconds_by_arm'];models={};opts={};sched={};state={};records={};checkpoint_index={}
 for seed in cfg['training_seeds']:
  group_start=time.monotonic();group=models_for_seed(seed);group_initialization_seconds=time.monotonic()-group_start
  for arm in cfg['primary_arms'] if seed==101 else cfg['replication_arms']:
   tag=f'{arm}_s{seed}';t=time.monotonic();p=group[arm].cuda()
   if isinstance(p,RoleMoE):p.role_mean.copy_(torch.tensor(calib['role_mean'],device='cuda',dtype=torch.float64));p.role_scale.copy_(torch.tensor(calib['role_scale'],device='cuda',dtype=torch.float64))
   models[tag]=p;opts[tag]=torch.optim.Adam(p.parameters(),lr=5e-5,betas=(.8,.998));sched[tag]=torch.optim.lr_scheduler.ExponentialLR(opts[tag],.98);torch.cuda.synchronize();state[tag]=dict(arm=arm,seed=seed,updates=0,charged_seconds=startup+group_initialization_seconds+pre.get(arm,0)+time.monotonic()-t,fresh_trajectories=0,logical_contrast_trajectories=0,distinct_contrast_draws=0,likelihood_stage_terms=0,status='READY',max_cycle_seconds=1.,next_compute_mark=900,counts=p.counts(),recipe='matched_high_contrast',discarded_prelaunch_seconds=pre.get(arm,0));records[tag]=[];checkpoint_index[tag]=[]
 if cfg.get('supplemental_native_v1'):
  tag='native_v1_s101';t=time.monotonic();torch.manual_seed(101);p=DeepSeekPolicy().cuda();models[tag]=p;opts[tag]=torch.optim.Adam(p.parameters(),lr=5e-5,betas=(.8,.998));sched[tag]=torch.optim.lr_scheduler.ExponentialLR(opts[tag],.98);state[tag]=dict(arm='native_v1',seed=101,updates=0,charged_seconds=startup+pre.get('native_v1',0)+time.monotonic()-t,fresh_trajectories=0,logical_contrast_trajectories=0,distinct_contrast_draws=0,likelihood_stage_terms=0,status='READY',max_cycle_seconds=1.,next_compute_mark=900,counts=p.counts(),recipe='SUPPLEMENTAL_NATIVE_B2000_L2000_NOT_MATCHED',discarded_prelaunch_seconds=pre.get('native_v1',0));records[tag]=[];checkpoint_index[tag]=[]
 write(C/'provenance/pilot_started.json',dict(PID=os.getpid(),UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),no_automatic_restart=True));manifest=json.loads((C/'provenance/source_sha256.json').read_text())
 def status(active,phase='RUNNING'):
  data=dict(state=phase,PID=os.getpid(),active_method=active,UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),arms=state,training_cap_per_arm=7200,source_manifest_sha256=sha(C/'provenance/source_sha256.json'));write(C/'status.json',data);write(ROOT/'reports/MOE_SBOED_V4_CURRENT.json',dict(campaign=str(C),**data))
 def save(tag,label):
  p=models[tag];s=state[tag];out=C/'models'/tag;out.mkdir(exist_ok=True);path=out/f'update{s["updates"]}_{label}.pt';t=time.monotonic();tmp=path.with_suffix('.partial');torch.save(dict(method='moe_sboed_v4' if s['arm']=='v4' else s['arm'],policy=copy.deepcopy(p).cpu(),optimizer=opts[tag].state_dict(),scheduler=sched[tag].state_dict(),state=s,configuration=cfg,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),routing_rng=None if getattr(p,'_routing_rng',None) is None else p._routing_rng.get_state(),source_manifest_sha256=sha(C/'provenance/source_sha256.json'),partition_sha256=sha(C/'provenance/partition.pt')),tmp);tmp.replace(path);digest=sha(path);s['charged_seconds']+=time.monotonic()-t;row=dict(path=str(path),sha256=digest,update=s['updates'],charged_seconds=s['charged_seconds'],eligible=s['charged_seconds']<=7200,label=label);checkpoint_index[tag].append(row);write(out/'index.json',checkpoint_index[tag])
  if row['eligible']:write(out/'latest_eligible.json',row)
  return row
 def stop_alarm(*args):raise BudgetEnd('Training time boundary')
 signal.signal(signal.SIGALRM,stop_alarm)
 try:
  for milestone in cfg['milestones']:
   for rel,digest in manifest.items():
    if sha(ROOT/rel)!=digest:raise RuntimeError('Frozen source changed: '+rel)
   for tag,p in models.items():
    s=state[tag]
    if s['status']=='BUDGET_COMPLETE':continue
    arm=s['arm'];p.train();s['status']='TRAINING';status(tag)
    while s['updates']<milestone:
     remaining=7200-s['charged_seconds']
     if remaining<max(30.,2*s['max_cycle_seconds']+5):s['status']='BUDGET_COMPLETE';break
     if s['next_compute_mark']<=7200 and s['charged_seconds']+1.5*s['max_cycle_seconds']+1>=s['next_compute_mark']:
      save(tag,f'time_before{s["next_compute_mark"]}');s['next_compute_mark']=next((x for x in cfg['compute_checkpoints_seconds'] if x>s['next_compute_mark']),7201)
     t=time.monotonic();charged_at_start=s['charged_seconds'];signal.setitimer(signal.ITIMER_REAL,remaining-10);u=s['updates']+1
     try:
      opts[tag].zero_grad(set_to_none=True)
      with torch.device('cuda:0'):
       loss,r,roleloss,context,cg=arm_loss(adapter.b,p,arm,s['seed'],u,cfg,partition,adapter,alpha)
       if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
       loss.backward()
       if cg is not None:context_record=apply_context_gradients(p,cg,context)
       gn=torch.nn.utils.clip_grad_norm_(p.parameters(),5,error_if_nonfinite=True);opts[tag].step()
       if (u-1)%1000==0:sched[tag].step()
       torch.cuda.synchronize()
      dt=time.monotonic()-t;s['charged_seconds']+=dt;s['max_cycle_seconds']=max(s['max_cycle_seconds'],dt);s['updates']=u;B,L=(2000,2000) if arm=='native_v1' else (cfg['batch_size'],cfg['contrasts']);s['fresh_trajectories']+=B;s['logical_contrast_trajectories']+=B*L;s['distinct_contrast_draws']+=B*L if arm=='native_v1' else L;s['likelihood_stage_terms']+=B*(L+1)*30;s['partition_reference_predictions']=s.get('partition_reference_predictions',0)+(len(adapter.designs)*(B+L) if arm in ['v4','uniform_top2'] else 0)
      if u==1 or u%25==0:
       row=dict(update=u,charged_seconds=s['charged_seconds'],cycle_seconds=dt,terminal_training_sPCE=float(r['info'][:,-1].mean().detach()),loss=float(loss.detach()),gradient_norm=float(gn),learning_rate=opts[tag].param_groups[0]['lr'],role_losses=None if roleloss is None else roleloss.detach().cpu().tolist(),context_gradient_applied=cg is not None)
       if arm in ['v4','uniform_top2']:row.update(mean_terminal_ESS=float(r['posterior_ESS'][:,-1].mean()),ceiling_saturation_fraction=float((r['info'][:,-1]>math.log(L+1)-.1).double().mean()),minimum_prior_group_particles=int(r['group_prior_counts'].min()))
       records[tag].append(row)
       with (C/'logs'/f'{tag}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
       print(tag,row,flush=True);status(tag)
      if u==1 or u%256==0:
       signal.setitimer(signal.ITIMER_REAL,0);s['charged_seconds']=charged_at_start+time.monotonic()-t;save(tag,'updates')
      s['charged_seconds']=charged_at_start+time.monotonic()-t
     except BudgetEnd:
      s['charged_seconds']=charged_at_start+time.monotonic()-t;s['status']='BUDGET_COMPLETE';s['discard_incomplete_update']=True;break
     finally:signal.setitimer(signal.ITIMER_REAL,0)
    if s['status']!='BUDGET_COMPLETE':s['status']='MILESTONE_COMPLETE';save(tag,'milestone')
    elif not s.get('discard_incomplete_update',False):save(tag,'budget_stop')
   from src.research.moe_v4_evaluation import evaluate_milestone
   status(None,'DEVELOPMENT_EVALUATION');evaluate_milestone(C,cfg,models,state,milestone)
   if all(s['status']=='BUDGET_COMPLETE' or s['updates']>=cfg['milestones'][-1] for s in state.values()):break
  status(None,'FINAL_DEVELOPMENT_EVALUATION')
  from src.research.moe_v4_evaluation import final_assessment
  final_assessment(C,cfg,checkpoint_index,state,partition,adapter);status(None,'DEVELOPMENT_COMPLETE')
 except Exception as e:write(C/'failure.json',dict(error=str(e),traceback=traceback.format_exc(),state=state));status(None,'FAILED_NO_AUTOMATIC_RESTART');raise

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--moe-sboed-v4-development',action='store_true');ap.add_argument('--campaign',required=True);ap.add_argument('--phase',choices=['cpu-prep','gpu-smoke','gpu-diagnostics','grid-smoke','pilot'],required=True);args=ap.parse_args();C=Path(args.campaign).resolve()
 if ROOT.name!='objective_power_grid_temp' or ROOT/'scratch_runs/moe_sboed_v4' not in C.parents:raise ValueError('V4 requires TEMP/scratch_runs/moe_sboed_v4')
 cfg=cfgread(C)
 if args.phase=='cpu-prep':prep(C,cfg)
 elif args.phase=='gpu-smoke':smoke(C,cfg)
 elif args.phase=='gpu-diagnostics':
  from src.research.moe_v4_smoke_diagnostics import check
  check(C,cfg)
 elif args.phase=='grid-smoke':
  locks=gpu_lock();import pytest
  rc=pytest.main(['tools/tests/test_moe_sboed_v4.py::test_grid_unchanged_simulator','-q','--disable-warnings','--junitxml='+str(C/'readiness/grid_numerical_tests.xml')]);write(C/'readiness/grid_numerical_status.json',dict(returncode=rc,compiler_available=__import__('shutil').which('nvcc')));raise SystemExit(rc)
 else:pilot(C,cfg)
if __name__=='__main__':main()
