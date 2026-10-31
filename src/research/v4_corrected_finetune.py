"""Corrected V4: identical historical V1 inference; offline expert gradient change."""
from pathlib import Path
import copy,json,hashlib,math,time,os,sys,subprocess,traceback
import numpy as np
import torch
from src.research.deepseek_sboed import DeepSeekPolicy
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout,generator
from src.research.context_expert_gradients import capture_context,partition_context,collect_context_gradients,apply_context_gradients
from src.research.v4_corrected_information_gradients import belief,context_from,collect,apply
from src.research.moe_v4_pilot import gpu_lock
from src.research.journal_validation import CommonValidation
R=Path(__file__).resolve().parents[2];C=R/'scratch_runs/moe_sboed_v4_corrected_v1_finetune';ENTRY=time.monotonic()
def write(p,x):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False,default=str));tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(seed=101):
 audit=json.loads((C/'lineage_audit.json').read_text());s=torch.load(audit['seed101']['matching_full_checkpoint'],map_location='cpu',weights_only=False)
 if seed==102:raise RuntimeError('Seed102 full optimizer lineage must be verified before replication')
 m=DeepSeekPolicy();m.load_state_dict(s['state_dict'],strict=True);o=torch.optim.Adam(m.parameters(),lr=5e-5,betas=(.8,.998));sch=torch.optim.lr_scheduler.ExponentialLR(o,.98);o.load_state_dict(s['optimizer']);sch.load_state_dict(s['scheduler']);return m,o,sch,s

def validate():
 locks=gpu_lock();torch.set_num_threads(4);m,_,_,s=load();m=m.cuda();b=make_benchmark(s['configuration']['benchmark']);tests={}
 with torch.device('cuda:0'):
  torch.manual_seed(393010);m.train();p=copy.deepcopy(m);p.train()
  with capture_context(m,1) as snap:r=rollout(b,m,30,64,393011,128)
  loss=-r['info'][:,-1].mean()+m.balance_loss();old=partition_context(snap);oldg=collect_context_gradients(m,r['info'][:,-1],m.balance_loss(),old)
  # Balanced synthetic context is only a reconstruction/gradient fixture.
  f=dict(group=torch.arange(64,device='cuda')%2,reliable=torch.ones(64,device='cuda',dtype=torch.bool));ctx=context_from(f,snap);ng=collect(m,r['info'][:,-1],m.balance_loss(),ctx)
  loss.backward();before={n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None and not n.startswith('routed.')};rec=apply(m,oldg,old,ng,ctx)
  assert all(torch.equal(before[n],p.grad) for n,p in m.named_parameters() if n in before);tests['unmodified_gradients_bitwise_equal']=True;tests['cohort_reconstruction_and_guards']=rec
  with torch.no_grad():
   for stage,f0 in enumerate(r['features']):assert torch.equal(m(f0,stage),p(f0,stage))
  tests['initial_action_equality_all30']=True;tests['architecture']=m.counts();tests['original_loss_reused']=True
  m.eval()
  with torch.no_grad():
   rr=rollout(b,m,30,128,393021,0,track_likelihood=False);particles=b.sample((128,2000),generator(393022));small=belief(b,particles,rr['actions'][:1],rr['observations'][:1]);refs=[]
   for i in range(0,128,4):
    bank=b.sample((4,131072),generator(393023+i));refs.append(belief(b,bank,[x[i:i+4] for x in rr['actions'][:1]],[x[i:i+4] for x in rr['observations'][:1]]))
   ref={k:torch.cat([x[k] for x in refs]) for k in refs[0]};eligible=small['reliable']&(ref['ess']>=512)&(ref['pmax']<=.02);N=int(eligible.sum());agreement=float(((small['value']>.75)==(ref['value']>.75))[eligible].double().mean()) if N else 0.;err=float((small['value']-ref['value']).abs()[eligible].quantile(.9)) if N else 1e9;counts=torch.bincount(small['group'][eligible],minlength=3).tolist();passed=N>=16 and min(counts[:2])>=4 and agreement>=.9 and err<=.25
   audit=dict(status='PASS' if passed else 'BLOCKED_INFORMATION_COHORT_VALIDATION',cases=128,particles=2000,independent_reference_particles=131072,stage=1,threshold=.75,minimum_ESS=64,maximum_weight=.1,split_bank_max_dispersion_difference=.2,classification_margin=.1,reference_minimum_ESS=512,eligible_cases=N,cohort_counts=counts,classification_agreement=agreement,absolute_feature_error_p90=err,reference_seed=393023,validation_seed=393021,scope='Prior-predictive dispersion ratio, not a full posterior fit or identifiability certification; only stage1 admitted, other stages fall back to historical V1',small_values=small['value'].tolist(),reference_values=ref['value'].tolist(),small_ESS=small['ess'].tolist(),reference_ESS=ref['ess'].tolist())
   write(C/'information_feature_validation.json',audit);tests['information_features']=audit;write(C/'correctness.json',tests)
   np.savez(C/'feature_validation_cases.npz',actions=torch.stack(rr['actions']).cpu().numpy(),observations=torch.stack(rr['observations']).cpu().numpy())
 print(json.dumps({k:v for k,v in audit.items() if not isinstance(v,list)},indent=2),flush=True)
 return passed

def stat(x):
 x=np.asarray(x);mu=float(x.mean());se=float(x.std(ddof=1)/math.sqrt(len(x)));return dict(mean=mu,MC_SE=se,CI95=[mu-1.96*se,mu+1.96*se])
def evaluate(models,b,cfg,u):
 with torch.no_grad():
  v=CommonValidation(cfg['benchmark'],30,393101,512,262144,batch=16,chunk=16384,device='cuda:0');scores={}
  for name,p in models.items():
   p.eval();a,meta=v(p);scores[name]=a;np.save(C/f'eval_{u}_{name}.npy',a)
  out={k:stat(a) for k,a in scores.items()};out['paired_v4_minus_v1']=stat(scores['v4']-scores['v1']);write(C/f'evaluation_update{u}.json',out);print('EVALUATION',u,json.dumps(out),flush=True);return out

def pilot():
 assert json.loads((C/'information_feature_validation.json').read_text())['status']=='PASS'
 if (C/'pilot_started.json').exists():raise RuntimeError('No automatic restart')
 locks=gpu_lock();torch.set_num_threads(4);m0,_,_,ss=load();cfg=ss['configuration'];b=make_benchmark(cfg['benchmark']);models={};opts={};sched={};state={};started=time.monotonic();manifest=json.loads((C/'source_sha256.json').read_text())
 for name in ['v1','v4']:
  m,o,sc,_=load();m=m.cuda()
  for vv in o.state.values():
   for k,v in vv.items():
    if torch.is_tensor(v):vv[k]=v.cuda()
  models[name]=m;opts[name]=o;sched[name]=sc;state[name]=dict(updates=0,charged_seconds=0.,specialization_seconds=0.,information_adjustments=0,reliable_information_steps=0)
 startup=time.monotonic()-ENTRY
 for name in state:state[name]['charged_seconds']=startup
 write(C/'pilot_started.json',dict(PID=os.getpid(),source_sha256=sha(C/'source_sha256.json'),state='RUNNING'))
 def save(name,u):
  t=time.monotonic();p=C/f'{name}_update{u}.pt';torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in models[name].state_dict().items()},optimizer=opts[name].state_dict(),scheduler=sched[name].state_dict(),state=state[name],initial_checkpoint_sha256=json.loads((C/'lineage_audit.json').read_text())['seed101']['sha256'],source_sha256=sha(C/'source_sha256.json'),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),p);state[name]['charged_seconds']+=time.monotonic()-t;write(C/f'{name}_update{u}.json',dict(path=str(p),sha256=sha(p),**state[name]))
 for name in models:save(name,0)
 evaluate(models,b,cfg,0)
 for milestone in [512,1024,2048]:
  for name,m in models.items():
   m.train();o=opts[name];sc=sched[name];s=state[name]
   while s['updates']<milestone:
    if s['charged_seconds']>3540:raise RuntimeError('Time cap approached; preserve matched checkpoint instead of overrun')
    for path,digest in manifest.items():
     if s['updates']%512==0 and sha(R/path)!=digest:raise RuntimeError('Frozen source changed')
    u=s['updates']+1;global_u=ss['update']+u;seed=101+u*104729;t=time.monotonic();o.zero_grad(set_to_none=True);special=global_u%10==0;stage=[1,10,20,29][(global_u//10-1)%4] if special else 1
    with torch.device('cuda:0'):
     with capture_context(m,stage) as snap:r=rollout(b,m,30,2000,seed,2000)
     info=r['info'][:,-1];bal=m.balance_loss();loss=-info.mean()+bal;cg=ctx=ng=nc=None;special_begin=time.monotonic()
     if special:
      ctx=partition_context(snap)
      if ctx['valid']:cg=collect_context_gradients(m,info,bal,ctx)
      if name=='v4' and stage==1:
       ft=belief(b,r['particles'].detach(),[a.detach() for a in r['actions'][:stage]],[y.detach() for y in r['observations'][:stage]]);nc=context_from(ft,snap)
       if nc['valid']:ng=collect(m,info,bal,nc);s['reliable_information_steps']+=1
     loss.backward();record=None
     if cg is not None:
      record=apply(m,cg,ctx,ng,nc) if name=='v4' else apply_context_gradients(m,cg,ctx)
      if name=='v4':s['information_adjustments']+=record['adjusted'] if ng is not None else 0
     if special:torch.cuda.synchronize();s['specialization_seconds']+=time.monotonic()-special_begin
     assert torch.isfinite(loss);gn=torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);o.step()
     if (global_u-1)%1000==0:sc.step()
     torch.cuda.synchronize()
    s['updates']=u;s['charged_seconds']+=time.monotonic()-t
    if u==1 or u%25==0:
     row=dict(arm=name,**s,training_sPCE=float(info.mean().detach()),saturation_fraction=float((info>math.log(2001)-.1).double().mean()),gradient_norm=float(gn),learning_rate=o.param_groups[0]['lr'],stage=stage if special else None,gradient_record=record)
     with (C/f'{name}_training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
     write(C/'status.json',dict(state='RUNNING',PID=os.getpid(),active_arm=name,arms=state));print(json.dumps(row),flush=True)
    del r,info,bal,loss,cg,ng
   save(name,milestone)
  evaluate(models,b,cfg,milestone)
 write(C/'status.json',dict(state='COMPLETE',PID=os.getpid(),arms=state));print('COMPLETE',flush=True)
if __name__=='__main__':
 try:
  if sys.argv[1]=='validate':validate()
  else:pilot()
 except Exception as e:
  write(C/'failure.json',dict(type=type(e).__name__,error=str(e),traceback=traceback.format_exc()));raise
