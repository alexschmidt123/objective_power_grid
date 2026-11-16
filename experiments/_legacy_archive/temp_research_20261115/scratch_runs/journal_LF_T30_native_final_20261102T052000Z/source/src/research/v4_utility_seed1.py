"""Four-hour seed1, historical V1 deployment plus training-only utility credit."""
import time
ENTRY=time.monotonic()
from pathlib import Path
import os,sys,json,hashlib,copy,math,subprocess,traceback
import torch,numpy as np
from src.research.deepseek_sboed import DeepSeekPolicy,load_deepseek
from src.research.engine import rollout,generator
from src.research.benchmarks import make_benchmark
from src.research.v4_utility_routing import paired_credit,route_output,swapped_ids,ranking_loss,add_credit
from src.research.compute_budget import ComputeClock
from src.research.moe_v4_pilot import gpu_lock
from src.research.journal_validation import CommonValidation
R=Path(__file__).resolve().parents[2];C=R/'scratch_runs/moe_sboed_v4_utility_routing_seed1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)
def config():return json.loads((C/'configuration.json').read_text())
def stat(a):
 a=np.asarray(a);mu=float(a.mean());se=float(a.std(ddof=1)/math.sqrt(len(a)));return dict(mean=mu,MC_SE=se,CI95=[mu-1.96*se,mu+1.96*se])
def prepare():
 C.mkdir(exist_ok=False);[(C/x).mkdir() for x in ['reports','checkpoints','logs','evaluations','provenance']]
 archive=R.parent/'objective_power_grid/experiments/_legacy_archive/labpc';cp=archive/'objective_power_grid_v1_specialization_integration/experiments/10262026/10262026_v1_cgs_seed101/models/full/selected.pt';assert sha(cp)=='b680081479f1ca0227f250265642cea0452ce57e110d26220746d7966b50edfd';old=torch.load(cp,map_location='cpu',weights_only=False)
 cfg=dict(method='moe_sboed_v4_utility_routing_seed1',status='DEVELOPMENT_UNVALIDATED',seed=1,initialization='FROM_SCRATCH_NO_HISTORICAL_WEIGHTS',benchmark=old['configuration']['benchmark'],horizon=30,batch=2000,contrasts=2000,learning_rate=5e-5,betas=[.8,.998],lr_decay=.98,lr_schedule_every=1000,gradient_clip=5,balance=.001,budget_seconds=14400,auxiliary_every=100,auxiliary_batch=64,auxiliary_contrasts=1024,auxiliary_router_gradient_fraction=.05,auxiliary_normalization='detached RMS, clamp[-1,1]',safety='skip nonfinite targets, RMS<1e-8 or RMS>100; never increase coefficient',checkpoint_updates=[1000,3000,5000],development_seed=394101,final_seed=394102,diagnostic_seed=394103,official_seed1001_unused=True,contextual_checkpoint=str(cp),contextual_sha256=sha(cp))
 # Locate the original reference by its registered immutable digest, not by EIG.
 import src.research.project_scope as scope,re
 code=Path(scope.__file__).read_text();match=re.search(r"hexdigest\(\)!='([0-9a-f]+)'",code[code.index('def load_current_policy'):]);digest=match.group(1) if match else None
 candidates=[p for p in archive.rglob('selected.pt') if 'deepseek' in str(p).lower() and 'v2' not in str(p).lower() and 'v3' not in str(p).lower()]
 for p in candidates:
  if sha(p)==digest:cfg.update(original_checkpoint=str(p),original_sha256=sha(p));break
 cfg['original_reference_lookup']=dict(registered_sha256=digest,candidates=len(candidates),found='original_checkpoint' in cfg)
 write(C/'configuration.json',cfg)
 for rel in ['src/research/deepseek_sboed.py','src/research/context_expert_gradients.py']:
  data=subprocess.check_output(['git','show','042915bfe8312faa333b1b825b94766c00c68dbf:'+rel],cwd=R);assert hashlib.sha256(data).hexdigest()==sha(R/rel)
 descriptions={
 'V4_ARCHITECTURE_EQUIVALENCE.md':'Unchanged pinned DeepSeekPolicy:8434total/6898active parameters,1shared+7routed SwiGLU(width8),top3,16features,original unnormalized selected softmax gates/residual/head/feasibility. Seed1 initialized from scratch; no historical weights loaded into training.',
 'V4_UTILITY_ROUTING_METHOD.md':'Original loss -mean terminal sequential sPCE +.001 router balance. Every100updates:64fresh joint histories,1024contrasts; one random decision per case and one selected/unselected replacement. Two full paired rollouts use identical truth/noise/contrast draws and identical observed prefixes. Only the selected decision is intervened; later policy resumes. Realized terminal advantage is detached,RMS-normalized,clipped[-1,1]. Router pairwise logistic loss favors replacement for positive utility and original for negative utility. Its additive gradient norm is capped at5% of primary router-gradient norm. This is biased/noisy offline credit, not exact conditional information-to-go or an unbiased EIG gradient. No auxiliary changes to expert/encoder/shared/head gradients; routed experts train through original terminal loss. No arbitrary context-gradient or role loss retained.',
 'V4_CORRECTNESS.md':'PENDING focused CPU tests and GPU native smoke. No information-cohort gate applies.',
 'V4_SEED1_TRAINING_STATUS.md':'PREPARING; training has not started.',
 'V4_COMPUTE.md':'14400charged training seconds including setup, discarded GPU smoke preparation, auxiliary paired simulations, gradients, logging, checkpoints. CPU correctness and independent development/final scoring separately recorded. No episode quota; checkpoint selection latest valid serialized within cap. L2000 ceiling log2001=7.6014023346.',
 'V4_EIG_COMPARISON.md':'PENDING frozen final checkpoint. Planned2000paired cases/1Mcontrasts, fresh seed394102, contextualV1 and originalV1 on identical draws. Historical11.04404166 is a target, not a new result or directly paired score. No official seed1001 used.',
 'V4_EXPERT_SPECIALIZATION.md':'PENDING independent diagnostics. Random top3 preserves raw gate probabilities/cardinality. Whole-routed masking/substitution are distribution-shifting frozen diagnostics, not proof of seven necessary semantic roles. Single seed cannot establish robust improvement.'}
 for name,txt in descriptions.items():(C/'reports'/name).write_text('# '+name.replace('.md','')+'\n\n'+txt+'\n')
 print('PREPARED',C,flush=True)

def tests(device='cpu',native=False):
 cfg=config();torch.set_num_threads(4);torch.manual_seed(1);m=DeepSeekPolicy();clone=DeepSeekPolicy();clone.load_state_dict(m.state_dict(),strict=True);m=m.to(device);clone=clone.to(device);m.eval();clone.eval();assert m.counts()['total_parameters']==8434 and m.counts()['activated_parameters_per_context']==6898
 with torch.device(device):
  b=make_benchmark(cfg['benchmark']);r=rollout(b,m,30,4,394001,8);records=[]
  for stage,f in enumerate(r['features']):
   a=m(f,stage);torch.testing.assert_close(a,clone(f,stage),rtol=0,atol=0);ids=m.last_dispatch['indices'];assert ids.shape==(4,3) and torch.nn.functional.one_hot(ids,7).sum(1).max()==1;out=route_output(m,m.last_features,ids);torch.testing.assert_close(out,a,rtol=1e-12,atol=1e-12)
   new,drop,add=swapped_ids(ids,torch.arange(4)%3,torch.arange(4)%4);assert ((ids[:,:,None]==new[:,None,:]).sum((1,2))==2).all()
  ck=C/'provenance'/('cpu_roundtrip.pt' if device=='cpu' else 'gpu_roundtrip.pt');torch.save(dict(state_dict=m.state_dict()),ck);clone.load_state_dict(torch.load(ck,map_location=device,weights_only=False)['state_dict'],strict=True)
  rec=paired_credit(b,m,394002,batch=8,contrasts=16);assert rec['same_prefix'] and not rec['advantage'].requires_grad and torch.isfinite(rec['advantage']).all()
  router=torch.nn.Linear(16,7,bias=False).double();z=torch.ones(1,16,dtype=torch.float64);adv=torch.tensor([1.],dtype=torch.float64,requires_grad=True);loss,_=ranking_loss(router,z,torch.tensor([0]),torch.tensor([1]),adv);g,leak=torch.autograd.grad(loss,(router.weight,adv),allow_unused=True);assert leak is None and g[1].sum()<0 and g[0].sum()>0
  loss,_=ranking_loss(router,z,torch.tensor([0]),torch.tensor([1]),-adv);gg=torch.autograd.grad(loss,router.weight)[0];assert gg[1].sum()>0 and gg[0].sum()<0
  m.train();rr=rollout(b,m,30,4,394003,8);primary=-rr['info'][:,-1].mean()+m.balance_loss();primary.backward();before={n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None};meta=add_credit(m,rec);assert all(torch.equal(p.grad,before[n]) for n,p in m.named_parameters() if n in before and n!='router.weight');assert torch.isfinite(primary)
  native_seconds=0
  if native:
   start=time.monotonic();torch.manual_seed(1);p=DeepSeekPolicy().to(device).train();o=torch.optim.Adam(p.parameters(),lr=5e-5,betas=(.8,.998));rr=rollout(b,p,30,2000,104730,2000);loss=-rr['info'][:,-1].mean()+p.balance_loss();loss.backward();credit=paired_credit(b,p,394004,64,1024);aux=add_credit(p,credit);torch.nn.utils.clip_grad_norm_(p.parameters(),5,error_if_nonfinite=True);o.step();torch.cuda.synchronize();native_seconds=time.monotonic()-start;assert torch.isfinite(loss)
  result=dict(status='PASS',device=device,architecture=m.counts(),strict_state_dict=True,forward_equality=True,top3=True,controlled_one_swap=True,same_prefix=True,terminal_full_horizon=True,router_gradient_direction=True,detached_utility_target=True,only_router_auxiliary_gradient=True,checkpoint_roundtrip=True,numerically_finite=True,native_full_update_seconds=native_seconds,auxiliary=meta,seconds=time.monotonic()-ENTRY)
  write(C/'provenance'/('gpu_smoke.json' if native else 'cpu_tests.json'),result);print(json.dumps(result,indent=2),flush=True)

def development(m,cfg,u):
 val=CommonValidation(cfg['benchmark'],30,cfg['development_seed'],128,65536,batch=16,chunk=16384,device='cuda:0');a,meta=val(m);np.save(C/'evaluations'/f'development_update{u}.npy',a);write(C/'evaluations'/f'development_update{u}.json',dict(**stat(a),metadata=meta,update=u,selection='monitoring only, never selected by EIG'))

def final_evaluation(m,cfg,checkpoint):
 val=CommonValidation(cfg['benchmark'],30,cfg['final_seed'],2000,1000000,batch=16,chunk=16384,device='cuda:0');np.savez(C/'evaluations/final_paired_cases.npz',truth=val.truth.numpy(),noise=val.noise.numpy(),bank=val.bank.numpy());policies={'v4':m,'contextual_v1':load_deepseek(cfg['contextual_checkpoint'],'cuda:0')}
 assert sha(cfg['contextual_checkpoint'])==cfg['contextual_sha256']
 if cfg.get('original_checkpoint'):
  assert sha(cfg['original_checkpoint'])==cfg['original_sha256'];policies['original_v1']=load_deepseek(cfg['original_checkpoint'],'cuda:0')
 scores={};results={}
 for name,p in policies.items():
  a,meta=val(p);scores[name]=a;results[name]=dict(**stat(a),metadata=meta);np.save(C/'evaluations'/f'final_{name}.npy',a);write(C/'evaluations/final_results.json',results)
 for name in policies:
  if name!='v4':results['v4_minus_'+name]=stat(scores['v4']-scores[name])
 write(C/'evaluations/final_comparison.json',results);lines=['# Frozen final EIG comparison','2000paired trajectories;1Mcontrasts; one seed; no official test randomness.','|Policy|EIG|MCSE|','|---|---:|---:|']
 for name in policies:lines.append(f"|{name}|{results[name]['mean']:.8f}|{results[name]['MC_SE']:.8f}|")
 lines.append('\nPaired V4-contextualV1: '+json.dumps(results['v4_minus_contextual_v1']));lines.append('Checkpoint: '+json.dumps(checkpoint));(C/'reports/V4_EIG_COMPARISON.md').write_text('\n'.join(lines));return results

class DiagnosticRouting(torch.nn.Module):
 kind='deepseek_moe_sboed';outputs_design=False
 def __init__(self,base,mode,seed=394201):
  super().__init__();self.base=copy.deepcopy(base).eval();self.mode=mode;self.seed=seed;self.rng=None
 def __getstate__(self):
  d=super().__getstate__();d['rng']=None;return d
 def forward(self,f,stage):
  z=self.base.encode_context(f,stage);scores=self.base.router(z).softmax(-1);ids=scores.topk(3,-1).indices
  if self.mode=='uniform':
   if self.rng is None:self.rng=torch.Generator(device=f.device).manual_seed(self.seed)
   ids=torch.rand(len(f),7,device=f.device,generator=self.rng).topk(3,-1).indices
  if self.mode=='mask_all':return self.base.emitter(z+self.base.shared(z))
  if self.mode.startswith('mask'):
   masked=int(self.mode[4:]);u=z+self.base.shared(z);v=z.new_zeros(z.shape)
   for e in ids.unique().tolist():
    if e==masked:continue
    rows,slots=(ids==e).nonzero(as_tuple=True);v.index_add_(0,rows,self.base.routed[e](z[rows])*scores[rows,e,None])
   return self.base.emitter(u+v)
  return route_output(self.base,z,ids)

def diagnostics(m,cfg):
 val=CommonValidation(cfg['benchmark'],30,cfg['diagnostic_seed'],512,262144,batch=16,chunk=16384,device='cuda:0');base,meta=val(m);out={}
 for mode in ['uniform','mask_all']+['mask'+str(i) for i in range(7)]:
  p=DiagnosticRouting(m,mode);x,md=val(p);out[mode]=dict(EIG=stat(x),full_minus_intervention=stat(base-x),metadata=md);np.save(C/'evaluations'/('diagnostic_'+mode+'.npy'),x)
 reps=[]
 with torch.device('cuda:0'):
  b=make_benchmark(cfg['benchmark'])
  for seed in [394301,394302]:
   r=paired_credit(b,m,seed,128,65536);scores=m.router(r['z']).detach().softmax(-1);sortedp=scores.sort(-1,descending=True).values;context=(sortedp[:,2]-sortedp[:,3])>.05;cohorts=[]
   for v in [False,True]:
    x=r['advantage'][context==v].cpu().numpy();cohorts.append(dict(router_margin_above_point05=v,N=len(x),utility=stat(x) if len(x)>=16 else None))
   reps.append(dict(seed=seed,paired_utility=stat(r['advantage'].cpu().numpy()),contexts=cohorts,context='Observable original-router third-versus-fourth gate margin>.05; no stage labels or uncertainty cohort training',interpretation='Random one-swap utility across joint histories, not precise posterior-conditional utility. Context summaries descriptive; no multiplicity-adjusted specialist-role claim.'))
 write(C/'evaluations/specialization_diagnostics.json',dict(frozen_interventions=out,independent_replicates=reps,limitations='Frozen masking causes distribution shift; random routing preserves3 and raw gate semantics. No retrained control or training-seed replication.'))
 (C/'reports/V4_EXPERT_SPECIALIZATION.md').write_text('# Measured frozen expert diagnostics\n\n'+json.dumps(dict(interventions=out,replicates=reps),indent=2)+'\nNo causal semantic-specialist claim follows from these frozen interventions.\n')

def train():
 cfg=config();assert json.loads((C/'provenance/cpu_tests.json').read_text())['status']=='PASS';smoke=json.loads((C/'provenance/gpu_smoke.json').read_text());assert smoke['status']=='PASS'
 if (C/'provenance/started.json').exists():raise RuntimeError('Already started; no automatic duplicate/restart')
 locks=gpu_lock();torch.set_num_threads(4);torch.cuda.init();torch.manual_seed(1);m=DeepSeekPolicy().cuda().train();o=torch.optim.Adam(m.parameters(),lr=5e-5,betas=(.8,.998));sch=torch.optim.lr_scheduler.ExponentialLR(o,.98);b=make_benchmark(cfg['benchmark']);clock=ComputeClock(C/'provenance/compute_clock.json',14400,credit_seconds=smoke['seconds'],hardware=torch.cuda.get_device_name(0),start=ENTRY);manifest=json.loads((C/'provenance/source_sha256.json').read_text());write(C/'provenance/started.json',dict(PID=os.getpid(),UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),seed=1,from_scratch=True,source_manifest_sha256=sha(C/'provenance/source_sha256.json'),configuration_sha256=sha(C/'configuration.json'),gpu=torch.cuda.get_device_name(0)))
 u=0;bound=max(1.,smoke['native_full_update_seconds']);next_recovery=300;latest=None;aux_seconds=0;rows=[]
 def status(state,**kw):
  record=dict(state=state,PID=os.getpid(),update=u,fresh_training_trajectories=u*2000,charged_seconds=clock.spent(),auxiliary_seconds=aux_seconds,latest_checkpoint=latest,**kw);write(C/'status.json',record);(C/'reports/V4_SEED1_TRAINING_STATUS.md').write_text('# Seed1 status\n\n'+json.dumps(record,indent=2))
 def save(label):
  nonlocal latest
  for rel,h in manifest.items():
   if sha(R/rel)!=h:raise RuntimeError('Frozen source changed: '+rel)
  p=C/'checkpoints'/f'update{u}_{label}.pt';temp=p.with_suffix('.partial');torch.save(dict(policy=copy.deepcopy(m).cpu(),state_dict={k:v.detach().cpu().clone() for k,v in m.state_dict().items()},optimizer=o.state_dict(),scheduler=sch.state_dict(),update=u,seed=1,configuration=cfg,source_manifest_sha256=sha(C/'provenance/source_sha256.json'),configuration_sha256=sha(C/'configuration.json'),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),auxiliary_seconds=aux_seconds),temp);temp.replace(p);h=sha(p);mark=clock.checkpoint(str(p));row=dict(path=str(p),sha256=h,update=u,charged_seconds=clock.spent(),eligible=clock.spent()<=14400)
  write(p.with_suffix('.json'),row)
  if row['eligible']:latest=row;write(C/'checkpoints/latest_eligible.json',row)
 save('initial');status('RUNNING')
 try:
  while clock.can_start(max(30.,2*bound)):
   start=time.monotonic();u+=1;o.zero_grad(set_to_none=True)
   with torch.device('cuda:0'):
    r=rollout(b,m,30,2000,1+u*104729,2000);info=r['info'][:,-1];value=float(info.mean().detach());sat=float((info>math.log(2001)-.1).double().mean());loss=-info.mean()+m.balance_loss();assert torch.isfinite(loss);loss.backward();routing=m.routing_statistics() if u==1 or u%25==0 else None;del r,info,loss;m._gate_cache=[];aux=None
    if u%100==0:
     t=time.monotonic();credit=paired_credit(b,m,39400000+u,64,1024);aux=add_credit(m,credit,.05);torch.cuda.synchronize();aux_seconds+=time.monotonic()-t
     aux.update(update=u,seconds=time.monotonic()-t,same_prefix_verified=True,bank_shape=[64,1024],seed=39400000+u,paired_randomness_sha256=hashlib.sha256(credit['truth'].cpu().numpy().tobytes()+credit['noise'].cpu().numpy().tobytes()+credit['particles'].cpu().numpy().tobytes()).hexdigest())
     with (C/'logs/utility_credit.jsonl').open('a') as f:f.write(json.dumps(aux)+'\n')
    gn=torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);o.step()
    if (u-1)%1000==0:sch.step()
    torch.cuda.synchronize()
   bound=max(bound,time.monotonic()-start)
   if u==1 or u%25==0:
    row=dict(update=u,charged_seconds=clock.spent(),training_sPCE=value,saturation_fraction=sat,gradient_norm=float(gn),learning_rate=o.param_groups[0]['lr'],auxiliary_seconds=aux_seconds,auxiliary=aux,routing=routing);rows.append(row)
    with (C/'logs/training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True);status('RUNNING')
   milestone=u in [1000,3000,5000] or (u>=10000 and u%5000==0)
   if u==1 or milestone or clock.spent()>=next_recovery:
    save('milestone' if milestone else 'recovery');next_recovery=clock.spent()+300
   if milestone:
    status('DEVELOPMENT_EVALUATION')
    with clock.exclude('validation'):development(m,cfg,u)
    m.train();status('RUNNING')
  save('final');report=clock.stop();write(C/'provenance/compute_final.json',dict(clock=report,auxiliary_seconds=aux_seconds,updates=u,fresh_primary_trajectories=u*2000,auxiliary_primary_trajectories=(u//100)*128,auxiliary_contrast_parameter_stages=(u//100)*2*64*1025*30));status('TRAINING_COMPLETE_FINAL_EVALUATION')
  assert latest and sha(latest['path'])==latest['sha256'];m=torch.load(latest['path'],map_location='cuda:0',weights_only=False)['policy'];final_evaluation(m,cfg,latest);status('SPECIALIZATION_DIAGNOSTICS');diagnostics(m,cfg);status('COMPLETE')
 except Exception as e:
  write(C/'failure.json',dict(error=str(e),traceback=traceback.format_exc(),update=u,last_eligible=latest,clock=clock.stop()));status('FAILED_NO_AUTOMATIC_RESTART');raise
if __name__=='__main__':
 phase=sys.argv[1]
 if phase=='prepare':prepare()
 elif phase=='cpu':tests()
 elif phase=='smoke':
  locks=gpu_lock();tests('cuda:0',True)
 elif phase=='train':train()
 else:raise ValueError(phase)
