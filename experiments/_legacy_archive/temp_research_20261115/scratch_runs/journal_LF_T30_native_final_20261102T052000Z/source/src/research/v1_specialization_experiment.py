
"""V1-only readiness, bounded development and promoted two-seed training."""
import argparse,copy,json,hashlib,math,subprocess,time
_PROCESS_BEGIN=time.monotonic()
from pathlib import Path
from contextlib import nullcontext
import numpy as np,torch,yaml
from src.research.deepseek_sboed import DeepSeekPolicy,load_deepseek
from src.research.deepseek_sboed_experiment import write,sha,norm
from src.research.context_expert_gradients import capture_context,partition_context,collect_context_gradients,apply_context_gradients
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout
from src.research.journal_validation import CommonValidation

FOUNDATION='13681d7091189a4e68056e6a2864dc58f9d5700b'
def configuration(path,seed=101):
 cfg=yaml.safe_load(Path(path).read_text());cfg['training_seed']=seed
 for key in ['corrected_DAD_checkpoint','canonical_evaluation_directory']:cfg[key]=str(Path(cfg[key]).expanduser())
 assert cfg['contrasts']==2000 and cfg['batch_size']==2000 and cfg['main_training_seconds']==14400
 return cfg
def paired(a,b):
 d=np.asarray(a)-np.asarray(b);se=float(d.std(ddof=1)/len(d)**.5)
 return dict(difference=float(d.mean()),paired_MC_SE=se,CI95=[float(d.mean()-1.96*se),float(d.mean()+1.96*se)],N=len(d))
def guard_foundation():
 for name in ['src/research/deepseek_sboed.py','src/research/deepseek_sboed_experiment.py','src/research/dad_location_policy.py','src/research/engine.py','src/research/benchmarks.py','src/research/journal_validation.py']:
  assert Path(name).read_bytes()==subprocess.check_output(['git','show',FOUNDATION+':'+name]),name

def step(model,optimizer,b,cfg,update,simulation_seed,treatment):
 special=treatment and update%cfg['specialization_every']==0
 stage=cfg['specialization_stages'][(update//cfg['specialization_every']-1)%len(cfg['specialization_stages'])] if special else 1
 optimizer.zero_grad(set_to_none=True);begin=time.perf_counter()
 with torch.device('cuda'):
  with (capture_context(model,stage) if special else nullcontext({})) as snapshot:
   r=rollout(b,model,30,cfg['batch_size'],simulation_seed,cfg['contrasts'])
  balance=model.balance_loss();loss=-r['info'][:,-1].mean()+balance
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite v1 full-horizon objective')
  context=partition_context(snapshot) if special else None;cg=None;extra_seconds=0.
  if context and context['valid']:
   torch.cuda.synchronize();extra_begin=time.perf_counter()
   cg=collect_context_gradients(model,r['info'][:,-1],balance,context)
   torch.cuda.synchronize();extra_seconds=time.perf_counter()-extra_begin
  loss.backward()
  unchanged={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None and not n.startswith('routed.')}
  before=norm(model.parameters());record=None
  if cg is not None:
   extra_begin=time.perf_counter();record=apply_context_gradients(model,cg,context);torch.cuda.synchronize();extra_seconds+=time.perf_counter()-extra_begin
  for n,p in model.named_parameters():
   if n in unchanged:assert torch.equal(unchanged[n],p.grad),n
  after=norm(model.parameters())
  if abs(after-before)>1e-9*max(1.,before):raise AssertionError('Specialization changed global preclip gradient norm')
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
  assert torch.stack(r['actions']).abs().max()<=4
  torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
  optimizer.step();torch.cuda.synchronize()
 return dict(update=update,SPCE=float(r['info'][:,-1].mean().detach()),preclip_gradient=before,post_specialization_gradient=after,context_gradient=record,extra_specialization_seconds=extra_seconds,seconds=time.perf_counter()-begin,routing=model.routing_statistics(),expert_gradient_norms=[norm(e.parameters()) for e in model.routed])

def smoke(C,cfg):
 from tools.journal_campaign import exclusive_device_lock
 lock=exclusive_device_lock(wait=True);torch.set_num_threads(2);guard_foundation();torch.manual_seed(cfg['training_seed'])
 model=DeepSeekPolicy().cuda().train();b=make_benchmark(cfg['benchmark']);optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['adam_betas']))
 # Force the first scheduled context step; smoke seed/data never initialize main.
 result=step(model,optimizer,b,cfg,10,97666001,True)
 assert model.counts()['total_parameters']==8434 and not model._forward_hooks
 result.update(PASS=True,counts=model.counts(),batch_size=2000,contrasts=2000,horizon=30,parameter_stage_propagations=b.calls,optimizer_updates=1,weights_discarded=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),foundation=FOUNDATION)
 write(C/'readiness/GPU_smoke.json',result)
 write(C/'readiness/native_full_minibatch_probe.json',dict(seconds=result['seconds'],batch_size=2000,contrasts=2000,weights_discarded=True))
 print('V1_SPECIALIZATION_GPU_SMOKE',json.dumps(result),flush=True);lock.close()

def pilot(C,cfg):
 assert json.loads((C/'readiness/v1_reproduction.json').read_text())['PASS']
 assert json.loads((C/'readiness/GPU_smoke.json').read_text())['PASS']
 assert '18 passed' in (C/'readiness/CPU_tests.txt').read_text()
 from tools.journal_campaign import exclusive_device_lock
 lock=exclusive_device_lock(wait=True);torch.set_num_threads(2);guard_foundation();start=time.perf_counter()
 baseline=load_deepseek('reference_artifacts/v1/selected.pt','cuda')
 models=[copy.deepcopy(baseline).train(),copy.deepcopy(baseline).train()]
 optimizers=[torch.optim.Adam(m.parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['adam_betas'])) for m in models]
 benchmarks=[make_benchmark(cfg['benchmark']) for _ in models];records=[];completed=0
 for update in range(1,cfg['pilot_updates_per_policy']+1):
  if time.perf_counter()-start>cfg['pilot_seconds']-5:break
  seed=97665000+104729*update;pair=[]
  for index,(m,o,b) in enumerate(zip(models,optimizers,benchmarks)):
   pair.append(step(m,o,b,cfg,update,seed,index==1))
  records.append(pair);completed=update
  if update%10==0:print('PILOT_UPDATE',update,'SPCE',[x['SPCE'] for x in pair],'context',pair[1]['context_gradient'],flush=True)
 val=CommonValidation(cfg['benchmark'],30,97665003,512,262144,batch=16,chunk=16384,device='cuda:0')
 scores=[];metadata=[]
 for label,m in zip(['frozen_v1','continued_v1','specialized_v1'],[baseline]+models):
  v,meta=val(m.eval());scores.append(v);metadata.append(meta);np.save(C/'readiness'/f'pilot_{label}_scores.npy',v)
 reports=[r[1]['context_gradient'] for r in records if r[1]['context_gradient']]
 adjustments=sum(r['adjusted'] for r in reports);stages=sorted(set(r['stage'] for r in reports if r['adjusted']>0))
 result=dict(PASS=bool(completed>0 and adjustments>0 and len(stages)>=2),foundation=FOUNDATION,updates_per_policy=completed,paired_training_randomness=True,main_weights_reused=False,specialization_adjustments=adjustments,adjusted_stages=stages,context_steps=reports,CGS_minus_matched_continued_v1=paired(scores[2],scores[1]),CGS_minus_frozen_v1=paired(scores[2],scores[0]),results={label:dict(EIG=float(v.mean()),MC_SE=float(v.std(ddof=1)/len(v)**.5),metadata=meta) for label,v,meta in zip(['frozen_v1','continued_v1','specialized_v1'],scores,metadata)},wall_seconds=time.perf_counter()-start,parameter_stage_propagations=[b.calls for b in benchmarks],specialization_extra_seconds=sum(r[1]['extra_specialization_seconds'] for r in records),selection='Predeclared single procedure; no hyperparameter selection; pilot weights discarded')
 write(C/'readiness/pilot_gate.json',result);write(C/'reports/pilot_updates.json',records);print('PILOT_GATE',json.dumps(result),flush=True);lock.close()
 if not result['PASS']:raise RuntimeError('Pilot did not demonstrate a functioning context-specialization update; do not promote')

def main_run(C,cfg):
 from tools.journal_campaign import exclusive_device_lock
 from src.research.v1_specialization_train import train_specialized
 queue_begin=time.monotonic();lock=exclusive_device_lock(wait=True);queue_seconds=time.monotonic()-queue_begin;torch.set_num_threads(2);guard_foundation()
 seal=json.loads((C/'metadata/source_freeze.json').read_text())
 for p,h in seal.items():assert sha(Path(p))==h,p
 assert json.loads((C/'readiness/promotion_tests.json').read_text())['PASS']
 write(C/'registry.json',dict(status='MAIN_TRAINING',training_seed=cfg['training_seed'],pid=__import__('os').getpid(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
 try:
  val=CommonValidation(cfg['benchmark'],30,cfg['development_seed'],128,262144,batch=16,chunk=16384,device='cuda:0')
  from src.research.v1_specialization_evaluation import install_arrays
  val=install_arrays(val,Path.home()/'Documents/deepseek_v1_specialization_banks/periodic')
  credit=json.loads((C/'readiness/native_full_minibatch_probe.json').read_text())['seconds']
  setup_seconds=time.monotonic()-_PROCESS_BEGIN-queue_seconds
  write(C/'costs/setup.json',dict(setup_seconds=setup_seconds,queue_seconds=queue_seconds,readiness_credit_seconds=credit,main_clock_credit_seconds=credit+setup_seconds))
  model,stop=train_specialized(C,cfg,'full',cfg['reference_optimizer_updates'],cfg['main_training_seconds'],val,credit=credit+setup_seconds)
  for p,h in seal.items():assert sha(Path(p))==h,p
  checkpoint=C/'models/full/selected.pt'
  write(C/'reports/checkpoint_freeze.json',dict(sha256=sha(checkpoint),training_seed=cfg['training_seed'],selection='Latest eligible v1 checkpoint',official_evaluation_seen=False,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
  write(C/'registry.json',dict(status='FROZEN_EVALUATION',training_seed=cfg['training_seed']))
  from src.research.v1_specialization_evaluation import finish
  finish(C,cfg,model)
  write(C/'registry.json',dict(status='COMPLETE_ONE_SEED',training_seed=cfg['training_seed']))
 except Exception:
  import traceback
  write(C/'registry.json',dict(status='FAILED_PRESERVED',traceback=traceback.format_exc(),training_seed=cfg['training_seed']));raise
 finally:lock.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--v1-specialization',action='store_true');p.add_argument('--phase',choices=['smoke','pilot','main'],required=True);p.add_argument('--campaign',required=True);p.add_argument('--config',default='reports/DEEPSEEK_V1_SPECIALIZATION_CONFIG.yaml');p.add_argument('--seed',type=int,default=101);a=p.parse_args()
 C=Path(a.campaign).resolve()
 for folder in ['readiness','costs','reports','models','evaluations','metadata','logs']:(C/folder).mkdir(parents=True,exist_ok=True)
 cfg=configuration(a.config,a.seed)
 if a.phase=='smoke':smoke(C,cfg)
 elif a.phase=='pilot':pilot(C,cfg)
 else:main_run(C,cfg)
if __name__=='__main__':main()
