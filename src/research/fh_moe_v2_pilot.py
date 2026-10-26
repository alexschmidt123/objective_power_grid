"""FH-MoE-v2 readiness and bounded pilot, founded only on original v1."""
import argparse,copy,json,math,time,subprocess
from pathlib import Path
from contextlib import nullcontext,ExitStack
import numpy as np,torch,yaml
from src.research.deepseek_sboed import DeepSeekPolicy,load_deepseek
from src.research.deepseek_sboed_experiment import write,norm,diagnostics
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout
from src.research.journal_validation import CommonValidation
from src.research.fh_moe_v2_responsibilities import *

def pair(a,b):
 d=np.asarray(a)-np.asarray(b);se=float(d.std(ddof=1)/len(d)**.5)
 return dict(difference=float(d.mean()),paired_MC_SE=se,CI95=[float(d.mean()-1.96*se),float(d.mean()+1.96*se)],N=len(d))
def guard():
 for name in ['deepseek_sboed','deepseek_sboed_experiment','engine','dad_location_policy','benchmarks','journal_validation']:
  p=Path('src/research')/(name+'.py')
  assert p.read_bytes()==subprocess.check_output(['git','show','13681d7091189a4e68056e6a2864dc58f9d5700b:'+str(p)]),str(p)
def step(m,opt,b,cfg,update,bank,treatment):
 opt.zero_grad(set_to_none=True);start=time.perf_counter()
 special=treatment and update%10==0;stage=[1,10,20,29][(update//10-1)%4] if special else 1
 with torch.device('cuda'):
  with (capture_terminal_context(m,stage) if special else nullcontext({})) as snap:
   r=rollout(b,m,30,2000,98120000+update*104729,2000)
  info=-r['info'][:,-1].mean();balance=m.balance_loss();record=None
  if special:
   record,d,targets=information_gradients(m,info,balance,snap,bank,stage)
   if targets is not None:bank.stages[stage]['training_responsibility_mean']=targets.mean(0).detach().cpu()
  else:(info+balance).backward()
  pre=norm(m.parameters());assert math.isfinite(pre)
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
  assert torch.stack(r['actions']).abs().max()<=4
  groups={name:norm(parameters) for name,parameters in [('shared',m.shared.parameters()),('router',m.router.parameters()),('routed',m.routed.parameters()),('encoder',m.encoder.parameters()),('head',m.emitter.parameters())]}
  torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step();torch.cuda.synchronize()
 return dict(update=update,training_sPCE=float(-info.detach()),preclip_norm=pre,group_gradients=groups,seconds=time.perf_counter()-start,stage=stage,reference=record)

class UniformSubset(torch.nn.Module):
 kind='deepseek_moe_sboed';outputs_design=False
 def __init__(self,m,seed):super().__init__();self.policy=copy.deepcopy(m).eval();self.seed=seed;self.g=None
 def forward(self,f,t):
  m=self.policy;z=m.encode_context(f,t);p=m.router(z).softmax(-1);mass=p.topk(3,-1).values.sum(-1,keepdim=True)/3
  if self.g is None:self.g=torch.Generator(device=z.device).manual_seed(self.seed)
  ids=torch.rand(len(z),7,device=z.device,generator=self.g).topk(3,-1).indices;route=z*0
  for e in ids.unique().tolist():
   rows,slots=(ids==e).nonzero(as_tuple=True);route.index_add_(0,rows,m.routed[e](z[rows])*mass[rows])
  return m.emitter(z+m.shared(z)+route)

def prediction_probe(m,bank,cfg):
 b=make_benchmark(cfg['benchmark']);m.train()
 with torch.device('cuda'),ExitStack() as stack:
  snapshots=[stack.enter_context(capture_terminal_context(m,t)) for t in [1,10,20,29]]
  r=rollout(b,m,30,512,98102604,2000)
  loss=-r['info'][:,-1].mean()
  gradients=torch.autograd.grad(loss,[s['u'] for s in snapshots])
 values=[];stages=[]
 for t,s,d in zip([1,10,20,29],snapshots,gradients):
  target,meta=bank.targets(t,d.detach()*512)
  if target is None:stages.append(dict(stage=t,status='NO_VALID_REFERENCES',reference=meta));continue
  prior=bank.stages[t].get('training_responsibility_mean')
  if prior is None:stages.append(dict(stage=t,status='NO_TRAINING_PRIOR'));continue
  prior=prior.to(target.device).clamp_min(1e-30);prior=prior/prior.sum()
  learned=-(target*s['logits'].detach().log_softmax(-1)).sum(-1)
  blind=-(target*prior.log()).sum(-1);values.append((learned-blind).cpu().numpy())
  stages.append(dict(stage=t,status='VALID_TARGETS',reference=meta))
 result=dict(stages=stages,parameter_stage_propagations=b.calls,valid_stages=len(values))
 if values:
  difference=np.stack(values).mean(0);result['learned_minus_context_blind_crossentropy']=pair(difference,np.zeros(512))
 return result

def run(C,cfg,phase):
 from tools.journal_campaign import exclusive_device_lock
 lock=exclusive_device_lock(wait=True);torch.set_num_threads(2);guard();started=time.perf_counter()
 try:
  torch.manual_seed(98102601);model=DeepSeekPolicy().cuda().train()
  if phase=='smoke':
   opt=torch.optim.Adam(model.parameters(),lr=5e-5,betas=(.8,.998));b=make_benchmark(cfg['benchmark'])
   record=step(model,opt,b,cfg,10,GradientResponsibilities(),True)
   write(C/'readiness/GPU_smoke.json',dict(PASS=True,record=record,peak_allocated_bytes=torch.cuda.max_memory_allocated(),counts=model.counts(),weights_discarded=True,wall_seconds=time.perf_counter()-started))
   return
  if phase=='stress':
   models=[copy.deepcopy(model),model];banks=[GradientResponsibilities(),GradientResponsibilities()]
   opts=[torch.optim.Adam(m.parameters(),lr=5e-5,betas=(.8,.998)) for m in models]
   benchmarks=[make_benchmark(cfg['benchmark']) for m in models];allocated=[]
   for update in range(1,31):
    for index in [0,1]:step(models[index],opts[index],benchmarks[index],cfg,update,banks[index],index==1)
    allocated.append(torch.cuda.memory_allocated())
   write(C/'readiness/repeated_native_GPU.json',dict(PASS=True,updates_per_arm=30,wall_seconds=time.perf_counter()-started,parameter_stage_propagations=[b.calls for b in benchmarks],peak_allocated_bytes=torch.cuda.max_memory_allocated(),end_allocated_bytes=allocated,weights_discarded=True))
   return
  assert json.loads((C/'readiness/GPU_smoke.json').read_text())['PASS']
  assert int(__import__('re').search(r'(\d+) passed',(C/'readiness/CPU_tests.txt').read_text()).group(1))>=24
  models=[copy.deepcopy(model),model];banks=[GradientResponsibilities(),GradientResponsibilities()]
  opts=[torch.optim.Adam(m.parameters(),lr=5e-5,betas=(.8,.998)) for m in models]
  schedulers=[torch.optim.lr_scheduler.ExponentialLR(o,.98) for o in opts]
  benchmarks=[make_benchmark(cfg['benchmark']) for _ in models];records=[]
  for update in range(1,301):
   if time.perf_counter()-started>600:break
   rows=[]
   for index in [0,1]:
    rows.append(step(models[index],opts[index],benchmarks[index],cfg,update,banks[index],index==1))
    if (update-1)%1000==0:schedulers[index].step()
   records.append(rows)
   if update%10==0:
    write(C/'reports/pilot_progress.json',dict(update=update,rows=rows,wall_seconds=time.perf_counter()-started))
    print('PILOT',update,json.dumps(rows),flush=True)
  # Freeze all pilot settings/weights before independent development scoring.
  write(C/'metadata/pilot_checkpoint_freeze.json',dict(updates=len(records),official_seed1001_used=False,pilot_weights_initialize_main=False))
  probe=prediction_probe(models[1],banks[1],cfg)
  torch.save(dict(state_dict=models[1].state_dict(),optimizer=opts[1].state_dict(),scheduler=schedulers[1].state_dict(),references=banks[1].state_dict(),updates=len(records),pilot_only=True),C/'models/pilot_state.pt')
  validator=CommonValidation(cfg['benchmark'],30,98102603,512,262144,batch=16,chunk=16384,device='cuda:0')
  values=[];metadata=[]
  for label,m in zip(['matched_native_v1','fh_moe_v2'],models):
   v,meta=validator(m.eval());values.append(v);metadata.append(meta);np.save(C/'evaluations'/f'{label}_scores.npy',v)
  original=Path.home()/'Documents/objective_power_grid_v1_specialization_run/reference_artifacts/v1/selected.pt'
  v1=load_deepseek(original,'cuda');v,meta=validator(v1);np.save(C/'evaluations/frozen_v1_scores.npy',v)
  frozen_reference=dict(EIG=float(v.mean()),MC_SE=float(v.std(ddof=1)/len(v)**.5),metadata=meta)
  uniform=[]
  for rep in range(3):
   v,meta=validator(UniformSubset(models[1],98102610+rep).cuda());uniform.append(v)
  uniform=np.mean(uniform,0);np.save(C/'evaluations/uniform_scores.npy',uniform)
  masked=copy.deepcopy(models[1]);masked.ablate_routed=True;v_mask,meta=validator(masked);np.save(C/'evaluations/masked_scores.npy',v_mask)
  applications=[r[1] for r in records if r[1]['reference'] and r[1]['reference'].get('auxiliary_applied')]
  active_stages=sorted(set(r['stage'] for r in applications))
  prediction=probe.get('learned_minus_context_blind_crossentropy')
  useful_routing=pair(values[1],uniform);contribution=pair(values[1],v_mask);versus_native=pair(values[1],values[0])
  conditions=dict(correctness=True,nontrivial_responsibility_updates=len(applications)>=10 and len(active_stages)>=2,
   independent_history_predictability=bool(prediction and prediction['CI95'][1]<0),
   positive_routing_point_estimate=useful_routing['difference']>0,
   positive_routed_contribution_point_estimate=contribution['difference']>0,
   no_clear_matched_pilot_regression=versus_native['CI95'][1]>=0)
  result=dict(PASS=all(conditions.values()),conditions=conditions,completed_updates_per_arm=len(records),auxiliary_updates=len(applications),active_stages=active_stages,
   prediction_probe=probe,learned_minus_uniform=useful_routing,full_minus_routed_mask=contribution,
   v2_minus_matched_native=versus_native,
   results={name:dict(EIG=float(v.mean()),MC_SE=float(v.std(ddof=1)/len(v)**.5),metadata=meta) for name,v,meta in zip(['matched_native_v1','fh_moe_v2'],values,metadata)},
   frozen_v1=frozen_reference,wall_seconds=time.perf_counter()-started,training_parameter_stage_propagations=[b.calls for b in benchmarks],
   official_evaluation_run=False,main_training_authorized_only_if_all_conditions_pass=True,
   accounting='Bounded pilot computation is EXTRA; all pilot weights are discarded. Any future primary main run includes all auxiliary optimization within14400 seconds. Ubuntu main matches primary update count without a4hour cutoff.')
  write(C/'reports/pilot_records.json',records);write(C/'readiness/pilot_gate.json',result)
  write(C/'registry.json',dict(status='PILOT_PASSED_READY_FOR_PROMOTION' if result['PASS'] else 'PILOT_FAILED_NO_MAIN_TRAINING',result=result))
  print('PILOT_GATE',json.dumps(result),flush=True)
 finally:lock.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--fh-moe-v2',action='store_true');p.add_argument('--phase',choices=['smoke','stress','pilot'],required=True);p.add_argument('--campaign',required=True);a=p.parse_args()
 C=Path(a.campaign).resolve()
 for d in ['readiness','metadata','reports','logs','models','evaluations']:(C/d).mkdir(parents=True,exist_ok=True)
 cfg=yaml.safe_load(Path('reports/FH_MOE_V2_CONFIG.yaml').read_text());run(C,cfg,a.phase)
if __name__=='__main__':main()
