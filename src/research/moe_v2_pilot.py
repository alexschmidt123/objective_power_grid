"""Predeclared matched progressive location pilot; no official evaluation."""
import argparse,copy,fcntl,hashlib,json,math,os,subprocess,time,traceback
from pathlib import Path
import numpy as np,torch,yaml
from src.research.moe_v2_decision import matched_models,DecisionMoE
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout
from src.research.journal_validation import CommonValidation
R=Path(__file__).resolve().parents[2];C=R/'experiments/10262026/10262026_moe_sboed_v2_decision_pilot'
ORDER=['v1','v2','dense','shared']
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');q.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def diff(x,y):
 d=np.asarray(x)-np.asarray(y);se=float(d.std(ddof=1)/len(d)**.5);return dict(mean=float(d.mean()),MC_SE=se,CI95=[float(d.mean()-1.96*se),float(d.mean()+1.96*se)])
def prepare():
 assert not C.exists(),'Preserve previous scientific pilot'
 for d in ['models','reports','evaluations','logs','metadata']:(C/d).mkdir(parents=True)
 cfg=yaml.safe_load((R/'configs/deepseek_sboed_location.yaml').read_text())
 plan=dict(method='MoE-sBOED-v2 candidate; unregistered scientific status',training_seed=201,initialization='Native v1 common modules copied to candidate/shared/dense; independent routed readouts; all train from scratch',batch_size=cfg['batch_size'],contrasts=cfg['contrasts'],horizon=30,optimizer=dict(name='Adam',lr=cfg['learning_rate'],betas=cfg['adam_betas'],lr_decay=.98,lr_decay_after_updates=[1,1001,2001,3001],clip=5),balance_factor=.001,matched_methods=ORDER,curve_updates=[128,512,2048,4096],maximum_updates=4096,training_GPU_seconds=7200,diagnostic_GPU_seconds=1800,maximum_total_GPU_seconds=9000,curve_seed=97402631,heldout_seed=97402632,confirmation_seed=97402633,context_fit_seed=97402634,curve_systems=128,curve_contrasts=65536,final_systems=512,final_contrasts=262144,context_fit_systems=512,intervention_stages=[1,10,20,29],contexts='Stage1 single negative-control bin; stages10/20/29 fitting-split median of observable mean previous log signal; no theta/expert/utility labels',intervention='Replace weakest routed slot by candidate expert unless already in top2; keep original slot weights, top3cardinality/shared/head; propagate full remaining horizon',multiplicity='Holm family-wise.05 for stage/context/expert differences and pairwise context interactions; independent sign and effect replication',minimum_interaction_effect=.02,minimum_meaningful_updates=2048,pilot_noninferiority_margin=.05,routing_promising_margin=.02,readiness='At least2048 matched updates, finite/stable gradients; heldout and confirmation v2-v1 lower95CI>-.05; routed-mask improvement lower95CI>0 and point>=.02; learned-uniform lower95CI>-.02 and point>=0; competence interactions exploratory Holm and independent replication; dense comparison disclosed. Otherwise harmful/inconclusive, never automatic full campaign',official_seed1001_used=False,full_LabPC_training_authorized_by_this_pilot=False,config_sha256=sha(R/'configs/deepseek_sboed_location.yaml'),source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip())
 write(C/'plan.json',plan)
 files=[p for p in R.glob('src/research/*.py')]+[R/'tools/tests/test_moe_v2_decision.py',R/'configs/deepseek_sboed_location.yaml']
 write(C/'source_manifest.json',dict(source_revision=plan['source_revision'],sha256={str(p.relative_to(R)):sha(p) for p in files}))
 write(C/'status.json',dict(state='PREDECLARED_READY_FOR_GPU_GATE',matched_updates=0));print(C)
def lock_device():
 path=Path.home()/'.cache/objective_power_grid/device_0.lock';path.parent.mkdir(parents=True,exist_ok=True)
 while True:
  lock=path.open('a+')
  try:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   if not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():return lock
  except BlockingIOError:pass
  lock.close();write(C/'status.json',dict(state='WAITING_UBUNTUPC_DEVICE_LOCK',pid=os.getpid()));time.sleep(15)
def score(model,seed,n,L):
 cfg=yaml.safe_load((R/'configs/deepseek_sboed_location.yaml').read_text())['benchmark'];v=CommonValidation(cfg,30,seed,n,L,batch=16,chunk=16384,device='cuda:0');values,meta=v(model);return values,meta

def run():
 plan=json.loads((C/'plan.json').read_text());manifest=json.loads((C/'source_manifest.json').read_text())
 for rel,digest in manifest['sha256'].items():assert sha(R/rel)==digest,rel
 with (C/'coordinator.lock').open('a+') as coordinator:
  fcntl.flock(coordinator,fcntl.LOCK_EX|fcntl.LOCK_NB);lock=lock_device();allocated=time.perf_counter();train_seconds=0.;diagnostic_seconds=0.;completed=0;bound=1.;records=[];failure=None
  try:
   # GPU full-horizon correctness test before any scientific optimizer step.
   with (C/'logs/GPU_correctness.log').open('x') as log:
    gate=subprocess.run([str(Path.home()/'miniconda3/envs/mocu_optimized/bin/python'),'-m','pytest','-q','tools/tests/test_moe_v2_decision.py'],cwd=R,env=dict(os.environ,OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT)
   assert gate.returncode==0,'GPU correctness gate failed'
   models=matched_models(plan['training_seed']);models={k:p.to('cuda:0').train() for k,p in models.items()}
   opts={k:torch.optim.Adam(p.parameters(),lr=plan['optimizer']['lr'],betas=tuple(plan['optimizer']['betas'])) for k,p in models.items()};schedulers={k:torch.optim.lr_scheduler.ExponentialLR(opts[k],.98) for k in ORDER};raw=yaml.safe_load((R/'configs/deepseek_sboed_location.yaml').read_text())['benchmark'];benches={k:make_benchmark(raw) for k in ORDER};curves=[]
   for update in range(1,plan['maximum_updates']+1):
    if train_seconds+bound*1.5>plan['training_GPU_seconds']:break
    if time.perf_counter()-allocated+bound*1.5>plan['maximum_total_GPU_seconds']:break
    start=time.perf_counter();round_data=[]
    # Rotate ordering to reduce systematic thermal/order differences.
    order=ORDER[(update-1)%4:]+ORDER[:(update-1)%4]
    for name in order:
     p=models[name];p.train();opt=opts[name];opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
     with torch.device('cuda:0'):
      r=rollout(benches[name],p,30,plan['batch_size'],plan['training_seed']+update*104729,plan['contrasts']);information=r['info'][:,-1].mean();loss=-information+p.balance_loss()
      if not torch.isfinite(loss):raise FloatingPointError('Nonfinite terminal sPCE')
      loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(p.parameters(),5,error_if_nonfinite=True));opt.step();torch.cuda.synchronize()
     if (update-1)%1000==0:schedulers[name].step()
     if update==1 or update%100==0:
      expert_norm=[float(torch.stack([q.grad.detach().norm() for q in e.parameters() if q.grad is not None]).norm()) if any(q.grad is not None for q in e.parameters()) else 0 for e in p.routed]
      round_data.append(dict(method=name,update=update,training_sPCE=float(information.detach()),loss=float(loss.detach()),preclip_gradient_norm=norm,clipped=norm>5,expert_gradient_norms=expert_norm,router_gradient_norm=float(p.router.weight.grad.norm()) if p.router is not None and p.router.weight.grad is not None else None,routing=p.routing_statistics(),peak_GPU_bytes=torch.cuda.max_memory_allocated(),learning_rate=opt.param_groups[0]['lr']))
     del r,loss,information
    torch.cuda.synchronize();elapsed=time.perf_counter()-start;train_seconds+=elapsed;bound=max(bound,elapsed);completed=update
    if round_data:
     records.extend(round_data);write(C/'reports/training_progress.json',dict(matched_updates=completed,training_seconds=train_seconds,records=records,primary_episodes_per_method=completed*plan['batch_size'],contrast_paths_per_method=completed*plan['batch_size']*plan['contrasts']));print('MATCHED_UPDATE',update,'TRAIN_SECONDS',train_seconds,flush=True)
    if update in plan['curve_updates']:
     start=time.perf_counter();values={}
     for name,p in models.items():
      v,meta=score(p.eval(),plan['curve_seed'],plan['curve_systems'],plan['curve_contrasts']);values[name]=v;np.save(C/f'evaluations/curve_{name}_{update}.npy',v);curves.append(dict(method=name,update=update,EIG=float(v.mean()),MC_SE=float(v.std(ddof=1)/len(v)**.5),metadata=meta));p.train()
     diagnostic_seconds+=time.perf_counter()-start;write(C/'reports/development_curves.json',curves)
     for name,p in models.items():torch.save(dict(method='moe_sboed_v2_candidate' if name=='v2' else name,update=update,state_dict={k:v.detach().cpu() for k,v in p.state_dict().items()},optimizer=opts[name].state_dict(),scheduler=schedulers[name].state_dict(),plan=plan),C/f'models/{name}_update{update}.pt')
     if update>=2048:
      d=diff(values['v2'],values['v1'])
      if d['CI95'][1]<-.1:failure='Predeclared clear harmful learning curve';break
   if completed<1:raise RuntimeError('No matched update completed')
   for name,p in models.items():torch.save(dict(method=name,update=completed,state_dict={k:v.detach().cpu() for k,v in p.state_dict().items()},plan=plan),C/f'models/{name}_final.pt')
   write(C/'metadata/final_checkpoint_freeze.json',dict(matched_updates=completed,selection='last common completed update; never best EIG',sha256={name:sha(C/f'models/{name}_final.pt') for name in ORDER}))
   start=time.perf_counter();results={};summaries=[]
   for split,seed in [('heldout',plan['heldout_seed']),('confirmation',plan['confirmation_seed'])]:
    values={}
    for name,p in models.items():
     v,meta=score(p.eval(),seed,plan['final_systems'],plan['final_contrasts']);values[name]=v;np.save(C/f'evaluations/{split}_{name}.npy',v)
    for control in ['masked','uniform']:
     p=copy.deepcopy(models['v2']).eval();p.ablate_routed=control=='masked';p.routing_mode='uniform' if control=='uniform' else 'learned';v,_=score(p,seed,plan['final_systems'],plan['final_contrasts']);values[control]=v;np.save(C/f'evaluations/{split}_{control}.npy',v)
    summary=dict(split=split,EIG={k:dict(mean=float(v.mean()),MC_SE=float(v.std(ddof=1)/len(v)**.5)) for k,v in values.items()},v2_minus_v1=diff(values['v2'],values['v1']),v2_minus_dense=diff(values['v2'],values['dense']),v2_minus_shared=diff(values['v2'],values['shared']),full_minus_mask=diff(values['v2'],values['masked']),learned_minus_uniform=diff(values['v2'],values['uniform']));summaries.append(summary)
   diagnostic_seconds+=time.perf_counter()-start
   ready=completed>=2048 and all(x['v2_minus_v1']['CI95'][0]>-.05 and x['full_minus_mask']['CI95'][0]>0 and x['full_minus_mask']['mean']>=.02 and x['learned_minus_uniform']['CI95'][0]>-.02 and x['learned_minus_uniform']['mean']>=0 for x in summaries)
   write(C/'reports/final_development.json',dict(status='PROMISING_NEEDS_CONTEXT_COMPETENCE' if ready else 'NOT_READY_OR_INCONCLUSIVE',matched_updates=completed,results=summaries,full_training_ready=False,context_interaction_status='PENDING_CONTROLLED_SUBSTITUTIONS_WITHIN_REMAINING_DIAGNOSTIC_BUDGET',no_official_EIG=True))
  except Exception as e:
   failure=repr(e);write(C/'reports/failure.json',dict(error=failure,traceback=traceback.format_exc(),matched_updates=completed));raise
  finally:
   write(C/'status.json',dict(state='FAILED_PRESERVED' if failure else 'DEVELOPMENT_COMPLETE',matched_updates=completed,training_GPU_seconds=train_seconds,diagnostic_GPU_seconds=diagnostic_seconds,total_GPU_locked_seconds=time.perf_counter()-allocated,failure=failure,source_revision=plan['source_revision'],no_LabPC_interruption=True));lock.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args();globals()[a.action]()
