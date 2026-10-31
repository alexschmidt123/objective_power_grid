"""Predeclared diagnostic only: frozen policies, full native horizon, no optimizer.
Contrast averaging is conditional on a COMPLETE joint-law trajectory, not a
fixed-prefix posterior calculation. Normalization is a separate nonlinear label.
"""
import argparse,copy,hashlib,json,math,time,subprocess,traceback,signal
from pathlib import Path
import torch
from src.research.deepseek_sboed import DeepSeekPolicy
from src.research.deepseek_sboed_experiment import write
from src.research.fh_moe_v3_discovery import RAW,guard
from src.research.fh_moe_v3_context import capture_contexts,history_covariates,normalized
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout
STAGES=[1,10,20,29]

def generator(device,seed):return torch.Generator(device=device).manual_seed(seed)
def joint_draws(n,seed,device):
 g=generator(device,seed)
 return torch.randn(n,4,generator=g,device=device,dtype=torch.double),torch.randn(n,30,1,generator=g,device=device,dtype=torch.double)
def bank_draws(n,L,seed,device):return torch.randn(n,L,4,generator=generator(device,seed),device=device,dtype=torch.double)

def extract(model,truth,noise,particles):
 """Every sample is an unconditional joint-law truth/noise draw unless a probe
 explicitly shares truth for variance decomposition. All paths are recomputed;
 useful history gradients are never detached. Contrasts do not affect actions.
 """
 b=make_benchmark(RAW);device=truth.device;B=len(truth)
 with torch.device(device),capture_contexts(model,STAGES) as states:
  r=rollout(b,model,30,B,0,particles.shape[1],truth=truth,noise=noise,prior_particles=particles)
  loss=-r['info'][:,-1].mean()
  nodes=[states[t]['u'] for t in STAGES]+[r['latents'][t] for t in STAGES]
  grads=torch.autograd.grad(loss,nodes)
  with torch.no_grad():x=torch.stack([history_covariates(model,r['features'][t],t) for t in STAGES],1)
  assert all(torch.isfinite(g).all() for g in grads)
  assert torch.stack(r['actions']).abs().max()<=4
  raw=torch.stack(grads[:4],1).detach()*B;action=torch.stack(grads[4:],1).detach()*B
  return dict(x=x.detach().cpu(),raw=raw.cpu(),action=action.cpu(),loss=float(loss.detach()),calls=b.calls)

def load_reference(path,device):
 m=DeepSeekPolicy();s=torch.load(path,map_location='cpu',weights_only=False);m.load_state_dict(s['state_dict']);m.ablate_routed=True
 return m.to(device).eval()

class Budget:
 def __init__(self,C,limit):self.C=C;self.start=time.perf_counter();self.limit=limit;self.records=[];self.longest=0.
 def call(self,name,fn):
  spent=time.perf_counter()-self.start
  if spent+max(60.,2*self.longest)>=self.limit:raise TimeoutError('Predeclared GPU budget safety stop')
  st=time.perf_counter();v=fn();torch.cuda.synchronize();dt=time.perf_counter()-st;self.longest=max(self.longest,dt)
  self.records.append(dict(task=name,seconds=dt,forward_parameter_stages=v.get('calls',0)))
  if len(self.records)%24==0:self.save('RUNNING')
  return v
 def save(self,status):
  write(self.C/'GPU_ledger.json',dict(status=status,GPU_locked_wall_seconds=time.perf_counter()-self.start,limit=self.limit,operations=len(self.records),records=self.records,forward_parameter_stages=sum(v['forward_parameter_stages'] for v in self.records),optimizer_updates=0,official_evaluation=False,peak_allocated_bytes=torch.cuda.max_memory_allocated(),hardware=torch.cuda.get_device_name(0),wall_not_kernel_busy_time=True))

def run(C,v3):
 plan=json.loads((C/'plan.json').read_text());assert plan['maximum_new_GPU_locked_seconds']==1800
 guard();paths=['src/research/gradient_variance_audit.py','src/research/gradient_variance_analysis.py','tools/tests/test_gradient_variance_audit.py']
 hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths};write(C/'source_freeze.json',hashes)
 assert not (C/'GPU_ledger.json').exists(),'Never overwrite GPU work'
 from tools.journal_campaign import exclusive_device_lock
 lock=exclusive_device_lock(wait=False)
 def hard_stop(signum,frame):raise TimeoutError('Hard1800-second GPU budget reached')
 signal.signal(signal.SIGALRM,hard_stop);signal.alarm(1800)
 budget=Budget(C,plan['soft_stop_seconds']);torch.set_num_threads(2);device=torch.device('cuda:0');status='FAILED'
 try:
  model=load_reference(v3/'shared_reference.pt',device)
  for split,seed in enumerate(plan['primary_split_seeds']):
   chunks=[]
   for k,start in enumerate(range(0,8000,500)):
    truth,noise=joint_draws(500,seed+k*100003,device);raws=[];partners=[];antis=[];x=None
    for rep in range(4):
     seedc=plan['contrast_seed_base']+split*1000003+k*1009+rep*13
     c=bank_draws(500,2000,seedc,device);d=bank_draws(500,2000,seedc+70000001,device)
     out=budget.call('primary_C',lambda:extract(model,truth,noise,c));raws.append(out['raw'])
     if x is None:x=out['x']
     else:assert torch.equal(x,out['x']),'Contrast bank incorrectly changed observed histories'
     out=budget.call('primary_D',lambda:extract(model,truth,noise,d));partners.append(out['raw']);assert torch.equal(x,out['x'])
     out=budget.call('primary_negative_C',lambda:extract(model,truth,noise,-c));antis.append(out['raw']);assert torch.equal(x,out['x'])
    chunks.append(dict(x=x,C=torch.stack(raws,1),D=torch.stack(partners,1),negative_C=torch.stack(antis,1)))
   data={key:torch.cat([d[key] for d in chunks]) for key in chunks[0]};torch.save(data,C/f'new_split{split}.pt');print('PRIMARY_SPLIT_COMPLETE',split,flush=True)
  # Larger contrast sensitivity: nested banks, identical complete paths, finite-L
  # targets differ. This does not correct the original objective unbiasedly.
  truth,noise=joint_draws(512,98263001,device);large=[];small=[]
  for rep in range(4):
   for start in range(0,512,128):
    particles=bank_draws(128,8000,98263100+rep*1009+start,device)
    a=budget.call('L2000_probe',lambda:extract(model,truth[start:start+128],noise[start:start+128],particles[:,:2000]));small.append(a['raw'])
    a=budget.call('L8000_probe',lambda:extract(model,truth[start:start+128],noise[start:start+128],particles));large.append(a['raw'])
  torch.save(dict(small=torch.stack(small).reshape(4,512,4,16),large=torch.stack(large).reshape(4,512,4,16)),C/'contrast_sensitivity.pt')
  # True parameters held fixed, observations independently resimulated. This
  # is NOT sampling a conditional law given an observed history.
  truth,_=joint_draws(256,98264001,device);obs=[]
  for obsrep in range(4):
   _,noise=joint_draws(256,98264100+obsrep,device);gs=[]
   for rep in range(4):
    particles=bank_draws(256,2000,98264200+rep+obsrep*100003,device)
    a=budget.call('observation_contrast_cross',lambda:extract(model,truth,noise,particles));gs.append(a['raw'])
   obs.append(torch.stack(gs,1))
  torch.save(torch.stack(obs,1),C/'observation_contrast.pt')
  # No new training: initial seeded reference and existing trained native masked.
  with torch.random.fork_rng(devices=[]):torch.manual_seed(98102711);initial=DeepSeekPolicy();initial.ablate_routed=True
  refs={'shared600':model,'initial_shared0':initial.to(device).eval(),'native600_masked_shared':load_reference(v3/'native_control.pt',device)}
  out={}
  for name,m in refs.items():
   parts=[]
   for k in range(4):
    truth,noise=joint_draws(256,98265001+k*100003,device);particles=bank_draws(256,2000,98265100+k*100003,device)
    parts.append(budget.call('reference_'+name,lambda:extract(m,truth,noise,particles)))
   out[name]={key:torch.cat([a[key] for a in parts]) for key in ['x','raw','action']}
  torch.save(out,C/'reference_sensitivity.pt')
  assert hashes=={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
  status='COMPLETE_NO_TRAINING'
 except Exception as e:
  write(C/'GPU_failure.json',dict(error=str(e),traceback=traceback.format_exc()));raise
 finally:signal.alarm(0);budget.save(status);lock.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--v3-campaign',required=True);a=p.parse_args();run(Path(a.campaign),Path(a.v3_campaign))
if __name__=='__main__':main()
