"""Bounded T3 evaluator, checkpoint and component-cost integration checks."""
import copy,json,time,math
from pathlib import Path
import numpy as np,torch,cloudpickle

def integration(C,cfg):
 from src.research.sir_t3_campaign import lock_gpu,write
 from src.research.sir_t3_common import T3SIR
 from src.research.sir_t3_evaluation import evaluate,cases
 from src.research.sir_evaluation import contrast_scores
 from src.research.engine import rollout
 from src.research.policies import Policy
 from src.research.sir_dad import SIRSumDAD
 from src.research.sir_t3_native_vsoed import T3InputUnits
 from src.research.sir_upstream_vsoed import UpstreamSIRActor
 from src.research.vendor.vsoed.utils import Net
 locks=lock_gpu();torch.set_num_threads(4);out=C/'profiling/integration';out.mkdir(exist_ok=False);records=[]
 with torch.device('cuda:0'):
  b=T3SIR(cfg['benchmark']);actor=Net([7,256,256,256,1],torch.nn.ReLU,torch.tensor([[0.,100.]]),'actor').float();actor.register_forward_pre_hook(T3InputUnits(3));v=UpstreamSIRActor(actor,3,b)
  for name,p in [('fixed',Policy(10,1,3,'fixed')),('dad',SIRSumDAD(3)),('vsoed',v),('random',None)]:
   x,meta=evaluate(cfg['benchmark'],p,2026102837,2,[8,16],out,name,method=name)
   truth,noise,bank,_=cases(cfg['benchmark'],2026102837,2,16,out/'cases_seed2026102837_N2_L16.npz');q=copy.deepcopy(p).cuda().eval() if p is not None else None
   r=rollout(b,q,3,2,2026102837,0,truth=truth.cuda(),noise=noise.cuda(),method='random' if name=='random' else None);ref,_=contrast_scores(b,r,bank,chunk=8,device='cuda:0');np.testing.assert_allclose(x[16],ref[:,-1].numpy(),rtol=1e-9,atol=1e-8)
   if p is not None:
    path=out/(name+'_roundtrip.pt');torch.save(copy.deepcopy(p).cpu(),path);q=torch.load(path,map_location='cuda:0',weights_only=False).eval();f=r['features'][1];torch.testing.assert_close(p(f,1),q(f,1))
   records.append(dict(test='terminal_scoring_and_checkpoint_'+name,status='PASS',metadata=meta))
  from src.research.sir_t3_native_rl import NativeSIRRLPolicy,algorithm,Replay
  p=NativeSIRRLPolicy(3,b);a=algorithm(p,3);replay=Replay(12,10);replay.data.zero_();state=dict(policy=p,algorithm=a,replay=replay);path=out/'native_rl_runtime.pkl'
  with path.open('wb') as f:cloudpickle.dump(state,f)
  with path.open('rb') as f:q=cloudpickle.load(f)
  assert q['replay'].capacity==12;torch.testing.assert_close(next(q['policy'].parameters()),next(p.parameters()));records.append(dict(test='native_rl_full_runtime_serialization',status='PASS'))
  th=b.sample((4096,),torch.Generator(device='cuda').manual_seed(2026102835)).requires_grad_();action=torch.full((4096,1),30.,dtype=torch.float64,requires_grad=True);t=time.monotonic();mean,state=b.forward(th,b.initial((4096,)),action);torch.cuda.synchronize();forward=time.monotonic()-t;t=time.monotonic();torch.autograd.grad(mean.sum(),(th,action));torch.cuda.synchronize();backward=time.monotonic()-t
  y=torch.randn(128,1,1,dtype=torch.float64);t=time.monotonic()
  for _ in range(100):ll=b.log_likelihood(y,mean.detach()[None])
  torch.cuda.synchronize();likelihood=(time.monotonic()-t)/100;records.append(dict(test='component_costs',ODE_batch=4096,ODE_forward_seconds=forward,ODE_all_parameter_and_action_sensitivity_seconds=backward,Gaussian_likelihood_128x4096_seconds=likelihood))
  # Native optimizer step costs, including the five posterior and critic updates.
  from src.research.sir_t3_native_vsoed import train_upstream_vsoed
  costs={'actor':[],'critic':[],'posterior':[]}
  class Observer:
   def set_runtime(self,rt):
    opts=[('actor',rt['actor_optimizer']),('critic',rt['critic_optimizer'])]+[('posterior',o) for group in rt['posterior'].poi_post_optimizers.values() for o in group]
    self.handles=[]
    for name,opt in opts:
     stamp=[0.]
     def pre(*args,key=name,stamp=stamp):torch.cuda.synchronize();stamp[0]=time.monotonic()
     def post(*args,key=name,stamp=stamp):torch.cuda.synchronize();costs[key].append(time.monotonic()-stamp[0])
     self.handles.extend([opt.register_step_pre_hook(pre),opt.register_step_post_hook(post)])
   def check(self,p,episodes,update,*args,**kw):return update>=3
   def finish(self,p):
    for handle in self.handles:handle.remove()
    return dict(selected_checkpoint=dict(update=3))
  directory=out/'vsoed';directory.mkdir();p,data=train_upstream_vsoed(T3SIR(cfg['benchmark']),3,{},101,directory,dict(cfg['methods']['vsoed'],updates=3),Observer());records.append(dict(test='native_vsoed_optimizer_costs',optimizer_step_seconds=costs,native_updates=data['native_optimizer_updates']))
 write(C/'manifests/integration_tests.json',dict(status='PASS',records=records))
 print(json.dumps(records,indent=2),flush=True)
