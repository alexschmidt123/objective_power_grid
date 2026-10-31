"""T3-only exact deterministic reuse for nonadaptive Fixed and campaign utilities."""
import math,torch
from torch import nn
from src.research.engine import rollout,replay_likelihood,generator

class UniformChronological(nn.Module):
 kind='random_t3';outputs_design=False
 def forward(self,f,stage):return f.new_zeros(len(f),1)
 # The campaign explicitly calls rollout(method='random'), using the common
 # independent decision RNG and uniform conditional feasible-time mapping.


def fixed_shared_spce(b,policy,B,L,seed,*,return_details=False):
 """Same expected terminal finite-L sPCE; shared bank changes batch covariance.

+    Conditional on any outer trajectory, L contrasts remain iid prior and
+    independent of its truth/noise. Identical nonadaptive actions permit exact
+    reuse of each contrast's deterministic states across all outer cases.
 """
 r=rollout(b,policy,3,B,seed,0)
 bank=b.sample((L,),generator(seed+71000003))
 actions=[x[:1] for x in r['actions']]
 # Every policy row must represent the SAME nonadaptive schedule.
 for x in r['actions']:torch.testing.assert_close(x,x[:1].expand_as(x))
 states=b.initial((L,));means=[]
 for a in actions:
  mean,states=b.forward(bank,states,a.expand(L,-1));means.append(mean[:,0])
 logc=r['truth'].new_zeros(B,L);logt=r['truth'].new_zeros(B)
 # Truth means are replayed with the exact same discrete solver. This is
 # redundant but small; no interpolation, surrogate, or parameter bank reuse
 # across optimizer updates occurs.
 true=b.initial((B,))
 for a,y,m in zip(r['actions'],r['observations'],means):
  tm,true=b.forward(r['truth'],true,a)
  logt=logt+b.log_likelihood(y,tm)
  logc=logc+b.log_likelihood(y[:,None,:],m[None,:,None])
 value=logt-torch.logaddexp(logt,torch.logsumexp(logc,-1))+math.log(L+1)
 if return_details:return value,dict(result=r,bank=bank,log_contrast=logc,log_truth=logt,means=means)
 return value

from src.research.benchmarks import SIR
class T3SIR(SIR):
 def propagate(self,theta,state,action):
  if not theta.is_cuda or self.execution_backend!='fused_rk4':return super().propagate(theta,state,action)
  duration=action[...,0]-state[...,3]
  counts=torch.maximum(torch.full_like(duration,self.steps,dtype=torch.long),torch.ceil(duration.detach()*theta.detach().sum(-1)/.2).long())
  self.ode_forward_steps+=int(counts.sum());maximum=int(counts.max())
  if torch.any(duration<0) or maximum>20000:raise ValueError('SIR invalid time or excessively stiff prior draw')
  from src.research.sir_t3_fused import fused_rk4
  current=fused_rk4(theta,state[...,:3],duration,counts,self.population,maximum)
  if not torch.isfinite(current).all() or torch.any(current < -1e-7):raise FloatingPointError('SIR integration failed; particles are not dropped')
  return current[...,1:2],torch.cat((current,action),-1)
