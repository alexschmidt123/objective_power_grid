"""Isolated improvement candidates; existing SIR source is unchanged."""
import torch
from src.research.sir_t3_common import T3SIR
class CountSortedSIR(T3SIR):
 def propagate(self,theta,state,action):
  if not theta.is_cuda or self.execution_backend!='fused_rk4':return super().propagate(theta,state,action)
  duration=action[...,0]-state[...,3];counts=torch.maximum(torch.full_like(duration,self.steps,dtype=torch.long),torch.ceil(duration.detach()*theta.detach().sum(-1)/.2).long());self.ode_forward_steps+=int(counts.sum());maximum=int(counts.max())
  if torch.any(duration<0) or maximum>20000:raise ValueError('Invalid SIR interval/rate')
  from src.research.sir_t3_fused import fused_rk4
  shape=state.shape[:-1];order=torch.argsort(counts.flatten(),stable=True);inverse=torch.empty_like(order);inverse[order]=torch.arange(len(order),device=order.device);th=theta.reshape(-1,2)[order];st=state[...,:3].reshape(-1,3)[order];du=duration.flatten()[order];co=counts.flatten()[order];sorted_state=fused_rk4(th,st,du,co,self.population,maximum);current=sorted_state[inverse].reshape(*shape,3)
  if not torch.isfinite(current).all() or torch.any(current < -1e-7):raise FloatingPointError('SIR integration failed')
  return current[...,1:2],torch.cat((current,action),-1)

def schedule_policy(b,times):
 from src.research.policies import Policy
 p=Policy(10,1,3,'fixed');prev=0.;lat=[]
 for i,t in enumerate(times):
  lo=prev+b.gap;hi=b.end-(2-i)*b.gap;q=(float(t)-lo)/(hi-lo)
  if not 0<q<1:raise ValueError('Schedule outside chronological interior')
  lat.append(torch.logit(torch.tensor(q,dtype=torch.float64)));prev=float(t)
 with torch.no_grad():p.sequence.copy_(torch.stack(lat).reshape_as(p.sequence))
 return p

def physical_times(b,p):
 from src.research.engine import rollout
 with torch.no_grad():r=rollout(b,p,3,1,2026103030,0)
 return [float(x[0,0]) for x in r['actions']]
