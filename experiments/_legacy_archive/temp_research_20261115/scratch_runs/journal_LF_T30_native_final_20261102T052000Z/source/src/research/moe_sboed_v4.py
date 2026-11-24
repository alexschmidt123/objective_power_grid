"""Four role-supervised feature experts. Deployment: observed history only."""
import torch
from torch import nn
from src.research.deepseek_sboed import FeatureFFN
METHOD='moe_sboed_v4';STATUS='DEVELOPMENT_UNVALIDATED'
ROLES=('immediate_coarse','immediate_fine','future_coarse','future_fine')
class ObservedHistory(nn.Module):
 def __init__(self,horizon,action_dim,obs_dim,action_scale=1.,obs_scale=1.,ordered=False):
  super().__init__();self.horizon=horizon;self.a=action_dim;self.o=obs_dim;self.action_scale=action_scale;self.obs_scale=obs_scale;self.ordered=ordered;self.encoder=nn.Sequential(nn.Linear(action_dim+obs_dim+int(ordered),256),nn.ReLU(),nn.Linear(256,16));self.norm=nn.LayerNorm(16);self.time_vector=nn.Parameter(torch.zeros(16))
 def forward(self,f,stage):
  slots=f[:,1:].reshape(len(f),self.horizon,self.a+self.o+1)
  if stage:
   x=torch.cat((slots[:,:stage,:self.a]*self.action_scale,slots[:,:stage,self.a:self.a+self.o]*self.obs_scale),-1)
   if self.ordered:x=torch.cat((x,(torch.arange(1,stage+1,device=x.device,dtype=x.dtype)/self.horizon)[None,:,None].expand(len(x),-1,1)),-1)
   z=self.encoder(x).sum(1)
  else:z=f.new_zeros(len(f),16)
  return self.norm(z)+(stage/self.horizon)*self.time_vector

class RoleExpert(nn.Module):
 def __init__(self):super().__init__();self.features=FeatureFFN(16,16);self.prediction=nn.Linear(16,1);nn.init.normal_(self.prediction.weight,std=.01);nn.init.zeros_(self.prediction.bias)
 def forward(self,z):return self.features(z)

class RoleMoE(nn.Module):
 kind=METHOD;outputs_design=False
 def __init__(self,horizon=30,action_dim=2,obs_dim=1,action_scale=4.,obs_scale=3.,routing='learned',dense_width=None,ordered_history=False):
  super().__init__();self.horizon=horizon;self.routing=routing;self.dense=dense_width is not None;self.history=ObservedHistory(horizon,action_dim,obs_dim,action_scale,obs_scale,ordered_history);self.experts=nn.ModuleList([RoleExpert() for _ in range(4)]) if not self.dense else nn.ModuleList();self.router=nn.Linear(16,4,bias=False) if not self.dense else None
  if self.router is not None:nn.init.normal_(self.router.weight,std=.01)
  self.dense_features=nn.Sequential(nn.Linear(16,dense_width),nn.SiLU(),nn.Linear(dense_width,64)) if self.dense else None
  self.design_head=nn.Sequential(nn.LayerNorm(80),nn.Linear(80,32),nn.SiLU(),nn.Linear(32,action_dim));nn.init.normal_(self.design_head[-1].weight,std=.01);nn.init.zeros_(self.design_head[-1].bias)
  self.register_buffer('role_mean',torch.zeros(4));self.register_buffer('role_scale',torch.ones(4));self._routing_rng=None;self.last_dispatch=None;self.mask_expert=None;self.mask_all=False;self.expert_permutation=None;self.routing_seed=86404991;self.double()
 def __getstate__(self):
  s=super().__getstate__();s['_routing_rng']=None;s['last_dispatch']=None;return s
 def forward(self,f,stage):
  z=self.history(f,stage)
  if self.dense:return self.design_head(torch.cat((z,self.dense_features(z)),-1))
  scores=self.router(z).softmax(-1)
  if self.routing=='uniform_top2':
   if self._routing_rng is None:self._routing_rng=torch.Generator(device=f.device).manual_seed(self.routing_seed)
   ids=torch.rand(len(z),4,device=z.device,generator=self._routing_rng).topk(2,-1).indices
  else:ids=scores.topk(2,-1).indices
  # Identical selected-softmax gate semantics for learned and uniform SUBSET
  # routing. Uniform selection does not mean dense averaging or changed mass.
  w=scores.gather(1,ids);w=w/w.sum(-1,keepdim=True);slots=z.new_zeros(len(z),4,16)
  if not self.mask_all:
   for e in ids.unique().tolist():
    if self.mask_expert==e:continue
    rows,col=(ids==e).nonzero(as_tuple=True);source=e if self.expert_permutation is None else self.expert_permutation[e];slots[rows,e]=self.experts[source](z[rows])*w[rows,col,None]
  self.last_dispatch=dict(indices=ids.detach(),weights=w.detach(),scores=scores.detach())
  return self.design_head(torch.cat((z,slots.flatten(1)),-1))
 def role_predictions(self,f,stage,detach_history=True):
  z=self.history(f.detach() if detach_history else f,stage)
  if detach_history:z=z.detach()
  pred=torch.cat([e.prediction(e(z)) for e in self.experts],-1);pred=pred*self.role_scale+self.role_mean
  # Empty future has exactly zero information, independent of parameters.
  if stage==self.horizon-1:pred=torch.cat((pred[:,:2],torch.zeros_like(pred[:,2:])),1)
  return pred
 def role_losses(self,features,targets):
  pred=torch.stack([self.role_predictions(f,t) for t,f in enumerate(features)],1);return ((pred-targets.detach())/self.role_scale).square().mean((0,1)),pred
 def counts(self):
  total=sum(p.numel() for p in self.parameters());aux=sum(p.numel() for e in self.experts for p in e.prediction.parameters());one=sum(p.numel() for p in self.experts[0].features.parameters()) if len(self.experts) else 0
  return dict(total_parameters=total,deployed_active_parameters=total-aux-2*one,routed_experts=len(self.experts),top_k=2 if len(self.experts) else 0,role_heads=4 if len(self.experts) else 0,always_active_FFN_experts=0,offline_auxiliary_experts_per_context=4 if len(self.experts) else 0)

def dense_control(target,**kw):
 probe=RoleMoE(dense_width=16,**kw);base=probe.counts()['total_parameters'];width=max(1,round(16+(target-base)/81));return RoleMoE(dense_width=width,**kw)
