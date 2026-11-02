"""V1-compatible policy; only training-time top-k selection explores."""
import math,torch
from src.research.deepseek_sboed import DeepSeekPolicy
class V6Policy(DeepSeekPolicy):
 def __init__(self,*a,routing_seed=7100101,**kw):
  super().__init__(*a,**kw);self.exploration_scale=0.;self.routing_seed=routing_seed;self.route_generator=None
  self.diagnostic_mode=None;self.swap_stage=None;self.swap_expert=None
 def __getstate__(self):
  s=super().__getstate__();s['route_generator']=None;return s
 def forward(self,features,stage):
  if stage==0 or (self.training and self._last_stage is not None and stage<=self._last_stage):self._gate_cache=[]
  self._last_stage=stage;z=self.encode_context(features,stage);u=z
  if self.shared is not None and not self.ablate_shared:u=u+self.shared(z)
  dispatch=None
  if self.router is not None and not self.ablate_routed:
   logits=self.router(z);scores=logits.softmax(-1)
   if self.training and self.exploration_scale>0:
    if self.route_generator is None:self.route_generator=torch.Generator(device=z.device).manual_seed(self.routing_seed)
    uniform=torch.rand(logits.shape,generator=self.route_generator,device=z.device,dtype=z.dtype).clamp(1e-12,1-1e-12)
    noisy=logits+self.exploration_scale*(-torch.log(-torch.log(uniform)))
    indices=noisy.topk(self.top_k,dim=-1).indices
   else:indices=scores.topk(self.top_k,dim=-1).indices
   if self.diagnostic_mode in ['fixed','stage_only']:
    start=0 if self.diagnostic_mode=='fixed' else stage%7
    indices=torch.tensor([(start+i)%7 for i in range(3)],device=z.device).expand(len(z),-1).clone()
   if self.swap_stage==stage and self.swap_expert is not None:
    indices=indices.clone();rows=(indices!=self.swap_expert).all(-1);indices[rows,0]=self.swap_expert
   weights=scores.gather(-1,indices);routed=z.new_zeros(z.shape)
   for e in indices.unique().tolist():
    rows,slots=(indices==e).nonzero(as_tuple=True)
    routed.index_add_(0,rows,self.routed[e](z[rows])*weights[rows,slots,None])
   u=u+routed;dispatch=dict(indices=indices.detach(),scores=scores.detach(),selected_weights=weights.detach())
   if self.training:self._gate_cache.append((scores,indices.detach()))
  self.last_dispatch=dispatch;self.last_features=z.detach();return self.emitter(u)
def exploration(update,total=18000):
 fraction=(update-1)/total
 if fraction<=.2:return .10
 if fraction>=.8:return 0.
 return .05*(1+math.cos(math.pi*(fraction-.2)/.6))
