"""Preserved competency channels plus EIG-adaptive deltas and learned retention."""
import sys,copy
sys.dont_write_bytecode=True
import torch
from torch import nn
def frozen_buffers(module):
 for child in module.modules():
  for name,p in list(child._parameters.items()):
   if p is not None:
    del child._parameters[name];child.register_buffer(name,p.detach().clone())
 return module.eval()

class SGActor(nn.Module):
 def __init__(self,parent,anchor_parent,anchored):
  super().__init__();self.anchored=anchored;self.encoder=parent.encoder;self.experts=parent.experts;self.router=parent.router;self.task_heads=parent.task_heads.requires_grad_(False);self.anchor_encoder=frozen_buffers(copy.deepcopy(anchor_parent.encoder));self.anchor_experts=frozen_buffers(copy.deepcopy(anchor_parent.experts));old=parent.decision[0];self.decision=parent.decision
  with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []):
   self.retention=nn.Linear(128,4,device=old.weight.device,dtype=old.weight.dtype)
  with torch.no_grad():self.retention.weight.zero_();self.retention.bias.zero_()
  with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []):self.delta_head=nn.Linear(256,128,bias=False,device=old.weight.device,dtype=old.weight.dtype)
  with torch.no_grad():self.delta_head.weight.copy_(2*old.weight)
 def components(self,x):
  s=self.encoder(x);live=torch.stack([e(s) for e in self.experts],-2)
  with torch.no_grad():a=self.anchor_encoder(x);anchor=torch.stack([e(a) for e in self.anchor_experts],-2)
  g=(10*torch.tanh(self.router(s)/10)).softmax(-1);r=self.retention(s).sigmoid();return live,anchor,g,r
 def parts(self,x,detach_encoder=False):
  s=self.encoder(x);s=s.detach() if detach_encoder else s;z=torch.stack([e(s) for e in self.experts],-2);g=(10*torch.tanh(self.router(s)/10)).softmax(-1);return s,z,g
 def forward(self,x,disable=None,stage_gates=None,uniform=False,route_override=None,replace_expert=None,donor_slots=None,mechanism=None):
  live,anchor,g,r=self.components(x)
  if route_override is not None:g,r=route_override
  if stage_gates is not None:g,r=stage_gates
  if uniform:g=torch.full_like(g,.25)
  if mechanism=='retention_constant':r=torch.full_like(r,.5)
  if mechanism=='adaptive_off':r=torch.zeros_like(r)
  if self.anchored:
   first=live if mechanism=='anchors_live' else anchor;delta=live-anchor
  else:first=live;delta=torch.zeros_like(live)
  if disable is not None:
   mask=torch.ones(4,device=g.device,dtype=g.dtype);mask[disable]=0;g=g*mask;g=g/g.sum(-1,keepdim=True)
  slots=torch.cat([first,r.unsqueeze(-1)*delta],-1)
  if replace_expert is not None:
   assert donor_slots is not None;slots=slots.clone();slots[...,replace_expert,:]=donor_slots[...,replace_expert,:]
  first=slots[...,:64];residual=slots[...,64:];pre=self.decision[0]((g.unsqueeze(-1)*first).flatten(-2))+self.delta_head((g.unsqueeze(-1)*residual).flatten(-2));return -4+8*self.decision[2](self.decision[1](pre)).sigmoid()

def migrate(a,anchored):
 old=a.actor_net;target=a.actor_target_net;anchor=copy.deepcopy(old);before_params={id(p) for g in a.actor_optimizer.param_groups for p in g['params']};a.actor_net=SGActor(old,anchor,anchored);a.actor_target_net=SGActor(target,anchor,anchored);a.actor_optimizer.param_groups[0]['params'].extend(list(a.actor_net.delta_head.parameters())+list(a.actor_net.retention.parameters()))
 assert all(p not in a.actor_optimizer.state for p in a.actor_net.delta_head.parameters())
 assert {id(p) for g in a.actor_optimizer.param_groups for p in g['params']}=={id(p) for name,p in a.actor_net.named_parameters() if not name.startswith(('anchor_encoder.','anchor_experts.'))}
 a._soft_tasks.enabled=False;a.discount=1.;return dict(original_head_and_Adam_objects_preserved=True,new_residual_head_shape=list(a.actor_net.delta_head.weight.shape),residual_weight_initialization='2*parent_head;retention.5',new_parameter_Adam_fresh=True,auxiliary_gradients_disabled=True)
