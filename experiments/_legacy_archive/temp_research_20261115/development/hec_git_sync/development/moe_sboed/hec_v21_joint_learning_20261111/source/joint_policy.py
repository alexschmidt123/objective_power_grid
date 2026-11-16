"""Route using experts' actual decision contributions,not standalone actions."""
import copy,torch
from torch import nn
from common import CFG,COST,jacobian
from v2_policy import V2Actor
from hec_policy import squash

def contributions(model,s,slots,g,a0):
 first,responses=slots[...,:64],slots[...,64:];W=model.fusion[0].weight[:,128:].reshape(128,4,64);V=model.response_fusion.weight.reshape(128,4,64);c=torch.einsum('bkc,okc->bko',first,W)+torch.einsum('bkc,okc->bko',responses,V)
 pre=model.fusion[0](torch.cat([s,(g[...,None]*first).flatten(1)],-1))+model.response_fusion((g[...,None]*responses).flatten(1));d=model.fusion[2](model.fusion[1](pre)).tanh();J=model.fusion[2].weight[None]*(pre>0)[:,None].to(pre.dtype);J=J*(1-d.square())[:,:,None]*(4-a0*d.sign())[:,:,None];v=torch.einsum('bdo,bko->bkd',J,c);vlogit=g[...,None]*(v-(g[...,None]*v).sum(1,keepdim=True));return c,pre,d,v,vlogit

def decision_features(model,x,info):
 s=info['history'];g0=info['gates'];slots=info['slots'];a0=info['probes'][:,0];c,pre,d,v,vlogit=contributions(model,s,slots,g0,a0);design=a0+4*d-a0*d.abs();nodes,pw=info['nodes'],info['pw'];Jtheta=jacobian(nodes,design);Jxi=-Jtheta.reshape(len(x),-1,2,2).sum(-2);directional=torch.einsum('bnd,bkd->bkn',Jxi,vlogit);response=4*(pw[:,None]*directional.square()).sum(-1)
 w,mu,sd=info['geometry_weights'],info['geometry_mu'],info['geometry_sd'];mean=(w[...,None]*mu).sum(1);var=((w[...,None]*(mu.square()+sd.square())).sum(1)-mean.square()).clamp_min(1e-5);context=torch.cat([s,mean/3,torch.log1p(var),info['semantic2'],info['semantic4']],-1)
 scalar=torch.stack([torch.log1p(c.square().mean(-1)),torch.log1p(vlogit.square().sum(-1)),torch.log1p(response)],-1);return context,c,vlogit,scalar,dict(old_gates=g0,decision_contributions=c,logit_action_tangents=vlogit,likelihood_directional_sensitivity=response,reference_pre=pre,reference_delta=d)

class JointHEC(V2Actor):
 def __init__(self,parent):
  nn.Module.__init__(self)
  for name,module in parent._modules.items():self.add_module(name,module)
  for name,buf in parent._buffers.items():self.register_buffer(name,buf)
  for name,param in parent._parameters.items():self.register_parameter(name,param)
  self.dynamic=parent.dynamic;self.stage_only=parent.stage_only;self.chunk=parent.chunk;assert self.dynamic and not self.stage_only;device=parent.fusion[0].weight.device
  with torch.random.fork_rng(devices=[0] if device.type=='cuda' else []):
   torch.manual_seed(CFG['seed']);self.contribution_projection=nn.Linear(128,16,bias=False).to(device);self.relevance=nn.Sequential(nn.Linear(185,64),nn.SiLU(),nn.Linear(64,1)).to(device)
  with torch.no_grad():self.relevance[-1].weight.zero_();self.relevance[-1].bias.zero_()
 def compute(self,x,w,mu,sd,a0,command_override=None,routing_override=None,disable=None,ordinary=False,return_info=False,response_off=None,replace_expert=None,donor_slots=None):
  assert not ordinary,'Ordinary alternative gate is not part of the locked candidate'
  _,info=super().compute(x,w,mu,sd,a0,command_override=command_override,routing_override=None,disable=None,ordinary=False,return_info=True,response_off=response_off,replace_expert=replace_expert,donor_slots=donor_slots)
  context,c,tangent,scalars,geometry=decision_features(self,x,info);key=self.contribution_projection(squash(c));identity=torch.eye(4,device=x.device,dtype=x.dtype)[None].expand(len(x),-1,-1);tokens=torch.cat([context[:,None].expand(-1,4,-1),key,tangent/4,scalars,identity],-1);score=self.relevance(tokens).squeeze(-1)
  oldraw=self.coordinator(torch.cat([info['history'],info['legacy_features'].flatten(1)],-1))+self.response_gate(info['responses'].flatten(1));oldlogits=10*torch.tanh(oldraw/10);g=(oldlogits+score).softmax(-1)
  if routing_override is not None:g=routing_override.expand_as(g)
  if disable is not None:
   mask=x.new_ones(4);mask[disable]=0;g=g*mask;g=g/g.sum(-1,keepdim=True)
  slots=info['slots'];pre=self.fusion[0](torch.cat([info['history'],(g[...,None]*slots[...,:64]).flatten(1)],-1))+self.response_fusion((g[...,None]*slots[...,64:]).flatten(1));delta=self.fusion[2](self.fusion[1](pre)).tanh();action=a0+4*delta-a0*delta.abs()
  if not return_info:return action
  info.update(geometry);info.update(gates=g,relevance_scores=score,relevance_tokens=tokens);return action,info

def install(a,candidate=True):
 if not candidate:return dict(method='ordinary_HECv2_continuation',all_original_parameters_and_Adam_preserved=True)
 old_params={id(p) for p in a.actor_net.parameters()};moments={id(p):v['exp_avg'].clone() for p,v in a.actor_optimizer.state.items() if 'exp_avg' in v};a.actor_net=JointHEC(a.actor_net);a.actor_target_net=JointHEC(a.actor_target_net);new=[p for p in a.actor_net.parameters() if id(p) not in old_params];a.actor_optimizer.param_groups[0]['params'].extend(new)
 assert old_params<={id(p) for p in a.actor_net.parameters()};assert all(torch.equal(v['exp_avg'],moments[id(p)]) for p,v in a.actor_optimizer.state.items() if 'exp_avg' in v);assert {id(p) for p in a.actor_net.parameters()}=={id(p) for group in a.actor_optimizer.param_groups for p in group['params']};assert all(p.shape==q.shape for p,q in zip(a.actor_net.parameters(),a.actor_target_net.parameters()));return dict(method='HECv21_decision_contribution_routing',all_original_parameters_and_Adam_preserved=True,new_parameter_elements=sum(p.numel() for p in new),new_moments_fresh=True)
