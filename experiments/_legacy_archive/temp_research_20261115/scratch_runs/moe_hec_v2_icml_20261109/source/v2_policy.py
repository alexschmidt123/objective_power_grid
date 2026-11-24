import copy,torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from hec_policy import HECActor,squash,upper,jacobian,mean_signal,COST
from sg_policy import frozen_buffers
from query_response import paired_discrimination,prospective_opportunity
class V2Actor(HECActor):
 def __init__(self,parent,stage_only=False,seed=202611099202):
  nn.Module.__init__(self)
  for name,module in parent._modules.items():self.add_module(name,module)
  self.dynamic=True;self.stage_only=stage_only;self.chunk=4096;self.anchor1=frozen_buffers(copy.deepcopy(parent.region));self.anchor3=frozen_buffers(copy.deepcopy(parent.sensitivity));device=parent.fusion[0].weight.device
  self.register_buffer('stage_context',torch.zeros(30,144,device=device));self.register_buffer('reference_logits',torch.zeros(30,20,device=device));self.register_buffer('stage_coord_context',torch.zeros(30,384,device=device));self.register_buffer('stage_response_context',torch.zeros(30,256,device=device));self.stage_logits=nn.Parameter(torch.zeros(30,20,device=device))
  with torch.random.fork_rng(devices=[0] if device.type=='cuda' else []):
   torch.manual_seed(seed);self.adapter2=nn.Sequential(nn.Linear(36,96),nn.SiLU(),nn.Linear(96,52)).to(device);self.adapter4=nn.Sequential(nn.Linear(20,96),nn.SiLU(),nn.Linear(96,52)).to(device);self.response_readouts=nn.ModuleList([nn.Linear(52,3),nn.Linear(52,6)]).to(device);self.response_gate=nn.Linear(256,4,bias=False).to(device);self.response_fusion=nn.Linear(256,128,bias=False).to(device)
  with torch.no_grad():
   self.response_gate.weight.zero_();self.response_fusion.weight.zero_()
   for k in [0,2]:self.response_fusion.weight[:,k*64:(k+1)*64].copy_(parent.fusion[0].weight[:,128+k*64:128+(k+1)*64])
 def compute(self,x,w,mu,sd,a0,command_override=None,routing_override=None,disable=None,ordinary=False,return_info=False,response_off=None,replace_expert=None,donor_slots=None):
  s,ctx=self.context(x,w,mu,sd,a0);stage=x[:,:30].argmax(-1);u=self.stage_context[stage] if self.stage_only else ctx;commands=(self.commander(u)+(self.stage_logits-self.reference_logits)[stage]).tanh()
  if command_override is not None:commands=command_override.expand_as(commands)
  c1,c2,c3,c4=commands.split([4,6,6,4],-1);nodes,pw=self.belief.nodes(w,mu,sd);mean=(w[...,None]*mu).sum(1);scale=((w[...,None]*(mu.square()+sd.square())).sum(1)-mean.square()).clamp_min(1e-5).sqrt();center=(mu-mean[:,None])/scale[:,None];attention=(w.clamp_min(1e-38).log()+(center*c1[:,None]).sum(-1)/2).softmax(-1);tokens=torch.cat([mu/3,sd.log(),w[...,None]],-1);z1=(attention[...,None]*self.region(tokens)).sum(1)
  direction=c2[:,:2]+x.new_tensor([1.,0.]);direction=direction/(direction.square().sum(-1,keepdim=True)+1e-6).sqrt();probes=(a0[:,None]+x.new_tensor([0.,1.,-1.])[None,:,None]*direction[:,None]).clamp(-4,4);pred=mean_signal(nodes,probes);tilt=(nodes-mean[:,None])/scale[:,None];hypattention=(pw.clamp_min(1e-38).log()+(tilt*c2[:,None,2:]).sum(-1)/2).softmax(-1);token2=torch.cat([pred.transpose(1,2)/5,hypattention[...,None]],-1);z2=(hypattention[...,None]*self.discriminate(token2)).sum(1)
  a3=(a0+c3[:,4:]).clamp(-4,4);J=jacobian(nodes,a3);F=4*torch.einsum('bn,bni,bnj->bij',pw,J,J);v=c3[:,:4]+x.new_tensor([1.,-1.,1.,-1.]);v=v/(v.square().sum(-1,keepdim=True)+1e-6).sqrt();z3=self.sensitivity(torch.cat([squash(upper(F)),v],-1))
  d4=c4[:,:2]+x.new_tensor([0.,1.]);d4=d4/(d4.square().sum(-1,keepdim=True)+1e-6).sqrt();radius=1+.5*c4[:,2:3];probe4=(a0[:,None]+x.new_tensor([0.,1.,-1.])[None,:,None]*radius[:,None]*d4[:,None]).clamp(-4,4);COST['frozen_opportunity_critic_rows']+=len(x)*3;q=self.opportunity(torch.cat([x[:,None].expand(-1,3,-1),probe4],-1)).squeeze(-1);qdiff=(q-q[:,:1])/10;pred4=mean_signal(nodes,probe4);pm=(pw[:,None]*pred4).sum(-1);pv=(pw[:,None]*(pred4-pm[:,:,None]).square()).sum(-1);remaining=(30-x[:,:30].argmax(-1)).to(x.dtype)[:,None]/30;features4=torch.cat([qdiff,pm/5,torch.log1p(pv),remaining,c4],-1);z4=self.sequential(features4);z=torch.stack([z1,z2,z3,z4],1)
  legacy=z;sem2,plus,minus,hp,hm=paired_discrimination(nodes,pw,pred,mean,scale,c2);in2=torch.cat([sem2,hp/3,hm/3,a0/4,c2,mean/3,scale.log()],-1);sem4=prospective_opportunity(nodes,pw,probe4,c4,stage);in4=torch.cat([sem4,qdiff,remaining,c4],-1);r2=torch.cat([sem2,self.adapter2(in2)],-1);r4=torch.cat([sem4,self.adapter4(in4)],-1);a1=(attention[...,None]*self.anchor1(tokens)).sum(1);a3=self.anchor3(torch.cat([squash(upper(F)),v],-1));first=torch.stack([a1,z2,a3,z4],1);responses=torch.stack([z1-a1,r2,z3-a3,r4],1)
  if response_off is not None:
   mask=x.new_ones(4);mask[response_off]=0;responses=responses*mask[None,:,None]
  coordctx=torch.cat([s,legacy.flatten(1)],-1);responsectx=responses.flatten(1)
  if self.stage_only:coordctx=self.stage_coord_context[stage];responsectx=self.stage_response_context[stage]
  raw=self.ordinary(s) if ordinary else self.coordinator(coordctx)+self.response_gate(responsectx);g=(10*torch.tanh(raw/10)).softmax(-1)
  if routing_override is not None:g=routing_override.expand_as(g)
  if disable is not None:
   mask=x.new_ones(4);mask[disable]=0;g=g*mask;g=g/g.sum(-1,keepdim=True)
  slots=torch.cat([first,responses],-1)
  if replace_expert is not None:
   assert donor_slots is not None;slots=slots.clone();slots[:,replace_expert]=donor_slots[:,replace_expert]
  first=slots[...,:64];responses=slots[...,64:];pre=self.fusion[0](torch.cat([s,(g[...,None]*first).flatten(1)],-1))+self.response_fusion((g[...,None]*responses).flatten(1));delta=self.fusion[2](self.fusion[1](pre)).tanh();action=a0+4*delta-a0*delta.abs()
  if not return_info:return action
  info=dict(history=s,legacy_features=legacy,features=first+responses,slots=slots,responses=responses,semantic2=sem2,semantic4=sem4,pair_plus=plus,pair_minus=minus,uncertainty_trace=scale.square().sum(-1),commands=commands,gates=g,attention=attention,hypattention=hypattention,F=F,direction=v,probe4=probe4,probes=probes,pred=pred,pw=pw,nodes=nodes,geometry_weights=w,geometry_mu=mu,geometry_sd=sd,geometry_attention=attention)
  return action,info
 def forward(self,x,command_override=None,routing_override=None,disable=None,ordinary=False,collect=False,response_off=None,replace_expert=None,donor_slots=None):
  shape=x.shape[:-1];flat=x.reshape(-1,117)
  with torch.no_grad():
   w,mu,sd=self.belief.params(flat);COST['frozen_SG_provisional_rows']+=len(flat);a0=torch.cat([self.base(y) for y in flat.split(8192)])
  outs=[];infos={}
  for i in range(0,len(flat),self.chunk):
   sl=slice(i,i+self.chunk);co=command_override;go=routing_override
   if co is not None and co.ndim>1 and len(co)==len(flat):co=co[sl]
   if go is not None and go.ndim>1 and len(go)==len(flat):go=go[sl]
   don=None if donor_slots is None else (donor_slots[sl] if len(donor_slots)==len(flat) else donor_slots)
   opts=dict(response_off=response_off,replace_expert=replace_expert,donor_slots=don,command_override=co,routing_override=go,disable=disable,ordinary=ordinary,return_info=collect)
   if torch.is_grad_enabled() and not collect:
    # Bind opts to this chunk for backward recomputation.
    def fn(xx,ww,mm,ss,aa,opts=opts):return self.compute(xx,ww,mm,ss,aa,**opts)
    out=checkpoint(fn,flat[sl],w[sl],mu[sl],sd[sl],a0[sl],use_reentrant=False)
   else:out=self.compute(flat[sl],w[sl],mu[sl],sd[sl],a0[sl],**opts)
   if collect:
    out,di=out
    for k,v in di.items():infos.setdefault(k,[]).append(v)
   outs.append(out)
  result=torch.cat(outs).reshape(*shape,2)
  if collect:return result,{k:torch.cat(v).reshape(*shape,*v[0].shape[1:]) for k,v in infos.items()}
  return result
