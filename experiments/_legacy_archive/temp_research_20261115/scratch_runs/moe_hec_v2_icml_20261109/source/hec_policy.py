"""Command-conditioned model queries and four heterogeneous feature experts."""
import copy,math,torch
from collections import defaultdict
COST=defaultdict(int)
from torch import nn
from torch.utils.checkpoint import checkpoint
from sg_policy import frozen_buffers

def squash(x):return x.sign()*torch.log1p(x.abs())
def mean_signal(theta,actions):
 # theta[B,N,4], actions[B,P,2] -> [B,P,N]
 COST['exact_signal_evaluations']+=len(theta)*theta.shape[1]*actions.shape[1]
 d=(theta[:,None].reshape(len(theta),1,-1,2,2)-actions[:,:,None,None]).square().sum(-1)
 return (.1+(d+1e-4).reciprocal().sum(-1)).log()
def jacobian(theta,action):
 COST['analytic_parameter_jacobians']+=len(theta)*theta.shape[1]
 diff=theta.reshape(len(theta),-1,2,2)-action[:,None,None];d=diff.square().sum(-1)+1e-4;signal=.1+d.reciprocal().sum(-1)
 return (-2*diff/d.square()[...,None]/signal[...,None,None]).flatten(-2)
def upper(m):return m[...,torch.triu_indices(4,4,device=m.device)[0],torch.triu_indices(4,4,device=m.device)[1]]
def advance(x,action,y):
 t=x[:,:30].argmax(-1);assert bool((t<29).all());out=x.clone();out[:,:30]=0;out.scatter_(1,(t+1)[:,None],1);idx=torch.arange(len(x),device=x.device);ds=out[:,30:88].reshape(-1,29,2);ds[idx,t]=action;obs=out[:,88:117];obs[idx,t]=y;return out

class Belief(nn.Module):
 def __init__(self,post):
  super().__init__();self.nets=nn.ModuleList([frozen_buffers(copy.deepcopy(post.poi_post_nets[t][0])) for t in range(30)])
  for net in self.nets:
   for name in ['mu_bounds','max_sigmas']:
    v=getattr(net,name);delattr(net,name);net.register_buffer(name,v.detach().clone())
  self.register_buffer('points',torch.cat([2*torch.eye(4),-2*torch.eye(4)]))
 @torch.no_grad()
 def params(self,x):
  t=x[:,:30].argmax(-1);COST['frozen_belief_rows']+=int((t>0).sum());b=len(x);w=x.new_zeros(b,8);w[:,0]=1;mu=x.new_zeros(b,8,4);sd=x.new_ones(b,8,4)
  for stage in t.unique().tolist():
   if stage==0:continue
   idx=(t==stage).nonzero().flatten();net=self.nets[stage-1];inp=torch.cat([x[idx,30:30+2*stage],x[idx,88:88+stage]],-1);f=net.feature_net(inp);w[idx]=net.weight_net(f).exp();mu[idx]=net.mu_net(f).sigmoid().reshape(-1,8,4)*(net.mu_bounds[:,1]-net.mu_bounds[:,0])+net.mu_bounds[:,0];sd[idx]=net.sigma_net(f).sigmoid().reshape(-1,8,4)*net.max_sigmas+1e-5
  return w,mu,sd
 def nodes(self,w,mu,sd):return (mu[:,:,None]+sd[:,:,None]*self.points[None,None]).flatten(1,2),(w[:,:,None].expand(-1,8,8)/8).flatten(1,2)
 def log_prob(self,x,theta):
  w,mu,sd=self.params(x);lp=torch.distributions.Normal(mu,sd).log_prob(theta[:,None]).sum(-1);core=torch.logsumexp(w.clamp_min(1e-38).log()+lp,-1);prior=-.5*theta.square().sum(-1)-2*math.log(2*math.pi);return torch.logaddexp(core,prior+math.log(1e-27))-math.log1p(1e-27)

class HECActor(nn.Module):
 def __init__(self,sg,post,critic,dynamic=True):
  super().__init__();self.dynamic=dynamic;self.base=frozen_buffers(copy.deepcopy(sg));self.belief=Belief(post);self.opportunity=frozen_buffers(copy.deepcopy(critic));self.history=copy.deepcopy(sg.encoder)
  self.commander=nn.Sequential(nn.Linear(144,64),nn.Tanh(),nn.Linear(64,20));self.region=nn.Sequential(nn.Linear(9,64),nn.SiLU(),nn.Linear(64,64));self.discriminate=nn.Sequential(nn.Linear(4,32),nn.SiLU(),nn.Linear(32,64));self.sensitivity=nn.Sequential(nn.Linear(14,96),nn.SiLU(),nn.Linear(96,64));self.sequential=nn.Sequential(nn.Linear(14,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,64));self.readouts=nn.ModuleList([nn.Linear(64,k) for k in [14,9,9,3]]);self.coordinator=nn.Sequential(nn.Linear(384,64),nn.Tanh(),nn.Linear(64,4));self.ordinary=nn.Linear(128,4);self.fusion=nn.Sequential(nn.Linear(384,128),nn.ReLU(),nn.Linear(128,2));self.chunk=4096
  with torch.no_grad():
   for layer in [self.commander[-1],self.coordinator[-1],self.ordinary,self.fusion[-1]]:layer.weight.zero_();layer.bias.zero_()
 def context(self,x,w,mu,sd,a0):
  mean=(w[...,None]*mu).sum(1);second=(w[...,None]*(mu.square()+sd.square())).sum(1);var=(second-mean.square()).clamp_min(1e-6);cov=torch.einsum('bm,bmi,bmj->bij',w,mu-mean[:,None],mu-mean[:,None]);pairs=torch.triu_indices(4,4,1,device=x.device);cross=cov[:,pairs[0],pairs[1]]/(var[:,pairs[0]]*var[:,pairs[1]]).sqrt();summary=torch.cat([mean,torch.log1p(var),cross],-1);s=self.history(x);return s,torch.cat([s,a0/4,summary],-1)
 def compute(self,x,w,mu,sd,a0,command_override=None,routing_override=None,disable=None,ordinary=False,return_info=False):
  s,ctx=self.context(x,w,mu,sd,a0);commands=self.commander(ctx).tanh() if self.dynamic else ctx.new_zeros(len(x),20)
  if command_override is not None:commands=command_override.expand_as(commands)
  c1,c2,c3,c4=commands.split([4,6,6,4],-1);nodes,pw=self.belief.nodes(w,mu,sd);mean=(w[...,None]*mu).sum(1);scale=((w[...,None]*(mu.square()+sd.square())).sum(1)-mean.square()).clamp_min(1e-5).sqrt();center=(mu-mean[:,None])/scale[:,None];attention=(w.clamp_min(1e-38).log()+(center*c1[:,None]).sum(-1)/2).softmax(-1);tokens=torch.cat([mu/3,sd.log(),w[...,None]],-1);z1=(attention[...,None]*self.region(tokens)).sum(1)
  direction=c2[:,:2]+x.new_tensor([1.,0.]);direction=direction/(direction.square().sum(-1,keepdim=True)+1e-6).sqrt();probes=(a0[:,None]+x.new_tensor([0.,1.,-1.])[None,:,None]*direction[:,None]).clamp(-4,4);pred=mean_signal(nodes,probes);tilt=(nodes-mean[:,None])/scale[:,None];hypattention=(pw.clamp_min(1e-38).log()+(tilt*c2[:,None,2:]).sum(-1)/2).softmax(-1);token2=torch.cat([pred.transpose(1,2)/5,hypattention[...,None]],-1);z2=(hypattention[...,None]*self.discriminate(token2)).sum(1)
  a3=(a0+c3[:,4:]).clamp(-4,4);J=jacobian(nodes,a3);F=4*torch.einsum('bn,bni,bnj->bij',pw,J,J);v=c3[:,:4]+x.new_tensor([1.,-1.,1.,-1.]);v=v/(v.square().sum(-1,keepdim=True)+1e-6).sqrt();z3=self.sensitivity(torch.cat([squash(upper(F)),v],-1))
  d4=c4[:,:2]+x.new_tensor([0.,1.]);d4=d4/(d4.square().sum(-1,keepdim=True)+1e-6).sqrt();radius=1+.5*c4[:,2:3];probe4=(a0[:,None]+x.new_tensor([0.,1.,-1.])[None,:,None]*radius[:,None]*d4[:,None]).clamp(-4,4);COST['frozen_opportunity_critic_rows']+=len(x)*3;q=self.opportunity(torch.cat([x[:,None].expand(-1,3,-1),probe4],-1)).squeeze(-1);qdiff=(q-q[:,:1])/10;pred4=mean_signal(nodes,probe4);pm=(pw[:,None]*pred4).sum(-1);pv=(pw[:,None]*(pred4-pm[:,:,None]).square()).sum(-1);remaining=(30-x[:,:30].argmax(-1)).to(x.dtype)[:,None]/30;features4=torch.cat([qdiff,pm/5,torch.log1p(pv),remaining,c4],-1);z4=self.sequential(features4);z=torch.stack([z1,z2,z3,z4],1)
  raw=self.ordinary(s) if (not self.dynamic or ordinary) else self.coordinator(torch.cat([s,z.flatten(1)],-1));g=(10*torch.tanh(raw/10)).softmax(-1)
  if routing_override is not None:g=routing_override.expand_as(g)
  if disable is not None:
   mask=x.new_ones(4);mask[disable]=0;g=g*mask;g=g/g.sum(-1,keepdim=True)
  delta=self.fusion(torch.cat([s,(g[...,None]*z).flatten(1)],-1)).tanh();action=a0+4*delta-a0*delta.abs()
  if not return_info:return action
  info=dict(commands=commands,gates=g,features=z,attention=attention,hypattention=hypattention,F=F,direction=v,probe4=probe4,probes=probes,pred=pred,pw=pw,nodes=nodes,geometry_weights=w,geometry_mu=mu,geometry_sd=sd,geometry_attention=attention)
  return action,info
 def forward(self,x,command_override=None,routing_override=None,disable=None,ordinary=False,collect=False):
  shape=x.shape[:-1];flat=x.reshape(-1,117)
  with torch.no_grad():
   w,mu,sd=self.belief.params(flat);COST['frozen_SG_provisional_rows']+=len(flat);a0=torch.cat([self.base(y) for y in flat.split(8192)])
  outs=[];infos={}
  for i in range(0,len(flat),self.chunk):
   sl=slice(i,i+self.chunk);co=command_override;go=routing_override
   if co is not None and co.ndim>1 and len(co)==len(flat):co=co[sl]
   if go is not None and go.ndim>1 and len(go)==len(flat):go=go[sl]
   opts=dict(command_override=co,routing_override=go,disable=disable,ordinary=ordinary,return_info=collect)
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
 def experts(self):return [self.region,self.discriminate,self.sensitivity,self.sequential]
 def set_phase(self,phase):
  for p in self.parameters():p.requires_grad_(True)
  for p in self.readouts.parameters():p.requires_grad_(False)
  if phase=='B':
   for module in [self.history,*self.experts()]:module.requires_grad_(False)
  elif phase!='C':raise ValueError(phase)
