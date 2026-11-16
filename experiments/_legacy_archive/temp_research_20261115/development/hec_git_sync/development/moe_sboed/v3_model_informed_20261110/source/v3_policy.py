"""Four complete stochastic designs; differentiable model-informed allocation."""
import copy,math,torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from common import CFG,COST,mean_signal,frozen_buffers

def proposal_features(nodes,pw,var,actions):
 b,n=nodes.shape[:2];p=actions.shape[1];pred=mean_signal(nodes,actions);pm=(pw[:,None]*pred).sum(-1);pv=(pw[:,None]*(pred-pm[:,:,None]).square()).sum(-1)
 diff=nodes.reshape(b,1,n,2,2)-actions[:,:,None,None,:];d=diff.square().sum(-1)+1e-4;signal=.1+d.reciprocal().sum(-1);J=(-2*diff/d.square()[...,None]/signal[...,None,None]).flatten(-2);COST['v3_analytic_parameter_jacobians']+=b*p*n
 F=4*torch.einsum('bn,bpni,bpnj->bpij',pw,J,J);scale=var.sqrt();Fw=F*scale[:,None,:,None]*scale[:,None,None,:];logdet=torch.linalg.slogdet(torch.eye(4,device=F.device,dtype=torch.float64)+Fw.double()).logabsdet.to(F.dtype)
 return torch.cat([pm[...,None]/5,torch.log1p(4*pv)[...,None],torch.log1p(Fw.diagonal(dim1=-2,dim2=-1).clamp_min(0)),logdet[...,None]/10,actions/4],-1)

class V3Actor(nn.Module):
 def __init__(self,parent,stage_only=False):
  super().__init__();self.stage_only=stage_only;self.reference=frozen_buffers(copy.deepcopy(parent));self.encoder=parent.history
  with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
   torch.manual_seed(CFG['seed']);self.experts=nn.ModuleList([nn.Sequential(nn.Linear(145,64),nn.SiLU(),nn.Linear(64,2)) for _ in range(4)]);self.router=nn.Sequential(nn.Linear(202,64),nn.Tanh(),nn.Linear(64,4))
  for m in [e[-1] for e in self.experts]+[self.router[-1]]:nn.init.zeros_(m.weight);nn.init.zeros_(m.bias)
  self.chunk=CFG['actor_chunk'];self.sigma=CFG['latent_sigma']
 @torch.no_grad()
 def provider(self,x):
  a0,info=self.reference(x,collect=True);w,mu,sd=info['geometry_weights'],info['geometry_mu'],info['geometry_sd'];mean=(w[...,None]*mu).sum(1);var=((w[...,None]*(mu.square()+sd.square())).sum(1)-mean.square()).clamp_min(1e-6);summary=torch.cat([mean/3,torch.log1p(var)],-1);nodes,pw=info['nodes'],info['pw'];base_features=proposal_features(nodes,pw,var,a0[:,None])[:,0];base_mu=torch.atanh((a0/4).clamp(-CFG['base_clip'],CFG['base_clip']));return base_mu,summary,nodes,pw,var,base_features
 def parts(self,x,model_features_override=None):
  base,summary,nodes,pw,var,bf=self.provider(x);z=self.encoder(x);inp=torch.cat([z,summary,bf],-1);mu=base[:,None]+CFG['residual_latent_bound']*torch.stack([e(inp) for e in self.experts],1).tanh();proposals=4*mu.tanh();pf=proposal_features(nodes,pw,var,proposals)
  if model_features_override is not None:pf=model_features_override.expand_as(pf)
  context=torch.cat([z,summary,pf.flatten(1)],-1)
  if self.stage_only:context=torch.zeros_like(context)
  logits=self.router(torch.cat([x[:,:30],context],-1));raw=logits.softmax(-1);g=(1-CFG['router_exploration'])*raw+CFG['router_exploration']/4
  return g,mu,dict(proposals=proposals,model_features=pf,summary=summary,variance=var,base_features=bf)
 def draw(self,x,u,eps,fixed=None,gates=None,model_features_override=None):
  g,mu,info=self.parts(x,model_features_override)
  if gates is not None:g=gates.expand_as(g)
  k=(u[:,None]>g.cumsum(-1)).sum(-1).clamp(max=3) if fixed is None else torch.full((len(x),),fixed,device=x.device,dtype=torch.long)
  z=mu[torch.arange(len(x),device=x.device),k]+self.sigma*eps;action=4*z.tanh();return action,dict(**info,gates=g,selected=k,latent=z,means=mu)
 def forward(self,x):
  shape=x.shape[:-1];xx=x.reshape(-1,117);outs=[]
  for q in xx.split(self.chunk):outs.append(self.draw(q,torch.rand(len(q),device=q.device),torch.randn(len(q),2,device=q.device))[0])
  return torch.cat(outs).reshape(*shape,2)
 def log_prob(self,action,g,mu):
  assert bool((action.abs()<4).all());z=torch.atanh(action/4);lj=(math.log(4)+2*(math.log(2)-z-torch.nn.functional.softplus(-2*z))).sum(-1);lp=torch.distributions.Normal(mu,self.sigma).log_prob(z[:,None]).sum(-1)-lj[:,None];marginal=torch.logsumexp(g.log()+lp,-1);r=(g.log()+lp-marginal[:,None]).exp();return marginal,r
 def actor_block_loss(self,x,critic,eps,u):
  g,mu,_=self.parts(x);ac=4*(mu+self.sigma*eps).tanh();xx=x[:,None].expand(-1,4,-1)
  with torch.no_grad():q=critic(torch.cat([xx,ac.detach()],-1)).squeeze(-1);adv=q-(g.detach()*q).sum(-1,keepdim=True)
  k=(u[:,None]>g.detach().cumsum(-1)).sum(-1).clamp(max=3);selected=ac[torch.arange(len(x),device=x.device),k];v=critic(torch.cat([x,selected],-1)).squeeze(-1)
  return -v.sum()-(g*adv).sum()
 def actor_loss(self,x,critic,generator):
  flat=x.reshape(-1,117);total=flat.new_zeros(())
  for xx in flat.split(self.chunk):
   eps=torch.randn(len(xx),4,2,device=xx.device,generator=generator);u=torch.rand(len(xx),device=xx.device,generator=generator)
   total=total+checkpoint(self.actor_block_loss,xx,critic,eps,u,use_reentrant=False)
  return total/len(flat)
 @torch.no_grad()
 def target_value(self,x,critic,generator):
  result=[]
  for xx in x.reshape(-1,117).split(CFG['target_chunk']):
   g,mu,_=self.parts(xx);eps=torch.randn(len(xx),4,2,device=xx.device,generator=generator);ac=4*(mu+self.sigma*eps).tanh();q=critic(torch.cat([xx[:,None].expand(-1,4,-1),ac],-1)).squeeze(-1);result.append((g*q).sum(-1))
  return torch.cat(result).reshape(x.shape[:-1])
