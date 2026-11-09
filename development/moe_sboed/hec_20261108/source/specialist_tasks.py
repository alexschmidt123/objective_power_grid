"""Offline approximate-belief tasks;never read replay latent labels."""
import torch,math
from hec_policy import mean_signal,advance,upper
@torch.no_grad()
def targets(actor,x,info,generator):
 w=info['geometry_attention'];mu=info['geometry_mu'];sd=info['geometry_sd'];mean=(w[...,None]*mu).sum(1);cov=torch.einsum('bm,bmi,bmj->bij',w,mu,mu)-mean[:,:,None]*mean[:,None,:]+torch.diag_embed((w[...,None]*sd.square()).sum(1));var=cov.diagonal(dim1=-2,dim2=-1).clamp_min(1e-6);cdf=(w[...,None]*(.5*(1+torch.erf((mean[:,None]-mu)/sd/math.sqrt(2))))).sum(1);pairs=torch.triu_indices(4,4,1,device=x.device);corr=cov[:,pairs[0],pairs[1]]/(var[:,pairs[0]]*var[:,pairs[1]]).sqrt();geo=torch.cat([cdf,torch.log1p(var),corr],-1)
 pred=info['pred'];pw=info['hypattention'];pm=(pw[:,None]*pred).sum(-1);cent=pred-pm[:,:,None];variance=(pw[:,None]*cent.square()).sum(-1);third=(pw[:,None]*cent.pow(3)).sum(-1)/(1+variance).pow(1.5);dis=torch.cat([pm/5,torch.log1p(variance/.25),third],-1)
 F=info['F'];eig,vec=torch.linalg.eigh(F.double());weak=vec[:,:,0].square().float();v=info['direction'];directional=torch.einsum('bi,bij,bj->b',v,F,v);sens=torch.cat([torch.log1p(eig.clamp_min(0)).float(),weak,torch.log1p(directional.clamp_min(0))[:,None]],-1)
 # Two genuine conditional predictive simulations under frozen approximate q.
 b=len(x);R=2;ww=info['geometry_weights'];ix=torch.multinomial(ww,R,replacement=True,generator=generator);row=torch.arange(b,device=x.device)[:,None];theta=mu[row,ix]+sd[row,ix]*torch.randn(b,R,4,device=x.device,generator=generator);theta=theta[:,None].expand(-1,3,-1,-1).reshape(-1,4);actions=info['probe4'][:,:,None].expand(-1,-1,R,-1).reshape(-1,2);xx=x[:,None,None].expand(-1,3,R,-1).reshape(-1,117);m=mean_signal(theta[:,None],actions[:,None])[:,0,0];eps=torch.randn(b,R,device=x.device,generator=generator)[:,None].expand(-1,3,-1).reshape(-1);y=m+.5*eps;t=xx[:,:30].argmax(-1);before=actor.belief.log_prob(xx,theta);after=before.clone();continuation=torch.zeros_like(before)
 for stage in t.unique().tolist():
  mask=t==stage;ids=mask.nonzero().flatten()
  if stage<29:
   nx=advance(xx[ids],actions[ids],y[ids]);after[ids]=actor.belief.log_prob(nx,theta[ids]);na=actor.base(nx);continuation[ids]=actor.opportunity(torch.cat([nx,na],-1)).squeeze(-1)
  else:
   net=actor.belief.nets[29];inp=torch.cat([xx[ids,30:88],actions[ids],xx[ids,88:117],y[ids,None]],-1);after[ids]=net(inp,theta[ids])
 value=(after-before+continuation).reshape(b,3,R).mean(-1);seq=(value-value[:,:1])/10
 return [geo,dis,sens,seq],dict(predictive_observations=b*3*R,predictive_continuation_policy_calls=int((t<29).sum()))

def supervised_loss(actor,x,commands,target,normalization=None):
 _,info=actor(x,command_override=commands,collect=True);losses=[]
 for k,head in enumerate(actor.readouts):
  prediction=head(info['features'][:,k]);d=prediction-target[k]
  if normalization is not None:d=d/normalization[k]
  losses.append(d.square().mean())
 return sum(losses),losses
