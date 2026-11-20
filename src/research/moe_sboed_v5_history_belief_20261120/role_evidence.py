"""Deterministic sBOED history–belief evidence;location-finding tensor adapter only."""
import math,torch
from frozen_feature_policy import observed_prefix
from hec_policy import mean_signal,jacobian

def summary(w,mu,sd):
 mean=(w[...,None]*mu).sum(1);center=mu-mean[:,None]
 cov=torch.einsum('bm,bmi,bmj->bij',w,center,center)+torch.diag_embed((w[...,None]*sd.square()).sum(1))
 scale=cov.diagonal(dim1=-2,dim2=-1).clamp_min(1e-5).sqrt();ij=torch.triu_indices(4,4,1,device=mu.device)
 corr=cov[:,ij[0],ij[1]]/(scale[:,ij[0]]*scale[:,ij[1]])
 return torch.cat([mean/3,scale.log(),corr],-1),mean,scale,cov

@torch.no_grad()
def evidence(c,belief):
 x=c['x'];B=len(x);stage=c['stage'];previous=(stage-1).clamp_min(0)
 xp=x.clone();xp[:,:30]=0;xp.scatter_(1,previous[:,None],1);xp=observed_prefix(xp)
 wp,mp,sp=belief.params(xp);before,meanp,scalep,covp=summary(wp,mp,sp)
 current,mean,scale,cov=summary(c['w'],c['mu'],c['sd'])
 designs=x[:,30:88].reshape(B,29,2);observations=x[:,88:117];idx=(stage-1).clamp(0,28)
 last=designs[torch.arange(B,device=x.device),idx];y=observations[torch.arange(B,device=x.device),idx]
 noise=c['prior'][:,8].clamp_min(1e-6);valid=stage>0
 # E1:current uncertainty geometry,conditioned jointly with existing ordered history.
 e1=current.tanh()
 # E2:the LAST completed measurement's predictive innovation under the BEFORE-measurement belief.
 nodes,pw=belief.nodes(wp,mp,sp);pred=mean_signal(nodes,last[:,None]).squeeze(1);pm=(pw*pred).sum(-1);pv=(pw*(pred-pm[:,None]).square()).sum(-1)
 comp=mean_signal(mp,last[:,None]).squeeze(1);cm=(wp*comp).sum(-1);between=(wp*(comp-cm[:,None]).square()).sum(-1)
 correlation=(pw[...,None]*(nodes-meanp[:,None])*(pred-pm[:,None])[...,None]).sum(1)/(scalep*(pv+1e-8).sqrt()[:,None])
 entropy=-(wp*wp.clamp_min(1e-30).log()).sum(-1)/math.log(8)
 standardized=(y-pm)/(pv+noise.square()).sqrt()
 mass_above=(pw*(pred>y[:,None])).sum(-1)
 absolute=(pw*(pred-y[:,None]).abs()).sum(-1)/(pv+noise.square()).sqrt()
 e2=torch.cat([standardized[:,None],torch.log1p(pv/noise.square())[:,None],torch.log1p(between/noise.square())[:,None],entropy[:,None],
  correlation.clamp(-1,1),((mean-meanp)/scalep).clamp(-5,5),mass_above[:,None],absolute[:,None]],-1).tanh()*valid[:,None]
 # E3:information supplied by OBSERVED designs,evaluated at CURRENT posterior component means.
 N=c['mu'].shape[1];expanded=c['mu'][:,None].expand(B,29,N,4).reshape(B*29,N,4)
 J=jacobian(expanded,designs.reshape(B*29,2)).reshape(B,29,N,4)
 mask=(torch.arange(29,device=x.device)[None,:]<stage[:,None]).to(x.dtype)
 F=torch.einsum('bt,bn,btni,btnj->bij',mask,c['w'],J,J)/noise[:,None,None].square()
 Cd=cov.double();eye=torch.eye(4,device=x.device,dtype=torch.float64).expand(B,4,4);L=torch.linalg.cholesky(Cd+1e-5*eye)
 A=L.transpose(-1,-2)@F.double()@L;unresolved=torch.linalg.solve(eye+A,eye)
 upper=torch.triu_indices(4,4,0,device=x.device)
 e3=torch.cat([unresolved[:,upper[0],upper[1]].to(x.dtype),scale.log()],-1).tanh()
 # E4:observed belief progress,with remaining horizon supplied separately to its existing conditioner.
 e4=(current-before).tanh()*valid[:,None]
 result=torch.stack([e1,e2,e3,e4],1)
 assert result.shape==(B,4,14) and torch.isfinite(result).all()
 return result
