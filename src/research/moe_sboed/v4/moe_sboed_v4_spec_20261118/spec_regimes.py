"""Detached, training-calibrated history regimes;never optimal-expert labels."""
import torch
from hec_policy import mean_signal,jacobian

@torch.no_grad()
def scores(c,opportunity):
    w,mu,sd=c['w'],c['mu'],c['sd'];nodes,pw=c['nodes'],c['pw']
    centered=mu-c['mean'][:,None]
    covariance=torch.einsum('bn,bni,bnj->bij',w,centered,centered)+torch.diag_embed((w[...,None]*sd.square()).sum(1))
    broad=torch.log1p(covariance.diagonal(dim1=-2,dim2=-1).sum(-1))
    offsets=mu.new_tensor([[0.,0.],[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    probes=(c['a0'][:,None]+offsets[None]).clamp(-4,4)
    predicted=mean_signal(mu,probes)
    pm=(predicted*w[:,None]).sum(-1,keepdim=True)
    separation=((predicted-pm).square()*w[:,None]).sum(-1).amax(-1)/.25
    entropy=-(w*w.clamp_min(1e-30).log()).sum(-1)/w.new_tensor(8.).log()
    discrimination=entropy*torch.log1p(separation)
    J=jacobian(nodes,c['a0'])
    fisher=4*torch.einsum('bn,bni,bnj->bij',pw,J,J)
    scaled=fisher*c['scale'][:,:,None]*c['scale'][:,None,:]
    weakest=torch.linalg.eigvalsh(scaled.double()).amin(-1).clamp_min(0)
    sensitivity=torch.log1p(1/(weakest+1e-8)).to(w.dtype)
    q=opportunity(torch.cat([c['x'][:,None].expand(-1,5,-1),probes],-1)).squeeze(-1)
    remaining=c['remaining'][:,0]
    opportunity_score=remaining*(q-q[:,:1]).clamp_min(0).amax(-1)
    out=torch.stack([broad,discrimination,sensitivity,opportunity_score],-1)
    assert torch.isfinite(out).all()
    return out,remaining

class Regimes:
    def __init__(self,thresholds):self.thresholds=list(thresholds)
    @torch.no_grad()
    def __call__(self,c,opportunity):
        value,remaining=scores(c,opportunity)
        masks=value>=value.new_tensor(self.thresholds)
        masks[:,3]&=remaining>=.5
        return masks,value
