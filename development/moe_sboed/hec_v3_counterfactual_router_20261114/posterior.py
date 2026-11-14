"""Chunked version of the audited prior/symmetrized-GMM importance sampler."""
import math
import numpy as np
import torch

@torch.no_grad()
def sample(context,x,ds,ys,stage,count,particles,seed,reference):
    rng=np.random.default_rng(seed);device=x.device
    w,mu,sd=context.belief.params(x);w=w[0].double();w/=w.sum();mu=mu[0].double();sd=sd[0].double()
    # Host storage bounds GPU allocation independently of the particle budget.
    pool=np.empty((particles,4),dtype=np.float64);logweight=np.empty(particles,dtype=np.float64)
    for first in range(0,particles,262144):
        size=min(262144,particles-first)
        ids=torch.tensor(rng.choice(len(w),size,p=w.cpu().numpy()),device=device)
        theta=mu[ids]+sd[ids]*torch.tensor(rng.standard_normal((size,4)),device=device)
        kind=rng.random(size);swap=torch.tensor(kind<.45,device=device);theta[swap]=theta[swap][:,[2,3,0,1]]
        prior=torch.tensor(kind>=.9,device=device);theta[prior]=torch.tensor(rng.standard_normal((int(prior.sum()),4)),device=device)
        lp=reference.prior_logprob(theta)
        lq=torch.logsumexp(torch.stack([lp+math.log(.1),
            reference.gaussian_mix_logprob(theta,w,mu,sd)+math.log(.45),
            reference.gaussian_mix_logprob(theta[:,[2,3,0,1]],w,mu,sd)+math.log(.45)]),0)
        likelihood=torch.zeros(size,device=device,dtype=torch.float64)
        for t in range(stage):
            prediction=reference.signal(theta.reshape(size,2,2),ds[0,t].double())
            likelihood+=-2*(ys[0,t]-prediction).square()
        pool[first:first+size]=theta.cpu().numpy();logweight[first:first+size]=(lp+likelihood-lq).cpu().numpy()
    weight=np.exp(logweight-logweight.max());weight/=weight.sum()
    ess=float(1/np.square(weight).sum());maximum=float(weight.max());ids=rng.choice(particles,count,p=weight)
    assert np.isfinite(weight).all() and abs(weight.sum()-1)<1e-10
    return torch.tensor(pool[ids],device=device),dict(seed=seed,particles=particles,ESS=ess,max_weight=maximum,
        selected_unique=len(np.unique(ids)),posterior_mean=(weight[:,None]*pool).sum(0).tolist(),
        approximate=True,proposal='0.1 prior +0.45 GMM +0.45 source-permuted GMM',GPU_chunk=262144)
