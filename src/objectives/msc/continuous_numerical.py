"""Antithetic numerical training of deterministic design policies.

For discontinuous finite-particle quantile utilities, do not substitute
stochastic-action REINFORCE for deterministic DAD. Keep xi=pi_phi(history).
Estimate the gradient of Gaussian-smoothed J_delta(phi)=E_z J(phi+delta*z):
  E[(J(phi+delta*z)-J(phi-delta*z))*z/(2*delta)].
Each rollout uses the exact control utility and paired model/noise draws.
This is a declared zeroth-order training adaptation, NOT original DAD's
pathwise estimator or an unbiased gradient of unsmoothed J at finite delta.
Reference: Salimans et al. (2017), https://arxiv.org/abs/1703.03864.
"""
import copy
import numpy as np
import torch


def improve_numerical(engine,policy,optimizer,rng,batch,*,scale=.01,directions=4,**kwargs):
    if scale<=0 or directions<1:raise ValueError('Positive numerical-gradient scale and direction count required')
    params=[p for name,p in policy.named_parameters() if name=='sequence' or name.startswith('network.')]
    originals=[p.detach().clone() for p in params]
    gradients=[torch.zeros_like(p) for p in params]
    paired=[]
    try:
        for _ in range(directions):
            perturbations=[torch.as_tensor(rng.normal(size=p.shape),dtype=p.dtype,device=p.device) for p in params]
            seed=int(rng.integers(2**31))
            scores=[]
            for sign in (1.,-1.):
                with torch.no_grad():
                    for p,base,z in zip(params,originals,perturbations):p.copy_(base+sign*scale*z)
                    result=engine.rollout(policy,np.random.default_rng(seed),batch,stochastic=False,**kwargs)
                    scores.append(float(np.mean(result['info'][:,-1])))
            if not np.isfinite(scores).all():raise ValueError('Nonfinite paired objective')
            for gradient,z in zip(gradients,perturbations):gradient.add_(z,alpha=(scores[0]-scores[1])/(2*scale*directions))
            paired.extend(scores)
    finally:
        with torch.no_grad():
            for p,base in zip(params,originals):p.copy_(base)
    optimizer.zero_grad(set_to_none=True)
    for p,gradient in zip(params,gradients):p.grad=-gradient
    torch.nn.utils.clip_grad_norm_(params,5.)
    optimizer.step()
    return float(np.mean(paired))
