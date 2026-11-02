"""Gated conditional auxiliary. Approximate posterior draws give no proved bound."""
from dataclasses import dataclass
import math
import torch

@dataclass(frozen=True)
class PosteriorQuality:
    conditioned_observations:int
    renewed:bool=True
    exact:bool=False
    validated:bool=False
    ess_fraction:float=0.
    unique_fraction:float=0.
    mixing_checked:bool=False
    symmetry_checked:bool=False
    proposal_weights_checked:bool=False
    sampling_gradient:str='stopped; not an unbiased full-policy gradient'

def require_quality(q,stage):
    if q.conditioned_observations!=stage or not q.renewed:
        raise ValueError('Negatives must be renewed conditional on past observations only')
    if not q.validated:raise ValueError('Unvalidated posterior sampler')
    if not q.exact and not (q.ess_fraction>=.5 and q.unique_fraction>=.5 and q.mixing_checked and q.symmetry_checked and q.proposal_weights_checked):
        raise ValueError('Approximate posterior ESS/coverage/mixing/proposal gate failed')

def conditional_scores(step_loglik,quality):
    if step_loglik.ndim!=3 or step_loglik.shape[-1]<2 or len(quality)!=step_loglik.shape[1]:
        raise ValueError('Conditional likelihood shape/quality mismatch')
    if not torch.isfinite(step_loglik).all():raise FloatingPointError('Nonfinite conditional likelihood')
    for stage,q in enumerate(quality):require_quality(q,stage)
    c=step_loglik[:,:,0]-torch.logsumexp(step_loglik,-1)+math.log(step_loglik.shape[-1])
    return c.sum(1),dict(estimator='conditional contrastive auxiliary',
        probability_bound='exact-posterior expectation bound' if all(q.exact for q in quality) else 'APPROXIMATE_HEURISTIC_NOT_A_PROVED_BOUND',
        gradient_claim='Sampling-gradient treatment must be reported separately; no general unbiased policy-gradient claim')

def combined_objective(main,alpha,conditional_thunk=None):
    # Exact short circuit: no extra random draw, computation, graph or rounding.
    if alpha==0:return main
    if not 0<alpha<=.1 or conditional_thunk is None:raise ValueError('Predeclared small auxiliary coefficient and samples required')
    return main+alpha*conditional_thunk()

def activation_gate(saturation,sampler_validated):
    return bool(saturation.get('measured_contrasts')==2000 and saturation.get('empirical_fraction_within_point10',0)>=.25 and sampler_validated)

def gaussian_posterior_draws(past,noise_variance,negatives,generator,prior_mean=0.,prior_variance=1.):
    """Exact independent Gaussian toy oracle; not a location posterior approximation."""
    if past.ndim!=2 or noise_variance<=0 or prior_variance<=0 or negatives<1:raise ValueError('Invalid Gaussian toy')
    stage=past.shape[1];variance=1/(1/prior_variance+stage/noise_variance)
    mean=variance*(prior_mean/prior_variance+past.sum(1)/noise_variance)
    draws=mean[:,None]+math.sqrt(variance)*torch.randn(len(past),negatives,dtype=past.dtype,device=past.device,generator=generator)
    quality=PosteriorQuality(stage,exact=True,validated=True,sampling_gradient='Analytic reparameterized Gaussian toy only')
    return draws,mean,variance,quality
