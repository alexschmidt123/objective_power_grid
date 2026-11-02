import math
import pytest,torch
from src.research.moe_v5_conditional import *

def test_exact_gaussian_posterior_and_no_future_conditioning():
    past=torch.tensor([[1.,2.],[-1.,.3]],dtype=torch.float64)
    x,mean,var,q=gaussian_posterior_draws(past,.5,16,torch.Generator().manual_seed(81))
    assert x.shape==(2,16);torch.testing.assert_close(mean,past.sum(1)/2.5)
    assert var==1/5 and q.exact and q.conditioned_observations==2
    future=torch.full((2,1),1e6);history=torch.cat((past,future),1)
    y,*_=gaussian_posterior_draws(history[:,:2],.5,16,torch.Generator().manual_seed(81));torch.testing.assert_close(x,y,rtol=0,atol=0)

def test_conditional_score_serial_vectorized_and_per_stage_ceiling():
    x=torch.randn(2,3,9,dtype=torch.float64,requires_grad=True)
    q=[PosteriorQuality(t,exact=True,validated=True) for t in range(3)]
    a,meta=conditional_scores(x,q)
    b=sum(x[:,t,0]-torch.logsumexp(x[:,t],-1)+math.log(9) for t in range(3))
    torch.testing.assert_close(a,b);assert (a<=3*math.log(9)).all()
    assert meta['probability_bound']=='exact-posterior expectation bound'
    a.mean().backward();assert x.grad.abs().sum()>0 and torch.isfinite(x.grad).all()

def test_approximate_quality_and_future_data_are_rejected():
    x=torch.zeros(1,2,3,dtype=torch.float64)
    with pytest.raises(ValueError):conditional_scores(x,[PosteriorQuality(0),PosteriorQuality(1)])
    with pytest.raises(ValueError):require_quality(PosteriorQuality(2,exact=True,validated=True),1)
    with pytest.raises(ValueError):require_quality(PosteriorQuality(1,validated=True,ess_fraction=.1),1)
    q=PosteriorQuality(1,validated=True,ess_fraction=.8,unique_fraction=.8,mixing_checked=True,symmetry_checked=True,proposal_weights_checked=True)
    require_quality(q,1)
    _,meta=conditional_scores(torch.zeros(1,1,3),[PosteriorQuality(0,validated=True,ess_fraction=.8,unique_fraction=.8,mixing_checked=True,symmetry_checked=True,proposal_weights_checked=True)])
    assert meta['probability_bound']=='APPROXIMATE_HEURISTIC_NOT_A_PROVED_BOUND'

def test_alpha_zero_exact_rng_and_graph_short_circuit():
    x=torch.tensor(2.,requires_grad=True);rng=torch.get_rng_state().clone()
    def forbidden():raise AssertionError('Alpha0 must not sample or execute conditional work')
    y=combined_objective(x,0,forbidden);assert y is x and torch.equal(rng,torch.get_rng_state())
    assert torch.autograd.grad(y,x)[0]==1
    assert not activation_gate({'measured_contrasts':32,'empirical_fraction_within_point10':1.},True)
    assert not activation_gate({'measured_contrasts':2000,'empirical_fraction_within_point10':.8},False)
