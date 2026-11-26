import math

import pytest
import torch
from torch.distributions import Independent, Normal

from src.research.benchmarks import LocationFinding
from src.research.moe_sboed.v6.adaptive_contrast import adaptive_scores, prior_over_proposal, source_proposal
from src.research.moe_sboed.v6.objective import cumulative_likelihood, scores, trajectory
from src.research.moe_sboed.v6.policy import METHODS, build_policy


def case(method, horizon=5):
    cfg = {"horizon": horizon, "architecture": "expert_history",
           "location_finding": {"design_bounds": [-4, 4]}}
    bench = LocationFinding(cfg)
    torch.manual_seed(101)
    return bench, build_policy(method, cfg, bench)


@pytest.mark.parametrize("method", METHODS)
def test_readers_match_masked_softmax_and_train_end_to_end(method):
    bench, model = case(method)
    theta = torch.randn(4, 4, dtype=torch.float64)
    tokens = torch.randn(4, 5, 3, dtype=torch.float64)
    state = model.start_history(theta)
    outputs = []
    for stage, token in enumerate(tokens.unbind(1)):
        state = model.update_history(state, token, stage)
        inp = torch.cat((token[:, :2], token[:, 2:].tanh(),
                         token.new_full((len(token), 1), (stage+1)/5)), -1)
        outputs.append(torch.stack([reader(inp) for reader in model.readers], dim=1))
        obs = torch.stack(outputs, dim=1)
        expected = (obs[..., :-1].tanh() * obs[..., -1:].softmax(dim=1)).sum(dim=1)
        torch.testing.assert_close(state[3], expected)
    if len(model.readers) > 1:
        assert not torch.allclose(state[3][:, 0], state[3][:, 1])
    bound = scores(bench, model, theta, torch.randn(4, 8, 4, dtype=theta.dtype), tokens[..., 2])
    (-bound.mean()).backward()
    for reader in model.readers:
        grad = reader[-1].weight.grad
        assert grad.isfinite().all() and grad[-1].abs().sum() > 0 and grad[:-1].abs().sum() > 0
    if model.router is not None:
        assert model.router[-1].weight.grad.abs().sum() > 0
    else:
        _, w = model.mixture(state, 5)
        torch.testing.assert_close(w, torch.full_like(w, 1/len(model.readers)))


def test_matched_ablations_and_causal_deployment():
    _, full = case(METHODS[0])
    _, uniform = case(METHODS[1])
    _, single = case(METHODS[4])
    torch.testing.assert_close(full.readers.state_dict(), uniform.readers.state_dict())
    torch.testing.assert_close(full.head.state_dict(), uniform.head.state_dict())
    torch.testing.assert_close(full.readers[2].state_dict(), single.readers[0].state_dict())
    x = torch.randn(4, 21, dtype=torch.float64)
    changed = x.clone()
    changed[:, 9:] = 9999
    torch.testing.assert_close(full(x,2), full(changed,2))
    torch.testing.assert_close(full(x,2), uniform(x,2))
    with torch.no_grad():
        full.router[-1].weight.normal_(0,.02)
    decoded = full(x,2)
    other = case(METHODS[0])[1]
    other.load_state_dict(full.state_dict())
    torch.testing.assert_close(decoded,other(x,2))


def test_importance_corrected_bound_against_distribution_api_and_finite_difference():
    bench, model = case(METHODS[0])
    theta, z, eps = [torch.randn(s, dtype=torch.float64) for s in ((3,4),(3,7,4),(3,5))]
    mix = .5
    mask = torch.rand(3,7,2) < mix
    def bound():
        return adaptive_scores(bench,model,theta,z,eps,mask,mix,.5)
    actual = bound()
    actions, obs = trajectory(bench,model,theta,eps)
    mu, sd = source_proposal(actions,obs,.5)
    bases = z.unflatten(-1,(2,2))
    proposals = torch.where(mask[...,None],bases,mu+sd*bases)
    pts = torch.cat((theta[:,None],proposals.flatten(-2)),1)
    sources = pts.unflatten(-1,(2,2))
    lp = Independent(Normal(torch.zeros_like(mu),torch.ones_like(sd)),1).log_prob(sources)
    local = Independent(Normal(mu,sd),1).log_prob(sources)
    lq = torch.logaddexp(lp + math.log(mix), local + math.log1p(-mix))
    ll = cumulative_likelihood(bench,pts,actions,obs)
    expected = ll[:,0] - torch.logsumexp(ll+(lp-lq).sum(-1),1) + math.log(pts.shape[1])
    torch.testing.assert_close(actual,expected)
    zero = prior_over_proposal(sources,torch.zeros_like(mu),torch.ones_like(sd),mix)
    torch.testing.assert_close(zero,torch.zeros_like(zero))
    grad = torch.autograd.grad(actual.mean(),model.head[-1].bias)[0][0]
    with torch.no_grad():
        v = model.head[-1].bias
        original=v.clone()
        h=1e-6
        v[0]+=h
        plus=bound().mean()
        v.copy_(original)
        v[0]-=h
        minus=bound().mean()
        v.copy_(original)
    torch.testing.assert_close(grad,(plus-minus)/(2*h),rtol=2e-5,atol=2e-5)
