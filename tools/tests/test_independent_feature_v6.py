import io

import pytest
import torch

from src.research.benchmarks import LocationFinding
from src.research.engine import rollout
from src.research.moe_sboed.v6.objective import scores
from src.research.moe_sboed.v6.policy import build_policy, METHODS
from src.research.moe_sboed.v6.feature_policy import FeaturePolicy


@pytest.mark.parametrize("method", METHODS)
def test_fresh_training_and_reload_without_archive(method, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = {"architecture":"independent_feature", "horizon":5, "location_finding":{"design_bounds":[-4,4]}}
    bench = LocationFinding(cfg)
    torch.manual_seed(101)
    model = build_policy(method, cfg, bench)
    theta, eps, particles = [torch.randn(shape, dtype=torch.float64)
                             for shape in ((5,4), (5,5), (5,11,4))]
    native = rollout(bench, model, 5, 5, 42, 11, truth=theta, noise=eps.unsqueeze(-1), prior_particles=particles)
    bound = scores(bench, model, theta, particles, eps)
    torch.testing.assert_close(bound, native["info"][:, -1])
    assert max(a.abs().max() for a in native["actions"]) <= 4
    state = model.start_history(theta)
    z, w = model.mixture(state, 0)
    torch.testing.assert_close(w, torch.full_like(w, 1/len(model.experts)))
    if len(model.experts)>1:
        assert not torch.equal(model.experts[0][0][0].weight, model.experts[1][0][0].weight)
    opt = torch.optim.Adam(model.parameters(), lr=5e-5)
    (-bound.mean()).backward()
    for e in model.experts:
        assert e[0][0].weight.grad.isfinite().all() and e[0][0].weight.grad.abs().sum() > 0
    if method == METHODS[0]:
        assert model.router[-1].weight.grad.abs().sum() > 0
    else:
        assert model.router is None
    opt.step()
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    copy = build_policy(method,cfg,bench)
    stream.seek(0)
    copy.load_state_dict(torch.load(stream, weights_only=True))
    torch.testing.assert_close(scores(bench,copy,theta,particles,eps), scores(bench,model,theta,particles,eps))


def test_causal_order_and_alternative_dimensions():
    torch.manual_seed(42)
    model = FeaturePolicy(METHODS[0], 10, action_dim=1, observation_dim=2)
    x = torch.randn(4, 41, dtype=torch.float64)
    y = x.clone()
    y[:, 9:] += 12345
    torch.testing.assert_close(model(x, 2), model(y, 2))
    y[:,1:9] = x[:,1:9].reshape(4,2,4).flip(1).reshape(4,8)
    assert model(x,2).shape == (4,1) and not torch.allclose(model(x,2),model(y,2))
