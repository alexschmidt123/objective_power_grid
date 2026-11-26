import pytest
import torch
from src.research.benchmarks import LocationFinding
from src.research.moe_sboed.v6.objective import cumulative_likelihood, scores
from src.research.moe_sboed.v6.policy import METHODS, build_policy


@pytest.mark.parametrize("method", METHODS)
def test_belief_matches_prefix_likelihood_and_trains(method):
    cfg = {"architecture": "belief_expert_history", "horizon": 4,
           "location_finding": {"design_bounds": [-4, 4]},
           "belief": {"particles": 32, "scramble_seed": 20471}}
    bench = LocationFinding(cfg)
    torch.manual_seed(101)
    model = build_policy(method, cfg, bench)
    theta, negatives, noise = [torch.randn(s, dtype=torch.float64)
                               for s in [(3, 4), (3, 16, 4), (3, 4)]]
    state = model.start_history(theta)
    torch.testing.assert_close(state, model.start_history(theta + 987))
    actions = torch.randn(3, 4, 2, dtype=theta.dtype)
    observations = noise.unsqueeze(-1)
    tokens = torch.cat((bench.encode_action(actions), observations / bench.obs_scale), -1)
    particles = model.belief_particles.expand(3, -1, -1)
    for t in range(4):
        state = model.update_history(state, tokens[:, t], t)
        expected = cumulative_likelihood(bench, particles, actions[:, :t+1], observations[:, :t+1])
        torch.testing.assert_close(state[-1], expected)
    torch.manual_seed(101)
    original = build_policy(method, dict(cfg, architecture="expert_history"), bench)
    history = torch.randn(3, 17, dtype=theta.dtype)
    torch.testing.assert_close(model(history, 2), original(history, 2))
    (-scores(bench, model, theta, negatives, noise).mean()).backward()
    grad = model.belief_projection.weight.grad
    assert grad.isfinite().all() and grad.abs().sum() > 0
    assert len(model.experts) == (4 if method in METHODS[:2] else 1)
    with torch.no_grad():
        model.belief_projection.weight.add_(-0.001 * grad)
    perturbed = history.clone()
    perturbed[:, 9:] += 100
    torch.testing.assert_close(model(history, 2), model(perturbed, 2))
    assert not torch.allclose(model(history, 2), original(history, 2))
    _, w = model.mixture(state, 4)
    if method != METHODS[0]:
        torch.testing.assert_close(w, torch.full_like(w, 1 / len(model.experts)))
