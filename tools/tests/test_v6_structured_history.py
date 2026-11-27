from pathlib import Path

import pytest
import torch
import yaml

from src.research.benchmarks import LocationFinding
from src.research.moe_sboed.v6.objective import scores
from src.research.moe_sboed.v6.policy import METHODS, build_policy


def make(architecture, method):
    cfg = yaml.safe_load(Path(f"configs/moe_sboed/v6/{architecture}_seed101.yaml").read_text())
    torch.manual_seed(cfg["seed"])
    bench = LocationFinding(cfg)
    return build_policy(method, cfg, bench), bench


def test_matched_initial_parameters_and_budget():
    for method in METHODS[:2]:
        structured, _ = make("structured_history", method)
        homogeneous, _ = make("observed_history", method)
        torch.testing.assert_close(structured.state_dict(), homogeneous.state_dict())
    configs = [yaml.safe_load(Path(f"configs/moe_sboed/v6/{name}_history_seed101.yaml").read_text())
               for name in ("observed", "structured")]
    for cfg in configs:
        cfg.pop("architecture")
    assert configs[0] == configs[1]


@pytest.mark.parametrize("method", METHODS[:2])
def test_causal_roles_and_full_horizon_gradients(method):
    model, bench = make("structured_history", method)
    theta = torch.randn(3, 4, dtype=torch.double)
    state = model.start_history(theta)
    torch.testing.assert_close(state, model.start_history(theta + 100))
    history = torch.randn(3, 30, 4, dtype=torch.double)
    for stage in range(5):
        state = model.update_history(state, history[:, stage, :3], stage)
    expected = model.design_logits(state, 5)
    history[:, 5:] = 1e6
    flat = torch.cat((theta[:, :1], history.flatten(1)), dim=1)
    torch.testing.assert_close(model(flat, 5), expected)
    weights = model.mixture(state, 5)[2]
    torch.testing.assert_close(weights, torch.full_like(weights, .25))
    assert (model.router is None) == (method == METHODS[1])
    reward = scores(bench, model, theta, torch.randn(3, 7, 4, dtype=torch.double),
                    torch.randn(3, 30, dtype=torch.double))
    reward.mean().backward()
    for expert in model.experts:
        assert expert.preference.weight.grad.norm() > 0
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    if model.router is not None:
        assert model.router[-1].weight.grad.norm() > 0


def test_physical_role_features():
    model, _ = make("structured_history", METHODS[0])
    previous = torch.tensor([[0., 0., 1.]], dtype=torch.double)
    current = torch.tensor([[.5, 0., 2.]], dtype=torch.double)
    views = model.role_inputs(current, previous, previous, 1)
    expected = [[.5, 0., .25], [.5, 0., 1.], [.5/.26, 0., .25], [2., 1., 1/30]]
    torch.testing.assert_close(views[0], current.new_tensor(expected))
    repeat = model.role_inputs(previous, previous, previous, 1)
    assert (repeat[:, 2] == 0).all()
