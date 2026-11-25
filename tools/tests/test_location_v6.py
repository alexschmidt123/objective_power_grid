import torch
import pytest

from src.research.benchmarks import LocationFinding
from src.research.engine import rollout
from src.research.moe_sboed.v6.objective import scores
from src.research.moe_sboed.v6.policy import LocationMoE, METHODS


def make(method):
    torch.manual_seed(101)
    return LocationMoE(method, 30)


@pytest.mark.parametrize("method", METHODS)
def test_fresh_arm_matches_native_terminal_loss_and_gradient(method):
    policy = make(method)
    full = make(METHODS[0])
    if method in METHODS[2:]:
        slot = METHODS.index(method) - 2
        torch.testing.assert_close(policy.experts[0].state_dict(), full.experts[slot].state_dict())
        assert len(policy.experts) == 1
    elif method == METHODS[1]:
        assert torch.equal(policy.weights(torch.ones(2,17,dtype=torch.float64)), torch.full((2,4),.25,dtype=torch.float64))
    assert (policy.router is None) == (method != METHODS[0])
    bench = LocationFinding({"location_finding":{"design_bounds":[-4,4]}})
    truth = torch.randn(3, 4, dtype=torch.float64)
    negatives = torch.randn(3, 7, 4, dtype=torch.float64)
    noise = torch.randn(3, 30, dtype=torch.float64)
    expected = rollout(bench, policy, 30, 3, 33, 7, truth=truth, noise=noise.unsqueeze(-1), prior_particles=negatives)
    assert len(expected["actions"]) == 30 and torch.stack(expected["actions"]).abs().max() <= 4
    expected = expected["info"][:, -1]
    actual = scores(bench, policy, truth, negatives, noise)
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
    grads = torch.autograd.grad(actual.mean(), tuple(policy.parameters()))
    gold = torch.autograd.grad(expected.mean(), tuple(policy.parameters()))
    for observed, reference in zip(grads, gold):
        assert observed.isfinite().all() and observed.abs().sum() > 0
        torch.testing.assert_close(observed, reference, atol=1e-8, rtol=1e-8)
