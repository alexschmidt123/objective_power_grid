import pytest
import torch

from src.research.benchmarks import LocationFinding
from src.research.engine import rollout
from src.research.moe_sboed.v6.objective import scores
from src.research.moe_sboed.v6.policy import METHODS, build_policy


def make(method, horizon):
    torch.manual_seed(101)
    benchmark = LocationFinding({"location_finding": {"design_bounds": [-4, 4]}})
    config = dict(architecture="observed_history_moe", horizon=horizon,
                  training=dict(objective="terminal_spce"))
    return build_policy(method, config, benchmark), benchmark


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("horizon", [15, 30])
def test_full_horizon_gradient_and_ablation(method, horizon):
    policy, bench = make(method, horizon)
    learned, _ = make(METHODS[0], horizon)
    torch.testing.assert_close(policy.head.state_dict(), learned.head.state_dict())
    torch.testing.assert_close(policy.encoder.state_dict(), learned.encoder.state_dict())
    if method in METHODS[2:]:
        assert len(policy.experts) == 1
        torch.testing.assert_close(policy.experts[0].state_dict(),
                                   learned.experts[METHODS.index(method)-2].state_dict())
    assert (policy.router is not None) == (method == METHODS[0])
    assert not torch.equal(learned.experts[0].preference.weight, learned.experts[1].preference.weight)
    theta = torch.randn(3,4,dtype=torch.double)
    negatives = torch.randn(3,7,4,dtype=torch.double)
    noise = torch.randn(3,horizon,dtype=torch.double)
    state = policy.start_history(theta)
    torch.testing.assert_close(state,policy.start_history(theta.flip(0)))
    _, _, weights = policy.mixture(state,0)
    torch.testing.assert_close(weights,torch.full_like(weights,1/len(policy.experts)))
    out = rollout(bench,policy,horizon,3,101,7,truth=theta,noise=noise.unsqueeze(-1),prior_particles=negatives)
    assert len(out['actions']) == horizon and torch.stack(out['actions']).abs().max() <= 4
    reward = scores(bench,policy,theta,negatives,noise)
    torch.testing.assert_close(reward,out['info'][:,-1])
    gradients = torch.autograd.grad(reward.mean(),tuple(policy.parameters()))
    reference = torch.autograd.grad(out['info'][:,-1].mean(),tuple(policy.parameters()))
    for a,b in zip(gradients,reference):
        torch.testing.assert_close(a,b,atol=1e-8,rtol=1e-8)
    by_name=dict(zip(dict(policy.named_parameters()),gradients))
    for index in range(len(policy.experts)):
        assert by_name[f'experts.{index}.preference.weight'].norm() > 0
    if policy.router is not None:
        assert by_name['router.2.weight'].norm() > 0


def test_location_pair_order_and_future_slots():
    model,_ = make(METHODS[0],15)
    x=torch.randn(4,61,dtype=torch.double)
    stage=5
    expected=model(x,stage)
    shuffled=x.clone()
    slots=shuffled[:,1:].reshape(4,15,4)
    slots[:,:stage]=slots[:,:stage].flip(1)
    slots[:,stage:]=12345
    torch.testing.assert_close(model(shuffled,stage),expected)


def test_appending_measurement_can_change_relative_attention_to_previous_measurements():
    policy, _ = make(METHODS[0],15)
    expert = policy.experts[0]
    state = policy.start_history(torch.empty(1))
    state = policy.update_history(state,torch.tensor([[.3,-.1,1.]],dtype=torch.double),0)
    state = policy.update_history(state,torch.tensor([[-.8,.5,.4]],dtype=torch.double),1)
    total, tokens = state
    original_difference = tokens[:,0] - tokens[:,1]

    def relative_log_weight(total):
        # Fix the stage coordinate to isolate what changed in the measurements.
        context = torch.cat((total,total.new_full((1,1),2/15)),1)
        return (original_difference * expert.preference(context)).sum() / 4

    before = relative_log_weight(total)
    total, _ = policy.update_history(state,torch.tensor([[.2,.7,-1.]],dtype=torch.double),2)
    assert not torch.isclose(relative_log_weight(total), before)
