import pytest
import torch
import yaml
from pathlib import Path

from src.research.benchmarks import LocationFinding
from src.research.moe_sboed.v6.local_precision import local_precision, signal_jacobian, training_terms
from src.research.moe_sboed.v6.objective import scores, cumulative_likelihood
from src.research.moe_sboed.v6.policy import build_policy


def config(arch):
    return yaml.safe_load(Path(f"configs/moe_sboed/v6/{arch}_precision_seed101.yaml").read_text())


def test_precision_matches_differentiated_physics_and_is_exchangeable():
    bench = LocationFinding(config("relative"))
    theta = torch.tensor([[.08,-.31,.79,-.2]],dtype=torch.float64,requires_grad=True)
    a = torch.tensor([[[-.4,.7],[.8,-.1],[.09,-.3]]],dtype=torch.float64,requires_grad=True)
    def signals(t):
        return bench.propagate(t[:,None,:],None,a)[0].flatten()
    actual = torch.autograd.functional.jacobian(signals,theta).squeeze(1)
    torch.testing.assert_close(signal_jacobian(theta,a)[0],actual)
    prec = torch.eye(4,dtype=a.dtype) + actual.T @ actual / bench.sigma**2
    gain = local_precision(theta,a,bench.sigma)
    torch.testing.assert_close(gain[0],torch.linalg.slogdet(prec).logabsdet / 2)
    torch.testing.assert_close(gain,local_precision(theta.roll(2,-1),a,bench.sigma))
    assert torch.isfinite(torch.autograd.grad(gain.sum(),a)[0]).all()


@pytest.mark.parametrize("arch", ["observed","relative"])
@pytest.mark.parametrize("method", ["moe-sboed","moe-sboed-uniform"])
def test_full_horizon_terms_and_causal_policy(arch,method):
    cfg=config(arch)
    bench=LocationFinding(cfg)
    torch.manual_seed(101)
    model=build_policy(method,cfg,bench)
    torch.manual_seed(101)
    other=build_policy("moe-sboed-uniform",cfg,bench)
    for key,val in other.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key],val)
    t=cfg["horizon"]
    theta=torch.randn(3,4,dtype=torch.double)
    noise=torch.randn(3,t,dtype=torch.double)
    neg=torch.randn(3,12,4,dtype=torch.double)
    history=torch.randn(3,t,4,dtype=torch.double)
    history[:,:,:2].tanh_()
    state=model.start_history(theta)
    torch.testing.assert_close(state,model.start_history(theta+7))
    for j in range(6):
        state=model.update_history(state,history[:,j,:3],j)
    flat=torch.cat((theta[:,:1],history.flatten(1)),1)
    reference=model.design_logits(state,6)
    torch.testing.assert_close(reference,model(flat,6))
    flat[:,1+4*6:]=1e6
    torch.testing.assert_close(reference,model(flat,6))
    c,p=training_terms(bench,model,theta,neg,noise,cumulative_likelihood)
    torch.testing.assert_close(c,scores(bench,model,theta,neg,noise))
    (c+cfg["training"]["local_precision_weight"]*p).mean().backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in model.parameters())
    if arch=="relative":
        assert model.reference.weight.grad.norm() > 0
