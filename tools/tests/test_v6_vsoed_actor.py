"""Exercise the native replay adapter against sequential V6 deployment."""
import pytest
import torch
from src.research.benchmarks import LocationFinding
from src.research.moe_sboed.v6.policy import build_policy
from src.research.moe_sboed.v6.variational import PackedActor


@pytest.mark.parametrize("method", ["moe-sboed","moe-sboed-uniform"])
def test_native_mixed_stages_and_gradients(method):
    torch.manual_seed(101)
    cfg = {"architecture":"independent_feature","training":{"objective":"vsoed_gmm_i"},
           "horizon":5,"location_finding":{"design_bounds":[-4,4]}}
    bench=LocationFinding(cfg)
    policy=build_policy(method,cfg,bench)
    packed=PackedActor(policy)
    t=cfg["horizon"]
    b=3
    d=torch.randn(b,t-1,2)
    y=torch.randn(b,t-1,1)
    state=policy.start_history(d.double())
    rows,expected=[],[]
    for k in range(t):
        stage=torch.eye(t)[k].repeat(b,1)
        # Fill even the unobserved suffix to test explicit causal masking.
        rows.append(torch.cat((stage,d.flatten(1),y.flatten(1)),-1))
        expected.append(bench.feasible(policy.design_logits(state,k),[],t))
        if k<t-1:
            state=policy.update_history(state,torch.cat((d[:,k]/4,y[:,k]/3),-1),k)
    expected=torch.stack(expected,1)
    x=torch.stack(rows,1)
    actual=packed(x)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
    ref=torch.autograd.grad(expected.sum(),tuple(policy.parameters()))
    grad=torch.autograd.grad(actual.sum(),tuple(policy.parameters()))
    for a,b in zip(grad,ref):
        torch.testing.assert_close(a,b,atol=4e-5,rtol=2e-5)
    # Exercise chunked recomputation used by the native 10k-by-T critic minibatch.
    expanded=x.repeat(140,1,1)
    out=packed(expanded)
    torch.testing.assert_close(out,actual.repeat(140,1,1),atol=2e-6,rtol=2e-6)
    assert torch.autograd.grad(out.mean(),policy.head[-1].weight)[0].isfinite().all()
