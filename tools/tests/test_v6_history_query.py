import torch
from src.research.benchmarks import LocationFinding
from src.research.moe_sboed.v6.policy import build_policy
from src.research.moe_sboed.v6.objective import scores
from src.research.moe_sboed.v6.history_query import training_prefix,update_posterior


def make(method):
    cfg = {"horizon":5,"architecture":"fresh_history_query","location_finding":{"design_bounds":[-4,4]}}
    bench = LocationFinding(cfg)
    torch.manual_seed(101)
    return bench,build_policy(method,cfg,bench)


def test_independent_routing_and_joint_provider():
    bench,m = make("moe-sboed")
    _,u = make("moe-sboed-uniform")
    theta = torch.randn(4,4,dtype=torch.double)
    initial = m.start_history(theta)
    torch.testing.assert_close(initial,m.start_history(theta+1000))
    state = m.update_history(initial,theta[:,:3],0)
    uniform_state = u.update_history(u.start_history(theta),theta[:,:3],0)
    torch.testing.assert_close(m.design_logits(state,1),u.design_logits(uniform_state,1))
    assert torch.equal(u.mixture(uniform_state,1)[1],torch.full((4,4),.25))
    torch.testing.assert_close(m.posterior.state_dict(),u.posterior.state_dict())
    assert not any(p.requires_grad for p in m.posterior.parameters())
    (-scores(bench,m,theta,torch.randn(4,16,4,dtype=torch.double),
             torch.randn(4,5,dtype=torch.double)).mean()).backward()
    for block in [*m.experts,m.head,m.router]:
        assert any(p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum()>0 for p in block.parameters())
    actions,obs = torch.randn(4,5,2),torch.randn(4,5,1)
    history = training_prefix(actions,obs,torch.Generator().manual_seed(301))
    lengths = history[:,:,-1].sum(-1).long()
    assert ((lengths >= 1) & (lengths < 5)).all()
    for i,n in enumerate(lengths):
        assert torch.equal(history[i,n:],torch.zeros_like(history[i,n:]))
    old = [p.clone() for p in m.posterior.parameters()]
    update_posterior(m.posterior,torch.optim.Adam(m.posterior.parameters()),history,theta.float())
    assert any(not torch.equal(a,b) for a,b in zip(old,m.posterior.parameters()))
    assert not any(p.requires_grad for p in m.posterior.parameters())
    u2 = make("moe-sboed")[1]
    u2.load_state_dict(m.state_dict())
    torch.testing.assert_close(m.design_logits(state,1),u2.design_logits(state,1))
