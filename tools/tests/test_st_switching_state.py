"""Informative hard-switch state fixtures; no statistical training claims."""
import copy
import pytest
import torch
from src.research.sparse_moe import DSelectGate
from src.research.strategy_moe import StrategyManager
from src.research.moe_family import MoESBOEDPolicy,BasePolicy,load_policy
from src.research.moe_architecture_controls import ArchitectureControl
from src.research.benchmarks import LocationFinding
from src.research.engine import rollout

def router():
    r=DSelectGate(2).double()
    with torch.no_grad():
        r.bits.weight.zero_();r.bits.weight[:,0]=.1;r.bits.bias.zero_()
    return r

def manager(bias):
    m=StrategyManager(2,initial_reroute_probability=.6)
    with torch.no_grad():m.switching_gate.logit.bias.fill_(bias)
    return m

def stage(m,r,x):return m(torch.tensor([[x,0.]],dtype=torch.float64),r,straight_through=True)[0]

@pytest.mark.parametrize('bias,expected',[(-.2,0),(.2,3)])
def test_informative_switch_forward_and_gate_derivative(bias,expected):
    r=router();m=manager(bias);stage(m,r,-1.)
    previous=m.state;previous.retain_grad()
    q=stage(m,r,1.)
    assert q.argmax(-1).item()==expected
    cost=q@torch.tensor([0.,1.,2.,3.],dtype=torch.float64)
    derivative=torch.autograd.grad(cost,m.switching_gate.logit.bias,retain_graph=True)[0]
    b=m.last_diagnostics['reroute_probability']
    # The gate surrogate compares HARD proposed and HARD previous experts.
    torch.testing.assert_close(derivative,3*b*(1-b),atol=1e-12,rtol=1e-12)
    cost.backward();assert previous.grad is not None
    if bias<0:assert previous.grad.abs().sum()>0
    assert m.change_count.item()==(expected==3)
    assert m.request_count.item()==(1+(bias>=0))
    assert m.relaxed_weights.grad_fn is not None
    assert not torch.equal(m.relaxed_weights,m.state)

def test_continuation_masks_current_router_surrogate_but_keeps_earlier_state_graph():
    r=router();m=manager(-.2)
    x=torch.tensor([[-1.,0.]],dtype=torch.float64,requires_grad=True)
    m(x,r,straight_through=True)
    y=torch.tensor([[1.,0.]],dtype=torch.float64,requires_grad=True)
    q=m(y,r,straight_through=True)[0]
    # Zero gate context weights isolate router/state gradients.
    loss=q@q.new_tensor([0.,1.,2.,3.])
    gx,gy=torch.autograd.grad(loss,(x,y))
    assert gx.abs().sum()>0
    assert torch.equal(gy,torch.zeros_like(gy))
    # The frozen previous surrogate weighted the new proposal by sigmoid(b),
    # even when the hard forward explicitly continued: this is a different VJP.
    old_router=copy.deepcopy(r);py=old_router(y,hard=False)[0]
    old_gy=torch.autograd.grad((m.last_diagnostics['reroute_probability'].detach()[:,None]*py@py.new_tensor([0.,1.,2.,3.])).sum(),y)[0]
    assert old_gy.abs().sum()>0

def test_reroute_request_same_expert_is_not_actual_change():
    r=router();m=manager(.2);stage(m,r,-1.);stage(m,r,-1.)
    assert m.request_count.item()==2 and m.change_count.item()==0 and m.duration.item()==2

@pytest.mark.parametrize('variant',['switchmoe','switchresmoe'])
def test_t30_hard_forward_loss_gradient_checkpoint_and_disabled_residual(variant,tmp_path):
    torch.manual_seed(101)
    b=LocationFinding({'location_finding':{'design_bounds':[-4,4]},'observation':{'noise_sigma':.5}})
    base=BasePolicy(ArchitectureControl('sum_single',30,64,4.,3.)) if 'resmoe' in variant else None
    p=MoESBOEDPolicy(variant,b,30,base=base,training_routing_mode='straight_through')
    # Keep selectors interior so this fixture can test nonzero gradients.
    with torch.no_grad():p.router.bits.weight.zero_();p.router.bits.bias.fill_(.1);p.manager.switching_gate.logit.bias.fill_(.2)
    hard=copy.deepcopy(p).eval();p.train()
    a=rollout(b,p,30,2,1001,8);z=rollout(b,hard,30,2,1001,8)
    torch.testing.assert_close(torch.stack(a['actions']),torch.stack(z['actions']),atol=1e-12,rtol=1e-12)
    assert (torch.stack(a['actions']).abs()<=4).all()
    (-a['info'][:,-1].mean()).backward()
    assert p.router.bits.weight.grad.abs().sum()>0
    assert all(torch.isfinite(v.grad).all() for v in p.parameters() if v.grad is not None)
    # Informative gate gradients are asserted analytically above; identical
    # experts/decisions in a rollout need not yield nonzero gate gradients.
    f=tmp_path/'policy.pt';torch.save(p.checkpoint(),f);loaded=load_policy(f,b)
    p.eval();new=rollout(b,loaded,30,2,1001,0);old=rollout(b,p,30,2,1001,0)
    torch.testing.assert_close(torch.stack(new['actions']),torch.stack(old['actions']),atol=0,rtol=0)
    if base is not None:
        p.core.alpha=0;p.train();a=rollout(b,p,30,2,1001,0);z=rollout(b,p.core.base.policy.eval(),30,2,1001,0)
        torch.testing.assert_close(torch.stack(a['actions']),torch.stack(z['actions']),atol=0,rtol=0)
