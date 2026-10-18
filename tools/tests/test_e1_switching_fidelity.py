"""Fixed-draw correctness fixtures, not experimental training runs."""
import copy
import pytest
import torch
from src.research.benchmarks import LocationFinding
from src.research.engine import rollout
from src.research.moe_family import MoESBOEDPolicy, BasePolicy
from src.research.moe_architecture_controls import ArchitectureControl


def make(variant,mode='learned_switch'):
    torch.manual_seed(103)
    b=LocationFinding({'location_finding':{'design_bounds':[-4,4]},'observation':{'noise_sigma':.5}})
    base=BasePolicy(ArchitectureControl('sum_single',5,64,4.,3.)) if 'resmoe' in variant else None
    p=MoESBOEDPolicy(variant,b,5,base=base,routing_mode=mode,training_routing_mode='straight_through')
    return b,p

@pytest.mark.parametrize('variant',['switchmoe','switchresmoe'])
@pytest.mark.parametrize('mode',['always_reroute','always_continue','learned_switch'])
def test_hard_training_inference_rollout_and_episode_reset(variant,mode):
    b,p=make(variant,mode);q=copy.deepcopy(p).eval();p.train()
    a=rollout(b,p,5,3,77101,8);z=rollout(b,q,5,3,77101,8)
    for key in ['actions','observations']:
        torch.testing.assert_close(torch.stack(a[key]),torch.stack(z[key]),atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(a['info'],z['info'],atol=1e-11,rtol=1e-11)
    if mode=='always_continue':
        assert (p.manager.request_count==1).all()
        assert (p.manager.duration==4).all()
        assert (p.manager.change_count==0).all()
    # A second identical episode must not carry previous episode state.
    a2=rollout(b,p,5,3,77101,8)
    torch.testing.assert_close(torch.stack(a['actions']),torch.stack(a2['actions']),atol=0,rtol=0)
    (-a2['info'][:,-1].mean()).backward()
    assert any(v.grad is not None and v.grad.abs().sum()>0 for v in p.experts.parameters())
    if mode=='learned_switch':
        assert sum(float(v.grad.abs().sum()) for v in p.manager.switching_gate.parameters() if v.grad is not None)>0
    assert all(torch.isfinite(v.grad).all() for v in p.parameters() if v.grad is not None)

@pytest.mark.parametrize('variant,plain',[('switchmoe','moe'),('switchresmoe','resmoe')])
def test_st_forced_reroute_matches_corresponding_nonswitching_policy(variant,plain):
    b,p=make(variant,'always_reroute');_,q=make(plain)
    state={k.replace('core.history_encoder.sum_encoder.','core.sum_encoder.') if plain=='moe' else k:v for k,v in p.state_dict().items() if '.manager.' not in k}
    result=q.load_state_dict(state,strict=False)
    assert not result.missing_keys and not result.unexpected_keys
    p.train();q.train()
    a=rollout(b,p,5,3,77101,8);z=rollout(b,q,5,3,77101,8)
    torch.testing.assert_close(torch.stack(a['actions']),torch.stack(z['actions']),atol=1e-12,rtol=1e-12)
