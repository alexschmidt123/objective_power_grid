"""Optional MoE training routing; default methods remain unchanged."""
import copy
import pytest
import torch
from src.research.benchmarks import LocationFinding
from src.research.engine import rollout
from src.research.moe_family import MoESBOEDPolicy,load_policy
from src.research.sparse_moe import DSelectGate

def benchmark():return LocationFinding({'location_finding':{'design_bounds':[-4,4]},'observation':{'noise_sigma':.5}})
def pair():
    torch.manual_seed(103);hard=MoESBOEDPolicy('moe',benchmark(),5)
    torch.manual_seed(103);st=MoESBOEDPolicy('moe',benchmark(),5,training_routing_mode='straight_through')
    return hard,st

@pytest.mark.parametrize('gamma',[1.,.1,.001])
def test_st_weights_match_native_hard_and_keep_exact_relaxed_derivative(gamma):
    torch.manual_seed(7);router=DSelectGate(193,1,gamma).double();c=torch.randn(8,193,dtype=torch.float64)
    w,soft,_,_=router.straight_through(c)
    assert torch.equal(w,router(c,hard=True)[0])
    native=router(c,hard=False)[0]
    assert torch.equal(soft,native)
    costs=torch.randn_like(w);params=tuple(router.parameters())
    a=torch.autograd.grad((w*costs).sum(),params,retain_graph=True)
    b=torch.autograd.grad((native*costs).sum(),params)
    assert all(torch.equal(x,y) for x,y in zip(a,b))

def test_st_native_zero_logit_ties():
    r=DSelectGate(193).double()
    with torch.no_grad():r.bits.weight.zero_();r.bits.bias.zero_()
    c=torch.zeros(2,193,dtype=torch.float64);w,*_=r.straight_through(c)
    assert torch.equal(w,r(c,hard=True)[0]) and (w.argmax(-1)==3).all()

def test_st_full_rollout_matches_hard_forward_and_default_initialization():
    hard,st=pair()
    assert all(torch.equal(v,hard.state_dict()[k]) for k,v in st.state_dict().items())
    hard.eval();st.train()
    a=rollout(benchmark(),hard,5,8,1001,8);b=rollout(benchmark(),st,5,8,1001,8)
    for key in ['actions','observations','latents']:
        torch.testing.assert_close(torch.stack(a[key]),torch.stack(b[key]),atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(a['info'],b['info'],atol=1e-11,rtol=1e-11)
    (-b['info'][:,-1].mean()).backward()
    assert st.router.bits.weight.grad.norm()>0
    assert next(st.core.sum_encoder.parameters()).grad.norm()>0
    assert all(torch.isfinite(p.grad).all() for p in st.parameters() if p.grad is not None)

def test_st_regularization_uses_unchanged_relaxed_probabilities():
    relaxed,st=pair();relaxed.train();st.train()
    r=rollout(benchmark(),relaxed,5,3,1001,0);f=r['features'][3]
    relaxed(f,3);st(f,3)
    for x,y in zip(relaxed.regularizers(),st.regularizers()):assert torch.equal(x,y)

def test_st_computes_unselected_proposals_for_router_but_does_not_train_them():
    _,p=pair();p.train();r=rollout(benchmark(),p,5,3,1001,0);f=r['features'][3].detach()
    with torch.no_grad():p.router.bits.weight.zero_();p.router.bits.bias.fill_(.1)
    p.zero_grad(set_to_none=True);calls=[]
    hooks=[e.register_forward_hook(lambda m,i,o,j=j:calls.append(j)) for j,e in enumerate(p.experts)]
    p(f,3).square().sum().backward()
    for h in hooks:h.remove()
    assert calls==[0,1,2,3]
    for j,e in enumerate(p.experts):
        norm=sum(float(v.grad.square().sum()) for v in e.parameters() if v.grad is not None)
        assert (norm>0)==(j==3)
    assert p.router.bits.weight.grad.norm()>0
    p.eval();calls=[];hooks=[e.register_forward_hook(lambda m,i,o,j=j:calls.append(j)) for j,e in enumerate(p.experts)]
    p(f[:1],3)
    for h in hooks:h.remove()
    assert calls==[3]

def test_st_switching_extension_is_available():
    p=MoESBOEDPolicy('switchmoe',benchmark(),5,training_routing_mode='straight_through')
    assert p.config['training_routing_mode']=='straight_through'


def test_unselected_expert_changes_router_surrogate_without_changing_hard_action():
    _,p=pair();p.train()
    f=rollout(benchmark(),p,5,3,1001,0)['features'][3].detach()
    with torch.no_grad():p.router.bits.weight.zero_();p.router.bits.bias.fill_(.1)
    first=p(f,3);g1=torch.autograd.grad(first.sum(),p.router.bits.bias)[0]
    with torch.no_grad():p.experts[0][-1].bias.add_(.5)
    second=p(f,3);g2=torch.autograd.grad(second.sum(),p.router.bits.bias)[0]
    assert torch.equal(first,second)
    assert not torch.allclose(g1,g2,atol=1e-12,rtol=1e-12)

def test_st_checkpoint_and_legacy_default_roundtrip(tmp_path):
    old,st=pair();b=benchmark()
    for name,p in [('legacy',old),('straight_through',st)]:
        path=tmp_path/(name+'.pt');torch.save(p.checkpoint(),path);loaded=load_policy(path,b)
        assert loaded.config.get('training_routing_mode','relaxed')==('relaxed' if name=='legacy' else 'straight_through')
        p.eval();f=rollout(b,p,5,3,1001,0)['features'][3]
        torch.testing.assert_close(p(f,3),loaded(f,3),atol=0,rtol=0)
