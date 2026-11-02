import copy,importlib.util
from pathlib import Path
import pytest,torch
from src.research.deepseek_sboed import DeepSeekPolicy
from src.research.benchmarks import LocationFinding
from src.research.moe_v5_credit import *

def test_swap_cardinality_and_replacement_expert_gradient_exposure():
    torch.manual_seed(91);p=DeepSeekPolicy();z=torch.randn(1,16,dtype=torch.float64)
    ids=torch.tensor([[0,1,2]]);changed,drop,add=swap_one(ids,torch.tensor([2]),torch.tensor([0]))
    assert set(changed[0].tolist())=={0,1,3} and int(add)==3 and int(drop)==2
    route_output(p,z,changed).square().sum().backward()
    assert p.shared.up.weight.grad.abs().sum()>0 and p.routed[3].up.weight.grad.abs().sum()>0
    assert p.routed[2].up.weight.grad is None

def test_confidence_weight_direction_detachment_and_noisy_rejection():
    router=torch.nn.Linear(16,7,bias=False).double();z=torch.ones(1,16,dtype=torch.float64)
    a=torch.ones(4,1,dtype=torch.float64,requires_grad=True)
    loss,meta=confidence_loss(router,z,torch.tensor([0]),torch.tensor([1]),a)
    g,leak=torch.autograd.grad(loss,(router.weight,a),allow_unused=True)
    assert leak is None and g[1].sum()<0 and g[0].sum()>0
    assert meta['weights'][0]>.99
    _,noise=confidence_loss(router,z,torch.tensor([0]),torch.tensor([1]),torch.tensor([[-1.],[1.],[-1.],[1.]],dtype=torch.float64))
    assert noise['weights']==[0.]

def test_same_history_replication_determinism_and_bounded_cost():
    torch.manual_seed(92);p=DeepSeekPolicy()
    make=lambda:LocationFinding({'location_finding':{'design_bounds':[-4,4]}})
    a=replicated_credit(make(),p,597771,batch=1,negatives=8,repeats=4)
    b=replicated_credit(make(),p,597771,batch=1,negatives=8,repeats=4)
    assert a['same_prefix'] and a['work'].trajectories==8
    torch.testing.assert_close(a['advantages'],b['advantages'],rtol=0,atol=0)
    with pytest.raises(ValueError):replicated_credit(make(),p,597771,batch=3,negatives=8,repeats=4)

def test_zero_refinement_is_exact_v4_and_only_router_is_modified():
    root=Path(__file__).resolve().parents[2]
    file=root/'scratch_runs/10312026_054200_moe_sboed_v5_ubuntu/labpc_v4_reference/source/src/research/v4_utility_routing.py'
    spec=importlib.util.spec_from_file_location('legacy_credit_test',file);legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    torch.manual_seed(93);a=DeepSeekPolicy();b=copy.deepcopy(a)
    for p,q in zip(a.parameters(),b.parameters()):p.grad=torch.ones_like(p);q.grad=torch.ones_like(q)
    record=dict(z=torch.randn(2,16,dtype=torch.float64),drop=torch.tensor([0,1]),add=torch.tensor([2,3]),advantage=torch.tensor([.1,-.2],dtype=torch.float64))
    old=legacy.add_credit(a,record);new=refine_router_credit(b,record,legacy,refinement=0)
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
    assert old==new
    record['advantages']=record['advantage'][None].expand(4,-1)
    unchanged={n:p.grad.clone() for n,p in b.named_parameters()}
    refine_router_credit(b,record,legacy,refinement=.1)
    delta=(b.router.weight.grad-unchanged['router.weight']).norm()
    assert delta<=.05*unchanged['router.weight'].norm()*(1+1e-10)
    for name,p in b.named_parameters():
        if name!='router.weight':torch.testing.assert_close(p.grad,unchanged[name],rtol=0,atol=0)
