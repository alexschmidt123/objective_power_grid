import copy
import io
import pytest
import torch
from src.research.benchmarks import LocationFinding,SIR
from src.research.engine import rollout
from src.research.moe_family import (MoESBOEDPolicy,BasePolicy,canonical_variant,component_groups,
    measured_parameters,load_policy,HistoryEncoder,EXPERT_IDS)
from src.research.moe_architecture_controls import ArchitectureControl
from src.research.sparse_moe import SparseMoESBOED,stabilize_gradients
from src.research.strategy_moe import StrategyMoEPolicy


def benchmark():return LocationFinding({'location_finding':{'design_bounds':[-4,4]},'observation':{'noise_sigma':.5}})
def base(h=5):return BasePolicy(ArchitectureControl('sum_single',h,64,4.,3.))
def model(v,h=5,**kw):
    torch.manual_seed(101)
    return MoESBOEDPolicy(v,benchmark(),h,base=base(h) if 'resmoe' in v else None,**kw)
def run(p,contrasts=8):return rollout(benchmark(),p,p.horizon,3,901,contrasts)
def equal(a,b):
    for key in ['actions','observations','latents']:
        torch.testing.assert_close(torch.stack(a[key]),torch.stack(b[key]),atol=1e-12,rtol=1e-12)
    if a['info'] is not None:torch.testing.assert_close(a['info'],b['info'],atol=1e-11,rtol=1e-11)

@pytest.mark.parametrize('v',['moe','resmoe','switchmoe','switchresmoe'])
def test_variants_rollouts_gradients_and_bounds(v):
    p=model(v).train();r=run(p);loss=-r['info'][:,-1].mean();loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(a).all() and (a.abs()<=4).all() for a in r['actions'])
    groups=component_groups(p);stabilize_gradients(p,groups,'component',5.)
    for name,params in groups.items():
        grads=[q.grad for q in params if q.requires_grad and q.grad is not None]
        if name=='base':assert not grads
        else:
            assert grads,name
            assert all(torch.isfinite(g).all() for g in grads)
            # DSelect mixing has k=1 and hence zero mixing gradients intrinsically;
            # the selector-bit weights must still receive informative gradients.
            assert sum(float(g.square().sum()) for g in grads)>0,name

@pytest.mark.parametrize('v',['resmoe','switchresmoe'])
def test_disabled_residual_exact_base(v):
    p=model(v,residual_scale=0.).eval();equal(run(p),run(p.core.base.policy.eval()))
    assert not any(q.requires_grad for q in p.core.base.parameters())

@pytest.mark.parametrize('relaxed',[False,True])
def test_switch_moe_original_equivalence(relaxed):
    torch.manual_seed(5);old=SparseMoESBOED(2,1,5,64,k=1,first_action=[0.,0.],history_encoder_revision=2,action_scale=4.,obs_scale=3.)
    new=StrategyMoEPolicy(5,reference=copy.deepcopy(old),mode='always_reroute')
    p=MoESBOEDPolicy('switchmoe',benchmark(),5,core=new)
    old.train(relaxed);p.train(relaxed)
    equal(run(old),run(p))

@pytest.mark.parametrize('relaxed',[False,True])
def test_switch_residual_equivalence(relaxed):
    a=model('resmoe');b=model('switchresmoe',routing_mode='always_reroute')
    state={k:v for k,v in a.state_dict().items()}
    b.load_state_dict(state,strict=False)
    a.train(relaxed);b.train(relaxed);equal(run(a),run(b))

@pytest.mark.parametrize('v',['moe','resmoe','switchmoe','switchresmoe'])
def test_single_expert_hard_inference_and_permutation(v):
    p=model(v).eval();counts=[0]*4
    handles=[]
    for j,e in enumerate(p.experts):
        def hook(m,i,o,j=j):counts[j]+=len(i[0])
        handles.append(e.register_forward_hook(hook))
    r=run(p,0)
    for h in handles:h.remove()
    assert sum(counts)==3*4
    f=r['features'][3];slots=f[:,1:].reshape(3,5,4).clone();slots[:,:3]=slots[:,[2,0,1]]
    perm=torch.cat((f[:,:1],slots.flatten(1)),1)
    torch.testing.assert_close(p.context(f),p.context(perm),atol=1e-12,rtol=1e-12)
    if hasattr(p.core,'base'):
        torch.testing.assert_close(p.core.base(f,3),p.core.base(perm,3),atol=1e-12,rtol=1e-12)
    assert p.last_route['strategy']['reroute_requests'].shape==(3,)

@pytest.mark.parametrize('v',['moe','resmoe','switchmoe','switchresmoe'])
def test_checkpoint_roundtrip(v,tmp_path):
    p=model(v).set_routing();expected=run(p)
    path=tmp_path/'policy.pt';torch.save(p.checkpoint(),path)
    q=load_policy(path,benchmark());equal(expected,run(q))

@pytest.mark.parametrize('v',['moe','resmoe','switchmoe','switchresmoe'])
def test_sir_dimension_aware_valid_rollout(v):
    b=SIR({'sir_ode':{'integration_steps_per_stage':32,'t_end':10.}})
    # Explicit shape-only fixture: not a claimed pretrained SIR performance model.
    h=3
    if 'resmoe' in v:
        from src.research.moe_family import _GenericSumCore
        core=_GenericSumCore(b,h)
        # A generic full-policy base for interface validation; no location checkpoint.
        bp=BasePolicy(core)
    else:bp=None
    p=MoESBOEDPolicy(v,b,h,base=bp)
    r=rollout(b,p,h,2,7,2);loss=-r['info'][:,-1].mean();loss.backward()
    assert torch.isfinite(loss)
    a=torch.stack(r['actions']);assert (a[1:]-a[:-1]>=b.gap-1e-10).all()
    assert (a>0).all() and (a<=10).all()


def test_aliases_and_parameter_measurement():
    assert canonical_variant('Strategy-MoE')=='switchmoe'
    assert canonical_variant('strategy_moe_sboed')=='switchmoe'
    assert canonical_variant('MoE-sBOED')=='moe'
    p=model('moe').eval();f=torch.zeros(1,21,dtype=torch.float64)
    n=measured_parameters(p,f,1)
    assert n['total']==72544 and n['active']==22424
    assert EXPERT_IDS==('E1','E2','E3','E4')


def test_request_distinct_from_change():
    p=model('switchmoe',routing_mode='always_reroute').eval()
    with torch.no_grad():
        p.router.bits.weight.zero_();p.router.bits.bias.fill_(1.)
    run(p,0);s=p.last_route['strategy']
    assert (s['reroute_requests']==4).all() and (s['strategy_changes']==0).all()
    assert (s['consecutive_duration']==4).all()


def test_wrong_benchmark_checkpoint_rejected(tmp_path):
    p=model('moe');path=tmp_path/'p.pt';torch.save(p.checkpoint(),path)
    changed=LocationFinding({'location_finding':{'design_bounds':[-3,3]}})
    with pytest.raises(ValueError,match='benchmark'):load_policy(path,changed)


def test_optional_real_historical_checkpoints(tmp_path):
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    checkpoint=root/'experiments/10142026/10142026_232132_strategy_moe_pilot/pilot/models/strategy_moe_sboed.pt'
    if not checkpoint.exists():pytest.skip('Historical model binaries are preserved locally, not committed')
    from tools.validate_family_checkpoints import validate
    validate(root,tmp_path)
