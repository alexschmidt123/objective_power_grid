import copy,math,inspect
from unittest.mock import patch
import torch,pytest
from src.research.sa_moe import *
from src.research.sa_moe_offline import conditional_rollout
from src.research.benchmarks import LocationFinding
from src.research.engine import rollout

def bmark():return LocationFinding({'location_finding':{'design_bounds':[-4,4]}})
def fixture():return torch.randn(5,121,dtype=torch.float64)

def test_deployment_only_selected_expert_and_no_search():
    torch.manual_seed(101);m=SAMoE().eval();f=fixture();f[:,0]=0;f[:,1:]=0;calls=[0]*4
    hooks=[net.register_forward_hook(lambda a,b,c,e=e:calls.__setitem__(e,calls[e]+1)) for e,net in enumerate(m.experts)]
    with patch.object(m,'all_proposals',side_effect=AssertionError('No all-expert action search')),patch('src.research.sa_moe_offline.measure_competence',side_effect=AssertionError('No offline evaluation online')):
        a=m(f[:1],0);again=m(f[:1],0)
    assert sum(calls)==2 and max(calls)==2;torch.testing.assert_close(a,again)
    assert a.abs().max()<=4 and torch.isfinite(a).all()
    for h in hooks:h.remove()
    source=inspect.getsource(__import__('src.research.sa_moe',fromlist=['SAMoE']))
    assert 'pimoe_ic' not in source and 'PIMoEPlanner' not in source

def test_capacity_match_and_all_conditional_expert_gradients():
    m=SAMoE();s=SAMoE(True);a=sum(p.numel() for p in m.parameters());b=sum(p.numel() for p in s.parameters());assert abs(a-b)/a<.005
    f=fixture();loss=sum(m.forced_first(f,e).square().mean() for e in range(4));z=m.context(f);r=torch.ones(5,4,dtype=torch.float64)/4;loss=loss-(r*m.probabilities(z).log()).sum(-1).mean();loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    for net in m.experts:assert sum(p.grad.abs().sum() for p in net.parameters())>0

def test_rare_expert_stability_and_tied_competence():
    q=torch.tensor([[0.,-100.,-500.,-800.]]*16,dtype=torch.float64);se=torch.zeros_like(q);log_r,r=competence_responsibilities(q,se)
    old_r=torch.full((16,),1e-12,dtype=torch.float64);old_ess=old_r.sum().square()/old_r.square().sum().clamp_min(1e-12);assert old_ess<1
    loss,ess=conditional_risk(log_r,torch.ones_like(q));assert torch.isfinite(loss);torch.testing.assert_close(ess,torch.ones(4,dtype=torch.float64)*16)
    q=torch.tensor([[1.,1.001,1.002,1.003]],dtype=torch.float64);_,r=competence_responsibilities(q,torch.ones_like(q));assert (r-.25).abs().max()<.001
    _,r=competence_responsibilities(q,torch.zeros_like(q),torch.tensor([True]));torch.testing.assert_close(r,torch.ones_like(q)/4)

def test_full_horizon_loss_gradients_checkpoint_and_physics(tmp_path):
    torch.manual_seed(101);m=SAMoE();b=bmark();r=rollout(b,m,30,2,1010,32);assert r['info'].shape==(2,30)
    loss=-r['info'][:,-1].mean();loss.backward();assert torch.isfinite(loss)
    path=tmp_path/'m.pt';torch.save(dict(method='SA-MoE-sBOED',single=False,state_dict=m.state_dict()),path);q=load_sa_checkpoint(path)
    m.eval();torch.testing.assert_close(m(r['features'][10],10),q(r['features'][10],10))
    assert all(a.abs().max()<=4 for a in r['actions'])

def test_counterfactual_runs_entire_remaining_horizon_and_crn():
    b=bmark();m=SAMoE().eval();posterior=torch.randn(256,4,dtype=torch.float64);a=[torch.zeros(2,dtype=torch.float64)]*27;y=[torch.tensor(0.,dtype=torch.float64)]*27
    stages=[];h=m.register_forward_hook(lambda mod,args,out:stages.append(args[1]))
    v=conditional_rollout(b,m,a,y,posterior,1,1012,4,64);again=conditional_rollout(b,m,a,y,posterior,1,1012,4,64)
    h.remove();assert stages==[28,29,28,29] and torch.isfinite(v).all();torch.testing.assert_close(v,again)


def test_official_noise_is_preserved_and_never_sampled_from_test_seed():
    b=bmark();m=SAMoE().eval();truth=torch.randn(2,4,dtype=torch.float64);noise=torch.randn(2,30,1,dtype=torch.float64)
    with torch.no_grad():r=rollout(b,m,30,2,10109998,0,truth=truth,noise=noise,track_likelihood=True)
    states=b.initial((2,1))
    for t,(action,y) in enumerate(zip(r['actions'],r['observations'])):
        mean,states=b.forward(truth[:,None],states,action[:,None]);torch.testing.assert_close(y,mean[:,0]+b.sigma*noise[:,t])
