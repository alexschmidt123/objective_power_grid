import copy,math,json
import numpy as np
import torch
from src.research.pimoe import *
from src.research.pimoe_ic import *
from src.research.benchmarks import LocationFinding

CFG=dict(conditional_ESS_shrinkage=16.,conditional_loss_fraction=.5,information_reference_SE_scale=.05,loss_info_weight=1.,loss_rank_weight=.1,loss_specialization_weight=0.)
def fixture():
 torch.manual_seed(101);b=LocationFinding({'location_finding':{'design_bounds':[-4,4]}});belief=AdaptiveBelief(b,256,1010);actions=torch.tensor([[-1.,0.],[0.,0.],[1.,1.]],dtype=DTYPE);f=state_features(b,belief,actions);g=rng(1011);indices=torch.multinomial(f['weights'][None].expand(3,-1),128,True,generator=g);y=f['means'].gather(1,indices)+.5*torch.randn(3,128,generator=g,dtype=DTYPE);ref=direct_reference(f['means'],f['weights'],.5,1012,512)
 return b,belief,InformationCalibratedPredictive(),dict(features=f,sample_indices=indices,y=y,reference=ref['EIG'],reference_se=ref['SE'])

def test_conditional_responsibilities_and_mixture_normalization():
 b,belief,m,item=fixture();z=m(**item['features']);torch.testing.assert_close(z['weights'].sum(-1),torch.ones(3,dtype=DTYPE));torch.testing.assert_close(z['mass'],(z['membership']*belief.weights[None,:,None]).sum(1));assert (z['weights']>=0).all()
 logq=conditional_log_prob(z,item['y']);combined=torch.logsumexp(logq+z['mass'][:,None].log(),-1);torch.testing.assert_close(combined,m.log_prob(z,item['y']),atol=1e-12,rtol=1e-12)
 grid=torch.linspace(-20,25,10000,dtype=DTYPE)[None].expand(3,-1);torch.testing.assert_close(torch.trapezoid(m.log_prob(z,grid).exp(),grid),torch.ones(3,dtype=DTYPE),atol=1e-5,rtol=0)

def test_cached_128_entropy_values_and_gradients_match_reference():
 _,_,m,item=fixture();z=m(**item['features']);a=m.entropy(z);b=PredictiveMoE.entropy(z,nodes=128);torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12);p=list(m.parameters());g1=torch.autograd.grad(a.sum(),p,retain_graph=True);g2=torch.autograd.grad(b.sum(),p)
 for x,y in zip(g1,g2):torch.testing.assert_close(x,y,atol=1e-11,rtol=1e-9)

def test_joint_loss_gradients_and_update():
 _,_,m,item=fixture();old=copy.deepcopy(m.state_dict());loss,row,_=information_calibrated_loss(m,item,CFG);assert torch.isfinite(loss);loss.backward()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters());assert sum(p.grad.abs().sum() for p in m.router.parameters())>0
 for expert in m.experts:assert sum(p.grad.abs().sum() for p in expert.parameters())>0
 opt=torch.optim.Adam(m.parameters(),lr=.001);opt.step();assert any(not torch.equal(v,old[k]) for k,v in m.state_dict().items())

def test_gaussian_entropy_information_and_tie_mask():
 d=dict(weights=torch.ones(1,1,dtype=DTYPE),means=torch.zeros(1,1,dtype=DTYPE),scales=torch.full((1,1),math.sqrt(1.25),dtype=DTYPE));h=InformationCalibratedPredictive.entropy(d);assert abs(float(h)-.5*math.log(2*math.pi*math.e*.25)-.5*math.log(5))<1e-12
 _,_,m,item=fixture();item['reference']=torch.zeros(3,dtype=DTYPE);item['reference_se']=torch.ones(3,dtype=DTYPE)*100;_,row,_=information_calibrated_loss(m,item,CFG);assert row['resolved_pairs']==0 and row['ranking_loss']==0

def test_source_symmetry_and_checkpoint_identity(tmp_path):
 b,p,m,item=fixture();f=item['features'];z=m(**f);p.theta=p.theta[:,[2,3,0,1]];swapped=m(**state_features(b,p,f['actions']));torch.testing.assert_close(z['means'],swapped['means'],atol=0,rtol=0)
 path=tmp_path/'ic.pt';torch.save(dict(model='PIMoE-sBOED-IC',training_numerics_revision=2,experts=4,width=48,components=3,state_dict=m.state_dict()),path);q=load_ic_checkpoint(path);torch.testing.assert_close(q(**f)['means'],z['means'],atol=0,rtol=0)
 assert sum(p.numel() for p in q.parameters())==13800

def test_budget_persistence_no_reset(tmp_path):
 p=tmp_path/'budget.json';p.write_text(json.dumps(dict(experimental_runtime_allowance_seconds=1.,optimization_allowance_seconds=.5,charged_experimental_seconds=0.,charged_optimization_seconds=0.,runtime_intervals=[])));x=ExperimentalLedger(p)
 with x.charge('test'):pass
 spent=x.read()['charged_experimental_seconds'];y=ExperimentalLedger(p);assert y.read()['charged_experimental_seconds']==spent;y.record_optimization(.1);assert y.read()['charged_optimization_seconds']==.1


def test_conditional_ESS_is_scale_invariant_even_tiny_mass():
 _,_,m,item=fixture()
 with torch.no_grad():m.router[-1].weight.zero_();m.router[-1].bias.copy_(torch.tensor([0.,-100.,-500.,-800.],dtype=DTYPE))
 loss,row,z=information_calibrated_loss(m,item,CFG)
 for value in row['conditional_effective_samples']:assert abs(value-128)<1e-8
 torch.testing.assert_close(z['posterior_region_weights'].sum(1),torch.ones(3,4,dtype=DTYPE),atol=1e-12,rtol=0)
 torch.testing.assert_close(z['expert_component_weights'].sum(-1),torch.ones(3,4,dtype=DTYPE),atol=1e-12,rtol=0)
 assert torch.isfinite(loss);loss.backward()
 for expert in m.experts:assert sum(p.grad.abs().sum() for p in expert.parameters())>0

def test_stable_regions_match_original_at_resolved_mass():
 _,_,m,item=fixture();old=PredictiveMoE();old.load_state_dict(m.state_dict());a=m(**item['features']);b=old(**item['features'])
 for key in ['weights','means','scales','membership','mass','region','regional_variance']:torch.testing.assert_close(a[key],b[key],rtol=1e-10,atol=1e-11)

def test_old_buggy_checkpoint_cannot_be_evaluated(tmp_path):
 _,_,m,item=fixture();p=tmp_path/'rejected.pt';torch.save(dict(model='PIMoE-sBOED-IC',experts=4,width=48,components=3,state_dict=m.state_dict()),p)
 import pytest
 with pytest.raises(ValueError,match='Rejected provisional'):load_ic_checkpoint(p)

def test_previous_ESS_formula_demonstrably_invalid():
 r=torch.full((128,),1e-12,dtype=DTYPE);bad=r.sum().square()/r.square().sum().clamp_min(1e-12);stable=(r.log().softmax(0).square().sum()).reciprocal()
 assert bad<1 and abs(float(stable)-128)<1e-10
