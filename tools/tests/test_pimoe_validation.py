import copy,math
import torch
from src.research.pimoe import *
from src.research.pimoe_validation import intervention,js_distance,distribution_mean_variance
from src.research.benchmarks import LocationFinding

def fixture():
 torch.manual_seed(101);b=LocationFinding({'location_finding':{'design_bounds':[-4,4]}});belief=AdaptiveBelief(b,128,1008);a=torch.tensor([[-1.,0.],[0.,0.],[1.,1.]],dtype=DTYPE);return PredictiveMoE(),state_features(b,belief,a)

def test_expert_removal_and_uniform_routing_are_normalized():
 model,f=fixture();before=copy.deepcopy(model.state_dict())
 for mode in ['full','uniform_mixture_weights','uniform_membership']+['expert_'+str(i) for i in range(4)]+['remove_'+str(i) for i in range(4)]:
  z=intervention(model,f,mode);torch.testing.assert_close(z['weights'].sum(-1),torch.ones(3,dtype=DTYPE),atol=1e-12,rtol=1e-12);assert (z['weights']>=0).all();assert torch.isfinite(PredictiveMoE.entropy(z)).all()
 z=intervention(model,f,'uniform_membership');torch.testing.assert_close(z['mass'],torch.full((3,4),.25,dtype=DTYPE))
 assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())

def test_JS_identical_symmetric_and_bounded():
 model,f=fixture();a=intervention(model,f,'expert_0');b=intervention(model,f,'expert_1')
 torch.testing.assert_close(js_distance(a,a),torch.zeros(3,dtype=DTYPE),atol=1e-12,rtol=0)
 torch.testing.assert_close(js_distance(a,b),js_distance(b,a),atol=1e-12,rtol=0)
 j=js_distance(a,b);assert (j>=-1e-8).all() and (j<=math.log(2)+1e-7).all()

def test_component_means_variances_match_moments():
 model,f=fixture();z=intervention(model,f,'full');mean,var=distribution_mean_variance(z);assert torch.isfinite(mean).all() and (var>=.25-1e-12).all()

def test_expert_index_permutation_does_not_change_full_density():
 model,f=fixture();q=copy.deepcopy(model);perm=[2,0,3,1]
 with torch.no_grad():q.router[-1].weight.copy_(model.router[-1].weight[perm]);q.router[-1].bias.copy_(model.router[-1].bias[perm])
 q.experts=torch.nn.ModuleList([copy.deepcopy(model.experts[e]) for e in perm]);z=model(**f);zz=q(**f);y=torch.linspace(-5,8,100,dtype=DTYPE)[None].expand(3,-1)
 torch.testing.assert_close(PredictiveMoE.log_prob(z,y),PredictiveMoE.log_prob(zz,y),atol=1e-12,rtol=1e-12)

def test_negligible_removal_mass_is_reported():
 from src.research.pimoe_validation import normalized_part
 z=dict(weights=torch.tensor([[1.,0.]],dtype=DTYPE),means=torch.zeros(1,2,dtype=DTYPE),scales=torch.ones(1,2,dtype=DTYPE))
 import pytest
 with pytest.raises(ValueError):normalized_part(z,[1])
