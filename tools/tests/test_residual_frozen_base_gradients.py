import torch
import pytest
from src.research.moe_family import BasePolicy,MoESBOEDPolicy
from src.research.moe_architecture_controls import ArchitectureControl
from src.research.benchmarks import LocationFinding
from src.research.engine import rollout


def test_frozen_base_retains_derivatives_with_respect_to_observed_history():
 torch.manual_seed(101);p=BasePolicy(ArchitectureControl('sum_single',5,64,4.,3.))
 f=torch.zeros(4,21,dtype=torch.float64);f[:,0]=.2;f[:,1:4]=torch.randn(4,3,dtype=torch.float64);f[:,4]=1;f.requires_grad_()
 p(f,1).sum().backward()
 assert f.grad[:,1:4].norm()>0
 assert all(not q.requires_grad and q.grad is None for q in p.parameters())

@pytest.mark.parametrize('variant',['resmoe','switchresmoe'])
def test_later_designs_differentiate_through_earlier_residual_measurements(variant):
 torch.manual_seed(101);b=LocationFinding({'location_finding':{'design_bounds':[-4,4]}})
 p=MoESBOEDPolicy(variant,b,5,base=BasePolicy(ArchitectureControl('sum_single',5,64,4.,3.))).train()
 r=rollout(b,p,5,4,101,8)
 gradient=torch.autograd.grad(r['latents'][3].sum(),r['latents'][1],retain_graph=True)[0]
 assert torch.isfinite(gradient).all() and gradient.norm()>0
 assert all(q.grad is None for q in p.core.base.parameters())
