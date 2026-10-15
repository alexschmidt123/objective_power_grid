import torch
import pytest
from tools.diagnose_router_contribution import HistoryIndependentRouting
from src.research.moe_family import MoESBOEDPolicy,BasePolicy
from src.research.moe_architecture_controls import ArchitectureControl
from src.research.benchmarks import LocationFinding
from src.research.engine import rollout

@pytest.mark.parametrize('variant',['moe','resmoe','switchmoe','switchresmoe'])
def test_constant_routing_preserves_first_and_executes_only_requested_expert(variant):
 torch.manual_seed(10);b=LocationFinding({'location_finding':{'design_bounds':[-4,4]}})
 bp=BasePolicy(ArchitectureControl('sum_single',5,64,4.,3.)) if 'resmoe' in variant else None
 p=MoESBOEDPolicy(variant,b,5,base=bp).eval();counts=[0]*4;hooks=[]
 for j,e in enumerate(p.experts):
  def hook(m,i,o,j=j):counts[j]+=len(i[0])
  hooks.append(e.register_forward_hook(hook))
 prob=torch.tensor([[0.,0.,1.,0.]]*4,dtype=torch.float64)
 with torch.no_grad():r=rollout(b,HistoryIndependentRouting(p,prob).eval(),5,2,101,0)
 for h in hooks:h.remove()
 assert counts==[0,0,8,0]
 assert p.manager is None or p.manager.mode=='learned'
 assert not torch.allclose(r['actions'][1],r['actions'][2])


def test_stage_samples_are_independent_of_batch_partition():
 # Integer case IDs and stage-specific generators make routing randomness
 # independent of batch size and observed data.
 seed=7100102
 def sample(start,n,stage):return [float(torch.rand((),generator=torch.Generator().manual_seed(seed+stage*104729+(start+i)*9709))) for i in range(n)]
 assert sample(0,8,3)==sample(0,4,3)+sample(4,4,3)
