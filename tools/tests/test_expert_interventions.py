import copy
import torch
import pytest
from src.research.engine import rollout
from src.research.benchmarks import LocationFinding
from src.research.moe_family import MoESBOEDPolicy,BasePolicy
from src.research.moe_architecture_controls import ArchitectureControl
from tools.diagnose_moe_expert_interventions import FrozenExpertIntervention,statistics

@pytest.mark.parametrize('variant',['moe','resmoe','switchmoe','switchresmoe'])
def test_one_decision_force_changes_only_target_and_preserves_selected_replay(variant):
    torch.manual_seed(101);b=LocationFinding({'location_finding':{'design_bounds':[-4,4]}})
    base=BasePolicy(ArchitectureControl('sum_single',5,64,4.,3.)) if 'resmoe' in variant else None
    p=MoESBOEDPolicy(variant,b,5,base=base).eval()
    with torch.no_grad():
        p.router.bits.weight.zero_();p.router.bits.bias.fill_(1.)
        normal=rollout(b,p,5,2,15,8)
        same=rollout(b,FrozenExpertIntervention(p,2,3).eval(),5,2,15,8)
    torch.testing.assert_close(torch.stack(normal['actions']),torch.stack(same['actions']),rtol=0,atol=0)
    torch.testing.assert_close(normal['info'],same['info'],rtol=0,atol=0)
    with torch.no_grad():changed=rollout(b,FrozenExpertIntervention(p,2,0).eval(),5,2,15,8)
    torch.testing.assert_close(torch.stack(normal['actions'][:2]),torch.stack(changed['actions'][:2]),rtol=0,atol=0)
    assert not torch.allclose(normal['actions'][2],changed['actions'][2])
    assert p.manager is None or p.manager.mode=='learned'


def test_statistics_known_paired_values():
    r=statistics([1.,2.,3.]);assert r['mean']==2.
    assert abs(r['standard_error']-1./3**.5)<1e-12
    assert statistics([])['mean'] is None
