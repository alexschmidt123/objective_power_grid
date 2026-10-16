import numpy as np
import pytest
import torch
from src.research.journal_validation import StoppingRule,Plateau,CommonValidation,ValidationObserver

def rule(**kwargs):return StoppingRule(**(dict(minimum_episodes=20,minimum_checkpoints=3,validation_every_episodes=10,patience=2,maximum_episodes=100,maximum_training_seconds=100)|kwargs))

def test_plateau_requires_minimum_experience_and_checkpoints():
 p=Plateau(rule());x=[1.,2.,3.,4.]
 assert p.update(x,5,1)['status']=='training'
 assert p.update(x,10,2)['status']=='training'
 assert p.update(x,20,3)['status']=='converged'

def test_small_noisy_mean_gain_does_not_reset_patience():
 p=Plateau(rule());p.update([1,2,3,4],10,1)
 r=p.update([0,3.1,2,5.1],20,2)
 assert r['paired_improvement_from_anchor']==pytest.approx(.05)
 assert r['improvement_threshold']>.05 and r['patience_used']==1

def test_meaningful_paired_gain_resets_anchor_and_patience():
 p=Plateau(rule());p.update([1,2,3,4],10,1)
 r=p.update([1.2,2.2,3.2,4.2],20,2)
 assert r['patience_used']==0 and r['selected']
 assert p.update([1.2,2.2,3.2,4.2],30,3)['status']=='training'

def test_safety_cap_is_not_called_convergence():
 p=Plateau(rule(maximum_episodes=20));p.update([1,2,3,4],10,1)
 assert p.update([2,3,4,5],20,2)['status']=='budget_limited'

def test_reserved_seed_rejected_before_any_draw():
 with pytest.raises(ValueError,match='Reserved'):CommonValidation({},5,8831001,4,8)

def test_scoring_nonleaf_switch_state_never_changes_training_model_or_rng(tmp_path):
 from src.research.benchmarks import LocationFinding
 from src.research.engine import rollout
 from src.research.moe_family import MoESBOEDPolicy
 raw=dict(system=dict(name='location_finding'),location_finding=dict(design_bounds=[-4,4]),observation=dict(noise_sigma=.5))
 b=LocationFinding(raw);p=MoESBOEDPolicy('switchmoe',b,3).train()
 r=rollout(b,p,3,4,901,8);r['info'][:,-1].mean().backward()
 old={k:v.clone() for k,v in p.state_dict().items()};state=p.manager.state;rng=torch.get_rng_state()
 v=CommonValidation(raw,3,771001,4,8,2,4);a,c=v(p);a2,c2=v(p)
 assert np.array_equal(a,a2) and c['paired_draws_sha256']==c2['paired_draws_sha256']
 assert p.training and p.manager.state is state
 assert torch.equal(rng,torch.get_rng_state())
 for k,x in old.items():assert torch.equal(x,p.state_dict()[k])

def test_nonfinite_or_unpaired_validation_cannot_stop():
 p=Plateau(rule())
 with pytest.raises(ValueError):p.update([1,float('nan')],10,1)
 p.update([1,2,3],10,1)
 with pytest.raises(ValueError,match='population'):p.update([1,2],20,2)

def test_rl_validation_preserves_native_stochastic_deployment(tmp_path):
 from src.research.benchmarks import LocationFinding
 from src.research.research_redq import RLPolicy
 from src.research.runner import evaluate
 raw=dict(system=dict(name='location_finding'),location_finding=dict(design_bounds=[-4,4]),observation=dict(noise_sigma=.5))
 torch.manual_seed(201);b=LocationFinding(raw);p=RLPolicy(b,3).eval()
 validator=CommonValidation(raw,3,771001,4,8,2,4);scores,cost=validator(p)
 (tmp_path/'eval').mkdir()
 settings=dict(eval_systems=4,evaluation_population_systems=4,eval_contrasts=8,evaluation_batch_size=2,evaluation_contrast_chunk=4)
 rows,summary=evaluate(b,p,'rl_sboed',3,settings,771001,{},tmp_path,paper=True,infer_parameters=False)
 np.testing.assert_allclose(scores,[r['spce'] for r in rows],atol=1e-12,rtol=1e-12)
 assert cost['mode']=='native_stochastic_policy'
