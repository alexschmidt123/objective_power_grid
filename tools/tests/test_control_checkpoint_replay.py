import json
import pytest
import torch
from src.research.benchmarks import LocationFinding
from src.research.engine import rollout
from src.research.matched_sum_single import MatchedSumSingle193
from src.research.moe_architecture_controls import ArchitectureControl
from tools.early_family_eig import load_single_policy

def make_saved(tmp_path,method):
    h=5;b=LocationFinding({'location_finding':{'design_bounds':[-4,4]},'observation':{'noise_sigma':.5}})
    torch.manual_seed(101)
    p=(ArchitectureControl('sum_single',h,64,b.action_scale,b.obs_scale) if method=='sum_single' else MatchedSumSingle193(h,64,b.action_scale,b.obs_scale)).eval()
    (tmp_path/'models').mkdir()
    raw=dict(system=dict(name='location_finding'),location_finding=dict(design_bounds=[-4,4]),observation=dict(noise_sigma=.5))
    (tmp_path/'run_config.json').write_text(json.dumps(dict(horizon=h,benchmark=raw)))
    (tmp_path/'models/sum_single_training.json').write_text(json.dumps(dict(generated_training_trajectories=8,training_seconds=1.)))
    path=tmp_path/'models'/f'{method}.pt';torch.save(dict(kind=method,horizon=h,hidden=64,state_dict=p.state_dict()),path)
    return b,p,path

@pytest.mark.parametrize('method',['sum_single','sum_single_193'])
def test_complete_control_checkpoint_replays_policy_and_rollout(method,tmp_path):
    b,p,path=make_saved(tmp_path,method);q=load_single_policy(path,b)
    a=rollout(b,p,5,3,902,8);c=rollout(b,q,5,3,902,8)
    for key in ['actions','observations','latents']:torch.testing.assert_close(torch.stack(a[key]),torch.stack(c[key]),atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(a['info'],c['info'],atol=1e-11,rtol=1e-11)

@pytest.mark.parametrize('method',['sum_single','sum_single_193'])
def test_control_checkpoint_rejects_different_physical_benchmark(method,tmp_path):
    _,_,path=make_saved(tmp_path,method)
    wrong=LocationFinding({'location_finding':{'design_bounds':[-3,3]},'observation':{'noise_sigma':.5}})
    with pytest.raises(ValueError,match='benchmark'):load_single_policy(path,wrong)


def test_frozen_evaluation_reads_actual_elapsed_budget_episode_count():
    from tools.early_family_eig import validate_training_budget
    t=dict(generated_training_trajectories=2345600,time_budget_seconds=1736.,time_budget_reached=True,training_seconds=1736.1,episode_safety_cap=5000000)
    assert validate_training_budget({},t)==2345600
    with pytest.raises(ValueError,match='not reached'):validate_training_budget({},dict(t,time_budget_reached=False))


def test_frozen_evaluation_rejects_incomplete_episode_budget():
    from tools.early_family_eig import validate_training_budget
    cfg=dict(benchmark=dict(research=dict(paper_protocol=dict(nominal_training_trajectories=1000000))))
    with pytest.raises(ValueError,match='not reached'):validate_training_budget(cfg,dict(generated_training_trajectories=8))
