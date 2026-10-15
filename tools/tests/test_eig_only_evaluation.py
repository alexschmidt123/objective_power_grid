import copy
import numpy as np
import pytest
import torch
from src.research.benchmarks import LocationFinding
from src.research.moe_architecture_controls import ArchitectureControl
from src.research.moe_family import MoESBOEDPolicy,BasePolicy
from src.research.runner import evaluate

@pytest.mark.parametrize('method',['resmoe','sum_single'])
def test_eig_only_preserves_common_actions_likelihoods_and_scores(method,tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    torch.manual_seed(101)
    b=LocationFinding({'location_finding':{'design_bounds':[-4,4]},'observation':{'noise_sigma':.5}})
    h=3;base=ArchitectureControl('sum_single',h,64,b.action_scale,b.obs_scale)
    p=MoESBOEDPolicy('resmoe',b,h,base=BasePolicy(base)) if method=='resmoe' else base
    p.eval()
    s=dict(eval_systems=4,eval_contrasts=8,evaluation_population_systems=4,evaluation_batch_size=2,
        evaluation_contrast_chunk=4,posterior_particles=16,posterior_steps=4,posterior_chains=2,
        posterior_proposal_scale=.15,smc_moves=2)
    full=tmp_path/'full';only=tmp_path/'eig_only'
    for path in [full,only]:(path/'eval').mkdir(parents=True)
    a,sa=evaluate(b,copy.deepcopy(p),method,h,s,1001,{},full,paper=True)
    # An inference call in EIG-only mode is a correctness error, not an optional fallback.
    import src.research.runner as runner
    monkeypatch.setattr(runner,'posterior_metrics',lambda *a,**k:pytest.fail('Secondary inference must not execute'))
    c,sc=evaluate(b,copy.deepcopy(p),method,h,s,1001,{},only,paper=True,infer_parameters=False)
    for x,y in zip(a,c):
        for key in ['truth','measurement_noise_standard','actions','observations','spce_by_stage','spce','nmc_upper_estimate','routing']:
            assert x[key]==y[key],key
        assert x['inference'] is not None and y['inference'] is None
    assert np.isfinite(sa['mean_mse']) and sa['parameter_inference_enabled'] is True
    for key in ['mean_mse','mean_normalized_mse','mean_importance_ess','inference_degenerate_fraction']:assert sc[key] is None
    for key in ['mean_spce','spce_mc_standard_error','mean_nmc_upper_estimate','eig_estimate_gap','contrasts','systems']:assert sa[key]==sc[key]
    assert sc['parameter_inference_enabled'] is False
    assert len(list((full/'eval').glob('*posterior_batch*.npz')))==2
    assert not list((only/'eval').glob('*posterior_batch*.npz'))
    assert (full/'eval/contrast_bank_seed1001.npy').read_bytes()==(only/'eval/contrast_bank_seed1001.npy').read_bytes()
