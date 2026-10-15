import pytest
from tools.report_equal_budget_dad import particle_cost,stats,paired

def test_cost_includes_truth_and_every_experiment_stage():
    c=particle_cost(1000000)
    assert c['contrast_trajectories']==1024000000
    assert c['truth_inclusive_likelihood_trajectories']==1025000000
    assert c['particle_stage_forward_likelihood_evaluations']==30750000000

def test_paired_interval_uses_case_differences():
    a=dict(case_sha256='c',bank_sha256='b',scores=[101,202,303])
    b=dict(case_sha256='c',bank_sha256='b',scores=[100,200,300])
    x=paired(a,b)
    assert x['mean']==2
    assert x['mc_standard_error']==pytest.approx(1/3**.5)

@pytest.mark.parametrize('key',['case_sha256','bank_sha256'])
def test_different_cases_or_bank_cannot_be_called_paired(key):
    a=dict(case_sha256='c',bank_sha256='b',scores=[2,3]);b=dict(a);b[key]='different'
    with pytest.raises(ValueError,match='Unpaired'):paired(a,b)

def test_nonfinite_scores_not_silently_filtered():
    with pytest.raises(ValueError):stats([1,float('nan')])
