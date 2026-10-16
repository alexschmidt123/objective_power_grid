import pytest
from tools.journal_campaign import make_config,METHODS,FAMILY,RESIDUAL,PAIRS
from src.research.journal_validation import Plateau,StoppingRule,apply_selection_mode

def test_exact_ten_methods_and_predeclared_six_contrasts():
    assert len(METHODS)==10 and len(set(METHODS))==10
    assert 'sum_single' not in METHODS and len(FAMILY)==4
    assert len(PAIRS)==6

@pytest.mark.parametrize('benchmark',['location','sir'])
def test_effective_router_schedules_and_native_budget_accounting(benchmark):
    raw=make_config(benchmark,1000000);r=raw['research'];p=r['journal_protocol']
    for method in FAMILY:
        effective=r['settings']|p['common_pathwise_settings']|p['method_settings'][method]
        assert effective['router_schedule_episodes']==(500000 if method in RESIDUAL else 1000000)
        assert effective['moe_router_warmup_fraction']==.3 and effective['moe_router_finish_fraction']==.95
    assert p['method_settings']['moe']==p['method_settings']['switchmoe']
    assert p['method_settings']['resmoe']==p['method_settings']['switchresmoe']
    v=p['native_vsoed_profile']
    assert v['initial_replay']+v['new_trajectories']*v['updates']==1000000
    score=p['method_settings']['scorebed'];assert score['scorebed_pretraining_episodes']==692990
    assert 10014+692990+296996==1000000
    assert r['settings']['rl_gradient_steps']==2

def test_isolated_jax_bridge_uses_same_final_selection_rule():
    rule=StoppingRule(minimum_episodes=1,minimum_checkpoints=1,validation_every_episodes=1,patience=1,maximum_episodes=3)
    plateau=Plateau(rule)
    first=apply_selection_mode(plateau.update([10.,10.],1,1.),1,rule,'final_fixed_budget')
    middle=apply_selection_mode(plateau.update([9.,9.],2,2.),2,rule,'final_fixed_budget')
    last=apply_selection_mode(plateau.update([8.,8.],3,3.),3,rule,'final_fixed_budget')
    assert not first['selected'] and not middle['selected'] and last['selected']
    assert middle['status']=='training' and last['status']=='fixed_budget_complete'
