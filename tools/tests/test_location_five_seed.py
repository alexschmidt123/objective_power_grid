import ast
from pathlib import Path
import pytest
from src.research.location_baseline_campaign import config, SEEDS, FINAL_SEED, VALIDATION_SEED
from src.research.compute_budget import primary_reference_schedule,spce_update_plan
from src.research.benchmarks import make_benchmark

def test_five_fresh_seeds_and_disjoint_evaluation():
    assert SEEDS==(101,102,103,104,105)
    assert FINAL_SEED!=VALIDATION_SEED
    assert FINAL_SEED not in {1001,1002,8831001}

@pytest.mark.parametrize('method',['dad','fixed','rl_sboed','vsoed','scorebed'])
def test_no_inherited_pilot_budget_or_arbitrary_update_target(method):
    raw=config(method,14400);r=raw['research'];p=r['journal_protocol']
    assert 'training_episode_cap' not in r and 'nominal_training_trajectories' not in r['paper_protocol']
    assert p['compute_budget']['seconds']==14400
    assert p['checkpoint_selection_mode']=='final_compute_budget'
    assert p['native_vsoed_profile']['updates']==10001
    assert set(p['method_settings'])=={'dad','fixed','rl_sboed','vsoed','scorebed'}
    b=make_benchmark(raw)
    assert b.name=='location_finding' and b.sigma==.5 and b.design_bounds==[-4.,4.]

@pytest.mark.parametrize('method',['dad','fixed'])
def test_corrected_primary_episode_count_and_native_recipe(method):
    raw=config(method,14400);r=raw['research'];s=r['settings']|r['journal_protocol']['method_settings'][method]
    cap=primary_reference_schedule(s,1000000000)
    assert cap==(100000000 if method=='dad' else 3200000) and spce_update_plan(s,cap)==(50000,'primary_episodes')
    assert s['batch_size']==(2000 if method=='dad' else 64)
    assert s['contrasts']==(2000 if method=='dad' else 1024)
    assert s['spce_gradient_clip']==(None if method=='dad' else 5.)
    assert s['learning_rate']==5e-5

def test_rl_native_reference_and_scorebed_all_stages():
    p=config('scorebed',14400)['research']['journal_protocol'];s=p['method_settings']['scorebed']
    assert p['method_settings']['rl_sboed']['rl_native_contrasts']==100000
    assert p['method_settings']['rl_sboed']['reference_schedule_episodes']==20001*100
    assert s['scorebed_restarts']==1 and s['scorebed_reference_sampler']
    assert s['score_schedule_reference_steps']==190000
    assert s['policy_schedule_reference_steps']==10000
    plan=s['scorebed_compute_phase_plan']
    assert plan['score_updates']==19*plan['policy_updates']
    assert plan['selection_used_EIG'] is False
    assert s['reference_schedule_episodes']==10014+(177365+9335)*1024

def test_native_worker_support_is_opt_in_and_targets_ema():
    root=Path(__file__).resolve().parents[2]
    source=(root/'src/research/scorebed_worker.py').read_text();ast.parse(source)
    assert "native=bool(journal_plan and journal_plan.get('scorebed_native_reference_schedule'))" in source
    assert 'use_ema=args.reference_sampler or native' in source
    assert 'Native ScoreBED full-stage accounting mismatch' in source
