import pytest
from src.research.compute_budget import ComputeClock,ComputeBudgetError,score_phase_plan
class FakeTime:
    value=0.
    def __call__(self):return self.value
    def advance(self,t):self.value+=t

def test_all_native_training_phases_and_shared_base_are_charged(tmp_path):
    t=FakeTime();c=ComputeClock(tmp_path/"clock.json",100,credit_seconds=7,now=t)
    phases={"setup_and_compilation":3,"normalization":4,"score_pretraining":20,"policy":10,"contrast_simulation":6,"posterior_learning":8,"RL_replay":9,"checkpoint_IO":2}
    for name,seconds in phases.items():
        t.advance(seconds);c.count("native_units",1,name)
    assert c.spent()==7+sum(phases.values())
    before=c.spent()
    with c.exclude("validation"):t.advance(13)
    assert c.spent()==before
    with pytest.raises(ComputeBudgetError):
        with c.exclude("score_pretraining"):pass
    report=c.report();assert set(report["components"])==set(phases)
    assert report["counts"]["native_units"]==len(phases)

def test_checkpoint_selection_uses_cost_not_EIG_and_records_overshoot(tmp_path):
    t=FakeTime();c=ComputeClock(tmp_path/"clock.json",10,now=t)
    t.advance(3);c.checkpoint("early_high_EIG")
    t.advance(6);c.checkpoint("late_lower_EIG")
    assert not c.can_start(2)
    t.advance(2);assert not c.checkpoint("over_budget")["eligible"]
    assert c.selected()["identifier"]=="late_lower_EIG"
    assert c.report()["charged_seconds"]==11 # overrun does not disappear

def test_cross_process_clock_identity(tmp_path):
    t=FakeTime();path=tmp_path/"clock.json";a=ComputeClock(path,20,now=t);t.advance(5)
    b=ComputeClock(path,20,now=t)
    with b.exclude("validation"):t.advance(2)
    t.advance(3);assert a.spent()==b.spent()==8
    with pytest.raises(ComputeBudgetError):ComputeClock(path,30,now=t)

def test_native_phase_ratio_and_infeasible_upfront_cost():
    p=score_phase_plan(100,10,1,2,19)
    assert p["score_updates"]==19*p["policy_updates"]
    assert p["estimated_seconds"]<=100
    with pytest.raises(ComputeBudgetError):score_phase_plan(10,9,1,2)

def test_real_observer_excludes_validation_selects_latest_and_stops(tmp_path,monkeypatch):
    import numpy as np,torch
    from src.research.journal_validation import StoppingRule,ValidationObserver
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    t=FakeTime();clock=ComputeClock(tmp_path/"clock.json",12,credit_seconds=2,now=t)
    class Validator:
        seed=1001
        def __call__(self,p):
            t.advance(5)
            return np.array([float(p.weight.item())]*2),dict(seconds=5.)
    rule=StoppingRule(minimum_episodes=1,maximum_episodes=1000,minimum_checkpoints=2,validation_every_episodes=100,patience=1)
    p=torch.nn.Linear(1,1,bias=False);o=ValidationObserver(Validator(),rule,tmp_path,"dad",selection_mode="final_compute_budget",compute_clock=clock,compute_checkpoint_seconds=1.)
    t.advance(3)
    with torch.no_grad():p.weight.fill_(10.)
    assert not o.check(p,4,1)
    t.advance(3)
    with torch.no_grad():p.weight.fill_(1.) # Lower EIG must still be selected as latest.
    assert not o.check(p,8,2)
    t.advance(2)
    assert not o.allow_next_unit(p,8,2)
    report=o.finish(p)
    assert report["selected_checkpoint"]["update"]==2
    assert p.weight.item()==1.
    assert clock.spent()==10 # Includes base2 + all native3+3+2, excludes validation15.
    assert report["checkpoint_selection"]=="latest_budget_eligible_checkpoint"

def test_finished_training_does_not_charge_final_evaluation(tmp_path):
    t=FakeTime();c=ComputeClock(tmp_path/"clock.json",100,now=t);t.advance(10)
    assert c.stop()["charged_seconds"]==10
    t.advance(1000)
    assert c.report()["charged_seconds"]==10
    assert not c.can_start()

def test_all_baseline_entrypoints_have_budget_hooks():
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]/"src/research"
    for name in ["training.py","family_training.py","native_rl_reference.py","sir_native_rl.py"]:
        assert "observer.allow_next_unit" in (root/name).read_text()
    for name in ["scorebed_worker.py","sir_scorebed_worker.py"]:
        code=(root/name).read_text();assert "compute_clock_from_plan" in code and "while score_spent<score_count" in code and "compute_clock.can_start" in code
    assert "observer.check(candidate" in (root/"upstream_vsoed.py").read_text()

def test_hardware_time_units_cannot_be_silently_changed(tmp_path):
    t=FakeTime();p=tmp_path/"clock.json";ComputeClock(p,10,hardware="RTX4090",now=t)
    with pytest.raises(ComputeBudgetError):ComputeClock(p,10,hardware="RTX5080",now=t)


def test_reference_schedule_not_legacy_one_million():
    from src.research.compute_budget import effective_episode_limit,classify_stop
    assert effective_episode_limit(1000000000,100000000)==100000000
    assert effective_episode_limit(1000000000,2058014)>1000000
    assert classify_stop('planned_schedule_complete_before_compute_cap',2058014,2058014,1000000000)=='reference_schedule_completion'
    assert classify_stop('compute_budget_complete',1200000,100000000,1000000000)=='compute_budget_exhaustion'
    assert classify_stop('training',1000000,100000000,1000000)=='episode_safety_cap_exhaustion'
