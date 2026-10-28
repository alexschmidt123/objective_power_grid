import pytest
from src.research.compute_budget import ComputeClock,ComputeBudgetError,native_update_plan,native_score_phase_plan,validate_native_score_phase_plan

class Time:
 def __init__(self):self.value=0.
 def __call__(self):return self.value
 def advance(self,x):self.value+=x

def test_open_validation_is_visible_to_external_watchdog(tmp_path):
 t=Time();p=tmp_path/'clock.json';a=ComputeClock(p,100,now=t)
 t.advance(10)
 with a.exclude('validation'):
  t.advance(50)
  b=ComputeClock(p,100,now=t)
  assert a.spent()==b.spent()==10
  assert b.report()['active_exclusion'][1]=='validation'
  with pytest.raises(ComputeBudgetError):
   with b.exclude('heldout_evaluation'):pass
 t.advance(3);assert a.spent()==13
 assert 'active_exclusion' not in a.report()

def test_finalization_reserve_and_post_serialization_eligibility(tmp_path):
 t=Time();c=ComputeClock(tmp_path/'clock.json',100,now=t,finalization_reserve_seconds=10)
 t.advance(85);assert c.can_start(4)
 assert not c.can_start(5)
 t.advance(7);assert c.checkpoint('native_checkpoint_finished')['eligible']
 t.advance(9);assert not c.checkpoint('late_native_checkpoint')['eligible']

def test_native_maximum_uses_time_not_equal_updates_or_EIG():
 p=native_update_plan(14400,35,50000,initialization_seconds=20,finalization_seconds=60)
 assert 0<p['maximum_updates']<50000
 assert p['estimated_charged_seconds']<=14400
 assert p['training_status']=='COMPUTE_LIMITED_NATIVE_IMPLEMENTATION'
 assert native_update_plan(100,1,3,finalization_seconds=1)['maximum_updates']==3

def test_score_phase_proportion_and_scheduler_identity():
 p=native_score_phase_plan(14400,200,.5,5,finalization_seconds=60)
 assert p['score_updates']==100*p['policy_updates']
 assert p['score_reference_steps']==50000 and p['policy_reference_steps']==500
 assert p['estimated_charged_seconds']<=14400
 allocation=validate_native_score_phase_plan(p,50000,500,100014,1024,100000000)
 assert allocation['total']==100014+(p['score_updates']+p['policy_updates'])*1024
 bad=p|{'score_updates':p['score_updates']-1}
 with pytest.raises(ValueError):validate_native_score_phase_plan(bad,50000,500,100014,1024,100000000)
 with pytest.raises(ValueError):validate_native_score_phase_plan(p,50000,500,100014,64,100000000)
 with pytest.raises(ComputeBudgetError):native_score_phase_plan(100,99,1,1,finalization_seconds=0)
