import pytest
from src.research.compute_budget import primary_reference_schedule,spce_update_plan,ComputeClock,classify_stop

def config():
 return dict(batch_size=2000,contrasts=2000,updates=50000,reference_optimizer_updates=50000,reference_training_episodes=100000000,reference_schedule_episodes=100000000,spce_trajectory_budget_unit='primary_episodes')

def test_native_reference_dimensions_and_no_legacy_flag():
 s=config();cap=primary_reference_schedule(s,1000000000)
 assert cap==100000000
 assert 'primary_episode_budget' not in s
 assert spce_update_plan(s,cap)==(50000,'primary_episodes')

def test_confusing_updates_with_episodes_rejected():
 s=config();s['reference_training_episodes']=50000
 with pytest.raises(ValueError):primary_reference_schedule(s,1000000000)

def test_missing_explicit_unit_fails_closed():
 s=config();del s['spce_trajectory_budget_unit']
 with pytest.raises(ValueError):primary_reference_schedule(s,1000000000)

def test_legacy_failure_reproduced_exactly():
 s=config();del s['spce_trajectory_budget_unit']
 assert spce_update_plan(s,100000000)[0]==25
 assert 100000000//(2000+1)==49975

def test_safety_guard_is_separate_and_stop_reason_distinct():
 s=config();assert primary_reference_schedule(s,60000)==60000
 assert spce_update_plan(s,60000)[0]==30
 assert classify_stop('training',60000,100000000,60000)=='episode_safety_cap_exhaustion'
 assert classify_stop('training',100000000,100000000,1000000000)=='reference_schedule_completion'
 assert classify_stop('compute_budget_complete',60000,100000000,1000000000)=='compute_budget_exhaustion'

def test_four_hour_limit_and_latest_eligible_checkpoint(tmp_path):
 t=[0.];c=ComputeClock(tmp_path/'clock.json',14400,now=lambda:t[0])
 t[0]=100;c.checkpoint('update30')
 t[0]=14399;c.checkpoint('latest')
 t[0]=14401;assert not c.can_start();assert not c.checkpoint('overshoot')['eligible']
 assert c.selected()['identifier']=='latest'
