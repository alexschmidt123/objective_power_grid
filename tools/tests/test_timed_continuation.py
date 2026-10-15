from types import SimpleNamespace
import pytest
import torch
from tools.tests.test_control_checkpoint_replay import make_saved
import src.research.moe_architecture_controls as controls

SETTINGS=dict(hidden=64,updates=4,batch_size=2,learning_rate=5e-5,contrasts=8,validate_every=100,moe_gradient_clip_mode='component')

def test_time_stopping_matches_same_completed_episode_updates(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    b,_,base=make_saved(tmp_path,'sum_single')
    ordinary=tmp_path/'ordinary';ordinary.mkdir()
    expected,em=controls.train_control(b,5,SETTINGS,101,ordinary,'sum_single',4,base)
    clock=iter([0.,.25,.25,.5,.5,.5,.5])
    monkeypatch.setattr(controls,'time',SimpleNamespace(perf_counter=lambda:next(clock)))
    timed=tmp_path/'timed';timed.mkdir()
    actual,am=controls.train_control(b,5,SETTINGS,101,timed,'sum_single',8,base,time_budget_seconds=.5)
    assert am['generated_training_trajectories']==4
    assert am['time_budget_reached'] is True
    assert am['history'][-1]['primary_episodes']==4
    assert am['stop_rule']=='first_completed_update_at_elapsed_budget'
    assert am['contrast_training_trajectories']==32
    for key,value in expected.state_dict().items():torch.testing.assert_close(value,actual.state_dict()[key],atol=0,rtol=0)

def test_episode_cap_cannot_silently_replace_requested_time(tmp_path,monkeypatch):
    b,_,base=make_saved(tmp_path,'sum_single');out=tmp_path/'out';out.mkdir()
    monkeypatch.setattr(controls,'time',SimpleNamespace(perf_counter=lambda:0.))
    with pytest.raises(RuntimeError,match='safety cap'):controls.train_control(b,5,SETTINGS,101,out,'sum_single',2,base,time_budget_seconds=1.)
    assert not (out/'sum_single.pt').exists()

@pytest.mark.parametrize('duration',[0.,-1.,float('inf'),float('nan')])
def test_invalid_time_budget_rejected(tmp_path,duration):
    with pytest.raises(ValueError,match='finite and positive'):controls.train_control(None,5,SETTINGS,101,tmp_path,'sum_single',8,'unused',time_budget_seconds=duration)

def test_timed_mode_cannot_train_unrelated_model(tmp_path):
    with pytest.raises(ValueError,match='explicit sum_single'):controls.train_control(None,5,SETTINGS,101,tmp_path,'sum_single',8,time_budget_seconds=1.)
