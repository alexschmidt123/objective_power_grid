import numpy as np
import pytest
import torch
from src.research.journal_validation import StoppingRule,ValidationObserver

class Validator:
    seed=771001
    def __call__(self,p):
        x=float(p.weight.detach().flatten()[0])
        return np.array([-x,-x]),dict(seconds=0.,seed=self.seed)

def test_fixed_budget_keeps_training_despite_validation_decline(tmp_path):
    rule=StoppingRule(minimum_episodes=1,minimum_checkpoints=1,validation_every_episodes=1,patience=1,maximum_episodes=3,maximum_training_seconds=60)
    p=torch.nn.Linear(1,1,bias=False)
    observer=ValidationObserver(Validator(),rule,tmp_path,'policy',selection_mode='final_fixed_budget')
    for episode in range(1,4):
        with torch.no_grad():p.weight.fill_(episode)
        assert observer.check(p,episode,episode)==(episode==3)
    result=observer.finish(p)
    assert float(p.weight)==3
    assert result['selected_checkpoint']['episodes']==3
    assert result['checkpoint_selection']=='final_predeclared_fixed_budget'
    assert result['convergence_status']=='fixed_budget_complete'

def test_fixed_budget_incomplete_is_not_a_selected_model(tmp_path):
    rule=StoppingRule(minimum_episodes=1,minimum_checkpoints=1,validation_every_episodes=1,patience=1,maximum_episodes=3,maximum_training_seconds=60)
    p=torch.nn.Linear(1,1,bias=False)
    observer=ValidationObserver(Validator(),rule,tmp_path,'policy',selection_mode='final_fixed_budget')
    observer.check(p,1,1)
    with pytest.raises(RuntimeError,match='No validation-selected checkpoint'):observer.finish(p)
