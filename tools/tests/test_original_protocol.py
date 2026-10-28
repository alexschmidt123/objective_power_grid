import math
import pytest
from src.research.original_protocol import score_allocation,vsoed_profile,sir_gmm_limits,dad_settings
@pytest.mark.parametrize('benchmark',['location_finding','sir_ode'])
@pytest.mark.parametrize('budget',[1000000,3000000,10000000])
def test_budget(benchmark,budget):
 p=score_allocation(budget,benchmark);assert p['initialization']+p['score']+p['policy']==budget
 assert p['batch']==1024 and 1<=p['policy_final_batch']<=1024
 v=vsoed_profile(benchmark,budget);assert v['initial_replay']+v['updates']*v['new_trajectories']==budget
 assert v['replay_batch']==10000 and v['replay_capacity']==1000000
 assert v['transition']==(10000 if benchmark=='sir_ode' else 0)
def test_small_SIR_policy_budget_honest():
 p=score_allocation(1000000,'sir_ode');assert p['policy']==2250 and p['policy_updates']==3 and p['policy_final_batch']==202
def test_coordinate_limits():
 mu=[math.log(.5),math.log(.1)];bounds,sig=sir_gmm_limits(mu,.5)
 for m,bb in zip(mu,bounds):assert [m+.5*x for x in bb]==pytest.approx([-6,4])
 assert sig==[1.,1.]
def test_DAD_reference():
 s=dad_settings();assert s['batch_size']==2000 and s['contrasts']==2000 and s['learning_rate']==5e-5
