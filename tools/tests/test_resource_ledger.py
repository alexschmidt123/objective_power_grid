import pytest
from src.research.resource_ledger import ResourceLedger,accounting,active_ledger

def test_units():
    l=ResourceLedger("training");l.record_paths(64,1024,30)
    assert l.counts["fresh_complete_episodes"]==64
    assert l.counts["contrast_parameter_stage_propagations"]==64*1024*30
    assert l.counts["scalar_conditional_log_density_evaluations"]==64*1025*30
    assert l.export()["ode_rhs_evaluations"] is None

def test_scope():
    a=ResourceLedger("validation");b=ResourceLedger("training")
    with accounting(a):
        assert active_ledger() is a
        with accounting(b):assert active_ledger() is b
        assert active_ledger() is a
    assert active_ledger() is None

def test_no_likelihood_and_bad_count():
    l=ResourceLedger("training");l.record_paths(2,0,5,False)
    assert "scalar_conditional_log_density_evaluations" not in l.counts
    with pytest.raises(ValueError):l.add("updates",-1)

from src.research.resource_ledger import with_training_ledger

def test_native_reconciliation():
    r=dict(training="official_ScoreBED_JAX",generated_training_trajectories=100,initialization_trajectories=10,score_training_trajectories=85,policy_training_trajectories=5,actual_policy_updates=1)
    l=with_training_ledger(r,5)["resource_ledger"]
    assert l["counts"]["true_parameter_stage_propagations"]==500
    assert l["counts"]["scalar_conditional_log_density_evaluations"] is None
    r["policy_training_trajectories"]=6
    with pytest.raises(ValueError):with_training_ledger(r,5)

def test_replay_not_new_data():
    r=dict(training="native_RL",generated_training_trajectories=100,contrast_training_trajectories=10000000,gradient_updates=64,replay_batch=4096,actor_updates=64)
    l=with_training_ledger(r,30)["resource_ledger"]
    assert l["counts"]["fresh_complete_episodes"]==100
    assert l["replay_transition_samples"]==64*4096
    assert l["counts"]["scalar_conditional_log_density_evaluations"]==10000100*30

def test_shared_rollout_accounting_preserves_outputs():
    import torch
    from src.research.benchmarks import LocationFinding
    from src.research.engine import rollout
    b=LocationFinding({"location_finding":{"design_bounds":[-4,4]},"observation":{"noise_sigma":.5}})
    a=rollout(b,None,3,2,1001,4,method="random")
    l=ResourceLedger("validation")
    with accounting(l):z=rollout(b,None,3,2,1001,4,method="random")
    torch.testing.assert_close(a["info"],z["info"],rtol=0,atol=0)
    torch.testing.assert_close(torch.stack(a["actions"]),torch.stack(z["actions"]),rtol=0,atol=0)
    assert l.counts["scalar_conditional_log_density_evaluations"]==2*5*3


def test_base_optimizer_updates_are_attributed():
    from src.research.resource_ledger import with_training_ledger
    row=with_training_ledger(dict(training='pathwise_sPCE_architecture_control',generated_training_trajectories=4096,contrast_training_trajectories=4096*1024,actual_updates=64),30)
    assert row['resource_ledger']['optimizer_updates']['policy']==64
