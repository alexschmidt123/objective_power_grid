"""Prepare the independent Ubuntu T=5 seed-101 campaign without training."""
import hashlib,json,subprocess,shutil
from pathlib import Path
import yaml
from tools.journal_campaign import make_config

ROOT=Path(__file__).resolve().parents[1]
C=ROOT/'experiments/10162026/10162026_ubuntu_sir_t5_seed101'
C.mkdir(parents=True,exist_ok=False)
shutil.copy2(ROOT/'docs/research_records/20261017_sir_execution/journal_protocol.md',C/'journal_protocol.md')
for cap,name in [(1000000,'sir_1m.yaml'),(500000,'sir_500k.yaml')]:
    cfg=make_config('sir',cap)
    cfg['sir_ode']['execution_backend']='fused_rk4'
    p=cfg['research']['journal_protocol']
    p.update(version='validation_convergence_v1',execution_protocol='sir_fixed_budget_execution_v1',status='development_readiness_pending',
             evaluation_seeds=[1001],reserved_final_evaluation_seed=8831001,
             development_resource_gate='full campaign only after numerical/native smoke checks and cost review',
             budget_unit='total_primary_episodes_including_pretraining')
    p['validation'].update(systems=64,contrasts=1024,batch_size=16,chunk=1024,
                           device='cuda_if_available')
    p['stopping_rule']['maximum_training_seconds']=86400
    p['common_pathwise_settings']['moe_router_minimum_gamma']=.001
    p['final_evaluation'].update(seed=1001,systems=2048,contrasts=16384,
                                 resolution_status='development_pending_resolution_check')
    if cap==500000:
        p['method_settings']['sum_single']['router_schedule_episodes']=500000
    cfg['research']['plan'].update(horizons=[5],scope='Ubuntu seed101 development; no T10 or replication launched')
    (C/name).write_text(yaml.safe_dump(cfg,sort_keys=False))
probe=ROOT/'experiments/10162026/10162026_ubuntu_sir_fused_probe/probe.json'
records=json.loads(probe.read_text())
eager=next(r for r in records if r['test']=='training_shape_T5_forward_backward' and r['backend']=='eager')
warm=[r for r in records if r['test']=='training_shape_T5_forward_backward' and r['backend']=='fused_rk4'][-1]
cost=dict(policy='SIRSumMLP',batch=32,contrasts=1024,horizon=5,
          reference_seconds=eager['seconds'],warm_seconds=warm['seconds'],
          measured_speedup=eager['seconds']/warm['seconds'],
          projected_1m_optimization_hours=warm['extrapolated_1m_optimization_hours'],
          projection_limit='one warm policy step; not full-method measured runtimes',
          total_campaign_planning_hours=[36,72],native_method_costs_pending=True,
          per_method_safety_cap_hours=24,training_budget_reduced=False)
(C/'cost_review.json').write_text(json.dumps(cost,indent=2)+'\n')
(C/'status.json').write_text(json.dumps(dict(state='readiness',full_training_started=False),indent=2))
(C/'git_origin.json').write_text(json.dumps(dict(
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    dirty_diff=subprocess.check_output(['git','diff'],cwd=ROOT,text=True),
    historical_results_preserved=True),indent=2))
print(str(C),flush=True)
