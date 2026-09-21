"""Equal-trajectory validation-only diagnostic for IEEE9 MoE optimization."""
from pathlib import Path
from types import SimpleNamespace
import json,hashlib
import numpy as np
from src.config import load_config
from src.domains.swing.continuous_rocof import EndpointRocofObserver
from src.objectives.eig.continuous_eig import OnlineEIG
from src.objectives.eig.continuous_moe import train_moe
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'experiments/09162026/09162026_ieee9_moe_stability_diagnostic'
out.mkdir(parents=True,exist_ok=False)
recipes=[('legacy32',32,.001,'legacy'),('stable32',32,.0003,'stable'),('stable128',128,.0003,'stable')]
(out/'protocol.json').write_text(json.dumps({'training_trajectories_per_recipe':8192,'recipes':recipes,
    'evaluation_data_used':False,'selection':'highest best validation EIG with finite training diagnostics; tied checkpoint frequency per 1024 trajectories',
    'source_sha256':hashlib.sha256((ROOT/'src/objectives/eig/continuous_moe.py').read_bytes()).hexdigest()},indent=2))
cfg=load_config(ROOT/'configs/ieee9_eig.yaml')
observer=EndpointRocofObserver(cfg,duration_bounds=(.2,3.),injection_bus=1,amplitude=.05,window=3.5)
engine=OnlineEIG(cfg,observer,horizon=5,sigma=.005,contrasts=128,min_separation=.01)
report={}
for name,batch,lr,mode in recipes:
    directory=out/name;directory.mkdir()
    args=SimpleNamespace(T=5,seed=101,learning_rate=lr,updates=8192//batch,batch_size=batch,
        validate_every=1024//batch,validation_systems=128,moe_training_mode=mode)
    p,elapsed=train_moe(engine,args,directory)
    history=json.loads((directory/'moe_sboed_training.json').read_text())
    updates=json.loads((directory/'moe_sboed_optimizer_diagnostics.json').read_text())
    steps=[m for u in updates for m in u['update_diagnostics']]
    report[name]={'best_validation':max(r['validation_utility'] for r in history),'last_validation':history[-1]['validation_utility'],
        'elapsed_seconds':elapsed,'learning_rate':lr,'batch_size':batch,'mode':mode,
        'attempted_kl_median':float(np.median([r['approximate_kl'] for r in steps])),
        'attempted_kl_max':max(r['approximate_kl'] for r in steps),
        'attempts_exceeding_kl_limit':sum(r['approximate_kl']>.02 for r in steps),
        'rejected_steps':sum(u['rejected_passes'] for u in updates),
        'accepted_steps':sum(u['accepted_passes'] for u in updates),
        'critic_to_actor_gradient_ratio_median':float(np.median([r['critic_gradient_norm']/max(r['actor_gradient_norm'],1e-12) for r in steps])),
        'expert_pair_switch_fraction_mean':float(np.mean([r['expert_pair_switch_fraction'] for r in steps]))}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('RECIPE_COMPLETE',name,json.dumps(report[name]),flush=True)
print('DIAGNOSTIC_COMPLETE',flush=True)
