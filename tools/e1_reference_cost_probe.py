"""Bounded native-size readiness; every execution uses the canonical run.sh."""
import argparse,json,subprocess,time,yaml,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.research.original_protocol import score_allocation
C=R/'experiments/10192026/10192026_e1_original_baseline_restoration'
p=argparse.ArgumentParser();p.add_argument('--benchmark',choices=['location','sir'],required=True);p.add_argument('--method',choices=['dad','rl_sboed','scorebed','vsoed'],required=True);a=p.parse_args()
cap={'dad':2000,'rl_sboed':100,'vsoed':21000,'scorebed':30494 if a.benchmark=='location' else 102190}[a.method]
cfg=yaml.safe_load((C/'configs'/f'{a.benchmark}_1000000.yaml').read_text());p=cfg['research']['journal_protocol'];p['stopping_rule'].update(minimum_episodes=cap,maximum_episodes=cap,maximum_training_seconds=600);p['validation'].update(systems=4,contrasts=8,batch_size=2,chunk=4);cfg['research']['settings'].update(eval_systems=2,eval_contrasts=8,evaluation_population_systems=2,evaluation_batch_size=2,evaluation_contrast_chunk=4);cfg['research']['paper_protocol']['nominal_training_trajectories']=cap
if a.method=='scorebed':p['method_settings']['scorebed']['scorebed_pretraining_episodes']=score_allocation(cap,'location_finding' if a.benchmark=='location' else 'sir_ode')['score']
path=C/'readiness'/f'{a.benchmark}_{a.method}_native_batch.yaml';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(yaml.safe_dump(cfg,sort_keys=False));out=C/'readiness'/f'{a.benchmark}_{a.method}_native_batch';log=C/'logs'/f'probe_{a.benchmark}_{a.method}.log';cmd=['bash','run.sh','--config',str(path),'--journal','--seed','101','--eval-seeds','1001','--eig-only','--methods',a.method,'--output',str(out)];start=time.monotonic()
try:
 with log.open('x') as f:proc=subprocess.run(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT,timeout=600)
 row=dict(method=a.method,benchmark=a.benchmark,complete_primary_episodes=cap,exit_code=proc.returncode,seconds=time.monotonic()-start,command=cmd,scope='readiness_only_not_E1_result')
except Exception as exc:row=dict(method=a.method,benchmark=a.benchmark,status='probe_failed_or_timeout',error=repr(exc),seconds=time.monotonic()-start,command=cmd)
(C/'reports'/f'cost_probe_{a.benchmark}_{a.method}.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
if row.get('exit_code')!=0:raise SystemExit(1)
