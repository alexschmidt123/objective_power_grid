"""Bounded full-setting Myopic trajectory timing, not a performance experiment."""
import json,time
from pathlib import Path
import torch,yaml
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout
from src.research.runner import DEFAULTS
root=Path(__file__).resolve().parents[1]
raw=yaml.safe_load((root/'configs/location_finding_eig.yaml').read_text())
s=DEFAULTS|raw['research']['settings'];torch.set_num_threads(4)
b=make_benchmark(raw)
start=time.perf_counter()
with torch.no_grad():result=rollout(b,None,30,1,1001,0,method='myopic',settings=s)
elapsed=time.perf_counter()-start
out=root/'experiments/10112026/10112026_new_myopic_moe_pilot_101_1001'
out.mkdir(parents=True,exist_ok=True)
record=dict(timing_seconds=elapsed,horizon=30,trajectories=1,planner_particles=s['planner_particles'],fantasies=s['fantasies'],search_candidates=s['search_candidates'],search_rounds=s['search_rounds'],projected_myopic_planning_hours=elapsed*2000/3600,performance_claim=False)
(out/'myopic_timing.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
