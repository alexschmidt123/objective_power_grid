"""CPU deployment latency including feature packing; no simulator allowed."""
if __name__ == '__main__':
    raise SystemExit('LEGACY_EXCLUDED_FROM_CURRENT_PROJECT: archived exploratory result generation')
import argparse,json,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch,yaml
from src.research.sa_moe import load_sa_checkpoint
from src.research.policies import history_features
from src.research.benchmarks import make_benchmark

def main():
 p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);a=p.parse_args();C=Path(a.campaign);cfg=yaml.safe_load((C/'configs/seed101.yaml').read_text());torch.set_num_threads(2)
 contexts=torch.load(C/'evaluations/competence_500.pt',map_location='cpu',weights_only=False)['contexts'];rows=[];start=time.perf_counter()
 for label in ['sa_moe','single']:
  path=C/'models'/f'{label}.pt'
  if not path.exists():continue
  model=load_sa_checkpoint(path);b=make_benchmark(cfg['benchmark'])
  for context in contexts:
   actions=[x.reshape(1,2) for x in context['actions']];observations=[x.reshape(1,1) for x in context['observations']];samples=[]
   with torch.inference_mode(),patch.object(b,'forward',side_effect=AssertionError('Deployment latency path must not simulate experiments')):
    for repeat in range(220):
     t=time.perf_counter();f=history_features(b,actions,observations,1,30);packed=time.perf_counter();a=model(f,len(actions));total=time.perf_counter()-t
     if repeat>=20:samples.append(dict(full_outer_seconds=total,feature_packing=packed-t,**model.inference_times[-1]))
   # Explicit sum includes feature packing, unlike original engine neural timer.
   total=[s['feature_packing']+s['encoder']+s['router']+s['expert'] for s in samples]
   rows.append(dict(model=label,stage=len(actions),feature_packing_seconds=float(np.mean([s['feature_packing'] for s in samples])),encoder_seconds=float(np.mean([s['encoder'] for s in samples])),router_seconds=float(np.mean([s['router'] for s in samples])),expert_seconds=float(np.mean([s['expert'] for s in samples])),total_online_decision_seconds=float(np.mean(total)),total_online_p95_seconds=float(np.percentile(total,95)),explicit_posterior_update_seconds=0.,posterior_update_definition='No numerical posterior update in this history-encoded architecture',simulator_calls=0,reference_contexts='Actual training-prefix contexts; timing benchmark, not quality-validation data'))
 (C/'reports/online_latency.json').write_text(json.dumps(dict(rows=rows,benchmark_seconds=time.perf_counter()-start,device='CPU2threads',deployment_forbidden_simulator_test_passed=True,readiness_and_diagnostics_cost=True),indent=2));print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
