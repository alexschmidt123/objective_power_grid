"""Evaluate a saved auxiliary posterior on completed development histories."""
import argparse,json,math,statistics
from pathlib import Path
import torch
from src.research.benchmarks import make_benchmark
from src.research.engine import history_features
from src.research.variational_pathwise import FullCovariancePosterior

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--method',default='moe_sboed');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4)
    cfg=json.loads((a.run/'run_config.json').read_text())
    rows=json.loads((a.run/'eval'/f'{a.method}_seed{a.seed}.json').read_text())
    checkpoint=torch.load(a.run/'models'/f'{a.method}.pt',map_location='cpu',weights_only=False)
    q=FullCovariancePosterior(**checkpoint['density_parameters']);q.load_state_dict(checkpoint['density_state']);q.eval()
    b=make_benchmark(cfg['benchmark']);n=len(rows);h=cfg['horizon']
    actions=torch.tensor([r['actions'] for r in rows],dtype=torch.float64)
    observations=torch.tensor([r['observations'] for r in rows],dtype=torch.float64)
    truth=torch.tensor([r['truth'] for r in rows],dtype=torch.float64)
    with torch.no_grad():
        features=history_features(b,list(actions.unbind(1)),list(observations.unbind(1)),n,h)
        z=b.transform(truth);bound=q.log_prob(features,z)-b.log_prior_z(z)
    values=bound.tolist();gap=[r['spce']-v for r,v in zip(rows,values)]
    result=dict(cases=n,evaluation_seed=a.seed,method=a.method,mean_heldout_variational_bound=statistics.mean(values),
         bound_mc_standard_error=statistics.stdev(values)/math.sqrt(n),mean_spce_minus_posterior_bound=statistics.mean(gap),
         policy_updates=0,posterior_updates=0,simulator_calls=0,
         interpretation='Auxiliary normalized density evaluated on development histories. Difference uses finite-sample sPCE, not exact EIG or certified posterior KL. Truth is used solely as a density argument after experiment selection.')
    if not all(math.isfinite(x) for x in values):raise ValueError('Nonfinite posterior density')
    if a.output.exists():raise FileExistsError(a.output)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
