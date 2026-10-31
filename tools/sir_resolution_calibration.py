"""Independent nested sPCE resolution diagnostic; no method ranking/tuning."""
import argparse,json,math,time,hashlib
from pathlib import Path
import numpy as np,torch,yaml
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout,replay_likelihood
from src.research.journal_checkpoints import load_checkpoint

@torch.no_grad()
def main(output,config,deadline):
 out=Path(output);out.mkdir(exist_ok=False,parents=True)
 raw=yaml.safe_load(Path(config).read_text());torch.set_num_threads(2)
 root=Path(__file__).resolve().parents[1];stable=root.with_name('objective_power_grid')
 old=stable/'experiments/_legacy_archive/labpc/objective_power_grid_sir_journal_r2_20261026/experiments/10262026/10262026_sir_journal_corrected_seed101_r2/runs/dad/models/dad.pt'
 with torch.device('cpu'):
  b=make_benchmark(raw);g=torch.Generator().manual_seed(97402621)
  truth=b.sample((256,),g);noise=torch.randn(256,5,1,generator=g,dtype=torch.float64);bank=b.sample((1000000,),g)
 p=load_checkpoint(old,b).cuda().eval();counts=[16384,65536,262144,1000000];arrays={};started=time.monotonic()
 with torch.device('cuda:0'):
  for method,policy in [('random',None),('dad',p)]:
   b=make_benchmark(raw);values={n:[] for n in counts}
   for first in range(0,256,16):
    if time.monotonic()>deadline:raise TimeoutError('Aggregate readiness deadline reached; retain partial resolution evidence')
    theta=truth[first:first+16].cuda();r=rollout(b,policy,5,len(theta),97402621+first*97451,0,truth=theta,noise=noise[first:first+16].cuda(),method=method)
    actions=r['actions'];ys=r['observations'];positive=replay_likelihood(b,theta[:,None],actions,ys)[0][:,0]
    total=torch.full_like(positive,-torch.inf);index=0
    for n in counts:
     while index<n:
      stop=min(n,index+4096);ll=replay_likelihood(b,bank[index:stop].cuda()[None].expand(len(theta),-1,-1),actions,ys)[0]
      total=torch.logaddexp(total,torch.logsumexp(ll,1));index=stop
     values[n].extend((positive-torch.logaddexp(positive,total)+math.log(n+1)).cpu().tolist())
    np.savez(out/(method+'_nested_partial.npz'),**{str(n):np.asarray(v) for n,v in values.items()})
    print('RESOLUTION',method,first+len(theta),time.monotonic()-started,flush=True)
   arrays[method]={n:np.asarray(v) for n,v in values.items()}
 records=[]
 for method,values in arrays.items():
  for n in counts:
   delta=values[n]-values[1000000]
   records.append(dict(method=method,contrasts=n,systems=len(delta),mean=float(values[n].mean()),mc_se=float(values[n].std(ddof=1)/len(delta)**.5),difference_to_common_1m=float(delta.mean()),paired_difference_se=float(delta.std(ddof=1)/len(delta)**.5)))
 result=dict(status='COMPLETE',seed=97402621,systems=256,records=records,frozen_common_contrasts=1000000,
  criterion='Largest predeclared candidate, never method-ranking based; report lower-bound tail sensitivity explicitly',
  residual_bias_certified=False,seconds=time.monotonic()-started,historical_DAD_checkpoint_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
  purpose='Estimator-only diagnostic on independent cases; historical policy is not a four-hour primary result')
 (out/'resolution.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--config',required=True);p.add_argument('--deadline',required=True,type=float);a=p.parse_args();main(a.output,a.config,a.deadline)
