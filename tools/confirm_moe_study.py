"""Prospective frozen-checkpoint EIG confirmation, common unchanged evaluator.

Run only after the one intervention and all comparison checkpoints are frozen.
This is an evaluation stage, not training or a new estimator. Seed1001 remains
used for smoke. The preregistered new seed is never used for policy selection.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout,generator
from src.research.location_evaluation import contrast_scores
from src.research.moe_family import load_policy,BasePolicy,benchmark_signature
from tools.report_four_variant_study import stats
from tools.early_family_eig import load_single_policy

@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--specs',required=True);ap.add_argument('--output',required=True);ap.add_argument('--seed',type=int,default=8831001);ap.add_argument('--smoke',action='store_true');args=ap.parse_args();torch.set_num_threads(1)
 specs=json.loads(Path(args.specs).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
 if args.smoke and args.seed!=8831001:raise ValueError('Smoke always uses development seed1001')
 seed=1001 if args.smoke else args.seed;n=4 if args.smoke else 2000;L=16 if args.smoke else 1000000
 cfg=json.loads((Path(specs[0]['run'])/'run_config.json').read_text());b=make_benchmark(cfg['benchmark']);T=cfg['horizon']
 assert b.name=='location_finding' and T==30 and b.design_bounds==[-4.,4.] and b.sigma==.5
 rng=generator(seed);truth=b.sample((2000,),rng);noise=torch.randn(2000,T,b.obs_dim,generator=rng,dtype=torch.float64);bank=b.sample((L,),rng)
 np.save(out/'contrast_bank.npy',bank.numpy());np.savez_compressed(out/'paired_cases.npz',truth=truth.numpy(),noise=noise.numpy())
 truth=truth[:n];noise=noise[:n]
 report=dict(seed=seed,smoke=args.smoke,systems=n,contrasts=L,training_updates=0,
  estimator='Unchanged src.research.location_evaluation.contrast_scores',
  selection_rule='All frozen comparisons reported; no training/tuning after examining this seed',
  limitation='One training seed: independent evaluation confirms frozen policies only, not training-seed reproducibility',models=[],paired_differences={})
 score_arrays={}
 for spec in specs:
  owncfg=json.loads((Path(spec['run'])/'run_config.json').read_text())
  if benchmark_signature(make_benchmark(owncfg['benchmark']))!=benchmark_signature(b):raise ValueError('Comparison physics/likelihood/feasibility mismatch')
  path=Path(spec['checkpoint']);before=hashlib.sha256(path.read_bytes()).hexdigest()
  if spec['type']=='family':p=load_policy(path,b)
  elif spec['type'] in {'dad_mlp','matched_single'}:p=load_single_policy(path,b)
  else:raise ValueError('Unknown checkpoint type')
  scores=[];started=time.perf_counter()
  for start in range(0,n,16):
   size=min(16,n-start);r=rollout(b,p,T,size,seed+start*97451,0,truth=truth[start:start+size],noise=noise[start:start+size])
   info,_=contrast_scores(b,r,bank,16384);scores.extend(info[:,-1].tolist())
   if start%512==0:print(spec['label'],start+size,flush=True)
  assert before==hashlib.sha256(path.read_bytes()).hexdigest()
  score_arrays[spec['label']]=np.asarray(scores)
  item=dict(label=spec['label'],canonical_variant=spec.get('variant'),checkpoint=str(path.resolve()),checkpoint_sha256=before,eig=stats(scores),seconds=time.perf_counter()-started)
  report['models'].append(item);(out/'partial.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 for i,(a,x) in enumerate(score_arrays.items()):
  for j,(c,y) in enumerate(score_arrays.items()):
   if i>j:report['paired_differences'][a+' minus '+c]=stats(x-y)
 (out/'scores.json').write_text(json.dumps({k:v.tolist() for k,v in score_arrays.items()},allow_nan=False)+'\n')
 report['bank_sha256']=hashlib.sha256((out/'contrast_bank.npy').read_bytes()).hexdigest()
 report['cases_sha256']=hashlib.sha256((out/'paired_cases.npz').read_bytes()).hexdigest()
 (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');(out/'exit_code').write_text('0\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
