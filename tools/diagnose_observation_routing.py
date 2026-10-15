"""Observation counterfactuals with designs, stage and previous expert fixed.

Only the physical simulator receives held-out truth. The policy/gate never does.
No posterior network, auxiliary training loss, or rollout-EIG claim is involved.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from src.research.benchmarks import make_benchmark
from src.research.moe_family import load_policy
from src.research.policies import history_features
from tools.report_four_variant_study import normalize_routes

@torch.no_grad()
def decision(policy,features,previous):
 c=policy.context(features).mean(1);proposal=policy.router(c,hard=True)[0].argmax(-1)
 if policy.manager is None or previous is None:return int(proposal[0]),1.,int(proposal[0])
 q=torch.nn.functional.one_hot(torch.tensor([previous]),4).double();b=float(policy.manager.switching_gate(c,q)[0,0])
 return (int(proposal[0]) if b>=.5 else previous),b,int(proposal[0])

@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--rows',required=True);ap.add_argument('--output',required=True);ap.add_argument('--cases',type=int,default=64);args=ap.parse_args();torch.set_num_threads(1)
 out=Path(args.output);out.mkdir(parents=True,exist_ok=False);cfg=json.loads((Path(args.run)/'run_config.json').read_text());b=make_benchmark(cfg['benchmark']);p=load_policy(args.checkpoint,b)
 assert b.name=='location_finding'
 rows=normalize_routes(json.loads(Path(args.rows).read_text()));rng=torch.Generator().manual_seed(7100103);records=[]
 for row in rows[:args.cases]:
  aa=[torch.tensor([x],dtype=torch.float64) for x in row['actions']];yy=[torch.tensor([x],dtype=torch.float64) for x in row['observations']]
  theta=torch.tensor([row['truth']],dtype=torch.float64)
  for stage in range(1,cfg['horizon']):
   f=history_features(b,aa[:stage],yy[:stage],1,cfg['horizon']);previous=row['routing'][stage-2]['selected_expert'] if stage>1 else None
   selected,prob,proposal=decision(p,f,previous)
   assert selected==row['routing'][stage-1]['selected_expert'],'Native history/previous-state replay mismatch'
   mean,_=b.forward(theta,b.initial((1,)),aa[stage-1]);values=[]
   for _ in range(4):
    y=mean+b.sigma*torch.randn(mean.shape,generator=rng,dtype=torch.float64)
    changed=f.clone();slots=changed[:,1:].reshape(1,cfg['horizon'],b.action_dim+b.obs_dim+1);slots[:,stage-1,b.action_dim:-1]=y/b.obs_scale
    z,bp,pr=decision(p,changed,previous);values.append(dict(selected_expert=z,reroute_probability=bp,proposal=pr))
   records.append(dict(stage=stage+1,previous_expert=previous,native_selected=selected,native_reroute_probability=prob,native_proposal=proposal,
    resampled=values,selected_flip_fraction=float(np.mean([v['selected_expert']!=selected for v in values])),
    proposal_flip_fraction=float(np.mean([v['proposal']!=proposal for v in values])),
    gate_flip_fraction=float(np.mean([(v['reroute_probability']>=.5)!=(prob>=.5) for v in values])),
    gate_probability_range=float(np.ptp([v['reroute_probability'] for v in values]))))
 summaries={}
 for key in ['selected_flip_fraction','proposal_flip_fraction','gate_flip_fraction','gate_probability_range']:
  summaries[key]=float(np.mean([r[key] for r in records]))
 summary=dict(variant=p.variant,interpretation='Counterfactual routing diagnostic, not rollout EIG; no policy training',
  source_cases=len(rows[:args.cases]),histories=len(records),observations_per_history=4,
  held_fixed='all designs, all other observations, experimental stage and previous expert identity',
  observation_source='independent Gaussian log-signal likelihood draws; truth used only by simulator',
  native_history_selection_replay_verified=True,aggregate=summaries,
  by_stage={str(s):{k:float(np.mean([r[k] for r in records if r['stage']==s])) for k in summaries} for s in range(2,cfg['horizon']+1)},
  checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),seed=7100103)
 (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');(out/'records.json').write_text(json.dumps(records,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='by_stage'},indent=2))
if __name__=='__main__':main()
