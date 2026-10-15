"""Frozen expert proposals and observation interventions, no EIG/training."""
import argparse,hashlib,json
from pathlib import Path
import torch
from src.research.benchmarks import make_benchmark
from src.research.moe_family import load_policy,EXPERT_IDS
from src.research.policies import history_features
from tools.report_four_variant_study import normalize_routes
from tools.report_moe_family import counterfactuals

@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--rows',required=True);ap.add_argument('--output',required=True);ap.add_argument('--cases',type=int,default=64);args=ap.parse_args()
 torch.set_num_threads(1);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
 cfg=json.loads((Path(args.run)/'run_config.json').read_text());b=make_benchmark(cfg['benchmark']);p=load_policy(args.checkpoint,b)
 rows=normalize_routes(json.loads(Path(args.rows).read_text()));proposals,cf=counterfactuals(b,p,rows,args.cases)
 ratios=[]
 if hasattr(p.core,'base'):
  for row in rows[:args.cases]:
   aa=[torch.tensor([x],dtype=torch.float64) for x in row['actions']];yy=[torch.tensor([x],dtype=torch.float64) for x in row['observations']]
   for stage in range(1,cfg['horizon']):
    f=history_features(b,aa[:stage],yy[:stage],1,cfg['horizon']);context=p.context(f);base=p.core.base(f,stage)[0];delta=p.core.alpha*torch.stack([e(context[:,j])[0] for j,e in enumerate(p.experts)])
    base_norm=base.norm();norm=delta.norm(dim=-1);cos=(delta*base).sum(-1)/(norm*base_norm).clamp_min(1e-6)
    ratios.append(dict(stage=stage+1,base_logit_norm=float(base_norm),correction_logit_norm=norm.tolist(),correction_over_base_norm=(norm/base_norm.clamp_min(1e-6)).tolist(),correction_cosine_to_base=cos.tolist()))
 result=dict(variant=p.variant,canonical_name=p.canonical_name,expert_ids=list(EXPERT_IDS),proposals=proposals,observation_counterfactuals=cf,residual_relative_to_base=ratios,
  checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),source_rows=str(Path(args.rows).resolve()),
  interpretation='Matched histories within each model; models can visit different histories. Counterfactual diagnostics are not rollout EIG.')
 (out/'geometry.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(p.canonical_name,proposals['histories'],proposals['pairwise_physical_distance_mean'])
if __name__=='__main__':main()
