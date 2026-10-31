"""Read-only early common EIG after a final checkpoint is saved.

This overlaps only the root runner's secondary posterior-inference stage. Cases,
noise and the whole bank are copied by identity from the existing development
population. Results must later match the completed root evaluator per trajectory.
"""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout
from src.research.location_evaluation import contrast_scores
from src.research.moe_family import load_policy,BasePolicy,benchmark_signature
from tools.report_four_variant_study import stats,case_identity,normalize_routes
from tools.report_moe_family import summarize_rows

def load_single_policy(path,benchmark):
 path=Path(path);saved=torch.load(path,map_location='cpu',weights_only=False)
 cfg=json.loads((path.parent.parent/'run_config.json').read_text())
 if benchmark_signature(make_benchmark(cfg['benchmark']))!=benchmark_signature(benchmark):raise ValueError('Control benchmark physics mismatch')
 if saved['horizon']!=cfg['horizon']:raise ValueError('Control checkpoint horizon mismatch')
 if saved.get('kind')=='sum_single':return BasePolicy.from_checkpoint(path,benchmark,saved['horizon']).policy.eval()
 if saved.get('kind')=='sum_single_193':
  from src.research.matched_sum_single import MatchedSumSingle193
  p=MatchedSumSingle193(saved['horizon'],saved['hidden'],benchmark.action_scale,benchmark.obs_scale)
  p.load_state_dict(saved['state_dict'],strict=True);return p.eval()
 raise ValueError('Expected complete sum_single or sum_single_193 checkpoint')

def validate_training_budget(cfg,training):
 count=training['generated_training_trajectories']
 if count<1:raise ValueError('Missing completed primary training episodes')
 target=training.get('time_budget_seconds')
 if target is not None:
  if not math.isfinite(target) or target<=0 or training.get('time_budget_reached') is not True or training['training_seconds']<target:
   raise ValueError('Elapsed training budget not reached')
  if count>training['episode_safety_cap']:raise ValueError('Exceeded episode safety cap')
 else:
  nominal=cfg.get('benchmark',{}).get('research',{}).get('paper_protocol',{}).get('nominal_training_trajectories',training.get('training_trajectory_budget',1000000))
  if count!=nominal:raise ValueError('Primary training episode budget not reached')
 return count

@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);group=ap.add_mutually_exclusive_group(required=True);group.add_argument('--variant',choices=['moe','resmoe','switchmoe','switchresmoe']);group.add_argument('--control-method',choices=['sum_single','sum_single_193']);ap.add_argument('--paired-source',required=True);ap.add_argument('--output',required=True);args=ap.parse_args();torch.set_num_threads(1)
 method=args.variant or args.control_method
 run=Path(args.run);source=Path(args.paired_source);out=Path(args.output)
 while not (run/'models'/f'{method}_training.json').exists():
  if (run/'failure.json').exists():raise RuntimeError((run/'failure.json').read_text())
  time.sleep(15)
 cfg=json.loads((run/'run_config.json').read_text());training=json.loads((run/'models'/f'{method}_training.json').read_text())
 training_episodes=validate_training_budget(cfg,training)
 checkpoint=run/'models'/f'{method}.pt';b=make_benchmark(cfg['benchmark']);p=load_policy(checkpoint,b) if args.variant else load_single_policy(checkpoint,b)
 native_source=json.loads((source/'eval/sum_single_seed1001.json').read_text())
 assert len(native_source)==cfg['settings']['eval_systems']==2000
 bank_path=source/'eval/contrast_bank_seed1001.npy';bank=torch.from_numpy(np.load(bank_path));assert len(bank)==1000000
 identity=case_identity(native_source);bank_sha=hashlib.sha256(bank_path.read_bytes()).hexdigest()
 while not (run/'eval/contrast_bank_seed1001.npy').exists():time.sleep(2)
 assert bank_sha==hashlib.sha256((run/'eval/contrast_bank_seed1001.npy').read_bytes()).hexdigest()
 out.mkdir(parents=True,exist_ok=False);scores=[];rows=[];started=time.perf_counter()
 for start in range(0,2000,16):
  n=min(16,2000-start);subset=native_source[start:start+n]
  truth=torch.tensor([r['truth'] for r in subset],dtype=torch.float64);noise=torch.tensor([r['measurement_noise_standard'] for r in subset],dtype=torch.float64);routes=[]
  def hook(module,inputs,output):
   route=module.last_route
   if route is not None:routes.append(dict(stage=inputs[1],weights=route['weights'].tolist(),strategy={k:v.tolist() for k,v in route['strategy'].items()}))
  h=p.register_forward_hook(hook)
  try:r=rollout(b,p,30,n,1001+start*97451,0,truth=truth,noise=noise)
  finally:h.remove()
  info,upper=contrast_scores(b,r,bank,cfg['settings']['evaluation_contrast_chunk'])
  for i in range(n):
   score=float(info[i,-1]);scores.append(score)
   rows.append(dict(trajectory_id=start+i,truth=subset[i]['truth'],measurement_noise_standard=subset[i]['measurement_noise_standard'],
    spce=score,spce_by_stage=info[i].tolist(),nmc_upper_estimate=float(upper[i]),actions=[x[i].tolist() for x in r['actions']],observations=[x[i].tolist() for x in r['observations']],
    routing=[dict(stage=z['stage'],weights=z['weights'][i],selected_expert=int(np.argmax(z['weights'][i])),strategy={k:v[i] for k,v in z['strategy'].items()}) for z in routes]))
  if start%256==0:print(method,start+n,flush=True)
 summary=dict(canonical_variant=args.variant,single_policy_control=args.control_method,eig=stats(scores),systems=2000,contrasts=1000000,training_episodes=training_episodes,
  case_identity_sha256=identity,bank_sha256=bank_sha,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),seconds=time.perf_counter()-started,
  paired_difference_from_base=stats(np.asarray(scores)-np.array([x['spce'] for x in native_source])),native_root_replay_verified=False,
  note='Common EIG-only stage complete; native root posterior diagnostics and score-equality check may still be pending; one training seed')
 (out/'rows.json').write_text(json.dumps(rows,allow_nan=False)+'\n');(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
 if args.variant:(out/'management.json').write_text(json.dumps(summarize_rows(rows),indent=2,allow_nan=False)+'\n')
 (out/'exit_code').write_text('0\n');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
