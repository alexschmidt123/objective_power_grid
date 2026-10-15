"""Frozen dominant-expert and calibrated stage-only routing diagnostics.

No retraining, no additional named method. All experts adapt to new histories;
only their selection is intervened on. Same common EIG evaluator and paired bank.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from src.research.moe_family import load_policy,EXPERT_IDS
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout
from src.research.location_evaluation import contrast_scores
from tools.report_four_variant_study import stats,normalize_routes

class HistoryIndependentRouting(nn.Module):
 kind='sparse_moe_prototype'
 def __init__(self,policy,probabilities,case_start=0,seed=7100102):
  super().__init__();self.policy=policy;self.probabilities=probabilities;self.case_start=case_start;self.seed=seed
 def forward(self,features,stage):
  if stage==0:return self.policy(features,stage)
  draws=torch.tensor([float(torch.rand((),generator=torch.Generator().manual_seed(self.seed+stage*104729+(self.case_start+i)*9709))) for i in range(len(features))],dtype=features.dtype)
  choice=torch.searchsorted(self.probabilities[stage-1].cumsum(-1),draws,right=True).clamp_max(3)
  def replace(router,inputs,output):
   w=torch.nn.functional.one_hot(choice,4).to(features.dtype)
   return w,w[:,None],router.codes[choice][:,None]
  manager=self.policy.manager;mode=manager.mode if manager is not None else None
  if manager is not None:manager.mode='always_reroute'
  hook=self.policy.router.register_forward_hook(replace)
  try:return self.policy(features,stage)
  finally:
   hook.remove()
   if manager is not None:manager.mode=mode

@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--rows',required=True);ap.add_argument('--bank',required=True);ap.add_argument('--output',required=True)
 ap.add_argument('--calibration-cases',type=int,default=512);args=ap.parse_args();torch.set_num_threads(1)
 run=Path(args.run);out=Path(args.output);out.mkdir(parents=True,exist_ok=False);cfg=json.loads((run/'run_config.json').read_text());b=make_benchmark(cfg['benchmark']);h=cfg['horizon']
 rows=normalize_routes(json.loads(Path(args.rows).read_text()));bank=torch.from_numpy(np.load(args.bank));counts=torch.zeros(h-1,4,dtype=torch.float64)
 p=load_policy(args.checkpoint,b)
 for start in range(0,args.calibration_cases,32):
  n=min(32,args.calibration_cases-start);active=[]
  def capture(module,inputs,output):
   if inputs[1]>0:active.append(module.last_route['weights'].argmax(-1))
  hook=p.register_forward_hook(capture)
  try:rollout(b,p,h,n,7100101+start*97451,0)
  finally:hook.remove()
  choices=torch.stack(active)
  counts+=torch.nn.functional.one_hot(choices,4).double().sum(1)
 probabilities=counts/args.calibration_cases
 dominant=int(np.argmax(np.bincount([r['selected_expert'] for row in rows for r in row['routing']],minlength=4)))
 fixed=torch.nn.functional.one_hot(torch.full((h-1,),dominant),4).double()
 truth=torch.tensor([r['truth'] for r in rows],dtype=torch.float64);noise=torch.tensor([r['measurement_noise_standard'] for r in rows],dtype=torch.float64);native=np.array([r['spce'] for r in rows]);results={};started=time.perf_counter()
 for mode,prob in [('fixed_dominant',fixed),('stage_only',probabilities)]:
  scores=[];selection=[]
  for start in range(0,len(rows),16):
   n=min(16,len(rows)-start);policy=HistoryIndependentRouting(load_policy(args.checkpoint,b),prob,start).eval();active=[]
   def capture(module,inputs,output):
    if inputs[1]>0:active.append(module.last_route['weights'].argmax(-1).tolist())
   hook=policy.policy.register_forward_hook(capture)
   try:r=rollout(b,policy,h,n,1001+start*97451,0,truth=truth[start:start+n],noise=noise[start:start+n])
   finally:hook.remove()
   info,_=contrast_scores(b,r,bank,cfg['settings']['evaluation_contrast_chunk']);scores.extend(info[:,-1].tolist());selection.extend(np.asarray(active).T.tolist())
   if start%512==0:print(mode,start+n,flush=True)
  scores=np.array(scores);results[mode]=dict(eig=stats(scores),learned_minus_intervention=stats(native-scores),scores=scores.tolist(),stage_expert_frequency=np.stack([(np.asarray(selection)==j).mean(0) for j in range(4)],-1).tolist())
  (out/'partial.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
 result=dict(canonical_variant=p.variant,dominant_expert=EXPERT_IDS[dominant],calibration_seed_rule='7100101+batch_start*97451; independent of seed1001 evaluation',
  calibration_cases=args.calibration_cases,stage_only_choice_seed=7100102,stage_probabilities=probabilities.tolist(),systems=len(rows),contrasts=len(bank),results=results,seconds=time.perf_counter()-started,
  checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),bank_sha256=hashlib.sha256(Path(args.bank).read_bytes()).hexdigest(),
  limitations=['Frozen interventions may create out-of-distribution histories','Stage-only probabilities fitted on independent simulator trajectories, not EIG outcomes','No new named method or trained alternative','Single training seed'])
 (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');(out/'exit_code').write_text('0\n');print(json.dumps({m:{k:v for k,v in d.items() if k not in ['scores','stage_expert_frequency']} for m,d in results.items()},indent=2))
if __name__=='__main__':main()
