"""Checkpoint-only IPC between unchanged native JAX updates and common EIG.

Producer imports no Torch, so the official isolated ScoreBED environment stays
independent. Consumer uses the project's Torch environment and existing scorer.
"""
import argparse,json,subprocess,time,shutil
from pathlib import Path
import numpy as np

def export_arrays(params):
 return {f'{section}/{layer}/{name}':np.asarray(value) for section,layers in params.items() for layer,values in layers.items() for name,value in values.items()}

def maybe_validate(params,plan_path,episodes,update,started,force=False):
 if not plan_path:return False
 plan_path=Path(plan_path);plan=json.loads(plan_path.read_text());out=plan_path.parent;progress=out/'scorebed_journal_progress.json'
 previous=json.loads(progress.read_text()) if progress.exists() else dict(records=[],validation_seconds=0.,next_check=plan['rule']['validation_every_episodes'])
 elapsed=time.perf_counter()-started-previous['validation_seconds']
 if not (force or episodes>=previous['next_check'] or elapsed>=plan['rule']['maximum_training_seconds']):return False
 candidate=out/'scorebed_journal_candidate.npz';np.savez(candidate,**export_arrays(params))
 command=[plan['torch_python'],'-m','src.research.journal_scorebed_bridge','--score','--plan',str(plan_path),'--candidate',str(candidate),'--episodes',str(episodes),'--update',str(update),'--elapsed',str(elapsed)]
 before=time.perf_counter();subprocess.run(command,cwd=plan['root'],check=True);wall=time.perf_counter()-before
 state=json.loads(progress.read_text());state['validation_seconds']=previous['validation_seconds']+wall
 state['records'][-1]['validation']['consumer_seconds']=state['records'][-1]['validation']['seconds'];state['records'][-1]['validation']['seconds']=wall
 progress.write_text(json.dumps(state,indent=2)+'\n')
 with (out/'scorebed_validation.jsonl').open('a') as f:f.write(json.dumps(state['records'][-1])+'\n')
 return state['records'][-1]['status']!='training'

def restored_params(params,plan_path):
 if not plan_path:return params
 import jax.numpy as jnp
 saved=np.load(Path(plan_path).parent/'scorebed_journal_best.npz')
 return {section:{layer:{name:jnp.asarray(saved[f'{section}/{layer}/{name}']) for name in values} for layer,values in layers.items()} for section,layers in params.items()}

def score(args):
 import torch
 from src.research.benchmarks import make_benchmark
 from src.research.journal_validation import StoppingRule,Plateau,CommonValidation,apply_selection_mode
 plan_path=Path(args.plan);plan=json.loads(plan_path.read_text());out=plan_path.parent;path=out/'scorebed_journal_progress.json'
 state=json.loads(path.read_text()) if path.exists() else dict(records=[],validation_seconds=0.)
 torch.set_num_threads(plan.get('threads',4));b=make_benchmark(plan['raw']);arrays=np.load(args.candidate)
 with torch.device('cpu'):
  if b.name=='sir_ode':
   from src.research.sir_scorebed import SIRScorePolicy
   p=SIRScorePolicy(arrays,plan['horizon'],b)
  else:
   from src.research.scorebed import OfficialScorePolicy
   p=OfficialScorePolicy(arrays,plan['horizon'],b.action_scale,b.obs_scale)
  v=plan['validation'];validator=CommonValidation(plan['raw'],plan['horizon'],v['seed'],v['systems'],v['contrasts'],v['batch_size'],v['chunk'])
  values,cost=validator(p)
 rule=StoppingRule(**plan['rule']);plateau=Plateau(rule);plateau.records=state['records'];plateau.best=state.get('best');plateau.anchor=None if state.get('anchor') is None else np.array(state['anchor']);plateau.stale=state.get('stale',0)
 row=plateau.update(values,args.episodes,args.elapsed);row.update(update=args.update,validation=cost)
 apply_selection_mode(row,args.episodes,rule,plan.get('selection_mode','best_validation'))
 if row['selected']:shutil.copyfile(args.candidate,out/'scorebed_journal_best.npz')
 state.update(records=plateau.records,best=plateau.best,anchor=plateau.anchor.tolist(),stale=plateau.stale,next_check=args.episodes+rule.validation_every_episodes)
 path.write_text(json.dumps(state,indent=2)+'\n');np.save(out/f'scorebed_validation_scores_{args.update}.npy',values)
 print('ScoreBED common journal validation',row,flush=True)

def write_plan(observer,raw,h,s,directory):
 from dataclasses import asdict
 import sys
 directory=Path(directory);path=directory/'scorebed_journal_plan.json';v=observer.validator
 plan=dict(selection_mode=observer.selection_mode,root=str(Path(__file__).resolve().parents[2]),torch_python=sys.executable,raw=raw,horizon=h,threads=s.get('threads',4),rule=asdict(observer.rule),validation=dict(seed=v.seed,systems=v.systems,contrasts=v.contrasts,batch_size=v.batch,chunk=v.chunk),score_pretraining_episodes=s.get('scorebed_pretraining_episodes',700000))
 path.write_text(json.dumps(plan,indent=2)+'\n');return path

def attach_summary(policy,observer,directory):
 if observer is None:return {}
 state=json.loads((Path(directory)/'scorebed_journal_progress.json').read_text());observer.plateau.records=state['records'];observer.validation_seconds=state['validation_seconds'];observer.best_state={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
 return observer.finish(policy)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--score',action='store_true');ap.add_argument('--plan',required=True);ap.add_argument('--candidate',required=True);ap.add_argument('--episodes',type=int,required=True);ap.add_argument('--update',type=int,required=True);ap.add_argument('--elapsed',type=float,required=True);score(ap.parse_args())
