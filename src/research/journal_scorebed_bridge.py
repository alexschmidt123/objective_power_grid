"""Checkpoint-only IPC between unchanged native JAX updates and common EIG.

Producer imports no Torch, so the official isolated ScoreBED environment stays
independent. Consumer uses the project's Torch environment and existing scorer.
"""
import argparse,json,subprocess,time,shutil
from pathlib import Path
import numpy as np

def export_arrays(params):
 arrays={}
 def walk(tree,path=()):
  if hasattr(tree,'items'):
   for key,value in tree.items():walk(value,path+(key,))
  else:arrays['/'.join(path)]=np.asarray(tree)
 walk(params)
 if 'log_std' in arrays:arrays['policy_architecture']=np.asarray('official_ScoreBED_IqbalGRU_SIR_v1')
 return arrays

def export_candidate(params,plan):
 """Preserve benchmark feasibility metadata in native validation IPC."""
 arrays=export_arrays(params)
 bounds=plan['raw'].get('location_finding',{}).get('design_bounds')
 if bounds is not None:
  bounds=np.asarray(bounds,dtype=float)
  if bounds.shape!=(2,) or not np.isfinite(bounds).all() or bounds[0]>=bounds[1]:
   raise ValueError('Invalid location design bounds')
  arrays['common_design_bounds']=bounds
 return arrays


def compute_clock_from_plan(plan):
    spec=plan.get("compute_budget")
    if not spec:return None
    from src.research.compute_budget import ComputeClock
    return ComputeClock(spec["path"],spec["seconds"],credit_seconds=spec["credit_seconds"],hardware=spec["hardware"])

def maybe_validate(params,plan_path,episodes,update,started,force=False):
 if not plan_path:return False
 plan_path=Path(plan_path);plan=json.loads(plan_path.read_text());out=plan_path.parent;progress=out/'scorebed_journal_progress.json'
 previous=json.loads(progress.read_text()) if progress.exists() else dict(records=[],validation_seconds=0.,next_check=plan['rule']['validation_every_episodes'])
 clock=compute_clock_from_plan(plan)
 if clock is not None and previous.get('records') and previous['records'][-1]['episodes']==episodes:
  if force:previous['records'][-1]['status']='planned_schedule_complete_before_compute_cap' if clock.can_start() else 'compute_budget_complete';progress.write_text(json.dumps(previous,indent=2))
  return force
 elapsed=clock.spent() if clock is not None else time.perf_counter()-started-previous['validation_seconds']
 compute_stop=clock is not None and not clock.can_start(plan['compute_budget'].get('native_unit_bound_seconds',0.))
 if clock is not None:
  if not(force or compute_stop or elapsed>=previous.get('next_compute_check',plan['compute_budget']['checkpoint_seconds'])):return False
 elif not (force or episodes>=previous['next_check'] or elapsed>=plan['rule']['maximum_training_seconds']):return False
 candidate=out/'scorebed_journal_candidate.npz';np.savez(candidate,**export_candidate(params,plan))
 if clock is not None:
  shutil.copyfile(candidate,out/f'scorebed_progress_{update}.npz')
  elapsed=clock.spent() # Include native checkpoint/export IO in eligibility.
 command=[plan['torch_python'],'-m','src.research.journal_scorebed_bridge','--score','--plan',str(plan_path),'--candidate',str(candidate),'--episodes',str(episodes),'--update',str(update),'--elapsed',str(elapsed)]
 if clock is not None and (compute_stop or force):command+=['--compute-stop']
 before=time.perf_counter()
 if clock is None:subprocess.run(command,cwd=plan['root'],check=True)
 else:
  with clock.exclude('validation'):subprocess.run(command,cwd=plan['root'],check=True)
 wall=time.perf_counter()-before
 state=json.loads(progress.read_text())
 if clock is not None and state['records'][-1]['selected']:shutil.copyfile(candidate,out/'scorebed_journal_best.npz')
 state=json.loads(progress.read_text());state['validation_seconds']=previous['validation_seconds']+wall
 state['records'][-1]['validation']['consumer_seconds']=state['records'][-1]['validation']['seconds'];state['records'][-1]['validation']['seconds']=wall
 progress.write_text(json.dumps(state,indent=2)+'\n')
 with (out/'scorebed_validation.jsonl').open('a') as f:f.write(json.dumps(state['records'][-1])+'\n')
 return state['records'][-1]['status']!='training'

def restored_params(params,plan_path):
 if not plan_path:return params
 import jax.numpy as jnp
 saved=np.load(Path(plan_path).parent/'scorebed_journal_best.npz')
 def rebuild(tree,path=()):
  if hasattr(tree,'items'):return {key:rebuild(value,path+(key,)) for key,value in tree.items()}
  return jnp.asarray(saved['/'.join(path)])
 return rebuild(params)

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
   if 'policy_architecture' in arrays and str(arrays['policy_architecture'])=='official_ScoreBED_IqbalGRU_SIR_v1':
    from src.research.sir_scorebed_gru import SIRScoreGRUPolicy
    p=SIRScoreGRUPolicy(arrays,plan['horizon'],b)
   else:p=SIRScorePolicy(arrays,plan['horizon'],b)
  else:
   from src.research.scorebed import OfficialScorePolicy
   p=OfficialScorePolicy(arrays,plan['horizon'],b.action_scale,b.obs_scale)
  v=plan['validation'];validator=CommonValidation(plan['raw'],plan['horizon'],v['seed'],v['systems'],v['contrasts'],v['batch_size'],v['chunk'],device=v.get('device','cpu'))
  values,cost=validator(p)
 rule=StoppingRule(**plan['rule']);plateau=Plateau(rule);plateau.records=state['records'];plateau.best=state.get('best');plateau.anchor=None if state.get('anchor') is None else np.array(state['anchor']);plateau.stale=state.get('stale',0)
 row=plateau.update(values,args.episodes,args.elapsed);row.update(update=args.update,validation=cost)
 apply_selection_mode(row,args.episodes,rule,plan.get('selection_mode','best_validation'))
 clock=compute_clock_from_plan(plan)
 if clock is not None:
  mark=clock.checkpoint(str(args.update),charged_seconds=args.elapsed);row.update(compute_checkpoint=mark);row['selected']=mark['eligible'];row['status']=('compute_budget_complete' if mark['eligible'] else 'compute_budget_exceeded') if getattr(args,'compute_stop',False) else 'training'
  state['next_compute_check']=args.elapsed+plan['compute_budget']['checkpoint_seconds']
 if row['selected'] and clock is None:shutil.copyfile(args.candidate,out/'scorebed_journal_best.npz')
 state.update(records=plateau.records,best=plateau.best,anchor=plateau.anchor.tolist(),stale=plateau.stale,next_check=args.episodes+rule.validation_every_episodes)
 path.write_text(json.dumps(state,indent=2)+'\n');np.save(out/f'scorebed_validation_scores_{args.update}.npy',values)
 print('ScoreBED common journal validation',row,flush=True)

def write_plan(observer,raw,h,s,directory):
 from dataclasses import asdict
 import sys
 directory=Path(directory);path=directory/'scorebed_journal_plan.json';v=observer.validator
 plan=dict(selection_mode=observer.selection_mode,root=str(Path(__file__).resolve().parents[2]),torch_python=sys.executable,raw=raw,horizon=h,threads=s.get('threads',4),rule=asdict(observer.rule),validation=dict(seed=v.seed,systems=v.systems,contrasts=v.contrasts,batch_size=v.batch,chunk=v.chunk,device=v.device),score_pretraining_episodes=s.get('scorebed_pretraining_episodes',700000))
 if observer.compute_clock is not None:
  if 'score_schedule_reference_steps' not in s or 'policy_schedule_reference_steps' not in s:raise ValueError('Original ScoreBED scheduler horizons must be explicit for matched-compute training')
  plan['score_schedule_reference_steps']=int(s['score_schedule_reference_steps']);plan['policy_schedule_reference_steps']=int(s['policy_schedule_reference_steps'])
  if 'scorebed_compute_phase_plan' not in s:raise ValueError('Measured frozen ScoreBED phase plan required in matched-compute mode')
  plan['scorebed_compute_phase_plan']=s['scorebed_compute_phase_plan']
  state=observer.compute_clock._read();plan['compute_budget']=dict(path=str(observer.compute_clock.path),seconds=state['budget_seconds'],credit_seconds=state['credit_seconds'],hardware=state['hardware'],checkpoint_seconds=observer.compute_checkpoint_seconds)
 path.write_text(json.dumps(plan,indent=2)+'\n');return path

def attach_summary(policy,observer,directory):
 if observer is None:return {}
 state=json.loads((Path(directory)/'scorebed_journal_progress.json').read_text());observer.plateau.records=state['records'];observer.validation_seconds=state['validation_seconds'];observer.best_state={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
 return observer.finish(policy)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--score',action='store_true');ap.add_argument('--compute-stop',action='store_true');ap.add_argument('--plan',required=True);ap.add_argument('--candidate',required=True);ap.add_argument('--episodes',type=int,required=True);ap.add_argument('--update',type=int,required=True);ap.add_argument('--elapsed',type=float,required=True);score(ap.parse_args())
