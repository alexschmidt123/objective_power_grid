"""Explicit four-hour COMPUTE_LIMITED ScoreBED: native score fitting + ten restarts.

Native50k-step score LR schedule and500-step policy schedules remain unchanged.
Only wall-time stage allocation is adapted; no official-case checkpoint selection.
"""
import time
ENTRY=time.monotonic()
import os,sys,json,argparse,fcntl,subprocess,traceback,hashlib,math
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');os.environ.setdefault('WANDB_MODE','disabled')
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/research/vendor/scorebed'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);ap.add_argument('--cpu-smoke',action='store_true');ap.add_argument('--export-intermediates',action='store_true');args=ap.parse_args();C=Path(args.campaign).resolve();smoke=args.cpu_smoke
 if ROOT.name!='objective_power_grid_temp' or ROOT/'scratch_runs/sir_ode_t3_baselines' not in C.parents:raise ValueError('TEMP T3 outputs only')
 if args.export_intermediates:
  import numpy as np
  from flax import serialization
  from flax.traverse_util import flatten_dict
  out=C/('cpu_smoke/scorebed' if smoke else 'checkpoints/scorebed');rows=[]
  for path in sorted(out.glob('restart0_round*.msgpack')):
   state=serialization.msgpack_restore(path.read_bytes());params=state['params_ema'];arrays={'/'.join(k):np.asarray(v) for k,v in flatten_dict(params).items()};target=path.with_suffix('.ema.npz');np.savez(target,**arrays);label=path.stem.removeprefix('restart0_');meta=out/f'common_{label}.json'
   record=json.loads(meta.read_text()) if meta.exists() else json.loads((out/'latest_common_round.json').read_text());rows.append(dict(path=str(target),round=int(state['step']),charged_seconds=record['charged_seconds'],sha256=sha(target)))
  write(out/'intermediate_exports.json',rows);print(json.dumps(rows,indent=2));return
 locks=[]
 if not smoke:
  for p in [Path.home()/'.cache/objective_power_grid/device_0.lock',ROOT/'.gpu0.lock',ROOT.parent/'objective_power_grid/.gpu0.lock']:
   f=p.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
  busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  if busy:raise RuntimeError('GPU busy: '+busy)
 import numpy as np,jax,jax.numpy as jnp,yaml
 from src.research.sir_scorebed_worker import config,check_tree
 from src.research.sir_scorebed_compat import sir_score_config
 from src.research.sir_scorebed_model import SIRScoreModel,SIRScoreGRUNet
 from policy_learning.loaders import load_score_trainer,load_policy_trainer
 from policy_learning.eig_estimators import get_grad_estimate_fn
 from policy_learning.utils.func import noop
 from policy_learning.utils.jax import derive_key
 from src.research.jax_checkpoint import save_state,load_state
 protocol=yaml.safe_load((C/'configs/sir_ode_t3_continuous_v1.yaml').read_text());s=protocol['full_testing']['scorebed'];out=C/('cpu_smoke/scorebed' if smoke else 'checkpoints/scorebed');out.mkdir(parents=True,exist_ok=False)
 assert jax.devices()[0].platform==('cpu' if smoke else 'gpu')
 cap=600 if smoke else 14400;phase_limit=300 if smoke else s['score_phase_charged_seconds_limit'];score_limit=2 if smoke else s['maximum_score_updates'];policy_limit=1 if smoke else 500;restarts=2 if smoke else 10;B=4 if smoke else 1024;stats_count=32 if smoke else 100000;rows=[];rounds=0;policies=[]
 def spent():return time.monotonic()-ENTRY
 def status(phase,**kw):
  record=dict(method='scorebed',state=phase,PID=os.getpid(),charged_seconds=spent(),score_updates=int(score.training_state.step) if 'score' in locals() else None,policy_rounds=rounds,restarts=restarts,batch_size=B,**kw);write(out/'status.json',record);print(record,flush=True)
 cfg=sir_score_config(config(3,50000,B));cfg['n_restarts']=restarts;cfg['policy']['num_iters']=500;cfg['score']['num_iters']=50000;cfg['score']['design_sampler_type']='random_sampler';cfg['score']['design_sampler_kwargs']={};sir=dict(population=500.,i0=2.,t_end=100.,min_spacing=.1,integration_steps_per_stage=512,beta_mean=.5,gamma_mean=.1,log_std=.5,sigma=1.);cfg['bed_model']=dict(sir,name='sir_ode',T=3)
 write(out/'native_effective_configuration.json',cfg)
 m=SIRScoreModel(T=3,init_seed=101,**sir);t=time.monotonic();m.compute_standardisation(stats_count);np.savez(out/'normalization.npz',**{k:np.asarray(v) for k,v in m.get_std_stats_dict().items()});initialization=time.monotonic()-t
 key=derive_key(101,'score_trainer');score,key=load_score_trainer(key,cfg,m,callbacks=[]);next_save=1800.;max_score_unit=30.
 status('SCORE_PRETRAINING',standardisation_seconds=initialization)
 try:
  for step in range(1,score_limit+1):
   if spent()+max(30.,1.5*max_score_unit)>=phase_limit:break
   t=time.monotonic();y,x,theta,aux,score.key=score._sample_target(score.key,(B,),use_fast_sampler=True);score.training_state,metrics=score.train_step(score.training_state,(y,x,theta,aux));check_tree(metrics,'score metrics');dt=time.monotonic()-t
   # Cold compilation is charged, but does not predict warm step throughput.
   if step>2:max_score_unit=max(.1,dt)
   if step==1 or step%100==0:
    row=dict(phase='score',update=step,charged_seconds=spent(),cycle_seconds=dt,metrics={k:float(np.asarray(v)) for k,v in metrics.items() if np.asarray(v).size==1});rows.append(row);write(out/'progress.json',rows);status('SCORE_PRETRAINING',actual_score_updates=step)
   if step==1 or spent()>=next_save:
    save_state(out/f'score_step{step}.msgpack',score.training_state);write(out/f'score_step{step}.json',dict(step=step,charged_seconds=spent(),key_data=np.asarray(jax.random.key_data(score.key)).tolist(),key_impl=str(jax.random.key_impl(score.key)),policy_EIG_available=False));next_save+=1800 if step!=1 else 0
  score_steps=int(score.training_state.step);save_state(out/'score_final.msgpack',score.training_state);restored=load_state(out/'score_final.msgpack',score.training_state)
  for a,b in zip(jax.tree_util.tree_leaves(restored),jax.tree_util.tree_leaves(score.training_state)):
   if hasattr(a,'dtype') and not jax.dtypes.issubdtype(a.dtype,jax.dtypes.prng_key):np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
  if score_steps==0:raise RuntimeError('No score optimization completed')
  data_score,design_score=score.get_unbatched_score_fns();net=SIRScoreGRUNet(encoder_shapes=[256,256],encoding_dim=64,rnn_hidden_dim=64,rnn_layers=2,enc_act='relu',emit_act='relu',emitter_shapes=[256,256],shape=(3,1),min_std=1e-4,deterministic=False,time_scale=m.end,count_scale=m.population)
  for r in range(restarts):
   p,key=load_policy_trainer(derive_key(101,'policy_trainer',r),cfg,m,grad_fn=noop,callbacks=[],policy_net=net);grad=get_grad_estimate_fn(p.policy_network,m,data_score,design_score,N=B,outer_batch_size=min(256,B),M=8 if smoke else 1024,estimate_type='score_chunked');p.update_grad_fn(grad);policies.append(p)
  status('POLICY_TRAINING',actual_score_updates=score_steps,primary_restart=0)
  max_round=180. if not smoke else 10.;last_saved=0
  def save_round(label):
   paths=[]
   for r,p in enumerate(policies):
    path=out/f'restart{r}_{label}.msgpack';save_state(path,p.training_state);paths.append(dict(restart=r,path=str(path),sha256=sha(path),updates=int(p.training_state.step)))
   eligible=spent()<=cap
   record=dict(round=rounds,charged_seconds=spent(),budget_seconds=cap,eligible=eligible,paths=paths);write(out/f'common_{label}.json',record);write(out/'latest_common_round.json',record)
   return paths
  while rounds<policy_limit:
   if spent()+max(120.,1.5*max_round)>=cap:break
   t=time.monotonic()
   for r,p in enumerate(policies):
    p.training_state,metrics,_=p.train_step(p.training_state);check_tree(metrics,f'policy restart{r}')
   rounds+=1;dt=time.monotonic()-t
   if rounds>1:max_round=dt
   if rounds==1 or rounds%10==0:
    rows.append(dict(phase='policy',round=rounds,updates_per_restart=rounds,restarts=restarts,charged_seconds=spent(),round_seconds=dt));write(out/'progress.json',rows);status('POLICY_TRAINING',actual_score_updates=score_steps)
   if rounds==1 or spent()-last_saved>=300:
    save_round(f'round{rounds}');last_saved=spent()
  if rounds<1:raise RuntimeError('No complete native policy update for all ten restarts')
  saved=save_round(f'round{rounds}_final')
  from flax.traverse_util import flatten_dict
  for r,p in enumerate(policies):
   params=p.training_state.params_ema;arrays={'/'.join(path):np.asarray(v) for path,v in flatten_dict(params).items()};key=jax.random.key(2026102835+r);keys=jax.random.split(key,3);x=jnp.broadcast_to(jnp.linspace(5.,95.,3)[None,:,None],(8,3,1));y=jax.random.uniform(keys[0],(8,3,1),dtype=jnp.float64)*500;z=jax.random.normal(keys[1],(8,3,1),dtype=jnp.float64);variables={'params':params};carry=net.apply(variables,(8,),method='init_gru_state');expected=[]
   for stage in range(3):
    raw=net.apply(variables,carry[-1],z=z[:,stage],method='emitter_from_state')[:,0];previous=jnp.zeros(8) if stage==0 else x[:,stage-1,0];expected.append(np.asarray(m.feasible(raw,previous,stage)[:,None]));token=net.apply(variables,y=y[:,stage],x=x[:,stage],method='encoder');carry,_=net.apply(variables,carry,token,method='gru_step')
   arrays.update(reference_x=np.asarray(x),reference_y=np.asarray(y),reference_noise=np.asarray(z),reference_actions=np.stack(expected,1));np.savez(out/f'restart{r}_ema_policy.npz',**arrays)
  total=spent()
  result=dict(status='COMPUTE_LIMITED' if not smoke else 'CPU_SMOKE_PASS',charged_training_seconds=total,budget_seconds=cap,budget_eligible=total<=cap,score_updates=score_steps,native_score_reference_updates=50000,policy_updates_per_restart=rounds,native_policy_reference_updates_per_restart=500,policy_restarts=restarts,standardisation_trajectories=stats_count,generated_training_trajectories=stats_count+14+(score_steps+rounds*restarts)*B,score_batch=B,policy_batch=B,training_seed=101,checkpoint_selection='Final budget-eligible EMA per restart; primary restart0 predeclared, all ten evaluated; no favorable EIG selection',native_schedule_complete=score_steps>=50000 and rounds>=500,allocation_rule=s,native_state_paths=saved,source_manifest_sha256=sha(C/'manifests/source_sha256.json') if not smoke else None)
  if not result['budget_eligible']:result['status']='INVALID_OR_INCOMPLETE_BUDGET_OVERRUN'
  write(out/'complete.json',result);status('TRAINING_COMPLETE',actual_score_updates=score_steps,result=result)
 except Exception as e:write(out/'failure.json',dict(error=str(e),traceback=traceback.format_exc(),charged_seconds=spent(),score_updates=int(score.training_state.step),completed_policy_rounds=rounds));raise
if __name__=='__main__':main()
