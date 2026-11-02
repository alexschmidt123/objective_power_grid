"""T3 native ScoreBED numerical/full-phase throughput probe; never a result policy."""
import os,time,json,sys,fcntl,subprocess,traceback
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('WANDB_MODE','disabled')
os.environ.setdefault('JAX_PLATFORMS','cuda')
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'src/research/vendor/scorebed'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def main():
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--sir-t3-scorebed-profile',action='store_true');a.add_argument('--campaign',required=True);args=a.parse_args();C=Path(args.campaign).resolve()
 if R.name!='objective_power_grid_temp' or R/'scratch_runs/sir_ode_t3_baselines' not in C.parents:raise ValueError('TEMP T3 only')
 locks=[]
 for p in [Path.home()/'.cache/objective_power_grid/device_0.lock',R/'.gpu0.lock',R.parent/'objective_power_grid/.gpu0.lock']:
  p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 if busy:raise RuntimeError('GPU busy: '+busy)
 out=C/'profiling/scorebed';out.mkdir(exist_ok=False);start=time.monotonic();rows=[]
 try:
  import numpy as np,jax,jax.numpy as jnp,yaml
  from src.research.sir_scorebed_worker import config,check_tree
  from src.research.sir_scorebed_compat import sir_score_config
  from src.research.sir_scorebed_model import SIRScoreModel,SIRScoreGRUNet
  from policy_learning.loaders import load_score_trainer,load_policy_trainer
  from policy_learning.eig_estimators import get_grad_estimate_fn
  from policy_learning.utils.func import noop
  from policy_learning.utils.jax import derive_key
  assert jax.devices()[0].platform=='gpu';cfg=sir_score_config(config(3,50000,1024));cfg['n_restarts']=10;cfg['score']['design_sampler_type']='random_sampler';cfg['score']['design_sampler_kwargs']={};sir=dict(population=500.,i0=2.,t_end=100.,min_spacing=.1,integration_steps_per_stage=512,beta_mean=.5,gamma_mean=.1,log_std=.5,sigma=1.);cfg['bed_model']=dict(sir,name='sir_ode',T=3);m=SIRScoreModel(T=3,init_seed=101,**sir)
  def record(label,t,**kw):
   row=dict(phase=label,seconds=time.monotonic()-t,**kw);rows.append(row);write(out/'progress.json',dict(PID=os.getpid(),elapsed=time.monotonic()-start,records=rows));print(row,flush=True)
  golden=np.load(C/'manifests/t3_numerical_oracle.npz');x=jnp.asarray(golden['actions']);th=jnp.asarray(golden['theta']);th=jnp.broadcast_to(th[:,None],(len(th),3,2));t=time.monotonic();y=jax.jit(m.means)(x,th).block_until_ready();g=jax.jit(jax.grad(lambda z:m.means(z,th).sum()))(x).block_until_ready();np.testing.assert_allclose(y,golden['means'],rtol=1e-9,atol=1e-8);np.testing.assert_allclose(g,golden['time_gradients'],rtol=2e-6,atol=1e-7);record('JAX_Torch_T3_ODE_and_gradient_equivalence',t,mean_error=float(np.max(abs(np.asarray(y)-golden['means']))),gradient_error=float(np.max(abs(np.asarray(g)-golden['time_gradients']))))
  t=time.monotonic();m.compute_standardisation(100000);record('standardisation_100000',t,trajectories=100000)
  key=derive_key(101,'score_trainer');t=time.monotonic();score,key=load_score_trainer(key,cfg,m,callbacks=[]);jax.block_until_ready(score.training_state);record('score_model_initialization',t)
  for i in range(5):
   t=time.monotonic();y,x,theta,aux,score.key=score._sample_target(score.key,(1024,),use_fast_sampler=True);score.training_state,metrics=score.train_step(score.training_state,(y,x,theta,aux));check_tree(metrics,'score');record('native_score_update',t,step=i+1,batch=1024,metrics={k:float(np.asarray(v)) for k,v in metrics.items() if np.asarray(v).size==1})
  data_score,design_score=score.get_unbatched_score_fns();net=SIRScoreGRUNet(encoder_shapes=[256,256],encoding_dim=64,rnn_hidden_dim=64,rnn_layers=2,enc_act='relu',emit_act='relu',emitter_shapes=[256,256],shape=(3,1),min_std=1e-4,deterministic=False,time_scale=m.end,count_scale=m.population)
  t=time.monotonic();policy,key=load_policy_trainer(derive_key(101,'policy_trainer',0),cfg,m,grad_fn=noop,callbacks=[],policy_net=net);grad=get_grad_estimate_fn(policy.policy_network,m,data_score,design_score,N=1024,outer_batch_size=256,M=1024,estimate_type='score_chunked');policy.update_grad_fn(grad);record('policy_initialization',t)
  for i in range(3):
   t=time.monotonic();policy.training_state,metrics,_=policy.train_step(policy.training_state);check_tree(metrics,'policy');record('native_policy_update_cost_only',t,step=i+1,batch=1024,untrained_score_for_cost_only=True)
  write(out/'effective_native_config.json',cfg);write(out/'result.json',dict(status='PASS',total_seconds=time.monotonic()-start,records=rows,devices=[str(x) for x in jax.devices()],jax=jax.__version__,not_a_trained_policy=True,required_schedule=dict(score_updates=50000,policy_updates_per_restart=500,restarts=10)))
 except Exception as e:write(out/'failure.json',dict(error=str(e),traceback=traceback.format_exc(),elapsed=time.monotonic()-start,records=rows));raise
if __name__=='__main__':main()
