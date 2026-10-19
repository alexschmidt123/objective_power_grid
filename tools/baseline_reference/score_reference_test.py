"""Run separately against official checkout and vendored modules; compare outputs."""
import os,sys,json
from pathlib import Path
C=Path('/home/qiqi/Documents/objective_power_grid/experiments/10172026/10172026_original_baseline_fidelity');R=Path('/home/qiqi/Documents/objective_power_grid_baseline_fidelity_20261017')
R=Path(os.environ.get('BASELINE_CHECKOUT',str(R)));C=Path(os.environ.get('BASELINE_AUDIT',str(C)));sys.path.insert(0,str(R))
mode=sys.argv[1];base=C/'upstream_references/scorebed' if mode=='upstream' else R/'src/research/vendor/scorebed';sys.path.insert(0,str(base));os.environ['JAX_PLATFORMS']='cpu';os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
import numpy as np,jax,jax.numpy as jnp
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
from policy_learning.bed_models import LocationFinding
from policy_learning.loaders import load_score_trainer,load_policy_trainer
from policy_learning.eig_estimators import get_grad_estimate_fn
from policy_learning.utils.func import noop
from functools import partial
class Bounded(LocationFinding):
 def feasible_design(self,x):return -4+8*jax.nn.sigmoid(x)
 def sample_designs(self,*a,**kw):
  x,aux,key=super().sample_designs(*a,**kw);return self.feasible_design(x),aux,key
 def get_policy_output_activation(self,kind):return self.feasible_design
 def _get_cached_fast_design_sampler(self,sampler_type,sampler_kwargs):
  key=(sampler_type,self._freeze_sampler_kwargs(sampler_kwargs))
  if key not in self._fast_design_sampler_cache:self._fast_design_sampler_cache[key]=partial(self.sample_designs,sampler_type=sampler_type,sampler_kwargs=sampler_kwargs)
  return self._fast_design_sampler_cache[key]
with initialize_config_dir(config_dir=str(base/'config'),version_base=None):cfg=OmegaConf.to_container(compose(config_name='config',overrides=['experiment=location_finding']),resolve=True)
cfg['bed_model']['T']=3;cfg['score'].update(num_iters=4,batch_size=2,design_sampler_type='time_corr_random_scale_gaussian',design_sampler_kwargs={'scale_min':.2,'scale_max':5.,'time_corr_rho_min':.7,'time_corr_rho_max':1.});cfg['policy'].update(grad_est_type='score',train_outer_samples=2,train_outer_batch_size=2)
target=Bounded(**cfg['bed_model']);target.compute_standardisation(4)
key=jax.random.key(101);trainer,key=load_score_trainer(key,cfg,target,callbacks=[])
y,x,th,aux,trainer.key=trainer._sample_target(trainer.key,(2,),use_fast_sampler=True)
assert (np.asarray(x)>=-4).all() and (np.asarray(x)<=4).all()
old=trainer.training_state.params;trainer.training_state,metrics=trainer.train_step(trainer.training_state,(y,x,th,aux));assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(metrics))
data_score,design_score=trainer.get_unbatched_score_fns();policy,key=load_policy_trainer(key,cfg,target,grad_fn=noop,callbacks=[])
grad=get_grad_estimate_fn(policy.policy_network,target,data_score,design_score,N=2,outer_batch_size=2,M=8,estimate_type='score');policy.update_grad_fn(grad)
score_hash=[np.asarray(v).copy() for v in jax.tree_util.tree_leaves(trainer.training_state.params)]
policy.training_state,pmetrics,_=policy.train_step(policy.training_state)
for a,b in zip(score_hash,jax.tree_util.tree_leaves(trainer.training_state.params)):np.testing.assert_array_equal(a,b)
# Fixed physical cases: include near-boundary designs and explicit Gaussian likelihood.
xi=jnp.array([[[-3.99,3.99],[3.9,-3.9],[0.,0.]]]);theta=jnp.broadcast_to(jnp.array([[[[.2,.4],[-.5,-.7]]]]),(1,3,2,2));eps=jnp.array([[[.1],[-.3],[.2]]]);obs,_=target.data_reparam(eps,xi,theta)
means=jnp.log(.1+jnp.sum(1/(1e-4+jnp.sum((xi[:,:,None,:]-theta)**2,axis=-1)),axis=-1))[...,None];np.testing.assert_allclose(obs,means+.5*eps,rtol=2e-6,atol=2e-6)
ll=target.per_step_data_log_lik(obs,xi,theta);expected=-.5*jnp.squeeze(eps**2,-1)-jnp.log(.5*jnp.sqrt(2*jnp.pi));np.testing.assert_allclose(ll,expected,rtol=2e-6,atol=2e-6)
g=jax.grad(lambda a:target.per_step_data_log_lik(obs,a,theta).sum())(xi)
arrays=dict(sample_x=np.asarray(x),sample_y=np.asarray(y),sample_theta=np.asarray(th),physical_obs=np.asarray(obs),likelihood=np.asarray(ll),likelihood_design_gradient=np.asarray(g))
for label,tree in [('score_params',trainer.training_state.params),('policy_params',policy.training_state.params),('policy_ema',policy.training_state.params_ema),('score_ema',trainer.training_state.params_ema),('score_metrics',metrics),('policy_metrics',pmetrics)]:
 for i,v in enumerate(jax.tree_util.tree_leaves(tree)):arrays[f'{label}_{i}']=np.asarray(v)
# Verify reporting and JAX-to-Torch inference against final EMA parameters.
from src.research.baseline_fidelity import scorebed_reported_params
reported=scorebed_reported_params(policy.training_state,True)
assert reported is policy.training_state.params_ema
assert any(not np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(jax.tree_util.tree_leaves(reported),jax.tree_util.tree_leaves(policy.training_state.params)))
flat={f'{section}/{layer}/{name}':np.asarray(value) for section,leaves in reported.items() for layer,values in leaves.items() for name,value in values.items()}
xx=np.asarray(x);yy=np.asarray(y);enc=jnp.ones((len(xx),16))*.01;expected=[]
for stage in range(3):
 expected.append(np.asarray(policy.policy_network.apply({'params':reported},enc=enc,method='emitter')))
 enc=enc+policy.policy_network.apply({'params':reported},y=jnp.asarray(yy[:,stage]),x=jnp.asarray(xx[:,stage]),method='encoder')
flat.update(reference_x=xx,reference_y=yy,reference_actions=np.stack(expected,1),common_design_bounds=np.array([-4.,4.]))
np.savez(C/'reports'/f'score_ema_bridge_{mode}.npz',**flat)
np.savez(C/'reports'/f'score_reference_{mode}.npz',**arrays)
(C/'reports'/f'score_reference_{mode}.json').write_text(json.dumps(dict(status='PASS',mode=mode,sampler=cfg['score']['design_sampler_type'],metrics={k:float(np.asarray(v)) for k,v in metrics.items() if np.asarray(v).size==1},score_model_frozen_during_policy_step=True,scope='Native marginal score loss, score optimizer update and score EIG policy gradient/update; bounded adaptation applied identically in both processes'),indent=2));print(mode,'PASS',flush=True)
