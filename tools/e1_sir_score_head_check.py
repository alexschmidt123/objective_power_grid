"""Check the pinned conservative network's documented two-channel LayerNorm degeneracy."""
import json,sys,os
from pathlib import Path
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'src/research/vendor/scorebed')]
import numpy as np,jax,jax.numpy as jnp
from flax import serialization
from policy_learning.loaders import load_score_trainer
from policy_learning.utils.jax import derive_key
from src.research.sir_scorebed_model import SIRScoreModel

def main(old,out):
 old=Path(old);out=Path(out);out.mkdir(parents=True,exist_ok=True);d=old/'sir/seed101/scorebed/models'
 if not d.exists():d=old/'worker_ubuntu/sir/seed101/scorebed/models'
 cfg=json.loads((d/'scorebed_official_config.json').read_text());sir=json.loads((d/'scorebed_sir_configuration.json').read_text());t=SIRScoreModel(T=5,init_seed=101,**sir);norm=np.load(d/'scorebed_normalization.npz')
 for k in norm.files:setattr(t,k,jnp.asarray(norm[k]))
 t.standardised=True;original,_=load_score_trainer(derive_key(101,'score_trainer'),cfg,t,callbacks=[]);variables=jax.tree_util.tree_map(jnp.asarray,serialization.msgpack_restore((d/'scorebed_official_score.msgpack').read_bytes()))
 rows=json.loads((d.parent/'eval/scorebed_seed1001.json').read_text())[:4];y=jnp.asarray([r['observations'] for r in rows]);x=jnp.asarray([r['actions'] for r in rows])
 def base(module,y,x):return module._base_score_net(y,x,sigma=None,aux_data={},train=False)
 vec=original.score_network.apply(variables,y,x,method=base)
 def ln(v):
  p=variables['params']['layer_norm'];mean=jnp.mean(v);var=jnp.mean((v-mean)**2);return (v-mean)/jnp.sqrt(var+1e-6)*p['scale']+p['bias']
 jac=jax.vmap(jax.vmap(jax.jacrev(ln)))(vec)
 oldscore=original._score_from_z_with_variables(variables,jnp.concatenate([y,x],-1),{})
 cfg2=json.loads(json.dumps(cfg));cfg2['score_network']['potential_hidden_dim']=256
 fixed,_=load_score_trainer(derive_key(101,'score_trainer'),cfg2,t,callbacks=[])
 # Identical trained Transformer; only the already-supported potential-head adaptation differs.
 vp=fixed._score_variables_from_state();params=dict(vp['params']);params['_base_score_net']=variables['params']['_base_score_net'];vp=dict(vp,params=params)
 if 'rff_freqs' in variables:vp['rff_freqs']=variables['rff_freqs']
 newscore=fixed._score_from_z_with_variables(vp,jnp.concatenate([y,x],-1),{})
 oldmax=float(jnp.max(jnp.abs(oldscore)));newmax=float(jnp.max(jnp.abs(newscore)))
 assert oldmax<1e-8 and newmax>1e-6,(oldmax,newmax)
 # Native score-matching one update: finite and informative parameter gradients.
 batch=fixed._sample_target(derive_key(101,'diagnostic_score_head'),(4,),use_fast_sampler=False);yy,xx,theta,aux,_=batch
 state=fixed.training_state
 def loss(params):return fixed.loss_fn(derive_key(101,'diagnostic_loss'),params,state.rff_freqs,state.batch_stats,(yy,xx,theta,aux),train=False)[0]
 value,grad=jax.value_and_grad(loss)(state.params);normgrad=float(jnp.sqrt(sum(jnp.sum(g*g) for g in jax.tree_util.tree_leaves(grad))))
 assert np.isfinite(float(value)) and np.isfinite(normgrad) and normgrad>1e-6
 updated,metrics=fixed.train_step(state,(yy,xx,theta,aux));assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(updated.params))
 result=dict(status='PASS',confirmed_defect='Two-output conservative-potential LayerNorm is numerically flat in trained SIR adapter',original_score_max_abs=oldmax,compatible_head_score_max_abs=newmax,trained_base_output_range=[float(vec.min()),float(vec.max())],original_layernorm_jacobian_max=float(jnp.max(jnp.abs(jac))),native_loss=float(value),native_gradient_norm=normgrad,correction='Enable pinned upstream potential_hidden_dim=256 only for scalar-observation/scalar-design SIR; unchanged Transformer, loss, policy and optimizers',upstream_source='policy_learning/score_nets.py:ConservativeScoreNetwork potential_hidden_dim',training_in_diagnostic='One native update on diagnostic clone, no policy retraining or checkpoint replacement')
 (out/'score_potential_head_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
