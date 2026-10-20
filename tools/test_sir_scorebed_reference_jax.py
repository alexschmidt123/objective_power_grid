"""Generate native-vs-adapter recurrent fixtures using the pinned JAX classes."""
import os,sys,json
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cpu')
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'src/research/vendor/scorebed')]
import numpy as np,jax,jax.numpy as jnp
from flax.traverse_util import flatten_dict
from policy_learning.policies import IqbalGRUNet
from src.research.sir_scorebed_model import SIRScoreModel,SIRScoreGRUNet
from src.research.sir_scorebed_worker import config
out=R/'experiments/10192026/10192026_e1_original_baseline_restoration/readiness';out.mkdir(parents=True,exist_ok=True)
b=SIRScoreModel(T=5);kw=dict(encoder_shapes=[256,256],encoding_dim=64,rnn_hidden_dim=64,rnn_layers=2,enc_act='relu',emit_act='relu',emitter_shapes=[256,256],shape=(5,1),deterministic=False,min_std=1e-4)
net=SIRScoreGRUNet(**kw);native=IqbalGRUNet(**kw);keys=jax.random.split(jax.random.key(101),4)
x=jnp.broadcast_to(jnp.linspace(1.,90.,5)[None,:,None],(3,5,1));y=jax.random.uniform(keys[0],(3,5,1))*500;z=jax.random.normal(keys[1],(3,5,1));variables=net.init(keys[2],y=y,x=x,z=z)
a=net.apply(variables,y=y,x=x,z=z);reference=native.apply(variables,y=y/500,x=x/100,z=z)
np.testing.assert_allclose(a,reference,rtol=1e-6,atol=1e-6)
ga=jax.grad(lambda p:net.apply({'params':p},y=y,x=x,z=z).sum())(variables['params']);gb=jax.grad(lambda p:native.apply({'params':p},y=y/500,x=x/100,z=z).sum())(variables['params'])
for a_,b_ in zip(jax.tree_util.tree_leaves(ga),jax.tree_util.tree_leaves(gb)):np.testing.assert_allclose(a_,b_,rtol=2e-5,atol=2e-5)
carry=net.apply(variables,(3,),method='init_gru_state');expected=[]
for t in range(5):
 raw=net.apply(variables,carry[-1],z=z[:,t],method='emitter_from_state')[:,0];prev=jnp.zeros((3,)) if t==0 else x[:,t-1,0];expected.append(b.feasible(raw,prev,t)[:,None]);token=net.apply(variables,y=y[:,t],x=x[:,t],method='encoder');carry,_=net.apply(variables,carry,token,method='gru_step')
arrays={'/'.join(k):np.asarray(v) for k,v in flatten_dict(variables['params']).items()};arrays.update(reference_x=np.asarray(x),reference_y=np.asarray(y),reference_noise=np.asarray(z),reference_actions=np.asarray(jnp.stack(expected,1)))
np.savez(out/'scorebed_gru_reference_fixture.npz',**arrays)
# Check every native component and the complete mixture: no pendulum simulation.
results=[]
for sampler in ['uniform','random_at_bounds','pseudo_random_independent','pseudo_random_pos_correlated','pseudo_random_neg_correlated','temp_corr','random_sampler']:
 raw,_,expected_key=b.reference_sampler.sample_designs(keys[3],(32,),sampler_type=sampler,sampler_kwargs={});times,_,actual_key=b.sample_designs(keys[3],(32,),sampler_type=sampler,sampler_kwargs={});last=jnp.zeros(32);forward=[]
 for t in range(5):last=b.feasible(raw[...,t,0],last,t);forward.append(last)
 np.testing.assert_array_equal(times[...,0],jnp.stack(forward,-1));np.testing.assert_array_equal(jax.random.key_data(actual_key),jax.random.key_data(expected_key));assert np.min(np.diff(np.asarray(times[...,0]),axis=1))>=.1-1e-10;results.append(sampler)
raw=jnp.linspace(-1.,1.,5)
def mapping(raw):
 last=0.;v=[]
 for t in range(5):last=b.feasible(raw[t],last,t);v.append(last)
 return jnp.stack(v)
times=mapping(raw)[:,None];_,logdet=b.raw_designs_and_log_jacobian(times);np.testing.assert_allclose(logdet,jnp.linalg.slogdet(jax.jacfwd(mapping)(raw))[1],atol=1e-10)
g=b.marginal_design_score(times);eps=1e-5;delta=jnp.zeros_like(times).at[2,0].set(eps);fd=(b.marginal_design_log_lik(times+delta)-b.marginal_design_log_lik(times-delta))/(2*eps);np.testing.assert_allclose(g[2,0],fd,atol=1e-5,rtol=1e-5)
cfg=config(5,877,1024);assert 'IqbalGRUNet' in cfg['policy_network']['_target_'];assert cfg['policy']['b1']==.9 and cfg['policy']['b2']==.999 and cfg['policy']['lr_schedule']['value']==1e-4;assert cfg['score']['y_score_loss_weight']==.02 and cfg['score']['clip_grad_norm']==10000;assert cfg['score_network']['base_score_network']['positional'] is True
(out/'scorebed_native_dynamics_reference_tests.json').write_text(json.dumps(dict(status='PASS',native_GRU_forward=True,native_GRU_parameter_gradients=True,sampler_components=results,triangular_Jacobian=True,uniform_component_density_gradient=True,native_optimizer_config=True,full_mixture_smooth_density='not defined for mixed atomic/continuous design measure; not required by native score objective'),indent=2));print('Native dynamics GRU, gradients, seven samplers, density and optimizer configuration PASS')
