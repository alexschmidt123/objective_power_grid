"""Frozen ScoreBED SIR adapter tests and reliability-screened marginal y-score diagnostics."""
import os;os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import argparse,hashlib,json,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'src/research/vendor/scorebed')]
import numpy as np,jax,jax.numpy as jnp
from flax import serialization
from policy_learning.loaders import load_score_trainer
from policy_learning.utils.jax import derive_key
from src.research.sir_scorebed_model import SIRScoreModel,SIRScorePolicyNet

def main(old,out):
 start=time.perf_counter();old=Path(old);out=Path(out);out.mkdir(parents=True,exist_ok=False)
 run=old/'sir/seed101/scorebed'
 if not run.exists():run=old/'worker_ubuntu/sir/seed101/scorebed'
 d=run/'models';cfg=json.loads((d/'scorebed_official_config.json').read_text());sir=json.loads((d/'scorebed_sir_configuration.json').read_text());target=SIRScoreModel(T=5,init_seed=101,**sir)
 norm=np.load(d/'scorebed_normalization.npz')
 for k in norm.files:setattr(target,k,jnp.asarray(norm[k]))
 target.standardised=True
 score,_=load_score_trainer(derive_key(101,'score_trainer'),cfg,target,callbacks=[])
 variables=serialization.msgpack_restore((d/'scorebed_official_score.msgpack').read_bytes())
 variables=jax.tree_util.tree_map(jnp.asarray,variables)
 score_fn=jax.jit(lambda y,x:score._score_from_z_with_variables(variables,jnp.concatenate([y,x],-1),{}))
 cases=json.loads((run/'eval/scorebed_seed1001.json').read_text())
 bank=np.load(old/'evaluations/development_bank/contrasts_resolution.npy');assert len(bank)>=131072
 allrows=[]
 # Paired independent prior banks: no reference accepted solely for a finite estimate.
 for idx in [0,1,7,31]:
  row=cases[idx];x=jnp.asarray(row['actions']);y=jnp.asarray(row['observations']);pred=np.asarray(score_fn(y[None],x[None]))[0,:,:1];refs=[];esses=[]
  for block in range(2):
   means=[]
   for begin in range(block*65536,(block+1)*65536,2048):
    th=jnp.broadcast_to(jnp.asarray(bank[begin:begin+2048])[:,None,:],(2048,5,2));xx=jnp.broadcast_to(x,(2048,5,1))
    means.append(np.asarray(jax.jit(target.means)(xx,th)))
   mu=np.concatenate(means);logw=-.5*np.square((np.asarray(y)[None]-mu)/target.sigma).sum((1,2));w=np.exp(logw-logw.max());w/=w.sum()
   refs.append(np.sum(w[:,None,None]*(mu-np.asarray(y))/target.sigma**2,0));esses.append(float(1/np.sum(w*w)))
  discrepancy=float(np.sqrt(np.mean((refs[0]-refs[1])**2)));reliable=min(esses)>=100 and discrepancy<=.05
  rec=dict(trajectory=idx,actions=np.asarray(x).tolist(),predicted_y_score=pred.tolist(),reference_y_scores=[r.tolist() for r in refs],reference_ESS=esses,reference_disagreement_RMSE=discrepancy,accepted_reference=reliable,y_score_RMSE=float(np.sqrt(np.mean((pred-(refs[0]+refs[1])/2)**2))) if reliable else None)
  allrows.append(rec);print(json.dumps(rec),flush=True)
 # Fixed physical-data likelihood and design derivatives: independently finite-differenced.
 x=jnp.asarray([[2.],[6.],[14.],[25.],[45.]]);th=jnp.broadcast_to(jnp.asarray([.5,.1]),(5,2));mu=target.means(x,th);y=mu+jnp.asarray([[.1],[-.2],[.3],[-.1],[.2]])
 logp=lambda xx:target.data_log_lik(y,xx,th)
 analytic=np.asarray(jax.grad(logp)(x));eps=1e-4;finite=[]
 for t in range(5):
  e=jnp.zeros_like(x).at[t,0].set(eps);finite.append(float((logp(x+e)-logp(x-e))/(2*eps)))
 np.testing.assert_allclose(analytic[:,0],finite,rtol=2e-4,atol=2e-5)
 # Policy VJP must include both changing observations and the chronological time map.
 from policy_learning.utils.func import noop
 from policy_learning.loaders import load_policy_trainer
 net=SIRScorePolicyNet(encoder_shapes=[256],encoding_dim=16,enc_act='relu',emit_act='relu',emitter_shapes=[],shape=(5,1),zero_init_final_layer=True,time_scale=target.end,count_scale=target.population)
 pol,_=load_policy_trainer(derive_key(101,'policy_trainer',0),cfg,target,grad_fn=noop,callbacks=[],policy_net=net)
 params=pol.training_state.params;theta=th;noise=jnp.array([[.1],[.2],[-.3],[.1],[.4]])
 def scalar(shift):
  pp=dict(params);em=dict(pp['emitter_net']);layer=dict(em['layers_0']);layer['bias']=layer['bias']+shift;em['layers_0']=layer;pp['emitter_net']=em
  yy,xx,_=target.reparam_sample(pp,net,theta,noise,None);return yy.sum()/500+xx.sum()/100
 vjp=float(jax.grad(scalar)(0.));fd=float((scalar(1e-3)-scalar(-1e-3))/(2e-3));np.testing.assert_allclose(vjp,fd,rtol=2e-3,atol=2e-4);assert abs(vjp)>1e-5
 record=dict(status='PASS',diagnostic_only=True,additional_training_episodes=0,diagnostic_initialization_episodes=14,source_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),score_bundle_sha256=hashlib.sha256((d/'scorebed_official_score.msgpack').read_bytes()).hexdigest(),normalization_sha256=hashlib.sha256((d/'scorebed_normalization.npz').read_bytes()).hexdigest(),likelihood_design_gradient_max_error=float(np.max(np.abs(analytic[:,0]-finite))),policy_vjp=vjp,policy_finite_difference=fd,marginal_y_score_cases=allrows,limitations='Marginal reference is importance sampling; rejected cases provide no score-accuracy conclusion. Design-score accuracy not established by marginal y-score check. Sampler is prior-independent sequential uniform feasible times, a SIR benchmark adaptation.',seconds=time.perf_counter()-start)
 (out/'score_adapter_audit.json').write_text(json.dumps(record,indent=2));print('PASS',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--old',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.old,a.output)
