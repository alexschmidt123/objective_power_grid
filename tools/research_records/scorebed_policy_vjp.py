import os,sys,pathlib,json,time
os.environ['JAX_PLATFORMS']='cpu';os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:])
R=pathlib.Path(os.environ.get('MOE_PROJECT_ROOT',pathlib.Path(__file__).resolve().parents[2]));C=R/'experiments/10162026/10162026_location_stability_correction';S=C/'source';O=C/'scorebed_training_diagnosis';OLD=R/'experiments/10152026/10152026_journal_five_seed_fixed_budget';sys.path.insert(0,str(S));sys.path.insert(0,str(S/'src/research/vendor/scorebed'))
import numpy as np
import jax,jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from src.research.scorebed_worker import CompatibleLocationFinding
from policy_learning.loaders import build_policy_network

def write(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False));os.replace(q,p)
def comparison(a,b):
 return dict(cosine=float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12)),relative_error=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12)),norm_ratio=float(np.linalg.norm(a)/max(np.linalg.norm(b),1e-12)))
def main():
 deadline=time.monotonic()+1800
 while not (O/'summary.json').exists():
  if (O/'failure.json').exists():raise RuntimeError('Score diagnostic failed')
  if time.monotonic()>deadline:raise RuntimeError('Score diagnostic wait cap')
  time.sleep(10)
 refs=json.loads((O/'score_reference_records.json').read_text());preds={s:np.load(O/f'score_predictions_seed{s}.npy')for s in[102,103,104]};records=[];summaries=[]
 for seed in[102,103]:
  run=OLD/f'location/seed{seed}/scorebed';rows=json.loads((run/'eval/scorebed_seed1001.json').read_text());cfg=json.loads((run/'models/scorebed_official_config.json').read_text());target=CompatibleLocationFinding(T=30,d=2,k=2,init_seed=seed,sigma=.5,standardise=False,design_bounds=[-4,4]);net=build_policy_network(target,cfg['policy_network']);arrays=dict(np.load(run/'models/scorebed_official_policy.npz'));params={}
  for k,v in arrays.items():
   if '/'in k:a,b,c=k.split('/');params.setdefault(a,{}).setdefault(b,{})[c]=jnp.asarray(v)
  @jax.jit
  def pull(theta,noise,cot):
   value,vjp=jax.vjp(lambda pp:target.reparam_sample(pp,net,theta,noise,jnp.zeros((30,2))),params)
   grad=vjp((cot[:,:1],cot[:,1:],{}))[0];flat,_=ravel_pytree(grad);return value[0],value[1],-flat
  learned=[];exact=[]
  for i in range(8):
   j=i if seed==102 else i+8;ref=refs[j];row=rows[i];theta=jnp.broadcast_to(jnp.array(row['truth']).reshape(2,2),(30,2,2));noise=jnp.array(row['measurement_noise_standard']);cot=jnp.array(ref['resolutions']['1000000']['score']);y,x,g=pull(theta,noise,cot);g=np.asarray(g);maxx=float(np.max(np.abs(np.asarray(x)-np.array(row['actions']))));maxy=float(np.max(np.abs(np.asarray(y)-np.array(row['observations']))));valid=ref['reliable_reference']and max(maxx,maxy)<1e-3;components={}
   for model in[102,103,104]:
    _,_,v=pull(theta,noise,jnp.array(preds[model][j]));v=np.asarray(v);components[str(model)]=comparison(v,g)
    if model==seed and valid:learned.append(v);exact.append(g)
   records.append(dict(policy_seed=seed,case=i,reference_reliable=ref['reliable_reference'],rollout_max_action_error=maxx,rollout_max_observation_error=maxy,valid_for_gradient_diagnosis=valid,score_model_gradient_comparisons=components))
   write(O/'policy_gradient_records.json',records);print('VJP',seed,i,'valid',valid,'own',components[str(seed)],flush=True)
  item=dict(policy_seed=seed,reliable_matched_cases=len(learned))
  if learned:item.update(mean_gradient_comparison=comparison(np.mean(learned,0),np.mean(exact,0)),case_cosines=[x['score_model_gradient_comparisons'][str(seed)]['cosine']for x in records if x['policy_seed']==seed and x['valid_for_gradient_diagnosis']])
  summaries.append(item)
 write(O/'policy_gradient_summary.json',dict(status='complete',results=summaries,meaning='Native policy VJP of learned marginal scores versus explicit-likelihood Fisher reference; not a new training objective or EIG estimator',limitations='Small held-out subset, posterior-importance reliability filter, conditional diagnostic not proof of global policy improvement'))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  import traceback;write(O/'policy_gradient_failure.json',dict(error=str(e),traceback=traceback.format_exc()));raise
