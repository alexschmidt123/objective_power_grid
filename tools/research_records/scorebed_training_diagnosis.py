"""CPU-only saved-score diagnostic; never trains or changes EIG results."""
import os,sys,pathlib,json,time,hashlib
os.environ['JAX_PLATFORMS']='cpu';os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
R=pathlib.Path(os.environ.get('MOE_PROJECT_ROOT',pathlib.Path(__file__).resolve().parents[2]));S=R/'experiments/10162026/10162026_location_stability_correction/source';sys.path.insert(0,str(S));sys.path.insert(0,str(S/'src/research/vendor/scorebed'))
import numpy as np
import jax,jax.numpy as jnp
from flax import serialization
from policy_learning.loaders import _build_score_network
from src.research.scorebed_worker import CompatibleLocationFinding
C=R/'experiments/10162026/10162026_location_stability_correction';O=C/'scorebed_training_diagnosis';O.mkdir(exist_ok=True);OLD=R/'experiments/10152026/10152026_journal_five_seed_fixed_budget'
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));os.replace(t,p)
def describe(x):
 x=np.asarray(x);return dict(mean=float(np.mean(x)),median=float(np.median(x)),minimum=float(np.min(x)),maximum=float(np.max(x)))
def reference(y,x,bank,resolutions=(65536,262144,1000000)):
 # Fisher identity: posterior average of conditional likelihood scores, for
 # diagnosis only. Report ESS/resolution; do not call low-ESS results truth.
 z=0.;z2=0.;scale=-np.inf;g=np.zeros((30,3));reports={};N=0
 for stop in resolutions:
  for st in range(N,stop,8192):
   theta=bank[st:min(st+8192,stop)].reshape(-1,2,2);diff=x[None,:,None,:]-theta[:,None,:,:];r2=1e-4+(diff**2).sum(-1);signal=.1+(1/r2).sum(-1);mu=np.log(signal);res=y[None,:,0]-mu;ll=-.5*(res**2).sum(-1)/.25;new=max(scale,float(ll.max()));factor=np.exp(scale-new);w=np.exp(ll-new);z=z*factor+w.sum();z2=z2*factor**2+(w*w).sum();dmu=(-2*diff/(r2*r2)[...,None]).sum(2)/signal[...,None];cond=np.concatenate((-res[...,None]/.25,res[...,None]/.25*dmu),-1);g=g*factor+np.einsum('n,ntd->td',w,cond);scale=new
  N=stop;reports[str(stop)]=dict(score=(g/z).tolist(),ess=float(z*z/z2))
 return reports

def main():
 start=time.monotonic();allrows={seed:json.loads((OLD/f'location/seed{seed}/scorebed/eval/scorebed_seed1001.json').read_text())for seed in[102,103,104]};bank=np.load(OLD/'location/seed102/scorebed/eval/contrast_bank_seed1001.npy');histories=[]
 for source in[102,103]:
  for i,row in enumerate(allrows[source][:8]):histories.append(dict(source_seed=source,case=i,y=np.array(row['observations'],dtype=np.float32),x=np.array(row['actions'],dtype=np.float32),truth=row['truth']))
 predictions={};checks=[];coverage=[]
 for seed in[102,103,104]:
  mp=OLD/f'location/seed{seed}/scorebed/models';cfg=json.loads((mp/'scorebed_official_config.json').read_text());target=CompatibleLocationFinding(T=30,d=2,k=2,init_seed=seed,sigma=.5,standardise=False,design_bounds=[-4,4]);target.compute_standardisation(10000);net=_build_score_network(cfg,target);variables=serialization.msgpack_restore((mp/'scorebed_official_score.msgpack').read_bytes());variables=jax.tree_util.tree_map(jnp.asarray,variables)
  @jax.jit
  def predict(y,x):return net.apply(variables,y=y,xi=x,sigma=jnp.ones(y.shape[:-2]),aux_data={},train=False)
  preds=[]
  for h in histories:
   a=np.asarray(predict(jnp.asarray(h['y'][None]),jnp.asarray(h['x'][None])))[0];assert a.shape==(30,3)and np.isfinite(a).all();preds.append(a)
  predictions[str(seed)]=np.array(preds);np.save(O/f'score_predictions_seed{seed}.npy',predictions[str(seed)])
  # Verify feasibility derivative is present and has expected chain rule.
  raw=jnp.array([-3.,-.5,0.,.5,3.]);jac=np.asarray(jax.jacfwd(target.feasible_design)(raw));expected=np.diag(np.asarray(8*jax.nn.sigmoid(raw)*(1-jax.nn.sigmoid(raw))));checks.append(dict(seed=seed,bounded_mapping_jacobian_error=float(np.max(np.abs(jac-expected))),standardisation_stats_finite=bool(all(np.isfinite(np.asarray(v)).all()for v in target.get_std_stats_dict().values()))))
  skw=cfg['score']['design_sampler_kwargs'];kind=cfg['score']['design_sampler_type'];y,x,theta,aux,key=target.fast_sample(jax.random.key(721009),(256,),sampler_type=kind,sampler_kwargs=skw);x=np.asarray(x);coverage.append(dict(seed=seed,score_training_distribution_boundary_coordinate_fraction=float(np.mean(np.abs(x)>3.8)),score_training_distribution_coordinate_sd=float(x.std()),saved_policy_boundary_fraction=float(np.mean(np.abs(np.array([r['actions']for r in allrows[seed]]))>3.8))))
  print('SCORE_MODEL_PREDICTIONS',seed,flush=True)
 write(O/'mapping_and_coverage.json',dict(checks=checks,coverage=coverage,diagnostic_only_simulations=3*(10000+256),score_model_architecture_unchanged=True))
 refs=[];errors=[]
 for j,h in enumerate(histories):
  rep=reference(h['y'].astype(float),h['x'].astype(float),bank);final=np.array(rep['1000000']['score']);earlier=np.array(rep['262144']['score']);resolution=float(np.linalg.norm(final-earlier)/max(np.linalg.norm(final),1e-12));reliable=rep['1000000']['ess']>=32 and resolution<=.2;refs.append(dict(source_seed=h['source_seed'],case=h['case'],ess=rep['1000000']['ess'],relative_resolution_change=resolution,reliable_reference=reliable,resolutions=rep))
  for seed,preds in predictions.items():
   p=preds[j];row=dict(score_training_seed=int(seed),history_source_seed=h['source_seed'],case=h['case'],reference_ess=rep['1000000']['ess'],reference_resolution_change=resolution,reliable_reference=reliable)
   for name,sl in [('joint',slice(None)),('data',slice(0,1)),('design',slice(1,3))]:
    a=p[:,sl].ravel();b=final[:,sl].ravel();row[name+'_relative_error']=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12));row[name+'_cosine']=float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12));row[name+'_norm_ratio']=float(np.linalg.norm(a)/max(np.linalg.norm(b),1e-12))
   errors.append(row)
  write(O/'score_reference_records.json',refs);write(O/'score_quality_records.json',errors);print('REFERENCE',j+1,'/16','ESS',rep['1000000']['ess'],'resolution',resolution,'reliable',reliable,flush=True)
 summary=[]
 for seed in[102,103,104]:
  for source in[102,103]:
   rr=[x for x in errors if x['score_training_seed']==seed and x['history_source_seed']==source];good=[x for x in rr if x['reliable_reference']];summary.append(dict(score_model_seed=seed,history_source_seed=source,cases=len(rr),reliable_cases=len(good),all_case_metrics={k:describe([x[k]for x in rr])for k in['joint_relative_error','joint_cosine','design_relative_error','design_cosine','data_relative_error','data_cosine']},reliable_case_metrics={k:describe([x[k]for x in good])for k in['joint_relative_error','joint_cosine','design_cosine']}if good else None))
 write(O/'summary.json',dict(status='complete',seconds=time.monotonic()-start,summary=summary,mapping_checks=checks,coverage=coverage,reference_method='Fisher identity explicit-likelihood prior-importance diagnostic; not a new policy training loss or EIG estimator',limitation='Only16 histories; reference reliability filtered byESS>=32 and <=20% change from262k to1M. No training correction selected automatically.'))
 print('SCOREBED_TRAINING_DIAGNOSIS_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  import traceback;write(O/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()));raise
