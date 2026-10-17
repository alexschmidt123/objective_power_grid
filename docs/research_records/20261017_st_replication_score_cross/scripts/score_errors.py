"""Frozen score-error diagnostics; no training or policy selection."""
import os,sys,json,time,hashlib
from pathlib import Path
os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2'
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:])
R=Path('/home/grads/g/g.lin/Documents/objective_power_grid');C=R/'experiments/10172026/10172026_st_replication_score_reverse_cross';S=C/'score_source';Q=R/'experiments/10162026/10162026_location_root_cause_v4';O=C/'reports';OLD=R/'experiments/10152026/10152026_journal_five_seed_fixed_budget/location'
sys.path[:0]=[str(S),str(S/'src/research/vendor/scorebed')]
import numpy as np,jax,jax.numpy as jnp
from src.research.score_frozen import restore
from src.research.scorebed_worker import CompatibleLocationFinding
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def reference(y,x,bank,resolutions=(65536,262144,1000000)):
 z=0.;z2=0.;scale=-np.inf;g=np.zeros((30,3));reports={};N=0
 for stop in resolutions:
  for st in range(N,stop,8192):
   theta=bank[st:min(st+8192,stop)].reshape(-1,2,2);diff=x[None,:,None,:]-theta[:,None,:,:];r2=1e-4+(diff**2).sum(-1);signal=.1+(1/r2).sum(-1);mu=np.log(signal);res=y[None,:,0]-mu;ll=-.5*(res**2).sum(-1)/.25;new=max(scale,float(ll.max()));factor=np.exp(scale-new);w=np.exp(ll-new);z=z*factor+w.sum();z2=z2*factor**2+(w*w).sum();dmu=(-2*diff/(r2*r2)[...,None]).sum(2)/signal[...,None];cond=np.concatenate((-res[...,None]/.25,res[...,None]/.25*dmu),-1);g=g*factor+np.einsum('n,ntd->td',w,cond);scale=new
  N=stop;reports[str(stop)]=dict(score=(g/z).tolist(),ess=float(z*z/z2))
 return reports
def log_marginal(y,x,bank):
 th=bank.reshape(-1,2,2);diff=x[None,:,None,:]-th[:,None,:,:];mu=np.log(.1+(1/(1e-4+(diff**2).sum(-1))).sum(-1));ll=-.5*((y[None,:,0]-mu)**2).sum(-1)/.25;m=ll.max();return float(m+np.log(np.exp(ll-m).mean()))
started=time.monotonic()
paths={(102,102):Q/'source/experiments/10162026/10162026_root_cause_v4/score_model102_policy102',(103,102):Q/'source/experiments/10162026/10162026_root_cause_v4/score_model103_policy102',(103,103):OLD/'seed103/scorebed',(102,103):C/'runs/score102_policy103'}
histories=[];coverage=[]
for (s,p),run in paths.items():
 rows=json.loads((run/'eval/scorebed_seed1001.json').read_text());fractions=np.array([np.mean(abs(np.array(r['actions']))>3.8) for r in rows]);order=np.argsort(fractions,kind='stable')
 coverage.append(dict(score_seed=s,policy_seed=p,mean_boundary_coordinate_fraction=float(fractions.mean()),fraction_episodes_touching_boundary=float((fractions>0).mean())))
 for label,indices in [('most_boundary',order[-2:]),('least_boundary',order[:2])]:
  for index in indices:
   r=rows[int(index)];histories.append(dict(score_seed=s,policy_seed=p,case=int(index),stratum=label,boundary_fraction=float(fractions[index]),y=np.array(r['observations'],float),x=np.array(r['actions'],float)))
bank=np.load(OLD/'seed102/scorebed/eval/contrast_bank_seed1001.npy')
# Independent finite-difference check of the Fisher identity reference.
h=histories[0];small=bank[:4096];ref=np.array(reference(h['y'],h['x'],small,(4096,))['4096']['score']);eps=1e-5;tests=[]
for channel in ['y','x']:
 x=h['x'].copy();y=h['y'].copy();a=y if channel=='y' else x;a[0,0]+=eps;plus=log_marginal(y,x,small);a[0,0]-=2*eps;minus=log_marginal(y,x,small);fd=(plus-minus)/(2*eps);target=ref[0,0 if channel=='y' else 1];assert abs(fd-target)<1e-4*max(1.,abs(target));tests.append(dict(channel=channel,finite_difference=fd,fisher_score=float(target)))
predictions={}
for seed in [102,103]:
 frozen,*_=restore(Q/f'bundles/score_{seed}.json',CompatibleLocationFinding)
 @jax.jit
 def predict(y,x):return frozen.score_network.apply(frozen.variables,y=y,xi=x,sigma=jnp.ones(y.shape[:-2]),aux_data={},train=False)
 values=[]
 for h in histories:
  v=np.asarray(predict(jnp.asarray(h['y'][None],dtype=jnp.float32),jnp.asarray(h['x'][None],dtype=jnp.float32)))[0];assert np.isfinite(v).all();values.append(v)
 predictions[seed]=np.array(values);np.save(O/f'score_error_predictions_{seed}.npy',predictions[seed])
records=[];refs=[]
for i,h in enumerate(histories):
 rr=reference(h['y'],h['x'],bank);final=np.array(rr['1000000']['score']);earlier=np.array(rr['262144']['score']);resolution=float(np.linalg.norm(final-earlier)/max(np.linalg.norm(final),1e-12));reliable=rr['1000000']['ess']>=32 and resolution<=.2
 info={k:v for k,v in h.items() if k not in ['x','y']};refs.append(dict(**info,ess=rr['1000000']['ess'],resolution_change=resolution,reliable=reliable,resolutions=rr))
 for seed in [102,103]:
  row=dict(**info,evaluated_score_bundle=seed,reference_reliable=reliable,reference_ess=rr['1000000']['ess'],reference_resolution_change=resolution)
  for label,sl in [('joint',slice(None)),('data',slice(0,1)),('design',slice(1,3))]:
   a=predictions[seed][i,:,sl].ravel();b=final[:,sl].ravel();row[label+'_relative_error']=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12));row[label+'_cosine']=float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12));row[label+'_norm_ratio']=float(np.linalg.norm(a)/max(np.linalg.norm(b),1e-12))
  records.append(row)
 write(O/'score_error_records.json',records);write(O/'score_error_references.json',refs);print('SCORE_REFERENCE_COMPLETE',i+1,len(histories),'reliable',reliable,flush=True)
summary=[]
for seed in [102,103]:
 for stratum in ['most_boundary','least_boundary']:
  rows=[r for r in records if r['evaluated_score_bundle']==seed and r['stratum']==stratum];good=[r for r in rows if r['reference_reliable']]
  summary.append(dict(score_bundle=seed,stratum=stratum,cases=len(rows),reliable_cases=len(good),reliable_mean_relative_error=float(np.mean([r['joint_relative_error'] for r in good])) if good else None,reliable_mean_design_cosine=float(np.mean([r['design_cosine'] for r in good])) if good else None))
result=dict(status='complete',seconds=time.monotonic()-started,histories=len(histories),model_predictions=32,new_primary_simulations=0,finite_difference_checks=tests,coverage=coverage,summary=summary,reference='Fisher identity explicit-likelihood prior importance, diagnostic only',reliability_rule='ESS>=32 and relative score change262144 to1M<=0.2',limitation='Extremal boundary strata are descriptive and not a representative population average. Low-ESS references cannot establish score error or its causal effect. No score-model hyperparameters changed.')
write(O/'score_errors_summary.json',result)
with (O/'score_report.md').open('a') as f:
 f.write('\n## Frozen score-error diagnosis\n\n16 saved histories: two most-boundary and two least-boundary cases for each crossing, both score bundles on identical histories. Native GPU normalization is loaded verbatim even though predictions run on CPU. No score or policy training. Fisher identity reference checked against finite differences; reliable only with ESS>=32 and <=20% score change from262144 to1M contrasts.\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n\nA score-training correction is not identified solely by crossing or training score loss. Do not substitute a favorable frozen bundle into primary results. If trustworthy references show a consistent boundary-specific deficit, the next single controlled intervention is increased feasible-boundary coverage in the score-training design sampler at the same episode budget. Otherwise retain the unresolved mechanism and avoid selecting a speculative correction. This is a recommendation gate, not an implemented change.\n')
print('SCORE_HISTORY_DIAGNOSIS_COMPLETE',flush=True)
