"""Read-only aggregation of matched MoE controls; never trains or selects models."""
import argparse,csv,json,math,pathlib
import numpy as np

def interval(x):
 x=np.asarray(x,float);n=len(x);mean=float(x.mean())
 if n<2:return dict(mean=mean,sd=None,ci95=None,n=n)
 from scipy.stats import t
 sd=float(x.std(ddof=1));half=float(t.ppf(.975,n-1))*sd/math.sqrt(n)
 return dict(mean=mean,sd=sd,ci95=[mean-half,mean+half],n=n)

def collect(campaign):
 campaign=pathlib.Path(campaign).resolve();out=campaign/'analysis';out.mkdir(exist_ok=True)
 runs=[];data={};training={};reference_identity=None
 for path in sorted(campaign.glob('location_finding_eig_T30_train*_Nobs0')):
  if not (path/'exit_code').exists() or (path/'exit_code').read_text().strip()!='0':continue
  cfg=json.loads((path/'run_config.json').read_text());seed=cfg['train_seed'];settings=cfg['settings']
  if settings['eval_contrasts']!=1000000 or settings['eval_systems']!=2000 or cfg['benchmark']['location_finding']['design_bounds']!=[-4.,4.]:raise ValueError('Mixed evaluation protocol')
  for row in json.loads((path/'summary.json').read_text()):
   method=row['method'];rows=json.loads((path/'eval'/f'{method}_seed1002.json').read_text())
   identity=[(r['truth'],r['measurement_noise_standard'],r['trajectory_id']) for r in rows]
   if reference_identity is None:reference_identity=identity
   elif identity!=reference_identity:raise ValueError('Evaluation cases are not paired')
   actions=np.asarray([r['actions'] for r in rows])
   if not np.isfinite(actions).all() or actions.min() < -4.000001 or actions.max()>4.000001:raise ValueError('Invalid design bounds')
   scores=np.asarray([r['spce'] for r in rows]);mse=np.asarray([r['inference']['mse'] for r in rows])
   if not np.isfinite(scores).all() or not np.isfinite(mse).all():raise ValueError('Nonfinite scores')
   tr=json.loads((path/'models'/f'{method}_training.json').read_text())
   if tr['generated_training_trajectories']!=1000000:raise ValueError('Training budgets differ')
   data[(method,seed)]=scores;training[(method,seed)]=tr
   routing=np.asarray([[q['weights'] for q in r['routing']] for r in rows]) if rows[0]['routing'] else None
   utilization=None if routing is None else dict(probabilities_by_stage=routing.mean(0).tolist(),execution_fraction_by_stage=(routing>0).mean(0).tolist(),hard_choice_by_stage=np.eye(4)[routing.argmax(-1)].mean(0).tolist())
   runs.append(dict(method=method,train_seed=seed,evaluation_seed=1002,eig=float(scores.mean()),mse=float(mse.mean()),conditional_eval_mc_se=float(scores.std(ddof=1)/math.sqrt(len(scores))),training_seconds=tr['training_seconds'],diagnostic_validation_seconds=tr.get('diagnostic_validation_seconds',0.),parameters=tr['policy_parameters'],routing=utilization,source=str(path)))
 methods=sorted({r['method'] for r in runs});summary=[]
 for method in methods:
  rr=[r for r in runs if r['method']==method]
  summary.append(dict(method=method,eig=interval([r['eig'] for r in rr]),secondary_mse=interval([r['mse'] for r in rr]),training_minutes=interval([r['training_seconds']/60 for r in rr]),parameters=[r['parameters'] for r in rr]))
 pairs=[('attention_single','sum_single','encoder'),('attention_dense_moe','attention_single','expert_architecture'),('attention_sparse_moe','attention_dense_moe','sparse_routing'),('attention_dense_moe','attention_matched_dense','dense_capacity_control'),('attention_sparse_moe','attention_matched_sparse','sparse_capacity_control'),('attention_dense_moe','attention_uniform_moe','learned_specialization'),('attention_sparse_moe','dad','original_DAD_reference')]
 differences=[]
 for a,b,label in pairs:
  seeds=sorted({s for m,s in data if m==a}&{s for m,s in data if m==b})
  if seeds:differences.append(dict(comparison=label,first=a,second=b,training_seeds=seeds,eig_difference=interval([float((data[(a,s)]-data[(b,s)]).mean()) for s in seeds])))
 result=dict(status='complete' if len(runs)==24 else 'partial',expected_trainings=24,completed_trainings=len(runs),primary_metric='EIG (nats; common held-out sPCE L=1000000)',seed_uncertainty='Student t interval across independent training seeds, conditional on paired evaluation bank; three seeds give exploratory evidence',evaluation_seed=1002,summary=summary,paired_differences=differences,runs=runs,limitations=['Small training-seed sample','Finite-contrast lower bound, not exact mutual information','Native sum encoder has16-dimensional bottleneck; same-head zero padding does not equalize encoder information capacity','MSE secondary and posterior convergence remains a separate check'])
 (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False))
 with (out/'results.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['method','train_seed','evaluation_seed','eig','mse','conditional_eval_mc_se','training_seconds','diagnostic_validation_seconds','parameters','source']);writer.writeheader();writer.writerows({k:v for k,v in r.items() if k!='routing'} for r in runs)
 if runs:
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  fig,ax=plt.subplots(figsize=(10,6))
  for method in methods:
   histories=[training[(method,s)].get('validation_history',[]) for m,s in data if m==method]
   histories=[h for h in histories if h]
   if not histories:continue
   curves=[]
   for h in histories:
    curves.append([next(x['eig'] for x in v['scores'] if x['routing']!='relaxed') for v in h])
   values=np.asarray(curves);x=np.asarray([v['primary_episodes'] for v in histories[0]])
   if all(np.array_equal(x,[v['primary_episodes'] for v in h]) for h in histories):ax.plot(x,values.mean(0),label=method)
  ax.set(xlabel='Primary training simulator episodes',ylabel='Held-out EIG estimate (nats; sPCE L=65536)',title='Diagnostic learning curves; final comparisons use L=1000000');ax.legend(fontsize=7);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=160);plt.close(fig)
 print(json.dumps(dict(status=result['status'],completed=result['completed_trainings'],paired_differences=differences),indent=2))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);collect(p.parse_args().campaign)
