"""Complete or explicitly partial report for the one-factor encoder study."""
import argparse,json,math
from pathlib import Path
import numpy as np

def read(path):return json.loads(path.read_text())
def paired(a,b):
 assert len(a)==len(b)==2000
 assert all(x['truth']==y['truth'] and x['measurement_noise_standard']==y['measurement_noise_standard'] and x['trajectory_id']==y['trajectory_id'] for x,y in zip(a,b))
 d=np.array([x['spce']-y['spce'] for x,y in zip(a,b)])
 return dict(mean=float(d.mean()),conditional_mc_se=float(d.std(ddof=1)/math.sqrt(len(d))))

def training_distribution(training):
 values=[v for r in training.get('history',[]) if r.get('primary_episodes',0)>=800000 for v in r.get('training_scores',[])]
 if not values:return dict(status='Historical per-trajectory scores not saved; use explicitly labeled fresh checkpoint probes.')
 x=np.array(values)
 return dict(n=len(x),sampling='Per-trajectory scores in periodically logged minibatches during last20percent training; not all1million scores.',
             quantiles=dict(zip(['min','p10','median','p90','p99','max'],np.quantile(x,[0,.1,.5,.9,.99,1]).tolist())),
             mean=float(x.mean()),fraction_within_01_of_ceiling=float(np.mean(x>math.log(1025)-.1)))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);args=ap.parse_args()
 root=Path.cwd();c=Path(args.campaign).resolve();s=c/'encoder_correction'
 report=dict(status=read(s/'status.json') if (s/'status.json').exists() else {'status':'pending'},
             root_cause='History encoder is the largest controlled EIG bottleneck in seed101; routing/clipping contributes but hard/relaxed inference mismatch is negligible.',
             selected_correction='Replace location attention history with original DAD sum16 set encoder, fixed padding and unchanged stage feature. Four expert heads and DSelect1 remain unchanged.',
             ablations={},seeds={},seed_aggregates={},comparisons={},expert_utilization={},
             interpretation_limits=['Evaluation1001 was used for development and is not an untouched test.',
                                    'Three training seeds provide limited training-variation evidence.',
                                    'A is a sum encoder with the common nonlinear control head; published DAD has its original emitter.',
                                    'A/B native context dimensions differ; A padding is fixed, not learned.',
                                    'B sees all four attention contexts; each C/D expert sees one.',
                                    'Improvement over original DAD alone does not establish a MoE advantage over the stronger single-policy A.',
                                    'Paired trajectory intervals condition on the shared contrast bank.',
                                    'Expert use alone does not prove useful specialization.'])
 for label,method in dict(A='sum_single',B='attention_single',C='attention_dense_moe',D='attention_sparse_moe',capacity='attention_matched_dense').items():
  f=c/'controls/eval'/f'{method}_seed1001.json'
  if f.exists():
   rows=read(f);train=read(c/'controls/models'/f'{method}_training.json')
   report['ablations'][label]=dict(eig=float(np.mean([r['spce'] for r in rows])),training_episodes=train['generated_training_trajectories'],training_seconds=train['training_seconds'],total_parameters=train['policy_parameters'],active_parameters_nonfirst=train['active_parameters_nonfirst'])
   report['ablations'][label]['training_score_distribution']=training_distribution(train)
 for name in ['implementation.json','pilot_cost_check.json','multiseed_cost_check.json','completed_control_evidence.json']:
  if (s/name).exists():report[name]=read(s/name)
 if len(report['ablations'])<5:
  report['root_cause']='Preliminary: history encoding has the largest observed loss among completed controls; sparse and capacity controls remain pending.'
 report['reused_baselines']={}
 baseline=root/'experiments/10132026/10132026_bounded_location_restart/pilot'
 for method in ['random','fixed','dad','rl_sboed','vsoed','scorebed']:
  source=baseline
  fresh=root/'experiments/10142026/10142026_vsoed_incremental_discount_correction/pilot'
  if method=='vsoed' and (fresh/'eval/vsoed_seed1001.json').exists():source=fresh
  f=source/'eval'/f'{method}_seed1001.json'
  if f.exists():
   rows=read(f);report['reused_baselines'][method]=dict(eig=float(np.mean([r['spce'] for r in rows])),mse=float(np.mean([r['inference']['mse'] for r in rows])),source=str(source.relative_to(root)),reused=True)
 diagnostics=read(c/'checkpoint_diagnostics/diagnosis.json')
 report['training_objective_distributions']=diagnostics['checkpoint_probes']
 report['mse_diagnosis']=diagnostics['mse_audit']
 report['mse_interpretation']='Policy failures are supported by paired information deficits and persistent errors under32768particles/32moves/3seeds. Posterior multimodality and budget sensitivity affect some point estimates; label-permutation error is already removed. Three selected cases cannot certify all2000 inferences.'
 report['permutation']=diagnostics['permutation']
 routing=c/'routing_eig_diagnostics/report.json'
 if routing.exists():
  r=read(routing);report['hard_relaxed']={name:{k:v for k,v in item.items() if k!='relaxed_scores'} for name,item in r['results'].items()}
 stored={}
 for seed in [101,102,103]:
  paths=dict(original=(root/'experiments/10142026/10142026_moe_component_clipping/pilot' if seed==101 else s/f'original_seed{seed}'),
             corrected=s/f'corrected_seed{seed}',dad=(root/'experiments/10132026/10132026_bounded_location_restart/pilot' if seed==101 else s/f'original_seed{seed}'))
  report['seeds'][str(seed)]={}
  for label,path in paths.items():
   method='dad' if label=='dad' else 'moe_sboed';f=path/'eval'/f'{method}_seed1001.json'
   if not f.exists():report['seeds'][str(seed)][label]={'status':'pending'};continue
   rows=read(f);assert len(rows)==2000
   x=np.array([r['spce'] for r in rows]);assert np.isfinite(x).all();stored[(label,seed)]=rows
   training=read(path/'models'/f'{method}_training.json')
   assert training['generated_training_trajectories']==1000000
   result=dict(status='complete',eig=float(x.mean()),mse=float(np.mean([r['inference']['mse'] for r in rows])),training_episodes=1000000,training_seconds=training['training_seconds'],parameters=training.get('policy_parameters'))
   if (path/'summary.json').exists():
    summary=next(r for r in read(path/'summary.json') if r['method']==method);result['evaluation_seconds']=summary['evaluation_seconds']
   result['active_parameters_nonfirst']=training.get('active_parameters_nonfirst')
   result['effective_nonpadding_parameters']=training.get('effective_nonpadding_parameters')
   if method=='moe_sboed':result['training_score_distribution']=training_distribution(training)
   report['seeds'][str(seed)][label]=result
   if label=='corrected':
    choices=[route['selected_expert'] for r in rows for route in r['routing'] if route['selected_expert'] is not None]
    if choices:report['expert_utilization'][str(seed)]=dict(hard_frequency=(np.bincount(choices,minlength=4)/len(choices)).tolist(),one_expert_per_stage=all(sum(w>0 for w in route['weights'])==1 for r in rows for route in r['routing']))
  for baseline in ['original','dad']:
   if ('corrected',seed) in stored and (baseline,seed) in stored:
    report['comparisons'][f'corrected_minus_{baseline}_seed{seed}']=paired(stored[('corrected',seed)],stored[(baseline,seed)])
 for label in ['original','corrected','dad']:
  vals=[r[label]['eig'] for r in report['seeds'].values() if r[label]['status']=='complete']
  report['seed_aggregates'][label]=dict(completed_training_seeds=len(vals),eig_mean=float(np.mean(vals)) if vals else None,eig_sample_sd=float(np.std(vals,ddof=1)) if len(vals)>1 else None)
 for baseline in ['original','dad']:
  vals=[report['comparisons'][f'corrected_minus_{baseline}_seed{seed}']['mean'] for seed in [101,102,103] if f'corrected_minus_{baseline}_seed{seed}' in report['comparisons']]
  if len(vals)==3:
   sd=float(np.std(vals,ddof=1));mean=float(np.mean(vals))
   report['comparisons'][f'training_seed_corrected_minus_{baseline}']=dict(mean=mean,sample_sd=sd,approximate_t95_interval=[mean-4.30265*sd/math.sqrt(3),mean+4.30265*sd/math.sqrt(3)],independent_trainings=3)
 complete=all(report['seed_aggregates'][k]['completed_training_seeds']==3 for k in ['original','corrected','dad'])
 report['three_seed_comparison_complete']=complete
 if 'implementation.json' not in report:
  report['selected_correction']='Prepared sum-encoder candidate; activation pending complete controlled evidence.'
 if report['status']['status']=='complete_controls_do_not_support_prepared_encoder_correction':
  report['root_cause']='Unresolved: completed controls did not satisfy the prepared encoder correction evidence guard.'
  report['selected_correction']=None
 corr=report['seed_aggregates']['corrected']['eig_mean']
 if corr is None:decision='Correction validation pending; no MoE advantage demonstrated.'
 elif not complete:decision='One-factor correction has partial results; independent training-seed confirmation pending.'
 else:
  dad=report['seed_aggregates']['dad']['eig_mean'];a=report['ablations']['A']['eig']
  decision=('Corrected MoE beats original DAD in mean; ' if corr>dad else 'Corrected MoE does not beat original DAD in mean; ')
  decision+=('seed101 sum-single control remains stronger. Sparse-MoE advantage is not demonstrated.' if report['seeds']['101']['corrected']['eig']<=a else 'MoE versus sum-single control requires independent-seed confirmation before claiming architecture advantage.')
 report['next_research_decision']=decision
 if report['status']['status']=='encoder_trial_has_not_demonstrated_a_DAD_win':
  decision='Encoder correction improves the original MoE but remains below DAD. Three-seed runs were not launched because the DAD-win criterion failed. No sparse-MoE advantage has been demonstrated; remaining expert/head/routing effects need matched controls.'
  report['next_research_decision']=decision
 (s/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 lines=['# MoE-sBOED encoder correction study','', 'Status: '+report['status']['status'],'',report['root_cause'],'',
        '| Control | EIG | Total parameters | Active parameters |','|---|---:|---:|---:|']
 for label,r in report['ablations'].items():lines.append(f"| {label} | {r['eig']:.5f} | {r['total_parameters']} | {r['active_parameters_nonfirst']} |")
 lines+=['','Correction: '+str(report['selected_correction']),'', '| Training seed | Original MoE | Corrected MoE | DAD |','|---|---:|---:|---:|']
 for seed,r in report['seeds'].items():
  values=[f"{r[label]['eig']:.5f}" if 'eig' in r[label] else 'Pending' for label in ['original','corrected','dad']]
  lines.append('| '+seed+' | '+' | '.join(values)+' |')
 lines+=['', '| Model | EIG mean across completed training seeds | Sample SD | Seeds |','|---|---:|---:|---:|']
 for label,r in report['seed_aggregates'].items():lines.append(f"| {label} | {r['eig_mean']} | {r['eig_sample_sd']} | {r['completed_training_seeds']} |")
 lines+=['','| Reused baseline, seed101 | EIG | MSE |','|---|---:|---:|']
 for label,r in report['reused_baselines'].items():lines.append(f"| {label} | {r['eig']:.5f} | {r['mse']:.5f} |")
 lines+=['',report['mse_interpretation'],'',decision,'','All formal runs retain1million training episodes, L1024 training and2000/L1000000evaluation. Timing and smoke results are correctness/cost diagnostics only.','', 'Implementation files: src/research/sparse_moe.py, src/research/moe_architecture_controls.py, src/research/runner.py. Original attention remains the default.']
 (s/'report.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(status=report['status'],seed_aggregates=report['seed_aggregates'],next_decision=decision),indent=2))

if __name__=='__main__':main()
