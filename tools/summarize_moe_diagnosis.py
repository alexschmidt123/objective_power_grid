"""Report complete/pending controls without training or selecting a correction."""
import json,math,argparse
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);args=p.parse_args();c=Path(args.campaign);root=Path.cwd()
 models=dict(A='sum_single',B='attention_single',C='attention_dense_moe',D='attention_sparse_moe',capacity='attention_matched_dense')
 result=dict(status='pending',root_cause='unresolved; routing optimization contributes, objective saturation and order sensitivity observed; controls pending',ablations={},differences={},selected_correction=None,next_decision='Do not claim sparse MoE advantage; select one modification only after complete controls and inference diagnostics; then decisive3training-seed comparison',training_L=1024,eval_L=1000000,train_seed=101,eval_seed=1001)
 for label,method in models.items():
  f=c/'controls/eval'/f'{method}_seed1001.json';t=c/'controls/models'/f'{method}_training.json'
  if f.exists():
   rows=json.loads(f.read_text());x=np.array([r['spce'] for r in rows]);r=dict(status='evaluation_complete',eig=float(x.mean()),eig_mc_standard_error=float(x.std(ddof=1)/math.sqrt(len(x))),mse=float(np.mean([r['inference']['mse'] for r in rows])),systems=len(rows))
  else:r=dict(status='pending')
  if t.exists():
   d=json.loads(t.read_text());r.update(training_seconds=d['training_seconds'],training_episodes=d['generated_training_trajectories'],total_parameters=d['policy_parameters'],active_parameters_nonfirst=d.get('active_parameters_nonfirst'),parameter_groups=d.get('parameter_groups'))
  result['ablations'][label]=r
 for first,second in [('A','B'),('B','C'),('C','D'),('capacity','C')]:
  files=[c/'controls/eval'/f'{models[x]}_seed1001.json' for x in [first,second]]
  if all(f.exists() for f in files):
   a,b=[json.loads(f.read_text()) for f in files];assert len(a)==len(b)==2000
   assert all(x['truth']==y['truth'] and x['measurement_noise_standard']==y['measurement_noise_standard'] for x,y in zip(a,b))
   d=np.array([y['spce']-x['spce'] for x,y in zip(a,b)]);se=float(d.std(ddof=1)/math.sqrt(len(d)))
   result['differences'][first+'->'+second]=dict(eig_difference=float(d.mean()),paired_mc_standard_error=se,approximate95_interval=[float(d.mean()-1.96*se),float(d.mean()+1.96*se)],not_training_seed_uncertainty=True)
 for name in ['preliminary.json','checkpoint_diagnostics/diagnosis.json','plan.json','cost_check.json']:
  if (c/name).exists():result[name]=json.loads((c/name).read_text())
 if (c/'pipeline_exit_code').exists() and (c/'pipeline_exit_code').read_text().strip()=='0':
  result['status']='initial_controls_complete; cause interpretation and decisive correction still required'
  changes={k:v['eig_difference'] for k,v in result['differences'].items() if not k.startswith('capacity')}
  result['largest_observed_architecture_difference']=max(changes,key=lambda k:abs(changes[k])) if changes else None
 historical={'DAD':('experiments/10132026/10132026_bounded_location_restart/pilot','dad'),'original MoE':('experiments/10132026/10132026_bounded_location_restart/pilot','moe_sboed'),'component MoE':('experiments/10142026/10142026_moe_component_clipping/pilot','moe_sboed'),'slower MoE':('experiments/10142026/10142026_moe_delayed_sharpening/pilot','moe_sboed')}
 result['historical']={}
 for n,(rel,method) in historical.items():
  f=root/rel/'eval'/f'{method}_seed1001.json'
  if f.exists():
   rows=json.loads(f.read_text());result['historical'][n]=dict(eig=float(np.mean([r['spce'] for r in rows])),mse=float(np.mean([r['inference']['mse'] for r in rows])),source=rel)
 lines=['# MoE-sBOED controlled diagnosis','', 'Status: '+result['status'], '', '## Root cause','',result['root_cause'],'','## A–D controls','','| Model | EIG | Status |','|---|---:|---|']
 for label,r in result['ablations'].items():lines.append('| '+label+' | '+(f"{r['eig']:.5f}" if 'eig' in r else 'Pending')+' | '+r['status']+' |')
 lines+=['','## Existing results','','| Policy | EIG | MSE |','|---|---:|---:|']
 for label,r in result['historical'].items():lines.append(f"| {label} | {r['eig']:.5f} | {r['mse']:.5f} |")
 lines+=['','## Interpretation limits','','A preserves the original DAD sum encoder but uses the common64/64tanh control head. The published DAD linear-emitter result is a separate baseline. Its16-dimensional sum output is zero padded, and unused padded weights are reported separately. Single attention sees all4 contexts; individual experts see193 features. Sparse D retains tiny intrinsic routing penalties. One-seed intervals describe paired evaluation Monte Carlo uncertainty, not training-seed variability.','', '## Correction and next decision','',result['next_decision']]
 (c/'report.json').write_text(json.dumps(result,indent=2));(c/'report.md').write_text('\n'.join(lines)+'\n');print(result['status'])
if __name__=='__main__':main()
