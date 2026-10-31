if __name__ == '__main__':
    raise SystemExit('LEGACY_EXCLUDED_FROM_CURRENT_PROJECT: historical result generation is archived')
from pathlib import Path
import json,csv,math,time
import numpy as np
import torch
from src.research.pimoe_validation import load_model,intervention,js_distance,density_metrics,csvwrite,write
from src.research.pimoe import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);args=p.parse_args();C=Path(args.campaign).resolve();reference=C/'reference';torch.set_num_threads(2);moe=load_model(reference/'models/pimoe_final.pt');single=load_model(C/'location/seed101/single/models/single_final.pt');states=torch.load(C/'evaluations/diagnostic_states_seed1006.pt',map_location='cpu',weights_only=False);rows=[];control=[];commonhead=[]
with torch.no_grad():
 for index,item in enumerate(states):
  f=item['features'];z=moe(**f);A,P=f['means'].shape;ref=direct_reference(f['means'],f['weights'],.5,1007000+index,1024);g=rng(1009000+index);ids=torch.multinomial(f['weights'],512,True,generator=g);y=f['means'][:,ids]+.5*torch.randn(A,512,generator=g,dtype=DTYPE)
  for label,m in [('moe',moe),('single',single)]:control.append(dict(state=index,episode=item['episode'],stage=item['stage'],model=label,**density_metrics(m(**f),y,ref['EIG'],ref['SE'],.5)))
  estimates={n:PredictiveMoE.entropy(z,nodes=n)-.5*math.log(2*math.pi*math.e*.25) for n in [24,64,128]}
  zz=intervention(moe,f,'uniform_membership');heads=[dict(weights=zz['weights'][:,e*3:(e+1)*3]*4,means=zz['means'][:,e*3:(e+1)*3],scales=zz['scales'][:,e*3:(e+1)*3]) for e in range(4)]
  for e in range(4):
   for j in range(e):commonhead.append(dict(state=index,stage=item['stage'],expert_i=e,expert_j=j,JS_identical_region_inputs=float(js_distance(heads[e],heads[j]).mean())))
  for a in range(A):rows.append(dict(state=index,stage=item['stage'],candidate=a,EIG_24=float(estimates[24][a]),EIG_64=float(estimates[64][a]),EIG_128=float(estimates[128][a]),reference_EIG=float(ref['EIG'][a]),reference_SE=float(ref['SE'][a]),difference_128_minus24=float(estimates[128][a]-estimates[24][a])))
csvwrite(C/'reports/entropy_resolution.csv',rows);csvwrite(C/'reports/matched_control_independent_states.csv',control);csvwrite(C/'reports/head_function_difference_same_region.csv',commonhead)
summary=[]
for label in ['moe','single']:
 a=[x for x in control if x['model']==label];n=sum(x['ranking_pairs'] for x in a);summary.append(dict(model=label,**{k:float(np.mean([x[k] for x in a])) for k in ['NLL','KS','coverage90','EIG_RMSE','EIG_bias','selected_reference_EIG','selected_regret']},ranking_accuracy=sum(x['ranking_correct'] for x in a)/n,ranking_pairs=n))
write(C/'reports/matched_control_independent_summary.json',summary);diff=np.asarray([r['difference_128_minus24'] for r in rows]);info=dict(maximum_abs_quadrature_128_minus24=float(np.max(np.abs(diff))),mean_abs_quadrature_difference=float(np.mean(np.abs(diff))),fraction_abs_difference_over005=float(np.mean(np.abs(diff)>.05)),same_input_head_JS_mean=float(np.mean([r['JS_identical_region_inputs'] for r in commonhead])),algorithm_changed=False,scope='diagnostic only; frozen planner still uses24nodes; no final-EIG-based tuning')
write(C/'reports/entropy_resolution_summary.json',info);print(json.dumps(dict(matched_control=summary,numerical_resolution=info),indent=2))
