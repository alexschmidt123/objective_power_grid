import json,hashlib,csv
from pathlib import Path
import numpy as np
R=Path('/home/grads/g/g.lin/Documents/objective_power_grid');C=R/'experiments/10172026/10172026_st_replication_score_reverse_cross';Q=R/'experiments/10162026/10162026_location_root_cause_v4';O=C/'reports';OLD=R/'experiments/10152026/10152026_journal_five_seed_fixed_budget/location'
def read(p):return json.loads(p.read_text())
def paired(a,b):
 d=a-b;m=float(d.mean());s=float(d.std(ddof=1)/len(d)**.5);return dict(difference=m,mc_se=s,ci95=[m-1.96*s,m+1.96*s])
paths={(102,102):Q/'source/experiments/10162026/10162026_root_cause_v4/score_model102_policy102',(103,102):Q/'source/experiments/10162026/10162026_root_cause_v4/score_model103_policy102',(103,103):OLD/'seed103/scorebed',(102,103):C/'runs/score102_policy103'}
records=[];scores={};identity=None;banksha=None
for (s,p),run in paths.items():
 rows=read(run/'eval/scorebed_seed1001.json');assert len(rows)==2000
 draws=[[r['truth'],r['measurement_noise_standard']] for r in rows]
 if identity is None:identity=draws
 else:assert draws==identity
 bank=np.load(run/'eval/contrast_bank_seed1001.npy');bh=hashlib.sha256(bank.tobytes()).hexdigest()
 if banksha is None:banksha=bh
 else:assert bh==banksha
 a=np.array([r['spce'] for r in rows]);scores[s,p]=a;np.save(O/f'score{s}_policy{p}_hard_scores.npy',a)
 tr=read(run/'models/scorebed_training.json');assert tr.get('complete_model_training_episodes',tr.get('generated_training_trajectories'))==1000000
 records.append(dict(score_bundle_seed=s,policy_training_seed=p,eig=float(a.mean()),mc_se=float(a.std(ddof=1)/len(a)**.5),complete_model_episodes=1000000,new_generated_episodes=tr.get('new_generated_training_trajectories'),training_seconds=tr.get('training_seconds'),run=str(run),historical_control=(s,p)==(103,103)))
cross=read(C/'runs/score102_policy103/models/initial_policy.json');assert cross==read(C/'score_readiness.json')['initial_policy']
contrasts={f'bundle103_minus102_policy{p}':paired(scores[103,p],scores[102,p]) for p in [102,103]}
contrasts['bundle_by_policy_interaction']=paired(scores[103,103]-scores[102,103],scores[103,102]-scores[102,102])
out=dict(status='complete',records=records,paired_comparisons=contrasts,evaluation_seed=1001,systems=2000,contrast_samples=1000000,contrast_bank_sha256=banksha,primary_results_replaced=False,reserved_final_seed_used=False,limitation='Two crossed policy initializations; bundle includes score normalization. MC intervals are conditional on trained policies, not training-seed uncertainty.')
(O/'score_cross_results.json').write_text(json.dumps(out,indent=2)+'\n')
with (O/'score_cross_results.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
lines=['# ScoreBED frozen-score crossed diagnosis','','Ordered labels: score-bundle seed / policy-training seed. All complete models account for exactly1M primary episodes; reused score pretraining702994 plus new policy initialization10 and policy training296996. Final-budget checkpoints, common2000/L1M development1001.','','| Score bundle | Policy seed | EIG | MC SE |','|---|---|---:|---:|']
for r in records:lines.append(f"| {r['score_bundle_seed']} | {r['policy_training_seed']} | {r['eig']:.5f} | {r['mc_se']:.5f} |")
lines+=['','Paired contrasts:','', '```json',json.dumps(contrasts,indent=2),'```','','Native policy103 initialization/RNG, normalization and frozen bundle102 prediction compatibility passed preflight. Historical103/103 is reused; no unnecessary new control training. This is a complete score-bundle intervention, not isolated score-weight causality. Score-error diagnostics and reliability filtering are reported separately.','']
effects=[contrasts[f'bundle103_minus102_policy{p}'] for p in [102,103]]
if all(v['ci95'][0]>0 for v in effects):lines+=['Bundle103 outperforms102 at both policy initializations on these paired cases. This supports a score-bundle contribution; differing effect sizes can still indicate interaction. It does not isolate the responsible score-training defect.']
else:lines+=['The bundle effect is not consistently positive at both policy initializations; the poor outcome cannot be attributed to bundle102 independently of policy initialization.']
lines+=['','No favorable score-bundle substitution into primary results. No targeted score-training correction is selected from crossing alone; require score-error evidence. Original five-seed results and reserved final evaluation seed8831001 remain untouched.']
(O/'score_report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(out,indent=2),flush=True)
