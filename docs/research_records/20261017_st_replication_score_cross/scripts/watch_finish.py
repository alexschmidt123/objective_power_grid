import os,json,time,subprocess,hashlib,traceback
from pathlib import Path
C=Path('/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_st_replication_score_reverse_cross');PY='/home/grads/g/g.lin/miniconda3/envs/mocu_optimized/bin/python'
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
started=time.monotonic()
try:
 while time.monotonic()-started<43200:
  state=json.loads((C/'status.json').read_text())
  if state['state']=='failed_preserved':raise RuntimeError('Coordinator failed; no automatic training retry: '+state.get('error',''))
  if state['state']=='complete':break
  pid=json.loads((C/'coordinator.json').read_text())['pid']
  try:os.kill(pid,0)
  except ProcessLookupError:
   if json.loads((C/'status.json').read_text())['state']=='complete':break
   raise RuntimeError('Coordinator exited before completing campaign; outputs preserved')
  time.sleep(20)
 else:raise TimeoutError('12h report watcher cap; no training launched by watcher')
 with (C/'logs/final_report.log').open('w') as f:
  subprocess.run([PY,str(C/'scripts/finalize.py')],cwd=C,stdout=f,stderr=subprocess.STDOUT,check=True)
 # Choose a single next ScoreBED investigation; do not claim a score-training
 # correction is supported unless a specific boundary error pattern is found.
 rows=json.loads((C/'reports/score_error_records.json').read_text());pairs={}
 for r in rows:
  if r['reference_reliable']:
   key=(r['score_seed'],r['policy_seed'],r['case']);pairs.setdefault(key,{})[r['evaluated_score_bundle']]=r
 boundary=[p for p in pairs.values() if len(p)==2 and p[102]['boundary_fraction']>0]
 central=[p for p in pairs.values() if len(p)==2 and p[102]['boundary_fraction']==0]
 def average(group,seed,field):return sum(p[seed][field] for p in group)/len(group) if group else None
 evidence=dict(reliable_boundary_histories=len(boundary),reliable_central_histories=len(central),boundary_mean_design_error_102=average(boundary,102,'design_relative_error'),boundary_mean_design_error_103=average(boundary,103,'design_relative_error'),boundary_mean_design_cosine_102=average(boundary,102,'design_cosine'),boundary_mean_design_cosine_103=average(boundary,103,'design_cosine'),central_mean_design_error_102=average(central,102,'design_relative_error'),central_mean_design_error_103=average(central,103,'design_relative_error'))
 supported=(len(boundary)>=4 and len(central)>=4 and evidence['boundary_mean_design_error_102']>1.25*evidence['boundary_mean_design_error_103'] and evidence['boundary_mean_design_cosine_102']<evidence['boundary_mean_design_cosine_103']-.1 and evidence['central_mean_design_error_102']<=1.1*evidence['central_mean_design_error_103'])
 # This descriptive evidence screen is not a hypothesis test or proof of causality.
 score_next=('Test increased feasible-boundary design coverage in score pretraining, changing only the sampler distribution at the same1M total episode budget. Keep native score/policy architecture, objective, optimizer and all five original seeds.' if supported else 'No score-training correction is justified yet. Next diagnostic: compare both frozen bundles\u2019 score-derived policy gradients at identical policy103 initialization on paired simulated histories against explicit-likelihood pathwise diagnostic gradients. Keep training and primary results unchanged; test gradient direction and scale before selecting a score-training correction.')
 recommendations=dict(moe='One controlled intervention: first100000 of the unchanged1M episodes use the native relaxed-forward mixture, then use unchanged ST routing for the remaining900000. Keep gamma, architecture, sPCE, optimizer, regularizers and clipping fixed. This targets documented unselected-expert starvation; predeclare and test all five original seeds. Recommendation only, not launched.',scorebed=score_next,score_boundary_defect_evidence=evidence,boundary_screen_descriptive_not_causal=True,training_launched_by_reporter=False)
 write(C/'reports/next_interventions.json',recommendations)
 for name in ['report.md','score_report.md']:
  with (C/'reports'/name).open('a') as f:f.write('\n## Next ScoreBED decision after reliability checks\n\n'+score_next+'\n\nMatched-history evidence: `'+json.dumps(evidence)+'`. This descriptive screen does not establish causality. No corrective training was launched.\n')
 write(C/'report_complete.json',dict(status='complete',report=str(C/'reports/report.md'),report_sha256=hashlib.sha256((C/'reports/report.md').read_bytes()).hexdigest(),training_launched=False,primary_results_replaced=False,reserved_final_seed_used=False))
 print('FINAL_REPORT_COMPLETE',flush=True)
except BaseException as exc:
 write(C/'report_failure.json',dict(error=str(exc),traceback=traceback.format_exc(),training_launched=False));raise
