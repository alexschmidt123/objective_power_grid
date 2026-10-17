"""Add derived reporting details; does not rescore, select or train models."""
from pathlib import Path
import json,shutil,hashlib
C=Path('/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_moe_st_routing_dev_103_105');R=C.parents[2];out=C/'reports'
read=lambda p:json.loads(Path(p).read_text())
assert read(C/'status.json')['state']=='complete'
curves=read(out/'development_curves.json');diag=read(out/'diagnostics.json');final=read(out/'final_eig.json')
report=out/'report.md';assert not (out/'report_generated.md').exists();shutil.copy2(report,out/'report_generated.md')
lines=['','## Additional interpretation and checkpoint evidence','',
'For straight-through training, **hard deployment is also the actual training forward policy**. Relaxed deployment is a counterfactual diagnostic, not its training forward policy. Therefore the soft/hard EIG difference should not automatically be called a literal training/deployment mismatch for the new mode. For the original relaxed trainer it does compare training-forward versus hard-deployment policy functions.','',
'| Seed | Configuration | Episodes | Hard development EIG | Relaxed development EIG | Relaxed − hard [95% paired MC interval] |','|---|---|---|---:|---:|---|']
for r in curves:
 if 'relaxed' not in r:continue
 t=r['relaxed_minus_hard'];ci=t['ci95'];lines.append(f"| {r['seed']} | {r['configuration']} | {r['episodes']:,} | {r['hard_eig']:.5f} | {r['relaxed']['eig']:.5f} | {t['difference']:+.5f} [{ci[0]:+.5f}, {ci[1]:+.5f}] |")
exposures=[]
for seed in [103,105]:
 for label in ['original','straight_through']:
  run=next(Path(r['checkpoint']).parent.parent for r in final if r['seed']==seed and r['configuration']==label)
  counts={'all':{},'final20':{}};n={'all':0,'final20':0}
  for line in (run/'models/moe_diagnostics.jsonl').open():
   r=json.loads(line)
   for scope in ['all']+(['final20'] if r['primary_episodes']>800000 else []):
    n[scope]+=1
    for g,v in r['gradient_before'].items():counts[scope][g]=counts[scope].get(g,0)+int(v>1e-8)
  assert n['all']==15625 and n['final20']==3125
  for scope in n:exposures.append(dict(seed=seed,configuration=label,scope=scope,updates=n[scope],gradient_exposure={k:v/n[scope] for k,v in counts[scope].items()}))
(out/'original_vs_st_gradient_exposure.json').write_text(json.dumps(exposures,indent=2)+'\n')
lines+=['','## Original versus straight-through gradient exposure','',
'Threshold: raw component gradient norm>1e-8. All updates means15,625 updates; final20% means the final3,125 updates with primary episode count>800,000. This uses complete native training logs from both configurations, not a console-log subsample.','',
'| Seed | Configuration | Updates | Router | E1 | E2 | E3 | E4 |','|---|---|---|---:|---:|---:|---:|---:|']
for r in exposures:
 e=r['gradient_exposure'];vals=[e[k] for k in ['router','expert_0','expert_1','expert_2','expert_3']]
 lines.append(f"| {r['seed']} | {r['configuration']} | {r['scope']} | "+' | '.join(f'{v:.2%}' for v in vals)+' |')
lines+=['','The original training action is relaxed: an argmax routing label there is a diagnostic proxy, not a single-expert forward execution. Straight-through training actions are hard. Gradient exposure and held-out hard selection are the comparable participation measurements. All four proposals are computed in straight-through training, despite unused expert parameters receiving zero gradient.','',
'## Outcome limitations','',
'The seed103 improvement is already substantial before800k, so its final gain cannot be attributed solely to preventing the late decline. The observed severe seed103 decline is absent in the new saved late checkpoints. This does not establish convergence or universal stability. Seed105 has temporary late dips and later recovery; the three-checkpoint screening result is reported above. Its historical intermediate checkpoint sampling is coarser, so these dips cannot be called newly introduced relative instability on that evidence alone.','',
'Most importantly, both corrected policies must be judged alongside their expert-use statistics. EIG improvement under the same architecture and budget is evidence for a routing-training configuration effect, **not evidence that four experts specialize usefully**. Single-expert dominance is a negative mechanism result. A five-seed replication would assess performance reproducibility, not by itself establish specialization. No default or primary journal result has been replaced.','',
'## Complete root-phase wall cost','',
'Native training-clock seconds exclude declared validation time but include some logging overhead. Development diagnostic seconds overlap validation costs and must not be added independently to the other clocks. Root-phase wall time below includes training, diagnostic validation, native final hard evaluation and launch/export overhead. Paired follow-up evaluations are additional and their individual costs are saved in the `*_cost.json` files.','',
'| Seed | Root-phase wall seconds | Native training-clock seconds | Validation seconds | Primary episodes |','|---|---:|---:|---:|---:|']
for seed in [103,105]:
 complete=read(C/f'completion/seed{seed}.json');r=next(r for r in final if r['seed']==seed and r['configuration']=='straight_through')
 lines.append(f"| {seed} | {complete['wall_seconds']:.2f} | {r['training_seconds']:.2f} | {r['validation_seconds']:.2f} | {r['training_episodes']:,} |")
lines+=['','Environment: `environment.json`. Scientific preflight192 primary rollouts are separately accounted in `correctness_costs.json`; only16 of those are optimizer-smoke episodes, and their model is not reused in either complete run. Public compatibility tests are separate CPU correctness overhead.','']
report.write_text(report.read_text()+'\n'.join(lines))
(out/'report_finalize_manifest.json').write_text(json.dumps(dict(training_or_scoring_performed=False,raw_report_sha256=hashlib.sha256((out/'report_generated.md').read_bytes()).hexdigest(),extended_report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
print('EXTENDED_REPORT_COMPLETE',flush=True)
