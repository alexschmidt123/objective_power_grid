import json,hashlib,shutil
from pathlib import Path
C=Path('/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_st_replication_score_reverse_cross');O=C/'reports';V=C/'report_versions';V.mkdir(exist_ok=True)
assert json.loads((C/'report_complete.json').read_text())['status']=='complete'
for name in ['report.md','score_report.md','next_interventions.json']:
 shutil.copy2(O/name,V/('automatic_'+name))
shutil.copy2(C/'report_complete.json',V/'automatic_report_complete.json')
nexts=json.loads((O/'next_interventions.json').read_text());nexts['moe']='One controlled intervention: native relaxed-forward expert mixture for the first1563 complete64-episode batches (100032 episodes), then unchanged ST routing for14062 batches (899968 episodes). Total remains exactly1M. Keep the original gamma schedule, architecture, sPCE, Adam, regularizers and clipping. Predeclare and compare all five original seeds. Recommendation only; not implemented or launched.'
(O/'next_interventions.json').write_text(json.dumps(nexts,indent=2)+'\n')
moe_note='''
## Final MoE interpretation

All five development-seed point differences are positive, with improved observed
mean, SD and minimum. However, the paired five-training-seed95% t interval for
the mean gain is[-0.85265,3.60873] nats, so it does not establish population
superiority. These are development seeds, not untouched confirmation.

Every final policy selects one expert for all58000 post-initial decisions.
Its dominant expert receives meaningful gradients on100% of updates. Other
experts receive at most9/15625 updates (0.0576%), with most receiving none;
the exact per-expert counts/fractions remain in moe_diagnostics.json. During
the final20%, every run has zero router-gradient exposure,100% selector
saturation and zero non-dominant-expert exposure. Hard/relaxed final scores
are identical case by case. This is strong single-active-expert performance,
not demonstrated useful four-expert specialization.

The next single intervention is1563 batches (100032 episodes) of native relaxed
expert warmup, followed by14062 batches (899968 episodes) of unchanged ST.
The total remains1M, with the original gamma schedule and all other settings.
This is the batch-exact version of the nominal10% warmup recommendation;
it is not implemented or launched.
'''
score_note='''
## Final ScoreBED root-cause assessment

Confirmed: bundle102 produces poor policies at both tested policy seeds;
bundle103 produces higher EIG with the same initialization/randomness within
each policy-seed comparison. The bundle-by-policy-seed interaction is also
large. This identifies a complete frozen-bundle contribution, including its
own normalization, rather than isolated weight causality or a universal
ranking of score-model seeds.

Boundary coordinates mean abs(design)>3.8. Bundle102 produces28.294% boundary
coordinates with policy102 and44.588% with policy103; bundle103 gives0% in
both. This is a behavioral association, not proof of a boundary score defect.

Only6/16 Fisher-reference histories pass ESS/resolution checks. On the four
reliable actual-boundary histories, bundle102 has mean design-score relative
error0.3181 and cosine0.9361, compared with bundle103's0.5262 and0.7835.
On the two reliable central histories, relative errors are0.3651 versus0.4772.
Thus these data do NOT support globally worse score accuracy for bundle102,
nor the proposed boundary-coverage training correction. The histories were
stratified extremes; these are not population-average accuracy estimates.

Unresolved: which early policy-gradient distortions or optimization paths
drive the boundary-attracted low-EIG outcome. One justified next diagnostic
is to compare both frozen bundles' score-derived policy gradients at identical
policy103 initialization, using paired histories and explicit-likelihood
pathwise diagnostic gradients. Test direction and scale before choosing a
score-training correction. No score-training modification is justified yet.
Original primary results remain unchanged; no favorable bundle is substituted.
'''
for name in ['report.md','score_report.md']:
 with (O/name).open('a') as f:f.write((moe_note if name=='report.md' else '')+score_note)
record=json.loads((C/'report_complete.json').read_text());record['report_sha256']=hashlib.sha256((O/'report.md').read_bytes()).hexdigest();record['human_reviewed_interpretation']=True;record['recommendation_batch_rounding']='1563+14062 batches=1M episodes';record['numeric_results_changed']=False
(C/'report_complete.json').write_text(json.dumps(record,indent=2)+'\n')
(C/'final_review_manifest.json').write_text(json.dumps(dict(numeric_results_changed=False,automatic_prose_preserved=str(V),source_or_checkpoint_changes=False,score_accuracy_claim_rejected=True,correction_training_launched=False),indent=2))
print('FINAL_INTERPRETATION_REVIEW_COMPLETE')
