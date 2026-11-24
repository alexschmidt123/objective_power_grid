"""Write outcome reports from actual receipts;never fabricate absent results."""
from native_common import *
def write():
 def read(n):
  p=O/n;return json.loads(p.read_text()) if p.exists() else None
 def save(n,s): (O/n).write_text(s+'\n')
 def dump(v):return '```json\n'+json.dumps(v,indent=2)+'\n```'
 completion=read('evidence/EXPERIMENT_COMPLETION.json');train={m:read(f'arms/{m}/RESULT.json') for m in ['control','candidate']};eig=read('evaluation/EIG_RESULTS.json');spec=read('evaluation/SPECIALIZATION_RESULTS.json');route=read('evaluation/ROUTING_RESULTS.json');mechanism=read('evaluation/MECHANISM_RESULTS.json');conditional=read('evaluation/CONDITIONAL_COMPETENCE.json');lat=read('evaluation/INFERENCE_LATENCY.json');evaluation=read('evaluation/COMPLETION.json')
 save('TRAINING_RESULTS.md','# Training results\n\n'+dump(train)+'\n\nOnly completed native optimizer updates count. Both arms use the saved parent replay/Adam/schedules, gamma=1, normalized posterior floor, no specialist auxiliary losses. Missing receipts mean INCOMPLETE. No training extension or replacement arm is authorized.')
 text='# Full-horizon EIG comparison\n\n'
 if eig:
  text+='Independent common bank: 2,000 trajectories, 1,000,000 prior negatives plus the positive once. Fixed finite-contrast information lower-bound estimator; values are not exact mutual information. SEs and intervals describe Monte Carlo uncertainty for these frozen checkpoints, not seed uncertainty.\n\n|Policy|Estimate (nats)|MC SE|\n|---|---:|---:|\n'
  for k,v in eig['methods'].items():text+=f"|{k}|{v['mean']:.6f}|{v['MCSE']:.6f}|\n"
  text+='\nPrimary paired candidate minus matched continuation:\n\n'+dump(eig['primary_candidate_minus_control'])+'\n\nSecondary frozen comparisons (Holm4 p-values; displayed 95% intervals are unadjusted):\n\n'+dump(eig['secondary'])+'\n\nOther comparisons are descriptive. Historical baselines have different training budgets; no compute-matched superiority claim is supported.'
 else:text+='INCOMPLETE: no complete common-bank result receipt. Individual completed policy scores, if any, remain in evaluation/. No cross-bank substitution.'
 save('EIG_COMPARISON.md',text)
 save('SPECIALIZATION_ABLATIONS.md','# Specialization and competency retention\n\n'+('INCOMPLETE: required receipts absent.' if not spec else dump(spec))+'\n\nConditional one-step donor interventions:\n\n'+dump(conditional)+'\n\nRetained anchors guarantee access to the old representations, not improved live competencies. Designated roles are tested rather than assumed. Native conditional return is a variational surrogate, not exact conditional EIG. Donor replacements and zero masking alter joint feature distributions; resulting sensitivity alone cannot prove distinct competencies.')
 save('ROUTING_ABLATIONS.md','# Routing and collaboration\n\n'+dump(route)+'\n\nMechanism interventions:\n\n'+dump(mechanism)+'\n\nStage/fixed/shuffled gates use calibration histories only. Changed actions trigger newly simulated observations. Ablations may be off manifold. The candidate has extra effective adaptive capacity, so the matched ordinary control alone cannot fully isolate the anchor mechanism.')
 costs=dict(training=train,latency=lat,evaluation=evaluation,total_owned_wall_seconds=(completion or {}).get('seconds'),wall_GPU_allocation_hours=(completion or {}).get('seconds',0)/3600)
 save('COMPUTATIONAL_COST.md','# Computational cost\n\n'+dump(costs)+'\n\nWall GPU allocation includes loading, checkpoint I/O and evaluation; it is not integrated GPU utilization. Parent training is inherited and excluded from incremental continuation time. Both arms receive 2,000 actor updates, 10,000 critic updates, 300,000 posterior updates, 2,000,000 new simulator episodes if complete. No additional preparation training. Each main policy requires 60,000 observations and 60 billion negative likelihood terms. Detailed per-intervention observations are in evaluation receipts.')
 decision='INCONCLUSIVE / INCOMPLETE'
 if eig and evaluation and evaluation['status']=='COMPLETE':
  d=eig['primary_candidate_minus_control'];gain=d['CI95'][0]>0
  tasks=(spec or {}).get('task_capabilities',{}).get('credible_distinct_task_experts',[])
  specialized=bool((spec or {}).get('all_experts_useful_by_donor_test')) and len(tasks)>=2
  routed=bool((route or {}).get('history_routing_beyond_stage'))
  decision='PROMISING SINGLE-SEED PILOT' if gain and specialized and routed else ('PARTIAL: EIG gain with incomplete mechanism evidence' if gain else 'UNSUCCESSFUL: no demonstrated EIG improvement beyond matched continuation')
 save('NEXT_RESEARCH_DECISION.md','# Research decision\n\n'+decision+'\n\n'+dump(dict(execution=completion,evaluation=evaluation,primary=(eig or {}).get('primary_candidate_minus_control')))+'\n\nNo additional training, seed campaign, architecture search or budget extension has been launched. Non-significance is not equivalence. Any positive result remains single seed and capacity-confounded. ICML novelty and superiority over compute-unmatched vsOED are not established. Require replicated EIG benefit, useful held-out conditional competencies, and capacity-matched mechanism evidence before a methodological claim. SIR transfer is a future proposal only.')
 save('RUN_INDEX.md','# Specialization-Guided EIG MoE\n\n'+decision+'\n\n'+dump(completion)+'\n\nSee EIG_COMPARISON.md, SPECIALIZATION_ABLATIONS.md, ROUTING_ABLATIONS.md and NEXT_RESEARCH_DECISION.md. All original experiments are read only; this directory owns the new artifacts. See PRETRAIN_MANIFEST.json and evidence/FINAL_PRESERVATION.json for provenance/preservation. No UbuntuPC access or final-test data.')
