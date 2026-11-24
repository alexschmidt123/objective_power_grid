"""Populate requested reports from actual receipts, preserving negative results."""
from native_common import *
def write():
 def read(n):
  p=O/n;return json.loads(p.read_text()) if p.exists() else None
 def save(n,text):(O/n).write_text(text+'\n')
 def block(v):return '```json\n'+json.dumps(v,indent=2)+'\n```'
 prep=read('preparation/RESULT.json');train={m:read(f'arms/{m}/RESULT.json') for m in ['control','candidate']};completion=read('evidence/EXPERIMENT_COMPLETION.json');evaluation=read('evaluation/COMPLETION.json');eig=read('evaluation/EIG_RESULTS.json');spec=read('evaluation/SPECIALIZATION_RESULTS.json');cmd=read('evaluation/COMMANDER_RESULTS.json');lat=read('evaluation/LATENCY.json')
 save('TRAINING_RESULTS.md','# Training results\n\nShared preparation:\n\n'+block(prep)+'\n\nNative matched continuations:\n\n'+block(train)+'\n\nPhase A uses supervised targets; Phases B/C use only native full-horizon variational actor–critic EIG. Missing receipts are INCOMPLETE, not zero-valued results. Both arms inherit the same SG replay, compatible history Adam and native critic/posterior schedules. New incompatible modules have fresh Adam. No extension/retraining occurred.')
 text='# Full-horizon information comparison\n\n'
 if eig:
  text+='New common development bank: 2,000 trajectories and 1,000,000 negatives plus the positive once. Estimates use the fixed finite-contrast lower bound, not exact EIG. SEs quantify trajectory Monte Carlo variation at one frozen training seed.\n\n|Policy|Estimate (nats)|MC SE|\n|---|---:|---:|\n'
  for k,v in eig['methods'].items():text+=f"|{k}|{v['mean']:.6f}|{v['MCSE']:.6f}|\n"
  text+='\nPrimary paired candidate minus matched heterogeneous control:\n\n'+block(eig['primary_candidate_minus_control'])+'\n\nFrozen secondary comparisons (Holm4 p-values; displayed 95% CIs are unadjusted):\n\n'+block(eig['secondary'])+'\n\nOther comparisons are descriptive. Historical methods are not compute matched; cross-bank numbers are never paired.'
 else:text+='INCOMPLETE: complete common-bank receipt absent. Any finished per-policy scores remain in evaluation/. No scores are invented or replaced by historical-bank values.'
 save('EIG_COMPARISON.md',text)
 save('SPECIALIZATION_AND_ROUTING_ABLATIONS.md','# Useful competencies and commanding\n\n'+block(dict(specialization=spec,commander_and_coordination=cmd))+'\n\nRoles were assigned by mathematical inputs; useful collaboration must be demonstrated. Live query changes and representation differences alone do not prove useful expertise. All policy interventions resimulate future observations. Masking can cause joint-distribution shift. The ordinary-router projection is a calibration diagnostic, while the matched control supplies the trained ordinary-router comparison. Task 4 labels are noisy approximate-belief bootstrap values, not exact conditional mutual information. Multiple tests use the predeclared Holm families.')
 costs=dict(shared_preparation=prep,arms=train,inference_latency=lat,evaluation=evaluation,total_owned_wall_seconds=(completion or {}).get('seconds'),GPU_wall_allocation_hours=(completion or {}).get('seconds',0)/3600)
 save('COMPUTE_LEDGER.md','# Compute ledger\n\n'+block(costs)+'\n\nPreparation is physically performed once and reused; comparisons charge each arm the same preparation opportunity. Real training simulation and synthetic teacher observations are separate. Model computation counters include deterministic signal evaluations, analytic Jacobians, frozen-belief rows, frozen-critic rows and provisional SG calls; they include checkpoint recomputation where executed. They are not hidden new simulator episodes. Buffers include substantial frozen posterior/critic weights and must be counted at deployment. GPU wall allocation includes serialization and evaluation; it is not integrated GPU utilization. Historical baseline training compute is inherited and not matched.')
 decision='INCOMPLETE / INCONCLUSIVE'
 positive=lambda d:any(k.startswith('Holm') and v<.05 for k,v in d.items()) and d.get('mean',0)>0
 if eig and evaluation and evaluation['status']=='COMPLETE':
  gain=eig['primary_candidate_minus_control']['CI95'][0]>0
  st=(spec or {}).get('competency',{});competencies=st.get('designated_expert_advantages',{});useful=(spec or {}).get('expert_removal',{});commands=(cmd or {}).get('commands',{});coord=(cmd or {}).get('coordination',{})
  competent=st.get('frozen_readout_vs_constant',{});all_competent=len(competent)==4 and all(positive(v) for v in competent.values())
  mechanism=all_competent and sum(positive(v) for v in competencies.values())>=2 and len(useful)==4 and all(positive(v) for v in useful.values()) and positive(commands.get('stage',{})) and positive(coord.get('stage',{}))
  decision='PROMISING SINGLE-SEED PILOT; REPLICATION REQUIRED' if gain and mechanism else ('PARTIAL: EIG gain but specialization/commanding mechanism remains incomplete' if gain else 'UNSUCCESSFUL: no demonstrated EIG improvement beyond matched heterogeneous control')
 save('RUN_INDEX.md','# HEC-MoE-sBOED\n\n'+decision+'\n\n'+block(completion)+'\n\nRead EIG_COMPARISON.md, SPECIALIZATION_AND_ROUTING_ABLATIONS.md, COMPUTE_LEDGER.md and ICML_NOVELTY_ASSESSMENT.md. One candidate/control pair only; original artifacts preserved. No UbuntuPC access, final-test bank, seed extension or SIR/grid campaign. See PRETRAIN_MANIFEST.json and evidence/FINAL_PRESERVATION.json.')
 novelty=(O/'ICML_NOVELTY_ASSESSMENT.md').read_text();save('ICML_NOVELTY_ASSESSMENT.md',novelty+'\n\n## Pilot outcome\n\n'+decision+'\n\nThis status does not establish ICML novelty. Require a targeted conditional-computation/model-based BOED prior-art comparison, multiple matched seeds and mechanism controls before claims. No further training is launched automatically.')
