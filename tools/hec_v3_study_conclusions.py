"""Read-only conclusions from the locked V3 study;does not alter running source."""
import json,pathlib
import numpy as np
from scipy.stats import t
M=pathlib.Path.home()/'Documents/objective_power_grid';R=M/'reports/hec_v3_study_20261115';O=M/'experiments/location_finding/hec_v3_study_20261115'
def stats(v):
 v=np.asarray(v,float);n=len(v)
 if n<5:return dict(status='PENDING',completed_seeds=n,required_seeds=5)
 se=float(v.std(ddof=1)/np.sqrt(n));mu=float(v.mean());w=float(t.ppf(.975,n-1))*se
 return dict(status='COMPLETE',n=n,mean=mu,seed_sd=float(v.std(ddof=1)),CI95=[mu-w,mu+w],note='Secondary seed-level interval;not an adjusted confirmatory hypothesis.')
def main():
 s=json.loads((R/'SUMMARY.json').read_text());primary=s['primary_seed_contrasts'];answers={}
 for q,k in [('1_learned_vs_trained_uniform','learned_minus_trained_uniform'),('2_learned_vs_v2_continuation','learned_minus_v2_continuation')]:
  d=primary.get(k)
  if not d or not d.get('all_five_seeds') or 'p_Holm' not in d:answers[q]=dict(status='PENDING',statement='Five paired seeds and the locked two-comparison correction are not yet complete.')
  else:
   supported=d['CI95'][0]>0 and d['p_Holm']<.05
   answers[q]=dict(status='SUPPORTED_CONDITIONAL_CONTINUATION_GAIN' if supported else 'NOT_DEMONSTRATED',statistics=d,
    statement='Evidence is conditional on shared pretrained V2/providers and the observed reference benchmark;not independently pretrained/fresh-test superiority.')
 results={};costs=[]
 for seed in [101,102,103,104,105]:
  for arm in ['learned','uniform','v2_control','router_only','encoders_only']:
   p=O/f'seed{seed}_{arm}'/'RESULTS.json'
   if p.exists():
    d=json.loads(p.read_text())
    if d['status']=='COMPLETE':results[seed,arm]=d
 learned=[d for (seed,arm),d in results.items() if arm=='learned']
 interactions=[];router=[];encoder=[];frozen=[]
 reference=json.loads((R/'COMPLETED_V3_AUDIT.json').read_text())
 original_v2=float(np.load(M/'experiments/_legacy_archive/temp_research_20261115/scratch_runs/moe_hec_v2_icml_20261109/evaluation/candidate_scores.npy').mean())
 for seed in [101,102,103,104,105]:
  if (seed,'learned') in results:
   a=results[seed,'learned']['evaluations']['learned']['mean'];frozen.append(a-original_v2)
   if (seed,'encoders_only') in results:router.append(a-results[seed,'encoders_only']['evaluations']['encoders_only']['mean'])
   if (seed,'router_only') in results:encoder.append(a-results[seed,'router_only']['evaluations']['router_only']['mean'])
   if all((seed,k) in results for k in ['encoders_only','router_only','v2_control']):interactions.append(a-results[seed,'encoders_only']['evaluations']['encoders_only']['mean']-results[seed,'router_only']['evaluations']['router_only']['mean']+results[seed,'v2_control']['evaluations']['v2_control']['mean'])
 answers['2b_learned_vs_frozen_original_v2']=stats(frozen)
 answers['3_attribution']=dict(router_given_encoders=stats(router),encoders_given_router=stats(encoder),interaction=stats(interactions),
  interpretation='Trained2x2 factorial. Original expert cores/head fine-tune in all cells;effects concern the added router and encoders. Frozen correction-off compares against the V3-refined inherited router,not original V2 weights.')
 functional={}
 for label in ['uniform','no_correction','stage_only','observation_swapped_gates','no_encoders','no_correction_no_encoders']:
  functional[label]=stats([d['inference_comparisons'][label]['mean'] for d in learned])
 answers['4_history_dependence']=dict(functional_EIG_gains=functional,interpretation='Stage-only gates fitted on independent diagnostic simulations;experts still use history. Observation swapping holds designs/stage fixed but is an artificial intervention. Weight variation alone is not evidence of EIG value.')
 answers['5_expert_contributions']={str(k):stats([d['inference_comparisons'][f'remove_feature_{k}']['mean'] for d in learned]) for k in range(1,5)}
 answers['5_expert_contributions_note']='Positive learned-minus-feature-removal EIG indicates a useful fusion contribution conditional on other experts. Router still sees that expert;this is not full information/path isolation. Feature similarities do not establish competency specialization.'
 for p in O.glob('*/RESOURCES.json'):
  d=json.loads(p.read_text());costs.append(dict(run=p.parent.name,**d))
 totals=dict(measured_resource_records=len(costs),exclusive_gpu_wall_seconds=sum(d['gpu_exclusive_wall_seconds'] for d in costs),
  sampled_gpu_busy_seconds_estimate=sum(d['sampled_gpu_busy_seconds_estimate'] for d in costs),cpu_user_seconds=sum(d['cpu_user_seconds'] for d in costs),cpu_system_seconds=sum(d['cpu_system_seconds'] for d in costs),
  new_training_episodes=sum(d['new_training_episodes'] for d in results.values()),new_training_observations=sum(d['new_training_observations'] for d in results.values()),
  resource_scope='Includes smoke/diagnosis where resource files exist;training episode totals include completed principal runs only. Busy utilization integral is not exact GPU kernel time. Historical provider/pretraining cost not reconstructed here.')
 out=dict(campaign_status=s['status'],completed_training_runs=s['completed_runs'],answers=answers,resources=totals,
  caveats=['Case CIs condition on shared contrast bank;five seeds estimate continuation variability only.','Canonical bank has informed development;no untouched-test claim.',
  'Wrapper computes instrumented descriptors for controls too;latency is not an optimized native V2 deployment benchmark.','No cost reward,balancing or diversity objective was introduced.'])
 (R/'SCIENTIFIC_CONCLUSIONS.json').write_text(json.dumps(out,indent=2)+'\n')
 text='# Scientific conclusions\n\nCampaign: '+s['status']+f"; {s['completed_runs']}/25 training runs complete.\n\n"
 text+='1. Learned versus independently trained uniform: '+answers['1_learned_vs_trained_uniform']['status']+'.\n\n'
 text+='2. Learned versus matched V2 continuation: '+answers['2_learned_vs_v2_continuation']['status']+'.\n\n'
 text+='3. Added-router/encoder attribution: trained factorial estimates are in SCIENTIFIC_CONCLUSIONS.json; pending entries are not findings.\n\n'
 text+='4. History-dependent value: assess stage-only/observation-swap EIG interventions and observation sensitivity together; varying weights alone is insufficient.\n\n'
 text+='5. Expert contribution: four feature-removal EIG effects are reported separately; they retain expert-derived router information and do not prove isolated competency.\n\n'
 text+='Five-seed intervals condition on common V2 pretraining and an already-observed benchmark. Original V2/V3 checkpoints remain frozen. No campaign success is declared while required comparisons remain pending.\n'
 (R/'SCIENTIFIC_CONCLUSIONS.md').write_text(text)
 print(json.dumps(dict(status=s['status'],completed=s['completed_runs'])))
if __name__=='__main__':main()
