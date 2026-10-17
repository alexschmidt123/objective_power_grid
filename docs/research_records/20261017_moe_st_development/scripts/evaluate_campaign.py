"""Paired development evaluation only; never trains or touches final test draws."""
import sys,json,time,hashlib,csv,math
from pathlib import Path
C=Path('/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_moe_st_routing_dev_103_105');S=C/'source';R=C.parents[2];sys.path.insert(0,str(S))
import torch,numpy as np,yaml
torch.set_num_threads(4)
from src.research.benchmarks import make_benchmark
from src.research.moe_family import load_policy
from src.research.journal_validation import CommonValidation
from src.research.engine import rollout
from src.research.location_evaluation import contrast_scores
raw=yaml.safe_load((C/'original.yaml').read_text());out=C/'reports';out.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');temp.replace(p)
def paired(a,b):
 d=np.asarray(a)-np.asarray(b);m=float(d.mean());se=float(d.std(ddof=1)/len(d)**.5)
 return dict(difference=m,mc_se=se,ci95=[m-1.96*se,m+1.96*se])
def summary(values):
 x=np.asarray(values);m=float(x.mean());se=float(x.std(ddof=1)/len(x)**.5)
 return dict(eig=m,mc_se=se,ci95=[m-1.96*se,m+1.96*se])
def soft_evaluate(checkpoint,validator,label):
 path=out/(label+'_scores.npy')
 if path.exists():return np.load(path)
 started=time.perf_counter();p=load_policy(checkpoint,make_benchmark(raw),relaxed=True);values=[]
 with torch.no_grad():
  b=make_benchmark(raw)
  for start in range(0,validator.systems,16):
   end=min(start+16,validator.systems)
   rr=rollout(b,p,30,end-start,validator.seed+start*97451,0,truth=validator.truth[start:end],noise=validator.noise[start:end])
   scores,_=contrast_scores(b,rr,validator.bank,16384);values.extend(scores[:,-1].tolist())
 np.save(path,values);write(out/(label+'_cost.json'),dict(seconds=time.perf_counter()-started,systems=validator.systems,contrasts=validator.contrasts,likelihood_particle_stages=validator.systems*(validator.contrasts+1)*30,paired_draws_sha256=validator.identity,training_episodes=0));print('RELAXED_COMPLETE',label,float(np.mean(values)),flush=True);return np.asarray(values)
def compact_trace(path):
 rows=[]
 with path.open() as f:
  for line in f:
   r=json.loads(line)
   rows.append({k:r[k] for k in ['primary_episodes','gamma','saturated_bit_fraction','fully_saturated_selector_fraction','logit_quantiles','component_gradient_before','component_gradient_after','actual_parameter_updates','training_spce','training_expert_usage']})
 return rows
def window_summary(trace):
 ans={}
 for lo,hi in [(0,200000),(200000,500000),(500000,750000),(750000,800000),(800000,850000),(850000,900000),(900000,1000001)]:
  rows=[r for r in trace if lo<=r['primary_episodes']<hi];components={}
  for g in rows[0]['component_gradient_before']:
   norms=np.array([r['component_gradient_before'][g] for r in rows])
   components[g]=dict(gradient_exposure_above_1e_minus8=float((norms>1e-8).mean()),negligible_gradient_fraction=float((norms<=1e-12).mean()),raw_gradient_quantiles=np.quantile(norms,[0,.5,.9,.99,1]).tolist(),clipped_gradient_max=max(r['component_gradient_after'][g] for r in rows),actual_step_quantiles=np.quantile([r['actual_parameter_updates'][g]['update_l2'] for r in rows],[0,.5,.9,.99,1]).tolist())
  ans[f'{lo}-{hi}']=dict(updates=len(rows),hard_selection_frequency=np.mean([r['training_expert_usage'] for r in rows],0).tolist(),saturated_bit_quantiles=np.quantile([r['saturated_bit_fraction'] for r in rows],[0,.5,.9,.99,1]).tolist(),selector_logit_quantile_medians=np.median([r['logit_quantiles'] for r in rows],0).tolist(),components=components)
 return ans
def onset(recs,scores):
 for i in range(1,len(recs)-2):
  a=max(recs[:i],key=lambda r:r['validation_eig']);tests=[paired(scores[r['primary_episodes']],scores[a['primary_episodes']]) for r in recs[i:i+3]]
  if all(t['difference']<-.3 and t['ci95'][1]<0 for t in tests):return dict(first_saved_deterioration=recs[i]['primary_episodes'],anchor=a['primary_episodes'],previous_saved_checkpoint=recs[i-1]['primary_episodes'],three_paired_changes=tests)
 return dict(first_saved_deterioration=None,note='No sustained drop satisfying the stated rule on saved checkpoints; not proof of general stability')
val=CommonValidation(raw,30,771001,512,1000000,16,16384);finalval=CommonValidation(raw,30,1001,2000,1000000,16,16384)
final=[];diagnostics={};curves=[];comparisons=[];all_scores={}
for seed in [103,105]:
 new=S/f'experiments/10172026/10172026_moe_st_dev/seed{seed}'
 old=R/f'experiments/10152026/10152026_journal_five_seed_fixed_budget/location/seed{seed}/moe'
 print('ANALYZE_SEED',seed,flush=True)
 new_settings=read(new/'models/moe_effective_settings.json');old_settings=read(old/'models/moe_effective_settings.json')
 differences={k:dict(original=old_settings.get(k),straight_through=new_settings.get(k)) for k in set(new_settings)|set(old_settings) if new_settings.get(k)!=old_settings.get(k)}
 assert set(differences)<={'moe_training_routing_mode','diagnostic_late_logging','diagnostic_start_episodes','diagnostic_checkpoint_episodes'},differences
 write(out/f'configuration_differences_{seed}.json',differences)
 for label,run in [('original',old),('straight_through',new)]:
  cases=read(run/'eval/moe_seed1001.json');hard=np.array([r['spce'] for r in cases]);checkpoint=run/'models/moe.pt'
  assert len(cases)==2000 and np.array_equal(np.array([r['truth'] for r in cases]),finalval.truth.numpy())
  assert np.array_equal(np.array([r['measurement_noise_standard'] for r in cases]),finalval.noise.numpy())
  assert np.array_equal(np.load(run/'eval/contrast_bank_seed1001.npy'),finalval.bank.numpy())
  # Independently verify checkpoint/action/score compatibility on first16 cases,
  # retaining the original2000-population bank rather than drawing a new bank.
  n=finalval.systems;finalval.systems=16
  p=load_policy(checkpoint,make_benchmark(raw));check,_=finalval(p);finalval.systems=n
  assert np.allclose(check,hard[:16],rtol=0,atol=1e-10)
  soft=soft_evaluate(checkpoint,finalval,f'{label}_{seed}_relaxed_final1001')
  all_scores[(seed,label)]=hard
  usage=np.bincount([r['selected_expert'] for c in cases for r in c['routing']],minlength=4)/58000
  tr=read(run/'models/moe_training.json')
  rec=dict(seed=seed,configuration=label,hard=summary(hard),relaxed=summary(soft),relaxed_minus_hard=paired(soft,hard),absolute_hard_relaxed_gap=abs(float(soft.mean()-hard.mean())),hard_expert_usage=usage.tolist(),training_episodes=tr['generated_training_trajectories'],training_seconds=tr['training_seconds'],optimization_seconds=tr.get('optimization_seconds'),validation_seconds=tr.get('validation_seconds'),diagnostic_seconds=tr.get('development_diagnostic_seconds',0),contrast_training_trajectories=tr['contrast_training_trajectories'],truth_inclusive_likelihood_particle_stages=tr['generated_training_trajectories']*1025*30,checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),paired_draws_sha256=finalval.identity)
  final.append(rec);write(out/'final_eig.json',final)
 comparison=dict(seed=seed,straight_through_minus_original=paired(all_scores[(seed,'straight_through')],all_scores[(seed,'original')]))
 comparisons.append(comparison);write(out/'paired_comparisons.json',comparisons)
 d=new/'models/late_diagnostics';recs=read(d/'checkpoints.json');scores={r['primary_episodes']:np.load(d/f"validation_scores_{r['primary_episodes']}.npy") for r in recs}
 assert all(r['validation']['paired_draws_sha256']==val.identity for r in recs)
 trace=compact_trace(d/'training_trace.jsonl');assert len(trace)==15625 and trace[-1]['primary_episodes']==1000000
 windows=window_summary(trace)
 soft_rows=[]
 chosen=[750016,800000,812544,825024,883200,1000000]
 for r in recs:
  ep=r['primary_episodes'];routes=r['probe_hard_routing'];choices=np.array([x['selected_experts'] for x in routes]);actions=np.array(r['probe_physical_actions']);logits=np.array(r['probe_hard_logits'])
  row=dict(seed=seed,configuration='straight_through',episodes=ep,hard_eig=r['validation_eig'],mc_se=r['mc_se'],gamma=r['gamma'],own_history_saturated_bit_fraction=float((abs(logits)>=r['gamma']/2).mean()),hard_expert_usage=[float((choices==j).mean()) for j in range(4)],mean_measurement_step=float(np.linalg.norm(np.diff(actions,axis=0),axis=-1).mean()),boundary_coordinate_fraction=float((abs(actions)>3.8).mean()))
  if ep in chosen:
   soft=soft_evaluate(r['checkpoint'],val,f'st_{seed}_relaxed_{ep}')
   row.update(relaxed=summary(soft),relaxed_minus_hard=paired(soft,scores[ep]),absolute_hard_relaxed_gap=abs(float(soft.mean()-r['validation_eig'])))
  curves.append(row)
 diagnostics[str(seed)]=dict(windows=windows,sustained_deterioration=onset(recs,scores),late_sustained_deterioration=onset([r for r in recs if r['primary_episodes']>=750000],scores),final_minus_pre800k=paired(scores[1000000],scores[750016]),final_minus_800k=paired(scores[1000000],scores[800000]))
 # Original diagnostic validation histories are retained at their actual budgets.
 for line in (old/'models/moe_validation.jsonl').read_text().splitlines():
  r=json.loads(line);curves.append(dict(seed=seed,configuration='original_native_validation',episodes=r['episodes'],hard_eig=r['eig'],mc_se=r['mc_standard_error']))
 if seed==103:
  od=R/'experiments/10162026/10162026_location_root_cause_v4/source/experiments/10162026/10162026_root_cause_v4/moe_instrumented_seed103/models/late_diagnostics'
  orecords=read(od/'checkpoints.json');assert all(r['validation']['paired_draws_sha256']==val.identity for r in orecords);oscores={r['primary_episodes']:np.load(od/f"validation_scores_{r['primary_episodes']}.npy") for r in orecords}
  for r in orecords:
   ep=r['primary_episodes'];row=dict(seed=seed,configuration='original_instrumented',episodes=ep,hard_eig=r['validation_eig'],mc_se=r['mc_se'],gamma=r['gamma'])
   if ep in chosen:
    cache=R/f'experiments/10162026/10162026_location_root_cause_v4/reports/component_analysis/moe_relaxed_{ep}.npy'
    soft=np.load(cache) if cache.exists() else soft_evaluate(r['checkpoint'],val,f'original_{seed}_relaxed_{ep}')
    row.update(relaxed=summary(soft),relaxed_minus_hard=paired(soft,oscores[ep]),absolute_hard_relaxed_gap=abs(float(soft.mean()-r['validation_eig'])))
   curves.append(row)
  diagnostics[str(seed)]['original_sustained_deterioration']=onset(orecords,oscores)
  diagnostics[str(seed)]['original_final_minus_pre800k']=paired(oscores[1000000],oscores[750016])
 write(out/'diagnostics.json',diagnostics);write(out/'development_curves.json',curves)
 # Per-stage final hard utilization and selection counts, complete2000 cases.
 for label,run in [('original',old),('straight_through',new)]:
  cases=read(run/'eval/moe_seed1001.json');selected=np.array([[r['selected_expert'] for r in c['routing']] for c in cases]);by_stage=np.stack([(selected==j).mean(0) for j in range(4)],-1)
  write(out/f'{label}_{seed}_utilization_by_stage.json',dict(excludes_first_design=True,stages=list(range(2,31)),E1_E2_E3_E4_frequencies=by_stage.tolist()))
# Tables include both signs and uncertainty; all original data remain separate.
with (out/'results.csv').open('w') as f:
 fields=['seed','configuration','hard_eig','hard_mc_se','relaxed_eig','relaxed_minus_hard','training_episodes','training_seconds','checkpoint'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for r in final:w.writerow(dict(seed=r['seed'],configuration=r['configuration'],hard_eig=r['hard']['eig'],hard_mc_se=r['hard']['mc_se'],relaxed_eig=r['relaxed']['eig'],relaxed_minus_hard=r['relaxed_minus_hard']['difference'],training_episodes=r['training_episodes'],training_seconds=r['training_seconds'],checkpoint=r['checkpoint']))
with (out/'paired_comparisons.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['seed','ST_minus_original','paired_mc_se','ci95_low','ci95_high'])
 for r in comparisons:
  t=r['straight_through_minus_original'];w.writerow([r['seed'],t['difference'],t['mc_se'],*t['ci95']])
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(2,3,figsize=(15,8))
for i,seed in enumerate([103,105]):
 for label,style in [('straight_through','o-'),('original_instrumented' if seed==103 else 'original_native_validation','s--')]:
  rows=sorted([r for r in curves if r['seed']==seed and r['configuration']==label],key=lambda r:r['episodes']);xs=[r['episodes'] for r in rows];ys=np.array([r['hard_eig'] for r in rows]);ses=np.array([r['mc_se'] for r in rows]);axs[i,0].plot(xs,ys,style,ms=3,label=label);axs[i,0].fill_between(xs,ys-1.96*ses,ys+1.96*ses,alpha=.12)
 axs[i,0].set_title(f'MoE-sBOED seed{seed}: hard development EIG');axs[i,0].legend(fontsize=7)
 strows=sorted([r for r in curves if r['seed']==seed and r['configuration']=='straight_through'],key=lambda r:r['episodes']);axs[i,1].stackplot([r['episodes'] for r in strows],np.array([r['hard_expert_usage'] for r in strows]).T,labels=['E1','E2','E3','E4']);axs[i,1].set_ylim(0,1);axs[i,1].set_title('Hard expert utilization,32 cases');axs[i,1].legend(fontsize=7)
 mixed=[r for r in strows if 'relaxed' in r];axs[i,2].plot([r['episodes'] for r in mixed],[r['hard_eig'] for r in mixed],'o-',label='hard');axs[i,2].plot([r['episodes'] for r in mixed],[r['relaxed']['eig'] for r in mixed],'s--',label='relaxed');axs[i,2].set_title('Paired hard versus relaxed EIG');axs[i,2].legend()
 for ax in axs[i]:ax.set_xlabel('Training episodes');ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
fig.tight_layout();fig.savefig(out/'development_diagnostics.png',dpi=200);fig.savefig(out/'development_diagnostics.pdf')
lines=['# MoE-sBOED straight-through routing development experiment','',
'Only training routing changed. Seeds103/105; exactly1M primary episodes each; final-budget checkpoint selection. Original runs, all other variants and final evaluation seed8831001 remain untouched. This is an optional MoE-sBOED training mode, not a new named variant.','',
'## Final paired development EIG','',
'2000 paired trajectories, L=1,000,000, seed1001. Native hard scores reused only after truth/noise/contrast-bank identity checks and independent16-case checkpoint score replay. Relaxed scores use the same draws and existing common scorer.','',
'| Seed | Training routing | Hard EIG ± MC SE | Relaxed EIG | Relaxed − hard [95% MC interval] | Training seconds |','|---|---|---:|---:|---|---:|']
for r in final:
 t=r['relaxed_minus_hard'];ci=t['ci95'];lines.append(f"| {r['seed']} | {r['configuration']} | {r['hard']['eig']:.5f} ± {r['hard']['mc_se']:.5f} | {r['relaxed']['eig']:.5f} | {t['difference']:+.5f} [{ci[0]:+.5f}, {ci[1]:+.5f}] | {r['training_seconds']:.2f} |")
lines+=['','| Seed | ST − original hard EIG | Paired95% MC interval |','|---|---:|---|']
for r in comparisons:
 t=r['straight_through_minus_original'];lines.append(f"| {r['seed']} | {t['difference']:+.5f} | [{t['ci95'][0]:+.5f}, {t['ci95'][1]:+.5f}] |")
lines+=['','These intervals measure trajectory-level Monte Carlo uncertainty, not variability over independent training runs. Two selected development seeds cannot establish general superiority. All negative differences are retained.','',
'## Late-stage behavior','']
for seed in [103,105]:
 d=diagnostics[str(seed)];lines+=['### Seed'+str(seed),'', 'All-checkpoint sustained-drop diagnostic: `'+json.dumps(d['sustained_deterioration'])+'`. Late-only750k–1M diagnostic: `'+json.dumps(d['late_sustained_deterioration'])+'`.','', 'Final minus750016-episode hard development EIG: `'+json.dumps(d['final_minus_pre800k'])+'`.','', 'Final minus800000-episode hard development EIG: `'+json.dumps(d['final_minus_800k'])+'`.','']
lines+=['The onset rule is exploratory and not adjusted for repeated looks. Original105 has no saved intermediate model weights; its original hard validation curves are shown at their actual episode counts, and an intermediate relaxed comparison is unavailable. No original-setting training was repeated to fill this gap.','',
'## Utilization and gradient exposure','',
'| Seed | Window | E1/E2/E3/E4 hard selection | E1/E2/E3/E4 gradient exposure | Median selector saturation | Median router step |','|---|---|---|---|---:|---:|']
for seed in [103,105]:
 for window,r in diagnostics[str(seed)]['windows'].items():
  usage='/'.join(f'{v:.1%}' for v in r['hard_selection_frequency']);exposure='/'.join(f"{r['components'][f'expert_{j}']['gradient_exposure_above_1e_minus8']:.1%}" for j in range(4));lines.append(f"| {seed} | {window} | {usage} | {exposure} | {r['saturated_bit_quantiles'][1]:.2%} | {r['components']['router']['actual_step_quantiles'][1]:.5g} |")
lines+=['','Gradient exposure threshold is norm>1e-8. JSON records zero-gradient fractions, raw/clipped gradient distributions, actual parameter increments and selector-logit quantiles. Full stage-wise selection frequencies are saved separately. In hard-forward training all four proposals are computed for router backward gradients, but only selected experts receive direct action gradients. A stable high-EIG policy dominated by one expert does not establish beneficial expert specialization.','',
'## Correctness and implementation','',
'13 focused checks passed. Exact hard routing weights at gamma1/.1/.001, native zero-logit ties, exact fixed-context relaxed-router gradient equivalence, unchanged initial parameters, full hard-forward rollout within1e-12, unchanged relaxed regularizers, selected-expert gradient flow, exclusion of other variants, historical103/105 checkpoint prediction compatibility, one-expert hard inference, checkpoint roundtrip and finite full-batch/L1024 sPCE gradients. Measured rollout differences: actions8.88e-16, observations1.07e-14, latents3.05e-16 (dense versus sparse batched arithmetic).','',
'The implementation uses w_ST=stop_gradient(w_hard)+(w_soft-stop_gradient(w_soft)); all four proposals enter the surrogate action expression. This is a biased routing-gradient surrogate, not a new EIG estimator. Native relaxed load/selector regularizers and all optimization settings are preserved. A selected-expert-only sparse dispatcher during training would incorrectly remove unselected proposal contributions to the routing gradient; it is deliberately not used here.','',
'## Interpretation and adoption','',
'Inspect both seed103 recovery and seed105 preservation; do not choose the favorable seed. If the joint results are promising, recommend an unchanged five-original-seed replication before changing the journal protocol. If one seed deteriorates or utilization/gradient exposure collapses, retain the negative result and do not claim the correction solved instability. An absence of a sustained late drop does not imply successful training if the policy was already poor. No further training or hyperparameter tuning is automatically launched.','',
'## Accounting and reproduction','',
'Two full runs use2,000,000 primary training episodes total,2,048,000,000 contrast trajectories, and61,500,000,000 truth-inclusive likelihood particle-stages. No pretraining. Diagnostic validation/evaluation and correctness generation are recorded separately. Original optimizer, learning-rate scheduler, regularization, clipping and gamma clock remain unchanged. Each run uses rootrun.sh; exact arguments are in `commands/seed103.json` and `commands/seed105.json`. Protocol, source hashes, original checkpoint hashes and patch are at the campaign root. Final checkpoint paths/hashes and wall times are in `completion/` and `final_eig.json`.','',
'Source changes: `src/research/sparse_moe.py`, `src/research/moe_family.py`, `src/research/family_training.py`, plus configurable diagnostic checkpoint observation in `src/research/moe_diagnostic_logging.py`. Historical source copies remain unchanged.','',
'![Development diagnostics](development_diagnostics.png)','',
'Machine-readable outputs: `results.csv`, `paired_comparisons.csv`, `final_eig.json`, `development_curves.json`, `diagnostics.json`, and stage-wise utilization JSONs. Saved hard/relaxed per-trajectory score arrays support paired uncertainty reconstruction.']
lines+=['','## Observed outcome, without checkpoint replacement','']
for seed in [103,105]:
 d=diagnostics[str(seed)];change=next(r['straight_through_minus_original'] for r in comparisons if r['seed']==seed);newrec=next(r for r in final if r['seed']==seed and r['configuration']=='straight_through');dominant=max(newrec['hard_expert_usage']);late=d['late_sustained_deterioration']['first_saved_deterioration'];pre=d['final_minus_pre800k']
 lines.append(f"Seed{seed}: final hard EIG change{change['difference']:+.5f}, paired95% interval{change['ci95']}; final-minus-pre800k change{pre['difference']:+.5f}, interval{pre['ci95']}; late sustained-drop onset={late}. Final hard expert frequencies={newrec['hard_expert_usage']}.")
 if dominant>=.99:lines.append('At least99% of final decisions use one expert. This is expert collapse/underutilization, not evidence of four useful specialized policies.')
 c103=next(r['straight_through_minus_original'] for r in comparisons if r['seed']==103);c105=next(r['straight_through_minus_original'] for r in comparisons if r['seed']==105)
if c103['ci95'][0]>0 and c105['ci95'][1]>=0:
 lines.append('The seed103 recovery is supported conditionally on these development cases, with no clear negative seed105 contrast at the nominal Monte Carlo level. This merits an unchanged five-seed development replication before adoption; it does not establish noninferiority, seed-level robustness or beneficial specialization.')
else:
 lines.append('These two development cases do not provide a clean recovery-with-preservation result. Retain the negative or inconclusive contrasts and do not adopt the correction as a demonstrated stability fix.')
(out/'report.md').write_text('\n'.join(lines)+'\n')
write(out/'evaluation_complete.json',dict(status='complete',training_launched_by_analysis=False,training_seeds=[103,105],development_eval_seed=1001,late_diagnostic_seed=771001,final_test_seed_used=False,primary_results_replaced=False))
print('PAIRED_ANALYSIS_COMPLETE',flush=True)
