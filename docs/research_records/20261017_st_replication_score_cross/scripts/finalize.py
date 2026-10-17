import json,hashlib,shutil,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/grads/g/g.lin/Documents/objective_power_grid');C=R/'experiments/10172026/10172026_st_replication_score_reverse_cross';O=C/'reports'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert read(C/'status.json')['state']=='complete'
for label in ['moe','score']:
 for rel,digest in read(C/f'{label}_source_manifest.json')['sha256'].items():assert sha(C/f'{label}_source'/rel)==digest,rel
for path,digest in read(C/'preserved_checkpoint_hashes.json').items():assert sha(Path(path))==digest,path
extra=read(C/'analysis_registry_amendment.json');assert sha(Path(extra['additional_preserved_checkpoint']))==extra['checkpoint_sha256']
assert not list(C.glob('runs/*/eval/*8831001*'))
moe=read(O/'moe_results.json');score=read(O/'score_cross_results.json');errors=read(O/'score_errors_summary.json');diag=read(O/'moe_diagnostics.json');stats=moe['statistics']
rows=moe['records'];srows=score['records'];s=stats['straight_through'];old=stats['original'];paired=stats['paired_seed_differences']
curves=[];onsets={};identities=set()
def difference(a,b):
 d=a-b;m=float(d.mean());se=float(d.std(ddof=1)/len(d)**.5);return dict(difference=m,mc_se=se,ci95=[m-1.96*se,m+1.96*se])
for row in rows:
 seed=row['seed'];folder=Path(row['checkpoint']).parent/'late_diagnostics';rr=read(folder/'checkpoints.json');rr.sort(key=lambda r:r['primary_episodes']);scores={}
 for r in rr:
  ep=r['primary_episodes'];identities.add(r['validation']['paired_draws_sha256']);scores[ep]=np.load(folder/f'validation_scores_{ep}.npy');curves.append(dict(seed=seed,episodes=ep,eig=r['validation_eig'],mc_se=r['mc_se'],gamma=r['gamma']))
 late=[r for r in rr if r['primary_episodes']>=750000];found=None
 for i in range(1,len(late)-2):
  anchor=max(late[:i],key=lambda r:r['validation_eig']);tests=[difference(scores[r['primary_episodes']],scores[anchor['primary_episodes']]) for r in late[i:i+3]]
  if all(t['difference']<-.3 and t['ci95'][1]<0 for t in tests):
   found=dict(first_checkpoint=late[i]['primary_episodes'],anchor=anchor['primary_episodes'],paired_changes=tests);break
 onsets[str(seed)]=dict(late_sustained_drop=found,final_minus_750016=difference(scores[1000000],scores[750016]),final_minus_800000=difference(scores[1000000],scores[800000]))
assert len(identities)==1,'Development validation cases must be paired across all seeds'
(O/'late_validation_diagnostics.json').write_text(json.dumps(dict(seeds=onsets,paired_validation_identity=next(iter(identities)),systems=512,contrasts=1000000,evaluation_seed=771001,rule='Exploratory first3 consecutive saved checks each>.3 below earlier running peak with paired95% upper<0; unadjusted for multiple looks'),indent=2)+'\n')
with (O/'validation_curves.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=curves[0].keys());w.writeheader();w.writerows(curves)
lines=['# MoE straight-through replication and ScoreBED crossed-score investigation','','Two separate location-finding development investigations. Every complete model is exactly1M-total. Final-budget checkpoints,2000 paired held-out trajectories,L1M,development seed1001. Original five-seed results are preserved. Final evaluation seed8831001 was not sampled. Score crossings are diagnostic interventions, never primary-result replacements.','','## A. Straight-through MoE: all five seeds','','| Training seed | Original hard EIG | ST hard EIG | ST relaxed EIG | Gain | E1/E2/E3/E4 usage |','|---|---:|---:|---:|---:|---|']
for r in rows:lines.append(f"| {r['seed']} | {r['original_eig']:.5f} | {r['hard_eig']:.5f} | {r['relaxed_eig']:.5f} | {r['paired_gain']['difference']:+.5f} | "+'/'.join(f'{100*x:.1f}%' for x in r['E1_E2_E3_E4'])+' |')
lines+=['',f"ST mean **{s['mean']:.5f}**, seed SD **{s['seed_sd']:.5f}**, minimum **{s['minimum']:.5f}**. Original mean {old['mean']:.5f}, SD {old['seed_sd']:.5f}, minimum {old['minimum']:.5f}.",'',f"Mean paired seed gain {paired['mean']:+.5f}, five-training-seed t interval {paired['seed_mean_t_ci95']}. Per-trajectory MC intervals are separately saved in moe_results.json; do not treat10000 trajectories as10000 independent trained policies.",'',f"Positive point differences in {stats['positive_differences']}/5 seeds; >=99.9% final single-expert selection in {stats['one_expert_collapsed_seeds']}/5 seeds.",'','The four64/64 tanh experts, sum encoder, native DSelect-1, gamma30%/95%/.001, optimizer, sPCE L1024, regularization and clipping were unchanged. Seeds103/105 are the preserved completed runs;101/102/104 are newly trained. ST forward is hard; the relaxed evaluation is a counterfactual, not the ST training forward.','']
lines+=['### Gradient exposure','','| Seed | Whole-run E1/E2/E3/E4 exposure | Final20% router exposure | Final20% selector saturation |','|---|---|---:|---:|']
for r in rows:
 d=diag[str(r['seed'])]['windows'];a=d['whole_run'];b=d['final20_percent'];ex='/'.join(f"{100*a['components'][f'expert_{j}']['gradient_exposure']:.2f}%" for j in range(4));lines.append(f"| {r['seed']} | {ex} | {100*b['components']['router']['gradient_exposure']:.2f}% | {100*b['mean_selector_saturation']:.2f}% |")
lines+=['','Gradient exposure means norm>1e-8. One-expert collapse is a negative mechanism result even if held-out EIG improves. The evidence cannot attribute a gain to useful four-expert specialization. A justified next controlled development intervention is temporary relaxed-forward expert warmup before unchanged ST routing, targeting undertrained experts. Freeze one duration, retain the original gamma schedule and episode budget, and test all five original seeds; this intervention is not implemented or launched.','','## B. ScoreBED: crossed frozen score bundles','','| Score bundle seed | Policy training seed | Held-out EIG | MC SE |','|---|---|---:|---:|']
for r in srows:lines.append(f"| {r['score_bundle_seed']} | {r['policy_training_seed']} | {r['eig']:.5f} | {r['mc_se']:.5f} |")
for key,r in score['paired_comparisons'].items():lines+=['',f"{key}: {r['difference']:+.5f}; paired95% MC interval {r['ci95']}."]
lines+=['','The missing crossing changes the frozen score bundle relative to native policy103: policy initialization/RNG follows103, score configuration/variables/normalization follows102. Initial parameters and keys match the native loader; exact bundle102 score predictions passed preflight. Both bundles were evaluated on identical physical histories. The intervention identifies a complete bundle effect and interaction, not isolated score-weight or source-normalization causality. **Budget audit:** every arm remains exactly1M-total. Policy-target normalization fields are unused by native DADNet initialization, physical simulation and score-derived gradients; replacing them with NaNs leaves initial parameters and native gradients bitwise identical. Only the frozen score network consumes its source calibration, already charged within702994 reused pretraining episodes. A provisional1.01M concern was rejected by this test and static call-path audit; all records are preserved. New policy initialization/training is297006 episodes, and no second calibration dataset is needed.','',
'Poor results with bundle102 at both policy seeds support a score-bundle contribution. Two policy initializations cannot establish a universal bundle ranking. Score-error references use Fisher identity, finite-difference verification, ESS>=32 and <=20% resolution change262k→1M. Low-ESS cases cannot support claims of score inaccuracy.','',
'Score-error summary:','','```json',json.dumps(errors,indent=2),'```','',
'No automatic score-training correction is justified by EIG crossing alone. If reliable matched-history errors establish a boundary-specific defect, the single candidate intervention is improved feasible-boundary coverage of score-training designs at the unchanged total episode budget. Otherwise recommend a matched policy-gradient error diagnostic before altering score training. Do not substitute bundle103 into primary seed102.','','## Cost and preservation','',
'New training:3M MoE primary episodes plus297006 new ScoreBED initialization/policy episodes. The ScoreBED complete-model budget includes702994 reused score-pretraining episodes. Reused MoE103/105 are not retrained. Correctness probes46 simulation episodes and validation/evaluation overhead are separate; no preflight weights are used for training. GPU phases are serialized. A14.43-second CPU-only dependency probe on two pinned cores overlapped seed102; retain this provenance rather than claiming perfectly identical wall-clock conditions. Measured root-phase wall times are in completion/, native optimization/validation times in each training record. Score pretraining wall time was not separately saved historically and is not invented.','',
'All original checkpoint hashes, both prior ST checkpoints, new frozen scientific sources and historical seed101 alias checkpoint were verified intact. Execution bookkeeping fixes (output path rejected before training, historical seed101 registry resolution, and a coordinator filename that shadowed the standard-library queue module during diagnostics) are preserved with provenance; none changed scientific training settings. The provisional extra-calibration concern was rejected by a native-gradient dependency test, with both initial and provisional records retained.','',
'Machine-readable results: moe_results.csv/.json, moe_diagnostics.json, score_cross_results.csv/.json, score_error_records.json and score_error_references.json. Exact commands/configurations/source and checkpoint hashes are retained in the campaign.','',
'![Investigation results](investigation_results.png)']
lines+=['','## Retained late-checkpoint behavior','','The full saved512-case/L1M validation curves use development771001 and are separate from the2000-case final development1001 scores. No intermediate checkpoint is selected. The sustained-drop screen is exploratory, unadjusted for repeated looks, and may identify temporary fluctuations rather than terminal failure.','','```json',json.dumps(onsets,indent=2),'```','','![Late validation curves](late_validation_curves.png)']
(O/'report.md').write_text('\n'.join(lines)+'\n')
fig_late,axis=plt.subplots(figsize=(9,4.5))
for seed in sorted(r['seed'] for r in rows):
 rr=[r for r in curves if r['seed']==seed and r['episodes']>=750000];xx=np.array([r['episodes'] for r in rr]);yy=np.array([r['eig'] for r in rr]);se=np.array([r['mc_se'] for r in rr]);axis.plot(xx,yy,'o-',ms=3,label=f'Seed {seed}');axis.fill_between(xx,yy-1.96*se,yy+1.96*se,alpha=.06)
axis.axvline(800000,color='black',linestyle=':',alpha=.4);axis.set_xlabel('Primary training episodes');axis.set_ylabel('Diagnostic validation EIG (nats)');axis.set_title('Retained ST late-stage curves:512 paired development cases, L=1M');axis.legend(ncol=5,fontsize=8);fig_late.tight_layout();fig_late.savefig(O/'late_validation_curves.png',dpi=220);fig_late.savefig(O/'late_validation_curves.pdf');plt.close(fig_late)
plt.rcParams.update({'font.size':9,'axes.titlesize':11})
fig,ax=plt.subplots(2,3,figsize=(15,8));seeds=[r['seed'] for r in rows];xs=np.arange(5)
ax[0,0].plot(xs,[r['original_eig'] for r in rows],'o--',label='Original relaxed training');ax[0,0].plot(xs,[r['hard_eig'] for r in rows],'o-',label='Straight-through training');ax[0,0].set_xticks(xs,seeds);ax[0,0].set_ylabel('Hard held-out EIG (nats)');ax[0,0].set_title('Paired training seeds');ax[0,0].legend(fontsize=8)
bottom=np.zeros(5)
for j in range(4):
 vals=np.array([r['E1_E2_E3_E4'][j] for r in rows]);ax[0,1].bar(xs,vals,bottom=bottom,label=f'E{j+1}');bottom+=vals
ax[0,1].set_xticks(xs,seeds);ax[0,1].set_ylim(0,1);ax[0,1].set_ylabel('Post-initial selection fraction');ax[0,1].set_title('ST expert utilization');ax[0,1].legend(ncol=4,fontsize=8)
exposure=np.array([[diag[str(seed)]['windows']['whole_run']['components'][f'expert_{j}']['gradient_exposure'] for j in range(4)] for seed in seeds]);im=ax[0,2].imshow(exposure,vmin=0,vmax=1,aspect='auto',cmap='Blues');ax[0,2].set_xticks(range(4),['E1','E2','E3','E4']);ax[0,2].set_yticks(range(5),seeds);ax[0,2].set_title('Training gradient exposure');fig.colorbar(im,ax=ax[0,2],label='Fraction of updates')
for i,p in enumerate([102,103]):
 rr=sorted([r for r in srows if r['policy_training_seed']==p],key=lambda r:r['score_bundle_seed']);ax[1,0].errorbar([0,1],[r['eig'] for r in rr],yerr=[1.96*r['mc_se'] for r in rr],marker='o',capsize=4,label=f'Policy seed {p}')
ax[1,0].set_xticks([0,1],['Score bundle 102','Score bundle 103']);ax[1,0].set_ylabel('Held-out EIG (nats)');ax[1,0].set_title('ScoreBED crossed diagnosis');ax[1,0].legend(fontsize=8)
er=read(O/'score_error_records.json')
for j,seed in enumerate([102,103]):
 rr=[r for r in er if r['evaluated_score_bundle']==seed and r['reference_reliable']];ax[1,1].scatter([r['boundary_fraction'] for r in rr],[r['design_relative_error'] for r in rr],label=f'Score bundle {seed}',marker=['o','x'][j],alpha=.8)
ax[1,1].set_xlabel('Boundary-coordinate fraction');ax[1,1].set_ylabel('Design-score relative error');ax[1,1].set_title('Reliable Fisher references only');ax[1,1].legend(fontsize=8)
ax[1,2].axis('off');ax[1,2].text(0,1,'Development evidence only\n\n5 training seeds; fixed 1M budget each\n2,000 paired trajectories; L=1,000,000\nEvaluation seed 1001\nFinal seed 8831001 untouched\n\nHigh EIG does not establish\nuseful multi-expert specialization.',va='top',fontsize=11)
fig.tight_layout();fig.savefig(O/'investigation_results.png',dpi=220);fig.savefig(O/'investigation_results.pdf');plt.close(fig)
(O/'integrity_verified.json').write_text(json.dumps(dict(status='passed',original_and_prior_st_checkpoint_count=len(read(C/'preserved_checkpoint_hashes.json'))+1,frozen_source_hashes_verified=True,reserved_final_seed_used=False,primary_results_replaced=False),indent=2))
print('REPORT_AND_FIGURES_COMPLETE',flush=True)
