import os
"""Read-only scientific audit; writes only a new report directory. No training/evaluation."""
import pathlib,json,hashlib,csv,statistics,math,collections,datetime,sys
import numpy as np
import torch
root=pathlib.Path(os.environ.get('MOE_PROJECT_ROOT',pathlib.Path(__file__).resolve().parents[2]));C=root/'experiments/10152026/10152026_journal_five_seed_fixed_budget'
out=C/'audit/troubleshooting_v2';out.mkdir(exist_ok=True)
registry=json.loads((C/'registry.json').read_text()); focus=['vsoed','scorebed','moe','switchmoe']; allrows=[];evidence=[];curves=[];inputs={}
def read(p):
 inputs[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
def finite(x):
 if torch.is_tensor(x):return bool(torch.isfinite(x).all())
 if isinstance(x,dict):return all(finite(y) for y in x.values())
 if isinstance(x,(list,tuple)):return all(finite(y) for y in x)
 return True
for e in registry:
 if e['benchmark']!='location' or e['status']!='complete' or e['method'] not in focus:continue
 m=e['method'];seed=e['seed'];n=e['native_method'];run=pathlib.Path(e['run']);mp=run/'models';tp=mp/(n+'_training.json');tr=read(tp);ev=run/'eval'/(n+'_seed1001.json');rows=read(ev)
 scores=np.array([r['spce'] for r in rows]);actions=np.array([r['actions'] for r in rows]);counts=np.zeros(4);requests=changes=decisions=0;stage=np.zeros((29,4));dur=[]
 for r in rows:
  for i,route in enumerate(r.get('routing',[])):
   j=route.get('selected_expert');
   if j is not None:counts[j]+=1;stage[i,j]+=1
   st=route.get('strategy',{})
   if st and not st.get('initialized',False):requests+=bool(st.get('reroute_requested'));changes+=bool(st.get('actual_strategy_change'));decisions+=1
  ids=[z['selected_expert'] for z in r.get('routing',[]) if 'selected_expert'in z]
  if ids:
   length=1
   for a,b in zip(ids,ids[1:]):
    if a==b:length+=1
    else:dur.append(length);length=1
   dur.append(length)
 ck=mp/(n+'.pt');check=dict(exists=ck.exists())
 if ck.exists():
  sha=hashlib.sha256(ck.read_bytes()).hexdigest();check.update(sha256=sha,registry_hash_matches=sha==e.get('checkpoint_sha256',e.get('audit',{}).get('checkpoint_sha256')),tensor_values_finite=finite(torch.load(ck,map_location='cpu',weights_only=False)))
 rec=dict(method=m,seed=seed,eig=float(scores.mean()),trajectory_mc_se=float(scores.std(ddof=1)/len(scores)**.5),episodes=tr.get('generated_training_trajectories'),checkpoint_selection=tr.get('checkpoint_selection'),boundary_coordinate_fraction=float(np.mean(np.abs(actions)>3.8)),mean_consecutive_action_distance=float(np.linalg.norm(np.diff(actions,axis=1),axis=2).mean()),expert_utilization=(counts/counts.sum()).tolist() if counts.sum() else None,reroute_fraction=requests/decisions if decisions else None,actual_change_fraction=changes/decisions if decisions else None,mean_expert_duration=statistics.mean(dur) if dur else None,checkpoint_check=check,run=str(run.relative_to(root)))
 vals=tr.get('journal_validation_history',[])
 if m=='scorebed':
  vp=mp/'scorebed_validation.jsonl';vals=[json.loads(l) for l in vp.read_text().splitlines()] if vp.exists() else [];rec['original_validation_invalid']=True
 rec['validation']=[dict(episodes=z['episodes'],eig=z['eig']) for z in vals]
 if m in ['moe','switchmoe']:
  path=pathlib.Path(tr['diagnostic_trace']);path=path if path.is_absolute() else root/path
  buckets=[collections.defaultdict(list) for _ in range(5)];last=None;first_large=None
  for line in path.open():
   x=json.loads(line);last=x;idx=min(4,(x['primary_episodes']-1)//200000);b=buckets[idx]
   for k,v in x.get('gradient_before',{}).items():b[k].append(v)
   b['total_raw_gradient'].append(x['gradient_norm']);b['training_spce'].append(x['training_eig'])
   if first_large is None and x.get('gradient_before',{}).get('router',0)>1e6:first_large=x['primary_episodes']
   if x['update']%100==0:curves.append(dict(method=m,seed=seed,episodes=x['primary_episodes'],training_spce=x['training_eig'],gamma=x['gamma'],router_raw_gradient=x.get('gradient_before',{}).get('router',0)))
  rec['gradient_by_fifth']=[{k:dict(mean=float(np.mean(v)),p99=float(np.quantile(v,.99)),maximum=float(max(v)),nonzero_fraction=float(np.mean(np.array(v)>1e-8))) for k,v in b.items()} for b in buckets]
  rec['first_router_gradient_above_1e6_episodes']=first_large;rec['final_training_scores']=last.get('training_score_distribution');rec['final_training_spce']=last['training_eig'];rec['final_gate_probability']=last.get('strategy_reroute_probability_mean');rec['trace_updates']=last['update'];rec['trace_episodes']=last['primary_episodes'];rec['stage_utilization']=(stage/2000).tolist()
 evidence.append(rec);allrows.append({k:rec[k] for k in ['method','seed','eig','trajectory_mc_se','episodes','checkpoint_selection','boundary_coordinate_fraction','mean_consecutive_action_distance']})
(out/'evidence.json').write_text(json.dumps(evidence,indent=2));(out/'source_hashes.json').write_text(json.dumps(inputs,indent=2))
with (out/'results.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
with (out/'training_curves.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
verification=read(C/'audit/instability_verification.json')
intro='''This is a development troubleshooting report, not final confirmatory evidence. Training seeds 101–105 are running under the frozen fixed-budget protocol; only completed methods enter this snapshot. Evaluation uses seed 1001, 2,000 paired trajectories, L=1,000,000, and the final 1M-budget checkpoint. No retraining, checkpoint reselection, algorithm change, or reserved-seed evaluation was performed for this audit.\n\nA nonzero gradient (>1e-8) is an exposure proxy, not proof of informative learning. Raw gradient norms are measured before component clipping; large raw values are not the applied parameter update. Correlation with sharpening does not establish the router as the sole cause.\n'''
reports={
'moe':('MoE-sBOED','Confirmed optimization breakdown in seed 103; root cause not fully isolated.', '''Seed 103 training sPCE falls during the last fifth while raw gradients explode. Its final held-out EIG is below Random's historical 8.26624 reference. The accepted run completes all 15,625 updates and 1M episodes; it is not a prematurely terminated model. Late relaxed-training deterioration means hard inference alone cannot explain every aspect of the failure. However, hard-versus-relaxed held-out EIG remains unmeasured in this audit.\n\nThe successful seed 104 despite highly concentrated routing shows that concentration alone is insufficient to explain failure. Assess starvation using exposure, norm distributions and proposal quality jointly.\n\nNext diagnostic after current runs: paired hard/relaxed evaluations of the same good and failed checkpoints; locate the deterioration against gamma and group gradient spikes. If a correction is justified, change one sharpening/optimization factor only in a separate development protocol. Preserve expert widths, encoder, objective, training count and evaluator. Do not choose a new setting simply because it repairs seed 103.'''),
'switchmoe':('SwitchMoE-sBOED','Confirmed seed-103 optimization breakdown and failure to demonstrate learned hard continuation.', '''The same late sPCE deterioration and gradient instability occurs in seed 103. Seeds 101–104 hard routing statistics are measured from saved deployed rollouts, excluding the initial routed initialization from continue/reroute denominators. A rerouting request may keep the same expert; actual changes are reported separately. Long expert durations do not prove that the gate chose continuation.\n\nThe gate is initialized near 0.99 rerouting. This is a plausible contributor to absence of persistence, not a demonstrated cause of EIG collapse. Do not simultaneously change gate initialization and router sharpening. First isolate the shared MoE instability. Gate-calibration interventions should be a separate subsequent development experiment with unchanged experts and router configuration.\n\nJournal claim restriction: current evidence supports the implemented switching architecture, but not useful learned temporal persistence or a switching-only performance advantage.'''),
'scorebed':('ScoreBED','Confirmed diagnostic validation defect; seed-102 final-policy weakness is real, underlying training cause unresolved.', '''`src/research/journal_scorebed_bridge.py::maybe_validate` saves `export_arrays(params)` without `common_design_bounds`. `OfficialScorePolicy` defaults to no bounding when this metadata is absent. By contrast, `scorebed_worker.py` includes bounds in the final official policy export. The final JAX-to-Torch bridge verifies bounded actions.\n\nConsequence: intermediate validation scores evaluate the wrong policy mapping. They must not be used as valid learning curves, convergence evidence or comparisons. Final checkpoint selection is fixed-budget, so weights are not chosen using these erroneous scores. No final held-out result is replaced by an unbounded result. A small prior smoke test passing was insufficient to establish bounded validation equivalence.\n\nSeed 102's boundary concentration and low EIG persist with the correct bounded final policy. The defect therefore does not establish why score-based training produced a weak seed. Frozen score-model quality, score-gradient direction on policy histories, and mismatch between score-training histories and policy histories remain untested.\n\nRequired correction after campaign completion: propagate bounds in intermediate exports; add action and complete-rollout parity tests between native JAX, validation and deployed Torch policies on identical histories and noise. Rescore saved candidates; missing historical candidates cannot be reconstructed from final weights. No full retraining is justified solely to repair diagnostic metadata. If the next audit finds a training-mechanism defect, declare it and rerun all affected seeds under one corrected configuration, retaining the originals.'''),
'vsoed':('vsOED','Seed sensitivity and incomplete convergence evidence; no catastrophic collapse found in inspected curves.', '''Seeds 102 and 103 improve through the final validation checkpoint. Seed 104 has a modest late decrease. The observed seed range is materially smaller than the ScoreBED/MoE failures, although larger than trajectory Monte Carlo standard errors. This is not proof that native actor–critic or variational posterior learning is numerically optimal.\n\nThe training reward is native variational reward and is not interchangeable with common held-out EIG. Incomplete critic/posterior loss records limit causal diagnosis; reward trends alone cannot isolate actor, critic or posterior approximation error. No extra primary episodes should be added to this method alone in the 1M study.\n\nNo rerun is currently justified. Report fixed-budget performance and unverified convergence. Only a demonstrated implementation/protocol defect would justify replacing the configuration, and affected seeds would need consistent treatment. Do not reuse the historical 5M result in the main comparison.''')}
for m,(name,status,body) in reports.items():
 rr=[x for x in evidence if x['method']==m];text=f'# {name}: troubleshooting report\n\n{intro}\n## Assessment\n\n{status}\n\n{body}\n\n## Measured evidence\n\n| Seed | EIG | Trajectory MC SE | Boundary coordinates | Final-budget selection | Checkpoint finite / hash matches |\n|---|---:|---:|---:|---|---|\n'
 for x in rr:
  ch=x['checkpoint_check'];text+=f"| {x['seed']} | {x['eig']:.5f} | {x['trajectory_mc_se']:.5f} | {x['boundary_coordinate_fraction']:.2%} | {x['checkpoint_selection']} | {ch.get('tensor_values_finite')} / {ch.get('registry_hash_matches')} |\n"
 if len(rr)>1:text+=f"\nAcross {len(rr)} completed training seeds: mean {statistics.mean(x['eig'] for x in rr):.5f}, sample SD {statistics.stdev(x['eig'] for x in rr):.5f}. This is an interim summary, not a five-seed result.\n"
 if m in ['moe','switchmoe']:
  text+='\n| Seed | Final-fifth mean raw norm | Router p99 raw norm | Router max raw norm | Final batch sPCE | Expert usage E1/E2/E3/E4 |\n|---|---:|---:|---:|---:|---|\n'
  for x in rr:
   g=x['gradient_by_fifth'][-1];text+=f"| {x['seed']} | {g['total_raw_gradient']['mean']:.3g} | {g['router']['p99']:.3g} | {g['router']['maximum']:.3g} | {x['final_training_spce']:.5f} | "+'/'.join(f'{z:.2%}' for z in x['expert_utilization'])+' |\n'
  text+='\nFinal-batch score quantiles are [minimum, p10, median, p90, p99, maximum]; they describe one batch, not the whole training distribution. Full traces contain per-update batch summaries.\n'
  for x in rr:text+=f"\n- Seed {x['seed']}: {x['final_training_scores']}.\n"
 if m=='switchmoe':
  for x in rr:text+=f"\n- Seed {x['seed']}: deployed reroute requests {x['reroute_fraction']:.2%}; actual changes {x['actual_change_fraction']:.2%}; mean expert run length {x['mean_expert_duration']:.3f}.\n"
 if m=='scorebed':
  text+='\nBounded replay on the independent validation population (512, seed 771001; not final evaluation):\n'
  for x in verification:
   if 'correct_bounded_validation_eig'in x:text+=f"\n- Seed {x['seed']}: incorrect logged {x['previous_unbounded_validation_eig']:.5f}; corrected final bounded replay {x['correct_bounded_validation_eig']:.5f}.\n"
 text+='\n## Reproducibility and limitations\n\nSee `evidence.json` for run paths, complete stage usage, validation history, gradients and checkpoint checks; `source_hashes.json` hashes inputs. Only saved complete runs are audited. The previous disk-full partial MoE-103 attempt remains archived and is not the accepted checkpoint. External GPU sharing affects wall-clock comparisons; no evidence identifies it as the cause of EIG degradation. No final seed 8831001 was used.\n'
 (out/(m+'_report.md')).write_text(text)
summary='# High-variance methods: journal readiness report\n\n'+intro+'\n## Evidence-based disposition\n\n| Method | Confirmed finding | Still unresolved | Immediate disposition |\n|---|---|---|---|\n| MoE-sBOED | Seed-103 late training breakdown | Router versus shared pathwise amplification; hard/relaxed mismatch | Finish original seeds; separate controlled diagnostic afterward |\n| SwitchMoE-sBOED | Same breakdown; no hard continuation demonstrated | Gate initialization effects and switching contribution | No temporal-persistence claim; diagnose shared failure first |\n| ScoreBED | Validation exports omit bounds; poor final seed102 | Score-gradient quality and training-distribution mismatch | Quarantine original validation curves; final EIG retained |\n| vsOED | Moderate seed spread, learning ongoing | Convergence and actor/critic/posterior contribution | Retain runs, label budget-limited/unverified convergence |\n\n## Conditions for journal claims\n\n1. Complete seeds101–105 without dropping failures. Publish per-seed outcomes, mean/SD and failure incidence; distinguish training uncertainty from trajectory MC error.\n2. Resolve demonstrated implementation defects with regression tests. Preserve original and corrected evidence separately.\n3. For any new MoE correction, predeclare one changed factor, equal total episode accounting and final-budget selection. Confirm across all declared seeds, not only a repaired failing seed.\n4. Demonstrate whether the correction reduces collapse as well as improves EIG. Report expert starvation and hard/relaxed gaps; do not infer informative specialization from utilization alone.\n5. Freeze settings before untouched final evaluation. Failed experiments remain part of development history. Five runs cannot establish that rare failures are impossible, even if a correction has no failures among five.\n6. Instability can be reported scientifically; the present evidence does not support claiming these configurations are robust or uniformly superior. There is no universal journal-acceptable SD cutoff: uncertainty, failure severity, reproducibility and supported claims matter.\n\n## Execution boundary\n\nThe active frozen seed101–105 campaign is unchanged. Corrections and additional training are deferred as requested. This report runs only CPU analysis of existing records. Seed105 and any incomplete methods are pending, not inferred. No final-confirmation results are claimed.\n\n## Files\n\n- moe_report.md\n- switchmoe_report.md\n- scorebed_report.md\n- vsoed_report.md\n- results.csv, training_curves.csv, evidence.json, source_hashes.json\n- instability_diagnostics.png/.pdf\n'
(out/'README.md').write_text(summary)
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
fig,ax=plt.subplots(2,2,figsize=(11,8))
for m in focus:
 rr=[x for x in evidence if x['method']==m];ax[0,0].errorbar([x['seed'] for x in rr],[x['eig'] for x in rr],yerr=[1.96*x['trajectory_mc_se'] for x in rr],marker='o',label=m)
ax[0,0].set(xlabel='Training seed',ylabel='Development EIG (95% MC interval)',xticks=[101,102,103,104,105]);ax[0,0].legend()
for x in evidence:
 if x['method']in ['moe','switchmoe'] and x['seed'] in [102,103]:
  label=f"{x['method']} {x['seed']}";ax[0,1].plot(np.arange(1,6)*.2,[b['router']['p99'] for b in x['gradient_by_fifth']],'-o',label=label)
  ax[1,0].plot(np.arange(1,6)*.2,[b['training_spce']['mean'] for b in x['gradient_by_fifth']],'-o',label=label)
ax[0,1].set(xlabel='Training fraction',ylabel='Pre-clipping router gradient p99',yscale='log');ax[0,1].legend()
ax[1,0].set(xlabel='Training fraction',ylabel='Mean training sPCE within fifth');ax[1,0].axhline(math.log(1025),ls='--',color='grey',label='log(1025)');ax[1,0].legend()
rr=[x for x in evidence if x['method']=='scorebed'];ax[1,1].bar([str(x['seed'])for x in rr],[100*x['boundary_coordinate_fraction']for x in rr]);ax[1,1].set(xlabel='ScoreBED training seed',ylabel='Action coordinates with |x| > 3.8 (%)')
fig.tight_layout();fig.savefig(out/'instability_diagnostics.png',dpi=220);fig.savefig(out/'instability_diagnostics.pdf');print(out);print([(x['method'],x['seed'],x['checkpoint_check'])for x in evidence])
