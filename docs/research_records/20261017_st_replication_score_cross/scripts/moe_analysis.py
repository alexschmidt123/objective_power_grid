import sys,json,time,hashlib,csv
from pathlib import Path
R=Path('/home/grads/g/g.lin/Documents/objective_power_grid');C=R/'experiments/10172026/10172026_st_replication_score_reverse_cross';P=R/'experiments/10172026/10172026_moe_st_routing_dev_103_105';S=C/'moe_source';O=C/'reports';sys.path.insert(0,str(S))
import torch,numpy as np,yaml
from scipy.stats import t
from src.research.benchmarks import make_benchmark
from src.research.moe_family import load_policy
from src.research.journal_validation import CommonValidation
from src.research.engine import rollout
from src.research.location_evaluation import contrast_scores
torch.set_num_threads(4)
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def paired(a,b):
 d=np.array(a)-np.array(b);m=float(d.mean());s=float(d.std(ddof=1)/len(d)**.5);return dict(difference=m,mc_se=s,ci95=[m-1.96*s,m+1.96*s])
raw=yaml.safe_load((S/'configs/moe_st_development.yaml').read_text());val=CommonValidation(raw,30,1001,2000,1000000,16,16384)
records=[];effects=[];diagnostics={};all_scores={};old_scores={}
for seed in range(101,106):
 run=P/f'source/experiments/10172026/10172026_moe_st_dev/seed{seed}' if seed in [103,105] else C/f'runs/moe_seed{seed}'
 registry=read(R/'experiments/10152026/10152026_journal_five_seed_fixed_budget/registry.json')
 origin=next(r for r in registry if r['method']=='moe' and r['seed']==seed and r['benchmark']=='location')
 old=Path(origin['run']);native=origin['native_method']
 tr=read(run/'models/moe_training.json');assert tr['generated_training_trajectories']==1000000 and tr['selected_checkpoint']['episodes']==1000000
 newcfg=read(run/'models/moe_effective_settings.json')
 if (old/'models/moe_effective_settings.json').exists():
  oldcfg=read(old/'models/moe_effective_settings.json')
  differences={k:{'old':oldcfg.get(k),'st':newcfg.get(k)} for k in newcfg.keys()|oldcfg.keys() if newcfg.get(k)!=oldcfg.get(k)}
  assert set(differences)<={'moe_training_routing_mode','diagnostic_late_logging','diagnostic_start_episodes','diagnostic_checkpoint_episodes'},differences
 else:
  assert seed==101 and origin['audit']['compatible']
  oldtr=read(old/f'models/{native}_training.json')
  assert oldtr['generated_training_trajectories']==1000000 and oldtr['routing_warmup_fraction']==.3 and oldtr['routing_finish_fraction']==.95 and oldtr['gradient_clip_mode']=='component'
  assert hashlib.sha256((old/f'models/{native}.pt').read_bytes()).hexdigest()==origin['audit']['checkpoint_sha256']
  differences=dict(historical_alias=native,compatibility_source='Existing frozen journal seed101 benchmark/budget/final-selection/native-settings/action-replay audit',prior_audit=origin['audit'])
 rows=read(run/'eval/moe_seed1001.json');previous=read(old/f'eval/{native}_seed1001.json');assert len(rows)==len(previous)==2000
 for rr in [rows,previous]:
  assert np.array_equal(np.array([r['truth'] for r in rr]),val.truth.numpy()) and np.array_equal(np.array([r['measurement_noise_standard'] for r in rr]),val.noise.numpy())
 assert np.array_equal(np.load(run/'eval/contrast_bank_seed1001.npy'),val.bank.numpy()) and np.array_equal(np.load(old/'eval/contrast_bank_seed1001.npy'),val.bank.numpy())
 hard=np.array([r['spce'] for r in rows]);oldhard=np.array([r['spce'] for r in previous]);all_scores[seed]=hard;old_scores[seed]=oldhard
 np.save(O/f'moe_st_{seed}_hard.npy',hard);np.save(O/f'moe_original_{seed}_hard.npy',oldhard)
 policy=load_policy(run/'models/moe.pt',make_benchmark(raw));count=val.systems;val.systems=16;check,_=val(policy);val.systems=count;assert np.allclose(check,hard[:16],rtol=0,atol=1e-10)
 softpath=O/f'moe_st_{seed}_relaxed.npy'
 if seed in [103,105]:soft=np.load(P/f'reports/straight_through_{seed}_relaxed_final1001_scores.npy');np.save(softpath,soft)
 elif softpath.exists():soft=np.load(softpath)
 else:
  policy=load_policy(run/'models/moe.pt',make_benchmark(raw),relaxed=True);vals=[];starttime=time.monotonic()
  with torch.no_grad():
   for start in range(0,2000,16):
    end=min(start+16,2000);b=make_benchmark(raw);rr=rollout(b,policy,30,end-start,val.seed+start*97451,0,truth=val.truth[start:end],noise=val.noise[start:end]);scores,_=contrast_scores(b,rr,val.bank,16384);vals.extend(scores[:,-1].tolist())
  soft=np.array(vals);np.save(softpath,soft);write(O/f'moe_st_{seed}_relaxed_cost.json',dict(seconds=time.monotonic()-starttime,likelihood_particle_stages=2000*1000001*30))
 selected=np.array([[r['selected_expert'] for r in c['routing']] for c in rows]);assert selected.shape==(2000,29)
 usage=[float((selected==j).mean()) for j in range(4)];by_stage=[[float((selected[:,stage]==j).mean()) for j in range(4)] for stage in range(29)]
 trace=run/'models/late_diagnostics/training_trace.jsonl';windows={}
 for line in trace.open():
  row=json.loads(line);ep=row['primary_episodes'];names=['whole_run']+(['final20_percent'] if ep>800000 else [])
  for name in names:
   w=windows.setdefault(name,dict(updates=0,usage_sum=np.zeros(4),components={},saturation_sum=0.));w['updates']+=1;w['usage_sum']+=row['training_expert_usage'];w['saturation_sum']+=row['saturated_bit_fraction']
   for k,norm in row['component_gradient_before'].items():
    d=w['components'].setdefault(k,dict(exposure_count=0,zero_count=0,norm_sum=0.,norm_max=0.,step_sum=0.));d['exposure_count']+=int(norm>1e-8);d['zero_count']+=int(norm<=1e-12);d['norm_sum']+=norm;d['norm_max']=max(d['norm_max'],norm);d['step_sum']+=row['actual_parameter_updates'][k]['update_l2']
 assert windows['whole_run']['updates']==15625
 for w in windows.values():
  n=w['updates'];w['expert_usage']=(w.pop('usage_sum')/n).tolist();w['mean_selector_saturation']=w.pop('saturation_sum')/n
  for d in w['components'].values():
   d.update(gradient_exposure=d.pop('exposure_count')/n,zero_gradient_fraction=d.pop('zero_count')/n,mean_raw_gradient_norm=d.pop('norm_sum')/n,mean_parameter_update=d.pop('step_sum')/n)
 diagnostics[str(seed)]=dict(windows=windows,final_usage=usage,stage_2_to_30_usage=by_stage,settings_differences=differences)
 effect=paired(hard,oldhard);effects.append(dict(seed=seed,**effect))
 records.append(dict(seed=seed,hard_eig=float(hard.mean()),mc_se=float(hard.std(ddof=1)/2000**.5),relaxed_eig=float(soft.mean()),soft_minus_hard=paired(soft,hard),maximum_case_soft_hard_difference=float(abs(soft-hard).max()),original_eig=float(oldhard.mean()),paired_gain=effect,E1_E2_E3_E4=usage,training_seconds=tr['training_seconds'],primary_episodes=1000000,checkpoint=str(run/'models/moe.pt'),checkpoint_sha256=hashlib.sha256((run/'models/moe.pt').read_bytes()).hexdigest()))
 write(O/'moe_partial_results.json',records);print('ANALYZED_MOE_SEED',seed,records[-1]['hard_eig'],flush=True)
def seeds_summary(x):
 x=np.array(x);mean=float(x.mean());sd=float(x.std(ddof=1));half=float(t.ppf(.975,4)*sd/5**.5);return dict(training_seeds=5,mean=mean,seed_sd=sd,minimum=float(x.min()),maximum=float(x.max()),seed_mean_t_ci95=[mean-half,mean+half])
stats=dict(straight_through=seeds_summary([r['hard_eig'] for r in records]),original=seeds_summary([r['original_eig'] for r in records]),paired_seed_differences=seeds_summary([r['paired_gain']['difference'] for r in records]),positive_differences=sum(r['paired_gain']['difference']>0 for r in records),one_expert_collapsed_seeds=sum(max(r['E1_E2_E3_E4'])>=.999 for r in records),inference_seed=1001,mc_uncertainty_is_not_seed_uncertainty=True)
write(O/'moe_results.json',dict(records=records,statistics=stats));write(O/'moe_diagnostics.json',diagnostics)
with (O/'moe_results.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['seed','original_eig','st_hard_eig','st_relaxed_eig','gain','paired_mc_ci_low','paired_mc_ci_high','E1','E2','E3','E4','training_seconds'])
 for r in records:w.writerow([r['seed'],r['original_eig'],r['hard_eig'],r['relaxed_eig'],r['paired_gain']['difference'],*r['paired_gain']['ci95'],*r['E1_E2_E3_E4'],r['training_seconds']])
lines=['# Straight-through MoE five-seed replication','','Seeds103/105 are preserved prior runs;101/102/104 use identical frozen scientific source/configuration. Exactly1M primary episodes each; final-budget checkpoint.2000 paired trajectories/L1M/development1001. No original result replacement.','','| Seed | Original EIG | ST hard EIG | ST relaxed EIG | Paired gain [95% MC interval] | E1/E2/E3/E4 usage |','|---|---:|---:|---:|---|---|']
for r in records:lines.append(f"| {r['seed']} | {r['original_eig']:.5f} | {r['hard_eig']:.5f} | {r['relaxed_eig']:.5f} | {r['paired_gain']['difference']:+.5f} {r['paired_gain']['ci95']} | {r['E1_E2_E3_E4']} |")
lines+=['','Five-seed summaries (t intervals use independent training-seed means, not10000 trajectories as replicates):','','```json',json.dumps(stats,indent=2),'```','','Stage frequencies exclude the separately learned first experiment. Whole-run/final20% exposure thresholds are norm>1e-8; near-zero<=1e-12. Full JSON reports router and encoder/expert gradients, actual parameter updates and selector saturation. Soft inference is a counterfactual policy; ST training forward is hard.','']
if stats['one_expert_collapsed_seeds']:
 lines+=['Observed expert collapse prevents interpreting an EIG improvement as evidence for useful four-expert specialization. One justified next controlled intervention is a temporary relaxed-forward expert-training warmup before returning to unchanged ST routing, targeting the documented unselected-expert starvation. Keep gamma/architecture/objective/optimizer/budget fixed and freeze one warmup duration before testing all original seeds; do not adopt or launch from this report.']
else:lines+=['No ubiquitous single-expert collapse at final evaluation. Assess gradient exposure and paired uncertainty before proposing a new correction; no architecture change follows automatically.']
lines+=['','Five original development seeds provide limited evidence and have already informed method development; they are not untouched confirmation. Reserved final evaluation seed8831001 remains unused.']
(O/'moe_report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(stats,indent=2),flush=True)
