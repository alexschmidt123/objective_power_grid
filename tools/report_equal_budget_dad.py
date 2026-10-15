"""Audit fixed-checkpoint, paired development EIG and DAD continuation costs.

Reads completed artifacts only; no training, posterior inference or EIG estimation.
"""
import argparse,csv,hashlib,json,math,shlex
from pathlib import Path
import numpy as np

BASE='experiments/10142026/10142026_moe_controlled_diagnosis/controls'
RES='experiments/10152026/10152026_004321_resmoe_switchresmoe_seed101/resmoe'
EPISODES='experiments/10152026/10152026_013344_moe_fairness_controls_seed101/continued_dad_mlp'
CAMPAIGN='experiments/10152026/10152026_equal_compute_dad_control_v2'
BASE_SHA='ec309d224a5c69a9b8ab9574b094f5d4225db506b2c352bbe88e4453865ac300'
RES_SHA='e9ac1228b83ecd459801c7802b1aa24e595b99a494307f068fae01bfa81ffcbb'
BANK_SHA='703671b0fb01ddf74dfe60d51e44914c71647c92e7e5ee51d111a9acfcecfde8'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(values):
 x=np.asarray(values,dtype=float)
 if len(x)<2 or not np.isfinite(x).all():raise ValueError('Need at least two finite scores')
 se=float(x.std(ddof=1)/math.sqrt(len(x)));m=float(x.mean())
 return dict(n=len(x),mean=m,mc_standard_error=se,conditional_95_interval=[m-1.96*se,m+1.96*se])
def identity(rows):return hashlib.sha256(json.dumps([[r['truth'],r['measurement_noise_standard']] for r in rows],separators=(',',':')).encode()).hexdigest()
def particle_cost(episodes,L=1024,T=30):
 return dict(primary_episodes=episodes,contrast_trajectories=episodes*L,truth_inclusive_likelihood_trajectories=episodes*(L+1),particle_stage_forward_likelihood_evaluations=episodes*(L+1)*T)
def paired(a,b):
 if a['case_sha256']!=b['case_sha256'] or a['bank_sha256']!=b['bank_sha256']:raise ValueError('Unpaired cases or contrast bank')
 return stats(np.asarray(a['scores'])-np.asarray(b['scores']))
def load(root,rel,method,name,base_seconds):
 p=root/rel;c=read(p/'run_config.json');t=read(p/'models'/f'{method}_training.json');rows=read(p/'eval'/f'{method}_seed1001.json')
 summaries=read(p/'summary.json');s=next(v for v in summaries if v['method']==method and v['eval_seed']==1001)
 assert c['train_seed']==101 and c['horizon']==30 and c['settings']['contrasts']==1024
 assert len(rows)==s['systems']==2000 and s['contrasts']==1000000
 assert all(np.isfinite(r['actions']).all() and np.max(np.abs(r['actions']))<=4+1e-12 for r in rows)
 score=stats([r['spce'] for r in rows]);assert abs(score['mean']-s['mean_spce'])<1e-11
 bank=sha(p/'eval/contrast_bank_seed1001.npy');assert bank==BANK_SHA
 checkpoint=p/'models'/f'{method}.pt';checkpoint_sha=sha(checkpoint)
 manifest=read(p/'source_manifest.json') if (p/'source_manifest.json').exists() else {}
 assert checkpoint_sha==t.get('checkpoint_sha256',checkpoint_sha)
 n=t['generated_training_trajectories'];is_base=rel==BASE
 provenance=t.get('base_checkpoint',t.get('base_pretraining',{}))
 if not is_base:assert provenance['sha256']==BASE_SHA and provenance['input_dim']==772
 effective=c['settings']|c['benchmark']['research'].get('training_profiles',{}).get('adaptive',{})
 selected={k:effective[k] for k in ['batch_size','learning_rate','hidden','contrasts','spce_adam_beta1','spce_adam_beta2','spce_lr_decay','spce_lr_schedule_every','moe_gradient_clip_mode','eval_systems','eval_contrasts','evaluation_batch_size','evaluation_contrast_chunk']}
 return dict(name=name,run=str(p.relative_to(root)),checkpoint=str(checkpoint.relative_to(root)),checkpoint_sha256=checkpoint_sha,
  source_git_commit=manifest.get('git_commit'),source_manifest=str((p/'source_manifest.json').relative_to(root)),config=str((p/'run_config.json').relative_to(root)),training_metadata=str((p/'models'/f'{method}_training.json').relative_to(root)),
  base_provenance=provenance,settings=selected,hardware=c.get('hardware'),eig=score,scores=[r['spce'] for r in rows],case_sha256=identity(rows),bank_sha256=bank,
  optimizer_updates=t.get('history',[{}])[-1].get('update'),training_cost=particle_cost(n),total_training_cost=particle_cost(n+(0 if is_base else 1000000)),
  additional_training_seconds=t['training_seconds'] if not is_base else 0.,base_training_seconds=base_seconds,
  total_training_seconds=t['training_seconds']+(0 if is_base else base_seconds),
  diagnostic_validation_seconds=t.get('diagnostic_validation_seconds',0.),
  training_budget_unit=t.get('training_budget_unit','primary_episodes'),requested_training_seconds=t.get('time_budget_seconds'),
  time_budget_reached=t.get('time_budget_reached'),checkpoint_selection=t.get('checkpoint_selection'),
  evaluation_seconds=s['evaluation_seconds'],secondary_parameter_inference=s.get('parameter_inference_enabled',True),
  mean_online_decision_seconds=s['mean_decision_seconds'],total_parameters=t['policy_parameters'],trainable_parameters=t['trainable_parameters'])

def main(argv=None):
 ap=argparse.ArgumentParser();ap.add_argument('--root',default='.');ap.add_argument('--output',required=True);args=ap.parse_args(argv)
 root=Path(args.root).resolve();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
 campaign=root/CAMPAIGN
 assert (campaign/'exit_code').read_text().strip()=='0','Timed continuation has not completed'
 assert read(campaign/'eig_only_replay_verification.json')['verified'] is True
 bt=read(root/BASE/'models/sum_single_training.json');base_seconds=bt['training_seconds']
 models=[load(root,BASE,'sum_single','Original DAD-MLP',base_seconds),load(root,EPISODES,'sum_single','DAD-MLP +1m episodes',base_seconds),load(root,CAMPAIGN+'/continued_dad_mlp_walltime','sum_single','DAD-MLP +matched elapsed time',base_seconds),load(root,RES,'resmoe','ResMoE-sBOED',base_seconds)]
 base,episode,timed,res=models
 assert base['checkpoint_sha256']==BASE_SHA and res['checkpoint_sha256']==RES_SHA
 assert timed['time_budget_reached'] is True
 assert timed['additional_training_seconds']>=res['additional_training_seconds']
 common=list(base['settings'])
 for m in models[1:]:assert m['settings']==base['settings'],(m['name'],'optimization/evaluation settings differ')
 comparisons=dict(resmoe_minus_original_dad=paired(res,base),resmoe_minus_equal_episode_dad=paired(res,episode),resmoe_minus_elapsed_matched_dad=paired(res,timed),elapsed_matched_dad_minus_original=paired(timed,base))
 winner=comparisons['resmoe_minus_elapsed_matched_dad'];lo,hi=winner['conditional_95_interval']
 if lo>0:interpretation='ResMoE retains a positive paired development EIG difference after elapsed-time matching for these checkpoints.'
 elif hi<0:interpretation='The elapsed-time-matched continued DAD-MLP exceeds ResMoE on the paired development cases.'
 else:interpretation='The paired development comparison does not resolve an EIG advantage after elapsed-time matching.'
 caveats=[
  'All results use training seed101 and repeatedly used development evaluation seed1001; seed8831001 remains unused. No final/publication result is claimed.',
  'Intervals are mean +/-1.96 paired trajectory standard errors, conditional on fixed checkpoints and one shared contrast bank. They exclude training-seed and contrast-bank variability.',
  'Elapsed training includes simulation, backward/optimizer work and diagnostics. The historical residual run overlapped CUDA posterior inference; the timed control waits for an idle GPU. Equal elapsed time is not exact GPU-FLOP matching.',
  'The additional one-million-episode control matches simulator/likelihood counts; the elapsed-time control may use more episodes. Both views are reported without presenting them as simultaneously matched.',
  'Fresh Adam is used for both additional-training phases because the saved base contains weights, not optimizer moments; LR decay restarts at the same initial value.',
  'ResMoE freezes the pretrained encoder, base head and first design; DAD continuation fine-tunes all original parameters. ResMoE also has more parameters and explicitly adds stage to its193-dimensional residual/router input; the original native772 DAD head keeps its original sum representation without adding a new stage input. A positive result alone cannot isolate sparse routing from residual parametrization, freezing or capacity.',
  'Both phases use seed101 and per-update seed101+(step+1)*104729; the first million additional draws reuse the original training random stream. Counts refer to generated episodes, not guaranteed unique draws across phases.',
  'Frozen dominant-expert routing matches learned ResMoE within paired uncertainty (difference +0.00826178, 95% interval [-0.04730444,+0.06382801]); useful learned sparse routing is not established. Forced routing can induce out-of-distribution histories.',
  'Training sPCE L1024 has ceiling log1025=6.93244789; high-precision held-out EIG uses L1000000. These are different bound resolutions.',
  'Only final predeclared checkpoints are evaluated; no EIG-based checkpoint selection or MSE training is used. Secondary MSE inference is omitted only for the new timed run and replay; EIG replay equivalence was verified on all2000 cases.'
 ]
 commands=read(campaign/'commands.json')
 commands['historical_resmoe_training_and_evaluation']=read(root/'experiments/10152026/10152026_004321_resmoe_switchresmoe_seed101/commands.json')[0]
 commands['historical_equal_episode_dad_training_and_evaluation']=read(root/'experiments/10152026/10152026_013344_moe_fairness_controls_seed101/commands.json')[0]
 report=dict(scope='Equal-additional-budget DAD-MLP control',interpretation=interpretation,models=[{k:v for k,v in m.items() if k!='scores'} for m in models],paired_differences=comparisons,shared_base_validation=dict(seconds=bt.get('diagnostic_validation_seconds'),cases=512,contrasts_per_case=65536,parameter_updates=0,truth_inclusive_likelihood_trajectories=512*65537,particle_stage_forward_likelihood_evaluations=512*65537*30,used_for_checkpoint_selection=False),eig_only_verification=read(campaign/'eig_only_replay_verification.json'),frozen_base_verification=read(campaign/'frozen_base_state_verification.json'),commands=commands,correctness=dict(focused_tests_passed=63,log=str((campaign/'focused_correctness.log').relative_to(root)),eig_only_full_replay_exact=True,frozen_base_tensors_exact=True),implementation_changes=['src/research/moe_architecture_controls.py: optional elapsed stopping only for explicit complete-base DAD continuation; default episode training unchanged','src/research/runner.py: continuation duration/safety cap and optional secondary-inference omission; common EIG unchanged','tools/early_family_eig.py: native complete-policy loading and validation of actual episode/time budget','tools/report_equal_budget_dad.py: strict fixed-artifact pairing, cost accounting and reproducible report','Focused continuation, EIG-only, checkpoint-replay and report tests; README and framework status updates'],caveats=caveats,next_experiment='Replicate this paired ResMoE versus continued DAD-MLP experiment for independent training seeds102 and103. Within each seed, share the same independently pretrained DAD-MLP base, freeze hyperparameters and budget rules beforehand, measure training costs without overlapping GPU jobs, and evaluate the frozen checkpoints on untouched paired cases. Report variation across complete training pipelines as well as trajectory MC uncertainty. Do not tune from the new evaluation outcomes. This is one replication experiment, not an architecture search.')
 (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 with (out/'paired_scores.csv').open('w') as f:
  w=csv.writer(f,lineterminator='\n');w.writerow(['case']+[m['name'] for m in models]+['ResMoE minus elapsed-matched DAD'])
  for i in range(2000):w.writerow([i]+[m['scores'][i] for m in models]+[res['scores'][i]-timed['scores'][i]])
 lines=['# Equal-budget DAD-MLP control','',interpretation,'','Development results only: training seed **101**, evaluation seed **1001**, 2,000 paired trajectories, L=1,000,000. Untouched seed **8831001** was not used.','', '| Model | EIG | 95% trajectory MC interval | Additional episodes | Additional training (s) | Total optimization (s) |','|---|---:|---|---:|---:|---:|']
 for m in models:
  a,b=m['eig']['conditional_95_interval'];n=0 if m is base else m['training_cost']['primary_episodes']
  lines.append(f"| {m['name']} | {m['eig']['mean']:.8f} | [{a:.8f}, {b:.8f}] | {n:,} | {m['additional_training_seconds']:.3f} | {m['total_training_seconds']:.3f} |")
 lines+=['','ResMoE minus elapsed-matched continued DAD-MLP: **%+.8f nats**, paired SE %.8f, conditional 95%% interval **[%+.8f, %+.8f]**.'%(winner['mean'],winner['mc_standard_error'],lo,hi),'','## Exact checkpoint provenance','',f'All paths below are relative to `{root}`.','']
 for m in models:lines += [f"- **{m['name']}**: `{m['checkpoint']}`; SHA256 `{m['checkpoint_sha256']}`. Configuration: `{m['config']}`. Training metadata: `{m['training_metadata']}`."]
 lines+=['','The exact complete DAD-MLP base uses its original permutation-invariant 3→256 ReLU→16 sum encoder, fixed zero padding to a native 772-dimensional head input, 64/64 tanh policy head, separately learned first design and original bounded action mapping. The residual/router input remains193; it is not silently substituted for the base input.','', '## Configuration and computational budget','', 'Common settings: batch64, full-horizon sPCE L1024, T30, Adam lr5e-5, betas(0.8,0.998), exponential decay0.98 every1000 updates, component clipping5. Standard-normal four-coordinate prior, two sources, log-signal Gaussian SD0.5, design box[-4,4]^2. Evaluator batch16, contrast chunk16384. No estimator, simulator or action transformation was changed.','', '| Model | Additional contrast trajectories | Additional truth-inclusive likelihood trajectories | Additional particle-stage likelihood evaluations | Total primary episodes including base |','|---|---:|---:|---:|---:|']
 for m in models[1:]:
  c=m['training_cost'];lines.append(f"| {m['name']} | {c['contrast_trajectories']:,} | {c['truth_inclusive_likelihood_trajectories']:,} | {c['particle_stage_forward_likelihood_evaluations']:,} | {m['total_training_cost']['primary_episodes']:,} |")
 lines += ['',f"Base pretraining: 1,000,000 episodes, 15,625 updates, {base_seconds:.6f}s optimization, plus {bt.get('diagnostic_validation_seconds',0):.6f}s diagnostic validation (512 trajectories, L65536, 33,554,944 truth-inclusive likelihood trajectories / 1,006,648,320 particle-stage evaluations, no updates, not used for checkpoint selection). Add this validation time separately to the optimization totals above. Include the base cost once per end-to-end model; it was shared in the actual campaign.",'','The elapsed control stops at the first completed update reaching1736.8317570788786s. Its5,000,000-episode cap is only a safety limit, not its actual episode budget. The predeclared stopping rule never examines held-out EIG. Reported particle-stage counts count complete truth-inclusive trajectory likelihoods at every stage; they do not equate backward-pass FLOPs or different neural-policy costs.','', '| Setting | Continued DAD-MLP | ResMoE-sBOED |','|---|---|---|','| Pretrained base | Exact original checkpoint | Same exact original checkpoint |','| Trainable parameters | 58,900 original parameters | 67,406 new residual/router parameters |','| Total parameters | 58,900 | 126,306 |','| Encoder/base/first design | Fine-tuned | Frozen |','| New experts/router | None | Four independent64/64 tanh residuals, DSelect-1 |','| Residual scale/initialization | Not applicable | alpha1; small nonidentical initialization std0.001 |','| Router schedule/regularization | Not applicable | gamma1 through30%, cosine to0.001 at95%; existing1e-4 penalties |','| Optimizer state | Fresh Adam from base weights | Fresh Adam for new components |','| Learning-rate schedule | Same initial LR and decay rule; more steps when time allows | Same rule over1m additional episodes |','','## Commands','', 'Commands were executed from the repository root. Each root command includes training (unless explicitly frozen) and the common held-out evaluation. Use a fresh output path when reproducing; existing outputs remain preserved.','']
 for name,cmd in commands.items():lines += [name+':','', '```bash',shlex.join(cmd),'```','']
 lines+=['Historical source-check manifests intentionally reject changed source; reproduce those commands from their frozen source snapshot or create a fresh manifest for the reviewed current code. The original equal-episode and ResMoE commands are also preserved in their campaign `commands.json` files. A frozen ResMoE evaluation through the new EIG-only option reproduces the old common evaluator within score tolerance1e-9 and action/observation tolerance1e-10 for all2,000 cases. See `eig_only_replay_verification.json` in the campaign.','','## Interpretation limits','']+['- '+x for x in caveats]
 lines+=['','## Implementation and correctness','', 'The optional elapsed-budget stop is confined to `train_control` in `src/research/moe_architecture_controls.py`; ordinary episode-budget training remains available. `src/research/runner.py` exposes the duration/safety cap and optional omission of secondary inference. `tools/early_family_eig.py` loads complete native controls and validates actual episode/time budgets. `tools/report_equal_budget_dad.py` reads fixed artifacts, validates pairing and counts, and produces this report. Residual architecture, simulator, likelihood, action mapping, sPCE implementation and existing checkpoints were not modified.','', '**63 focused tests passed**, including forced-routing/base equivalences, bounded rollouts, gradient reachability, exact ordinary-versus-timed training for matched updates, checkpoint round trips, EIG-only equality and paired-report accounting. The full ResMoE EIG-only replay produced zero difference in all2000 actions, observations and scores; all11 pretrained-base tensors in ResMoE are exactly identical to the original base. Test log: `experiments/10152026/10152026_equal_compute_dad_control_v2/focused_correctness.log`.','']
 lines+=['','## One recommended next experiment','',report['next_experiment'],'','The current comparison tests whether extra training explains the observed development gain. It cannot by itself demonstrate that sparse expert specialization causes that gain. All original checkpoints and results are preserved.','']
 (out/'report.md').write_text('\n'.join(lines));print(json.dumps(dict(interpretation=interpretation,paired_difference=winner,models=[dict(name=m['name'],eig=m['eig']['mean'],episodes=m['training_cost']['primary_episodes'],seconds=m['additional_training_seconds']) for m in models]),indent=2))
if __name__=='__main__':main()
