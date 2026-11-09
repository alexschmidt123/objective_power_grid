"""Generate requested reports from the completed, read-only LabPC audit."""
import pathlib,json,hashlib,re
R=pathlib.Path.home()/'Documents/objective_power_grid_temp/scratch_runs';A=R/'moe_hec_runtime_efficiency_audit_20261109';S=json.loads((A/'AUDIT_SUMMARY.json').read_text());O=pathlib.Path.home()/'Documents/objective_power_grid_temp/development/hec_git_sync/development/moe_sboed/audits/hec_runtime_20261109';O.mkdir(parents=True,exist_ok=True)
C,G,H=[S[n] for n in ['collaborative','sg','hec']]
assert G['config']['parent_actor_sha256']==C['arms']['candidate']['result']['actor_sha256'];assert H['config']['parent_actor_sha256']==G['arms']['candidate']['result']['actor_sha256'];assert G['config']['parent_runtime_sha256']==C['arms']['candidate']['result']['runtime_sha256'];assert H['config']['parent_runtime_sha256']==G['arms']['candidate']['result']['runtime_sha256']
assert C['arms']['control']['start']['effective_actor_digest']==C['arms']['candidate']['start']['effective_actor_digest'];assert G['arms']['control']['start']['initial_morphed_digest']==G['arms']['candidate']['start']['initial_morphed_digest'];assert H['arms']['control']['start']['initial_actor_digest']==H['arms']['candidate']['start']['initial_actor_digest']
def duration(x):
 n=round(x);return f'{n//3600}h {(n%3600)//60:02d}m {n%60:02d}s'
def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)+'\n'
def ci(v):return f"{v['mean']:+.5f} [{v['CI95'][0]:+.5f}, {v['CI95'][1]:+.5f}]"
intro='''Audit date: 2026-11-09. Authoritative inputs are the saved LabPC configurations, JSONL training logs, per-arm RESULT/STARTED receipts, actor checkpoints, full-runtime hashes, frozen-policy score vectors and evaluation receipts. This audit performed **zero training updates, zero neural inference calls and zero new simulator calls**, and used CPU only. Actor tensor dictionaries were read on CPU with restricted PyTorch loading; full runtimes were hashed without deserializing training objects. All 141 read input files, including six terminal actors and six full runtimes, retained their SHA256 values after the audit.

Source roots, relative to LabPC `~/Documents/objective_power_grid_temp/scratch_runs/`:

- C: `moe_collaborative_soft_queue_20261108/`.
- S: `moe_specialization_guided_eig_20261108/`.
- H: `moe_hec_sboed_20261108/`.

Audit evidence is isolated in `scratch_runs/moe_hec_runtime_efficiency_audit_20261109/{AUDIT_SUMMARY.json,RAW_RECEIPTS_AND_LOGS.json,INPUT_PRESERVATION.json}`. Reproduce with `python tools/audit_hec_runtime.py --output ~/Documents/objective_power_grid_temp/scratch_runs/NEW_AUDIT_DIRECTORY`; the collector requires a new directory and refuses to overwrite an existing audit. No existing experiment, checkpoint, evaluation rule or model was changed. The current machine-access workflow supersedes old access restrictions, without editing historical protocol files.
'''
runtime='# Runtime and EIG efficiency analysis\n\n'+intro+'''
## Main finding

The quoted 3h57m, 53m and 1h58m are **two-arm experiment durations**, not the training time of one candidate. HEC used five times fewer native updates per arm than fresh Collaborative, while doing about 2.5 times as much wall-time work per update. That explains its shorter total versus Collaborative. Relative to SG, HEC used the same 2,000-update continuation budget but substantially more model-query and gradient work; it was slower, not a demonstrated implementation speedup.

## Measured time decomposition

All three ran sequential arms on the same recorded GPU type, NVIDIA RTX 4090. The scientific-driver timer begins after acquisition of the GPU lease. The completion receipt is written before final report generation and preservation sealing; queue wait, implementation time, preflight checks and final sealing are outside these headline durations. Arm times include training/checkpoint serialization within their timer; terminal hash calculations and some loading are in the remaining driver time. These are wall-time allocations, not measured SM-busy GPU hours or energy.

'''
rows=[]
for name,s in [('Collaborative',C),('SG',G),('HEC',H)]:
 rows.append([name,duration(s['arms']['control']['result']['seconds']),duration(s['arms']['candidate']['result']['seconds']),f"{s['prep_seconds']:.3f}",f"{s['evaluation']['seconds']:.3f}",f"{s['unallocated_driver_seconds']:.3f}",duration(s['completion']['seconds'])])
runtime+=table(['Experiment','Control','Candidate','Shared preparation (s)','Evaluation (s)','Other driver (s)','Whole pair'],rows)
runtime+='\nSources: each root’s `arms/*/RESULT.json`, `preparation/RESULT.json` where present, `evaluation/COMPLETION.json`, `evidence/EXPERIMENT_COMPLETION.json` and `source/run_experiment.py`. HEC preparation’s 5.934s is physically performed once, copied to both arms and charged equally to them in conditional efficiency calculations. It is not 300 native updates or 300 new simulator batches.\n\n## Training budgets and actual observation simulation\n\n'
runtime+=table(['Per arm','Collaborative','SG','HEC'],[
 ['Initialization','Fresh actor/critic/posteriors/Adam/replay','Trained Collaborative candidate','Trained SG candidate'],
 ['Native update counter','0 → 10,001','10,001 → 12,001','12,001 → 14,001'],
 ['New actor updates','10,001','2,000','2,000'],
 ['Critic / posterior-net optimizer steps','50,005 / 1,500,150','10,000 / 300,000','10,000 / 300,000'],
 ['New primary episodes incl. initial replay','10,021,000','2,000,000','2,000,000'],
 ['New primary observation simulations (T=30)','300,630,000','60,000,000','60,000,000'],
 ['Fresh initial replay episodes','20,000','0; inherited 1M buffer','0; inherited 1M buffer'],
 ['Replay minibatch / capacity','10,000 / 1M','10,000 / 1M','10,000 / 1M'],
 ['Actor design rows per native update','10,000 × 30','10,000 × 30','10,000 × 30'],
 ['Discount / final specialist loss','0.9 / ongoing auxiliary','1.0 / none','1.0 / none in B/C'],
 ['Expert-learning schedule','Joint plus task gradients throughout','Joint EIG for all 2,000','300 shared supervised steps; freeze experts in B=200; joint C=1,800'],
 ['Training prior contrasts','0; variational AC','0; variational AC','0; variational AC']])
runtime+='''
The native profile uses five critic updates and five updates of each of 30 posterior nets per actor iteration. Its fixed replay batch is 10,000, not the EIG-evaluation batch of 16. HEC’s 4,096-row gradient-checkpoint chunks partition the 300,000-row actor calculation; they do **not** reduce its statistical minibatch. The fresh 20,000-episode replay accounts for 600,000 observations per Collaborative arm.

HEC additionally generated **27,648 synthetic one-step predictive observations** for shared specialist preparation from the saved approximate belief, not true-latent labels read from replay. These are separate from 60M native observations per arm and were physically generated once for the pair. Preparation had 4,096 training and 512 validation prefixes, 300 Adam steps × 256 samples = 76,800 reused training-prefix presentations, and no new real episodes. Direct teacher-provider accounting adds 26,682 SG-policy rows, 26,682 frozen-critic rows and 966 terminal-posterior rows to the policy-forward counters (`H/evidence/PREPARATION_DIRECT_PROVIDER_COUNTS.json`). Neither a quadrature signal evaluation nor a replay presentation is a fresh experimental episode.

The initial native learning rates were approximately 0.001, 0.000367824 and 0.000301146 respectively; exploration was 0.5, 0.183912 and 0.150573. Both continue their saved .9999 decays, and all use the same native Polyak rate 0.1. Collaborative had ongoing expert auxiliary gradients and gamma0.9. SG explicitly changed both arms to gamma1 and a normalized prior-mixture density safeguard; HEC inherits those choices. Therefore the three campaigns are not identical training-objective experiments.

Sources: locked configs, per-arm STARTED/RESULT receipts, JSONL optimizer counts, `C/source/train_soft_pair.py`, `S/source/train_sg_pair.py`, `H/source/{train_pair,migration,prepare_specialists}.py`, and the imported direct-head `frozen_protocol.json` native profile. Checkpoint metadata confirms terminal counters 10001,12001,14001 and their distinct candidate/control method flags.

## Throughput, phase timings and GPU memory

'''
rows=[]
for name,s in [('Collaborative',C),('SG',G),('HEC',H)]:
 for arm,r in s['arms'].items():rows.append([name,arm,f"{r['mean_seconds_per_update']:.6f}",f"{r['median_logged_block_seconds_per_update']:.6f}",f"{r['result']['gpu_peak_bytes']/2**30:.3f}"])
runtime+=table(['Experiment','Arm','Wall s/update','Median logged interval s/update','Peak torch allocated GiB'],rows)
runtime+='\nThe mean divides the arm RESULT wall time by completed native updates; the interval median uses successive saved JSONL records (usually 50 updates), not unsaved per-step kernel measurements. The final logger-to-result gap is only about 0.5–1.0s per arm.\n\n'
rows=[]
for arm,r in H['arms'].items():rows.append([arm,f"{r['phase_B']['seconds']:.3f}",f"{r['phase_B']['seconds_per_update']:.6f}",f"{r['phase_C']['seconds']:.3f}",f"{r['phase_C']['seconds_per_update']:.6f}"])
runtime+=table(['HEC arm','B=200 time (s)','B s/update','C=1800 logged time (s)','C s/update'],rows)
runtime+='''
Phase-C numbers end at the terminal log and exclude the sub-second terminal arm-serialization gap. Logs show zero expert/history gradient allocation during B by design, then nonzero gradients for all four experts/history during C; candidate commander/coordinator gradients are nonzero, while the control’s unused commander/coordinator gradients remain zero.

HEC candidate is **13.04% slower than its matched control** (3521.274 vs3115.058s). Its forward primitive counters match, but the dynamic commander differentiates through commanded likelihood/sensitivity/critic-input queries and uses feature-aware coordination. These extra gradient paths are verified by code and logged gradients. They plausibly explain the added cost; no profiler isolates their exact share from synchronization, kernels or contention.

HEC candidate is **2.512× SG candidate’s s/update**. Pair training is 2.378× SG and whole driver time is 2.211× SG. SG candidate’s 0.701s/update is close to Collaborative’s 0.706s/update. Therefore SG’s much shorter completion is overwhelmingly a reduced continuation budget, not a measured faster kernel. Removing continuous auxiliary work is a real code difference, but Collaborative’s recorded auxiliary host section totals only 293.259s (4.15% of its candidate arm); it cannot explain the roughly 5× reduction in update count. This host timing is not a synchronized GPU-kernel attribution.

The logs record peak `torch.cuda.max_memory_allocated`, not total device reservation, context memory or average GPU utilization. No complete utilization/power trace or kernel profiler was saved; current idle GPU readings cannot reconstruct historical GPU occupancy. HEC’s lower 2.505GiB allocation is consistent with source-level gradient checkpointing and its different live/frozen graph; **no on/off ablation proves the exact memory saving or any throughput gain**. It trades recomputation for retained activation memory. There is no evidence HEC was faster because of higher GPU utilization, faster hardware, a smaller statistical batch or an undocumented budget change.

## HEC’s extra work and deployed model footprint

Each HEC arm’s actual instrumentation records 706,560,000,000 exact signal evaluations, 117,760,000,000 analytic parameter Jacobians, 1,218,000,000 frozen-belief rows, 1,240,000,000 frozen-SG provisional rows and 5,520,000,000 frozen-critic rows. Both arms have identical forward counts; these include actor recomputation. They are model-derived query work, not billions of independently observed experiments. Native actor/critic/posterior work is additional and is accounted by the optimizer/simulator budget above.

`H/source/hec_policy.py` uses 64 deterministic GMM sigma points, exact likelihood-derived features, analytic Fisher queries, three frozen-critic probes, stage-wise belief precomputation and checkpointed live computation. The analytic Fisher avoids unnecessary reverse-mode Jacobian calls, but this audit did not benchmark another implementation. It cannot claim any quantitative speedup from those engineering choices.

Checkpoint tensor inspection gives:

'''
runtime+=table(['Checkpoint state','Collaborative candidate','SG candidate','HEC candidate/control'],[
 ['All stored tensor elements','166,308','330,280','15,368,770'],
 ['Allocated new/live parameter elements incl. readouts','166,308','199,592','139,297'],
 ['Frozen providers / anchor state','0','130,688','15,229,473'],
 ['Policy-function active live parameters (phase C)','164,358','197,642','136,506 candidate / 101,542 control']])
runtime+='''
HEC buffers include the SG policy (330,280 elements), belief providers (14,736,632 including small constants) and critic (162,561). Stored elements are not all trainable weights. HEC’s 139,297 allocated parameters include 2,275 diagnostic readout parameters and unused alternative-routing modules. Its candidate has 34,964 more policy-function active parameters than its control (**34.43%**); identical tensor allocation is not identical effective capacity. The EIG receipt’s `network_parameters=0` for frozen SG reflects conversion to buffers, not a zero-cost/zero-size policy.

On the HEC evaluation’s same saved histories, inference timings are HEC 2.5655ms/design, control 2.5211ms and frozen SG 0.4521ms: HEC is approximately **5.675×** slower than SG in this single-case benchmark. Earlier recorded Collaborative and SG candidate timings were 0.2256ms and 0.4434ms/design, with different repetition schedules; compare the same-run HEC/SG timing first. HEC is not an established deployment-efficiency improvement.

## Evaluation workloads

All use 2,000 new full trajectories, T30, 1M negative prior contrasts, one positive included once (1,000,001 denominator terms), evaluation batch16/chunk16384 and the frozen scorer. Distinct banks are never paired across campaigns.

'''
rows=[]
for name,s in [('Collaborative',C),('SG',G),('HEC',H)]:
 count=len(s['evaluation']['main_policies_completed']);rows.append([name,count,f'{count*60000:,}',f"{s['rollout_observations_in_receipts']:,}",f"{s['evaluation']['seconds']:.3f}"])
runtime+=table(['Experiment','Main frozen policies','Main observation simulations','All full-rollout receipts: observations','All evaluation wall s'],rows)
runtime+='''
Collaborative evaluated four main policies, a 512-case diagnostic rollout and six 256-case intervention rollouts. SG evaluated ten main policies, the diagnostic base and sixteen 256-case interventions; it additionally generated 132,288 suffix observations for its separate conditional native-return panel. HEC evaluated eleven main policies, the diagnostic base and ten 256-case interventions; its competency teacher adds 9,216 synthetic one-step observations (1,536 prefixes ×3 queries ×2 replicas). These extra native-return/teacher observations are not contrast-scored full trajectories. Thus observation totals represented here are 301,440 (C), 870,528 (S including conditional panel) and 761,376 (H including teacher), under these explicit definitions.

The summed full-rollout receipts take 151.859, 396.260 and 426.187s. Remaining evaluation time also covers bank construction/loading, model construction, probes, teacher/conditional work and latency timing; no finer exact timing is available. HEC vsSG adds one main policy but removes six full-rollout interventions. Its full contrast workload rises only from 738.24B to752.16B negative observation-likelihood terms (1.89%); evaluation wall rises6.25%. HEC’s 25.6s extra evaluation explains little of the 3,878.7s whole-driver gap versus SG. It does explain why a comparison using only update counts misses some cost.

## Inherited costs and meaningful efficiency comparisons

The saved parent hashes confirm the lineage **fresh Collaborative candidate → SG candidate → HEC**. HEC inherits the trained SG provisional policy, live history initialization and compatible Adam moments, trained native critic/posteriors, schedules and replay. The four new heterogeneous experts and incompatible heads do not inherit old expert weights/Adam; they receive shared fresh preparation, then 1,800 joint EIG updates. HEC is a warm-started system, not a fresh four-expert training result.

'''
rows=[]
for i,name in enumerate(['Collaborative candidate','SG lineage','HEC lineage']):rows.append([name,f"{S['lineage']['candidate_native_counter'][i]:,}",f"{S['lineage']['candidate_real_episodes'][i]:,}",duration(S['lineage']['candidate_only_training_seconds'][i])])
runtime+=table(['Candidate-producing lineage','Accumulated native counter','Accumulated new primary episodes','Training wall (excluding controls/evaluation)'],rows)
runtime+='''
HEC’s candidate lineage is 14,021,000 real episodes /420,630,000 observations, plus its teacher preparation, and about3h20m candidate-producing training. This does not mean every HEC parameter received14,001 updates. Funding all three two-arm driver campaigns used6h48m34s, excluding preflights/queue/seals. Counting only the latest1h58m as the full development cost would omit inherited training. Parent costs can be treated as sunk only for a clearly labeled incremental continuation decision.

For HEC **on its own common bank**, conditional gain over the same frozen SG parent can be divided by each observed training-plus-preparation time. These are descriptive endpoint amortization ratios, not a learning-curve derivative, runtime-confidence bound or matched-time experiment. Preparation is charged in full to each arm for standalone use; physically the pair prepared it once. Evaluation time and inherited costs are excluded consistently. Control-vsSG intervals below are audit-derived, unadjusted descriptive paired intervals, not a new predeclared primary family.

'''
rows=[]
for arm in ['control','candidate']:
 e=S['conditional_efficiency']['hec'][arm];rows.append([arm,f"{H['eig']['methods'][arm]['mean']:.5f}",ci(e['gain_vs_same_bank_frozen_parent']),f"{e['incremental_training_plus_prep_GPU_wall_hours']:.6f}",f"{e['conditional_gain_nats_per_GPU_wall_hour']:.5f} [{e['rate_CI95'][0]:.5f}, {e['rate_CI95'][1]:.5f}]"])
runtime+=table(['HEC arm','Estimate (nats)','Gain vs frozen SG [95% CI]','Incremental GPU-wall h','Conditional gain nats/GPU-wall h [95% CI]'],rows)
runtime+='''
The matched-update result is candidate−control **+0.07211 [0.01889,0.12532] nats**, p=.00794, with the same2M new episodes but13% different wall times. This supports an improvement per matched new-episode/update budget for the complete package in one seed. It does not show a superior equal-time learning curve; giving the control the candidate’s extra406.2s was not tested. Dividing that gain by the time difference would not establish the causal marginal value of computation.

The analogous SG-on-its-own-bank rates are0.15452 (control) and0.34646 (candidate) gain nats/GPU-wall h relative to its frozen Collaborative parent; they are conditioned on another parent/bank and common gamma/density-floor corrections. **Do not rank HEC against SG from these different-stage rates.** We do not report absolute final EIG/hour as an across-campaign winner, because inherited policy quality, objectives, extra providers, training opportunities and estimator banks differ. No compute-matched superiority over native vsOED or fresh-training-efficiency claim is justified.

## Which components plausibly explain HEC’s EIG gain?

On the HEC bank: frozen SG12.77406±0.03566; fixed-command heterogeneous continuation12.88750±0.03378; HEC12.95961±0.03374 (nats±MCSE). The frozen-SG gap is+0.18556 [0.12445,0.24666], Holm4 p=3.06e−9. The fixed-command control already gains+0.11345 over frozen SG. Arithmetically that is61.1% of the HEC point gap; the remaining38.9% is the candidate-control difference. This is **not a causal component decomposition**.

Verified: both new arms use the same saved SG parent, prepared specialist tensors, replay and native update budget; initial state digests match. They differ in dynamic commands, feature-aware coordination and effective active capacity/gradient cost. Their package contrast is positive. Additional continuation and a learned residual correction around a strong SG policy are available to both arms and plausibly explain part of both gains over frozen SG.

Unestablished: whether distinct specialist semantics, command adaptation within a stage, feature-aware coordination alone, the extra34,964 active parameters, or optimization noise caused the remaining gain. A history-only SG residual control, ordinary SG continuation from the same parent, and a command×coordination factorial are missing. Inference ablations are informative but are not replacements for matched training controls. The specific ICML claims are assessed separately in ICML_SPECIALIZATION_AND_ROUTING_ASSESSMENT.md.

## Provenance and limits

Every actor/runtime SHA matches its RESULT receipt; parent hashes match the next config; native terminal counters agree with logs/checkpoints; both arms’ initial state digests match in every campaign. The audit recomputed same-bank paired contrasts from saved2000-case score arrays and matched the stored SG/HEC primary means to1e−12. No new trajectory, estimator, checkpoint selection or model fit was used. Full hashes are in INPUT_PRESERVATION.json; the six actor/runtime hashes are also in each original RESULT.json.

Primary information values are finite-contrast lower-bound estimates, not exact EIG. Their Monte Carlo intervals do not include training-seed variability, finite-contrast bias or systematic model error. Three differing campaign scores cannot be turned into a controlled architecture-efficiency comparison merely by normalizing wall time.
'''
runtime=re.sub(r'\b(experts?|Expert|gamma|its|approximately|over|about|than|by|with|and|from|after|before|per|on|the|vs)(?=\d)',r'\1 ',runtime);runtime=runtime.replace('by+','by +').replace('nats,95%','nats, 95%')
(O/'RUNTIME_AND_EIG_EFFICIENCY_ANALYSIS.md').write_text(runtime)
assessment='# ICML specialization and routing assessment\n\n'+intro+'''
## Verdict

**Not yet sufficiently established for a competitive ICML submission whose central claim is four complementary specialists intelligently commanded by history-dependent information needs.** This is a promising one-seed EIG improvement with a mathematically nontrivial query architecture. Its strongest mechanism claims remain unsupported by the current ablations. That is an evidence assessment, not a prediction of an acceptance decision or a demand for a theorem.

The package improves the matched-update heterogeneous control by+0.07211 nats,95%CI[0.01889,0.12532],p=.00794. It also improves the frozen SG checkpoint by+0.18556[0.12445,0.24666],Holm4p=3.06e−9. These are valuable finite-contrast estimates. They do not by themselves establish all-four specialization, observation/uncertainty-dependent routing benefit, specialization×commanding synergy, or algorithmic novelty.

## What the implemented algorithm actually learns

The deployed policy has a frozen strong SG provisional action a0, frozen SG variational GMM belief providers and a frozen SG critic, plus trainable new features and a residual decision head. Native actor–critic/posterior training continues with gamma1 and no specialist auxiliary loss in B/C. The frozen providers are approximations inherited from SG, not exact posterior inference or exact conditional-information oracles.

The commander receives a learned history representation, a0 and approximate-belief means, log variances and correlations. Its outputs alter mixture-region attention, hypothetical design directions, hypothesis weights, Fisher query locations/directions and opportunity probes **before** feature fusion. The exact likelihood and analytic sensitivities really are used. Source and no-update checks establish a differentiable computation path from the native action-value objective to commands and all four experts. Final logs show nonzero candidate commander/coordinator gradients.

Four specialist responsibilities and their different architectures/inputs are **manually specified**. Their representations, command values and integration are learned. This is not discovery of four unconstrained semantic roles, and the paper should not describe it that way. All feature pathways are available, but availability is not demonstrated necessity. A shared history-to-fusion path and the strong a0 can sustain good performance even when an expert is removed.

## Expert specialization: evidence and limits

Phase-A preparation was shared by both arms. On its512 held-out replay prefixes, all four normalized task MSE point estimates improved versus initialization and the training-mean constant predictor:

'''
p=H['preparation'];roles=['Posterior geometry','Likelihood discrimination','Local sensitivity/identifiability','Sequential opportunities']
assessment+=table(['Expert / role','Initial MSE','Prepared MSE','Constant MSE'],[[f'{k+1}: {role}',f"{p['before_normalized_MSE'][k]:.5f}",f"{p['after_normalized_MSE'][k]:.5f}",f"{p['constant_normalized_MSE'][k]:.5f}"] for k,role in enumerate(roles)])
assessment+='''
These are point-estimate preparation checks, not a full claim of statistically proven downstream design competence. Geometry labels describe the existing approximate belief rather than calibrated true posterior regions. Likelihood/Fisher tasks are meaningful mathematical calculations, but learning to encode them does not establish that the downstream policy exploits their distinctions. Expert4 has a two-replica **one-step variational-information plus SG-critic bootstrap** target; it does not demonstrate a learned distinct multi-step strategy or exact expected suffix EIG.

On the final diagnostic bank, calibration-only ridge probes were fitted separately to each frozen representation/task, rival selection used calibration histories, and designated-expert advantages were tested on256 separate held-out histories with Holm4 correction. Positive values mean smaller normalized task loss than the calibration-selected rival:

'''
tasks=H['COMPETENCY_RESULTS.json'];advantages=tasks['designated_expert_advantages'];removal=H['specialization']['expert_removal']
assessment+=table(['Expert','Designated task advantage [95% CI]','Holm4 p','Removal EIG penalty [95% CI]','Removal Holm4 p'],[[str(k+1),ci(advantages[f'expert{k+1}_designated_advantage']),f"{advantages[f'expert{k+1}_designated_advantage']['Holm4_p']:.3g}",ci(removal[f'expert{k+1}']),f"{removal[f'expert{k+1}']['Holm4_p']:.3g}"] for k in range(4)])
assessment+='''
**Established:** experts1 and3 have distinct recoverable strengths on their designated mathematical tasks relative to other HEC representations. This goes beyond merely different output vectors. **Not established:** distinct designated advantages for experts2/4, individually credible EIG usefulness of every expert, four-way complementarity, or conditional information-state advantages that can guide a deployable selector. No expert-removal penalty survives correction; their wide intervals do not prove the experts are useless or equal. Removal also renormalizes the remaining gates and shifts the joint feature distribution, so even positive ablations would need interpretation.

Fixed specialist readouts after joint EIG training are another caution:

'''
assessment+=table(['Expert','Constant loss minus frozen-readout loss [95% CI]','Holm4 p','Interpretation'],[[str(k+1),ci(tasks['frozen_readout_vs_constant'][f'expert{k+1}_beats_constant']),f"{tasks['frozen_readout_vs_constant'][f'expert{k+1}_beats_constant']['Holm4_p']:.3g}",'Readout beats constant' if tasks['frozen_readout_vs_constant'][f'expert{k+1}_beats_constant']['mean']>0 else 'Readout worse than constant'] for k in range(4)])
assessment+='''
Three fixed readouts no longer beat a constant predictor on the final histories. This is **not proof of expert collapse**: feature coordinates can rotate, features/fusion can coadapt, and final histories/commands differ from preparation data. Refitted probes are therefore essential. On the same final histories/commands, prepared-versus-live refitted probe differences show a modest disadvantage for live expert1 (−0.01061 normalized loss advantage,Holm4p=1.72e−12); expert3’s apparent disadvantage does not survive Holm4(p=.0953), and experts2/4 differences are inconclusive. Thus there is no demonstrated across-expert competency improvement from final EIG adaptation, even though final EIG improves.

Task targets are heterogeneous by construction, not four independent discoveries. Correlated statistics can still make their decision contributions redundant. The saved tests do not measure coalition interaction/synergy, the conditional information value of one expert’s query on particular histories, or whether another expert/history-only head can replace it at matched capacity. Those are the missing links between representation specialization and BOED utility.

## Intelligent commanding and coordination

Observed within-stage command standard deviations range approximately0.032–0.218; gate SDs0.023–0.047. The commander therefore varies with histories beyond stage in a descriptive sense. Architecture gives it approximate latent-uncertainty summaries, but nonzero variation and nonzero weights/gradients do not show that uncertainty causally drives useful command choices. No controlled same-stage uncertainty-summary ablation or conditional uncertainty-versus-command-utility analysis was performed.

Each following effect is **learned full HEC minus the indicated frozen-policy intervention**, with full trajectory resimulation, common latent/noise draws and256 held-out histories. Commands and routing families have separate Holm3 corrections. Displayed95% intervals are unadjusted:

'''
commands=H['COMMANDER_RESULTS.json']['commands'];coord=H['COMMANDER_RESULTS.json']['coordination']
assessment+=table(['Intervention','HEC EIG advantage [95% CI]','Holm3 p','Supported inference'],[
 ['Fixed zero commands',ci(commands['fixed']),f"{commands['fixed']['Holm3_p']:.4g}",'Positive; learned query settings matter in this coadapted policy'],
 ['Stage-only commands',ci(commands['stage']),f"{commands['stage']['Holm3_p']:.4g}",'No demonstrated history benefit beyond stage'],
 ['Stage-matched shuffled commands',ci(commands['shuffle']),f"{commands['shuffle']['Holm3_p']:.4g}",'Positive point estimate; inconclusive'],
 ['Conventional linear soft-routing projection',ci(coord['ordinary']),f"{coord['ordinary']['Holm3_p']:.4g}",'No demonstrated improvement over this projection'],
 ['Uniform coordination',ci(coord['uniform']),f"{coord['uniform']['Holm3_p']:.4g}",'No demonstrated benefit; point estimate favors uniform'],
 ['Stage-only coordination',ci(coord['stage']),f"{coord['stage']['Holm3_p']:.4g}",'No demonstrated history-aware coordination benefit']])
assessment+='''
The fixed-command intervention is **zero commands**, not the best learned constant command or best stage schedule. Even though zero occurred in preparation, switching a trained coadapted system to zero changes its feature inputs. Its positive effect does not establish that history-conditioned selection is required. Stage means preserve learned stage settings and are a stronger current challenge to the dynamic-history claim. Non-significance does not prove that stage-only and learned policies are equal; the current diagnostic uncertainty is substantial.

The “ordinary routing” intervention is a calibration-only ridge projection of learned log gate weights onto the candidate’s history encoding. It was not optimized for EIG. The trained matched heterogeneous control is the real ordinary-router comparison, but it also changes the commanding mechanism and effective active capacity. The positive candidate/control primary result supports the **whole changed package**, not isolated superiority of the commander or coordinator. Uniform tests replace coordination, not all command computations; there is no separately tested uniform-command mechanism to report.

Therefore the correct answer to “does it outperform fixed, uniform, stage-only and conventional routing?” is: it beats its tested zero-command intervention and its complete matched-update control; it has **not** demonstrated superiority to stage-only commands, shuffled commands, uniform/stage-only coordination or the linear ordinary-routing projection.

## Joint contribution to full-horizon EIG

All information comparisons use the same new2000-case bank and frozen finite-contrast estimator. On this bank HEC12.95961, heterogeneous control12.88750 and frozen SG12.77406. Candidate−control is positive, while the control already captures+0.11345 of the+0.18556 gap over frozen SG. Both new arms receive2,000 native updates and new residual-policy capacity; frozen SG receives none. Thus ordinary continuation/residual adaptation, extra data and additional model-derived features remain plausible explanations for much of the frozen-baseline gap.

Current evidence cannot isolate specialization×commanding interaction. Shared preparation is present in both arms, and there is no matched no-preparation or generic-feature control. Dynamic commands and feature-aware coordination change simultaneously; their presence/absence does not form a2×2 factorial. Candidate active capacity is34,964 parameters larger than control and training wall time13% longer. No matched ordinary SG continuation from this same SG checkpoint or history-only residual baseline is available. The earlier SG ordinary control starts from another parent and is not a replacement.

Useful joint contribution would require more than all three components being differentiable. For example, define fixed-commander/ordinary-coordinator value J00, dynamic-only J10, feature-coordinator-only J01 and full J11 under the same specialist preparation and budget. The interaction J11−J10−J01+J00 would test an explicit synergy claim; main effects may be useful even without positive synergy. A claim that specialist semantics cause that benefit additionally needs an information/capacity-matched representation/preparation control. None of these missing contrasts can be recovered by subtracting current frozen-policy ablations.

The matched HEC result is **sample/update matched, not exactly wall-time or effective-capacity matched**. Single-seed Monte Carlo CIs do not estimate training-seed variance. Higher frozen-vsOED EIG on this bank is observed(+0.51789,CI[0.42619,0.60959]), but extra inherited/continuation computation prevents a compute-matched superiority claim.

## Algorithmic novelty: substantive differences and inherited mechanisms

The prior-art comparison is targeted rather than exhaustive; it cannot establish novelty by absence of an exact name match.

'''
assessment+=table(['Prior art','Established contribution','HEC difference and remaining gap'],[
 ['[MMoE](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/) / [PLE](https://doi.org/10.1145/3383313.3412236)','Learned expert combinations and structured task/shared representations.','HEC uses four model-derived statistical query operators for a single sequential-information objective; heterogeneous tasks and soft fusion alone are inherited ideas.'],
 ['[Expert Choice](https://arxiv.org/abs/2202.09368) / [DeepSeekMoE](https://arxiv.org/abs/2401.06066)','Expert-token allocation, fine-grained segmentation/shared-expert mechanisms.','HEC changes what dense experts compute rather than capacity-limited token selection. It needs decision-competence evidence, not a claim that different roles prove specialization.'],
 ['[Guo et al.](https://arxiv.org/abs/2505.22323)','Orthogonality/routing-variance auxiliary objectives.','HEC uses pretraining then pure native information-value gradients, without final diversity/load-balancing rewards. Avoiding those auxiliaries is a distinction, not novelty proof.'],
 ['[FiLM](https://arxiv.org/abs/1709.07871) / [HyperNetworks](https://arxiv.org/abs/1609.09106)','Conditioned feature modulation / generated network weights.','HEC commands query locations, belief attention and directions before feature encoding. It is not literally a hypernetwork, but learned conditional computation is established.'],
 ['[Residual Policy Learning](https://arxiv.org/abs/1812.06298)','Learn corrections around an existing controller.','HEC corrects a frozen SG design using command-conditioned statistical features. Strong warm-start residual adaptation is prior art, not a new contribution by itself.'],
 ['[DAD](https://proceedings.mlr.press/v139/foster21a.html) / [vsOED](https://arxiv.org/abs/2306.10430)','Amortized sequential design; variational actor–critic information optimization.','HEC inherits the sequential design framework and native gradient estimator. Its candidate novelty must be the query-learning/coordination mechanism, not EIG or AC.'],
 ['[ScoreBED](https://arxiv.org/abs/2607.08335)','Separate policy-independent score learning from policy optimization.','HEC has no corresponding score-matching contribution and currently has higher per-update/inference costs than SG.'],
 ['Project selection-credit/CIAS/Collaborative/SG','Prior competitive credit failures; successful feature collaboration and anchored adaptation.','HEC removes a responsibility teacher and selects model-derived queries, but current evidence does not show new complementary full-horizon competencies or observation-dependent commanding benefit.']])
assessment+='''
A defensible **candidate** methodological contribution is: native full-horizon information-value gradients learn the content of heterogeneous belief/likelihood/sensitivity/opportunity queries, then integrate their representations around a strong sequential design policy. This is more specific than a renamed softmax router. It is presently a new composition/application hypothesis, not an established new learning estimator, theory, general BOED specialization principle or ICML-level result. Exact likelihood access, Fisher formulas, quadrature, four experts, task labels, frozen providers and bounded residuals are individually established ingredients.

## Claim ledger

'''
assessment+=table(['Claim','Current status'],[
 ['Correct bounded sequential policy with model-derived heterogeneous inputs','Supported structurally and by saved no-update correctness checks.'],
 ['One-seed improvement beyond fixed-command heterogeneous control at equal new episodes/native updates','Supported for the finite-contrast estimate; not equal wall time/active capacity.'],
 ['Experts1 and3 encode their designated statistics better than other experts','Supported by held-out corrected probe tests.'],
 ['All four experts acquire distinct useful BOED competencies','Unproven: two designated-role tests inconclusive; no corrected positive individual-removal EIG effects.'],
 ['Final joint EIG training preserves/improves every prepared competency','Unproven; fixed readouts degrade and no across-expert probe improvement is established.'],
 ['Commands vary within stage and are mathematically more than routing weights','Supported; descriptive variation and changed model queries are real.'],
 ['Latent uncertainty causes beneficial history-specific command allocation','Unproven: possible by inputs, not isolated empirically.'],
 ['Dynamic commanding outperforms stage-only commands and ordinary/uniform coordination','Unproven in current diagnostic families.'],
 ['Specialization and dynamic coordination jointly cause the EIG gain','Unproven: factorial, capacity and history-only residual controls missing.'],
 ['Higher same-bank score than frozen SG/Option/vsOED','Supported as a frozen-checkpoint comparison, with unequal inherited/training compute.'],
 ['Faster or more computationally efficient learning from scratch','Unsupported: warm starts, reduced update budgets,2.5× higher per-update cost and5.7× higher inference latency.'],
 ['Generalization and a competitive ICML mechanism claim','Not established: one task/seed, incomplete mechanism evidence and unverified algorithmic novelty.']])
assessment+='''
## Minimum additional evidence, ordered to avoid unnecessary training

These are **recommendations only**. This audit launches none of them and does not change historical evaluation rules or checkpoints.

1. **Strengthen frozen diagnostics before spending training compute.** Predeclare a fresh diagnostic bank and multiplicity families, separate calibration/validation, and resimulate all changed-policy observations. Test learned vsstage-only commands/coordinators as the key history-benefit contrast; compare optimized calibration-only constant commands as well as zero. Replace/mask each expert using stage-matched donor representations, evaluate leave-one-out/selected coalition interactions, and stratify by decision-time information summaries rather than true latent parameters. Use common randomness where valid. Quantify finite-contrast sensitivity at predeclared contrast counts instead of selecting favorable expert rankings.
2. **Establish semantic content and uncertainty sensitivity on common histories.** Compare prepared/live refitted probes, calibration-only transferred readouts and generic-feature controls using the same histories/commands/target generator. For expert1 validate belief geometry against independently reliable model-specific references on a small prefix panel; for expert2/3 distinguish nonlocal discrimination from local sensitivity; for expert4 validate its bootstrap opportunity rankings against separately resimulated continuation returns on a small tractable panel. A critic/pretext score is not exact conditional EIG. Perturb or stage-match the commander’s uncertainty-summary input separately from its history and action inputs, with an explicit covariate-shift caveat, to test whether this information actually changes useful query selection.
3. **Only if frozen evidence is promising, request a small controlled training study.** From the same SG parent include ordinary SG continuation and an information/capacity-matched history-only residual control. Use the command×coordination2×2 with shared preparation to separate the two mechanisms; reuse the existing two cells only where the frozen protocol and parent match. Include an appropriate no-specialist-preparation or generic representation control to isolate semantic preparation rather than extra information/capacity. Match episodes/updates, active parameters and report all operator/frozen-provider cost; add an equal-wall-time control or fixed cost/return curve for an efficiency claim. Do not call differently effective modules “matched capacity” merely because both instantiate unused tensors.
4. **Replicate the predeclared central comparison across independent training seeds.** At least three independent runs per arm are an initial variability check; a power-planned five-seed comparison is preferable if variability requires it. Report run-level uncertainty and paired trajectory uncertainty separately, terminal-only selection, all negatives and inherited compute. Extra evaluation trajectories alone cannot replace independent training runs.
5. **For a broad ICML method, test portability after the mechanism survives controls.** One additional scientifically distinct problem is a minimum external check of the general query-learning claim; deterministic SIR ODE and dynamic swing-equation campaigns are planned but unexecuted. Reuse one algorithm with only justified likelihood/sensitivity/feasibility operators changed. A strong single-benchmark paper would require correspondingly deeper identification and mechanism evidence; multiple benchmarks alone cannot repair a missing causal mechanism.

Diagnostic sample sizes need a power plan. For illustration only, the current stage-command paired SE0.03997 with256 cases implies a case-level SD about0.6395 nats. A conventional two-sided80%-power test for a0.03-nat effect would need roughly3,600 independent cases before Holm/multiple-seed considerations, assuming that SD transfers. This is planning arithmetic, not a recommendation to optimize another reused bank or a promise that a larger test will turn positive. Avoid reading non-significance as equality.

A publishable claim should survive both plausible outcomes: if a stage schedule matches the commander, simplify the claim to learned stage-dependent query design; if only two specialists are necessary, do not maintain a four-essential-expert narrative. Preserve negative outcomes rather than enlarging budgets until they disappear.

## Research decision

Keep HEC as a reproducible **partial positive EIG pilot**. Do not yet claim four genuinely complementary experts, uncertainty-sensitive intelligent routing, jointly identified specialization/commanding synergy, compute-matched vsOED superiority or ICML-ready novelty. The most economical next authorized step would be sharper frozen-policy identification of what the useful query changes are; training a more complicated architecture is not justified by this audit alone.
'''
assessment=re.sub(r'\b(experts?|Expert|gamma|its|approximately|over|about|than|by|with|and|from|after|before|per|on|the|vs)(?=\d)',r'\1 ',assessment);assessment=assessment.replace('by+','by +').replace('nats,95%','nats, 95%')
(O/'ICML_SPECIALIZATION_AND_ROUTING_ASSESSMENT.md').write_text(assessment)
# Save report copies separately from all protected experiment roots.
for n in ['RUNTIME_AND_EIG_EFFICIENCY_ANALYSIS.md','ICML_SPECIALIZATION_AND_ROUTING_ASSESSMENT.md']:(A/n).write_bytes((O/n).read_bytes())
(A/'REPORT_SHA256.json').write_text(json.dumps({n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in ['RUNTIME_AND_EIG_EFFICIENCY_ANALYSIS.md','ICML_SPECIALIZATION_AND_ROUTING_ASSESSMENT.md']},indent=2));print(json.dumps(dict(status='COMPLETE',reports=str(O),existing_models_modified=False,new_training_runs=0)))
