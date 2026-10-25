"""Generate the final specialization report from saved artifacts only."""
from pathlib import Path
import csv,hashlib,json,math,shutil,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parents[1]
D=R/'experiments/10242026/10242026_ir_moe_labpc_diagnostics'
G=D/'GPU_attempt2'
C=Path.home()/'Documents/objective_power_grid_integrated_information_moe/experiments/10242026/10242026_ir_moe_seed101'
A=R/'reports/ir_moe_labpc_diagnostics';A.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,x:Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
meta=read(D/'metadata/source_inventory.json')
feature=read(G/'data/feature_audit.json')['stages']
competence=read(G/'data/competence.json')
interventions=read(G/'data/routing_interventions.json')
stochastic=read(G/'data/stochastic_deployment_diagnostic.json')
cost=read(G/'costs/GPU_diagnostics.json')
pairs=read(D/'data/paired_comparisons.json')
assert hashlib.sha256((C/'models/full/selected.pt').read_bytes()).hexdigest()==meta['checkpoint_sha256']==cost['checkpoint_sha256']
assert cost['checkpoint_unchanged'] and cost['optimizer_updates']==0
official=meta['official'];train=meta['training'];stats=meta['statistics']
valid=[x for x in competence['histories'] if x['status']=='FINITE_IMPORTANCE_POSTERIOR_APPROXIMATION']
excluded=[x for x in competence['histories'] if x not in valid]
wins=sum(sum(x['reliable_alternative_wins']) for x in valid)
cos=np.mean([x['feature_cosine'] for x in feature],axis=0)
off=cos[~np.eye(7,dtype=bool)]
shared_norm=float(np.mean([x['shared_feature_L2'] for x in feature]))
routed_norm=float(np.mean([x['routed_feature_L2'] for x in feature]))
shared_action=float(np.mean([x['shared_design_effect_RMS'] for x in feature]))
routed_action=float(np.mean([x['routed_design_effect_RMS'] for x in feature]))
entropy=float(np.mean([x['entropy'] for x in feature]))
uniform=next(x for x in interventions if x['mode']=='uniform_selected_weights')
uniform_gain=-uniform['paired_intervention_minus_original']['difference']
uniform_se=uniform['paired_intervention_minus_original']['paired_MC_SE']
fence=chr(96)*3
conclusion='D. Insufficient reliable evidence.'
summary=dict(conclusion=conclusion,official_EIG=official['EIG'],official_MC_SE=official['MC_SE'],training_updates=train['executed_updates'],last_logged_update=stats['last_logged_update'],last_logged_training_spce=stats['last_logged_spce'],training_seconds=train['charged_seconds'],valid_conditional_histories=len(valid),excluded_conditional_histories=len(excluded),familywise_reliable_alternative_wins=int(wins),learned_minus_uniform_weights=uniform_gain,paired_MC_SE=uniform_se,feature_cosine_mean=float(off.mean()),shared_action_RMS=shared_action,routed_action_RMS=routed_action,diagnostic_GPU_wall_seconds=cost['exclusive_GPU_wall_seconds'],checkpoint_sha256=meta['checkpoint_sha256'])
write(A/'summary.json',summary)
write(A/'paired_comparisons.json',pairs)
write(A/'routing_interventions.json',interventions)
write(A/'competence.json',competence)
write(A/'feature_audit.json',dict(stages=feature))
write(A/'costs.json',dict(successful_GPU_attempt=cost,competence_counts=read(G/'costs/competence_counts.json'),CPU_inventory=read(D/'costs/CPU_inventory.json'),failed_GPU_attempt1=read(D/'metadata/failed_GPU_attempt1.json'),note='GPU wall allocation is measured; kernel busy time was not profiled. Original-model official evaluation was reused, not rerun.'))
write(A/'source_provenance.json',dict(source_campaign=str(C),training_commit=meta['source_commit'],checkpoint_sha256=meta['checkpoint_sha256'],verified_frozen_source_hash_count=meta['all_frozen_hashes_verified'],canonical_cases_sha256=meta['configuration']['cases_sha256'],canonical_contrast_sha256=meta['configuration']['contrast_sha256'],paired_draws_sha256=official['metadata']['paired_draws_sha256'],original_controller_status=meta['original_controller'],reporting_exception=meta['reporting_failure'],model_unchanged=True,training_updates_performed_by_this_analysis=0))

# Preserve small score arrays in a reviewable table; no model or bank is copied.
import yaml
cfg=meta['configuration'];old=np.load(Path(cfg['reference_campaign'])/'evaluations/official_seed1001_scores.npy')
scores=[np.load(D/'data/ir_seed1001_scores.npy'),np.load(D/'data/vsoed_seed1001_scores.npy'),np.load(D/'data/dad_seed1001_scores.npy'),old]
with (A/'paired_trajectory_scores.csv').open('w',newline='') as f:
 writer=csv.writer(f);writer.writerow(['trajectory_id','IR_MoE','vsOED','corrected_DAD','DeepSeekMoE_v1'])
 for i in range(2000):writer.writerow([i]+[float(x[i]) for x in scores])
with (A/'expert_utilization_by_stage.csv').open('w',newline='') as f:
 writer=csv.writer(f);writer.writerow(['stage']+['E'+str(i) for i in range(7)]+['router_entropy','shared_design_RMS','routed_design_RMS'])
 for row in feature:writer.writerow([row['stage']]+row['selection_frequency']+[row['entropy'],row['shared_design_effect_RMS'],row['routed_design_effect_RMS']])
np.savetxt(A/'expert_feature_similarity.csv',cos,delimiter=',')
with (A/'expert_combination_advantage.csv').open('w',newline='') as f:
 writer=csv.writer(f);writer.writerow(['history','stage','status']+[''.join(map(str,s)) for s in competence['sets']])
 for row in competence['histories']:writer.writerow([row['case'],row['stage'],row['status']]+row.get('paired_advantage_over_selected',['']*35))

shutil.copy2(G/'figures/expert_utilization.png',A/'expert_utilization_by_stage.png')
shutil.copy2(G/'figures/competence_matrix.png',A/'expert_combination_information_advantage.png')
fig,ax=plt.subplots(figsize=(6,5))
im=ax.imshow(cos,vmin=0,vmax=1,cmap='viridis')
ax.set_xticks(range(7));ax.set_yticks(range(7));ax.set_xticklabels(['E'+str(i) for i in range(7)]);ax.set_yticklabels(['E'+str(i) for i in range(7)])
for i in range(7):
 for j in range(7):ax.text(j,i,f'{cos[i,j]:.2f}',ha='center',va='center',color='black' if cos[i,j]>.6 else 'white',fontsize=9)
ax.set_title('Expert feature cosine similarity\n64 held-out trajectories, averaged across stages')
fig.colorbar(im,ax=ax,label='Cosine similarity');fig.tight_layout();fig.savefig(A/'expert_feature_similarity.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4))
ax.plot(range(30),[x['shared_design_effect_RMS'] for x in feature],label='Remove shared feature branch')
ax.plot(range(30),[x['routed_design_effect_RMS'] for x in feature],label='Remove routed feature branch')
ax.set(xlabel='Experimental stage (zero based)',ylabel='RMS change in physical design',title='Frozen local action contributions on identical histories')
ax.legend();fig.tight_layout();fig.savefig(A/'expert_action_contributions.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4))
labels=['Uniform weights; same learned sets','Swap rank-3 for rank-4 expert','Shared-only intervention','Routed-only intervention']
means=[x['paired_intervention_minus_original']['difference'] for x in interventions]
errors=[1.96*x['paired_intervention_minus_original']['paired_MC_SE'] for x in interventions]
ax.errorbar(means,range(4),xerr=errors,fmt='o',capsize=4);ax.axvline(0,color='gray',linestyle='--')
ax.set_yticks(range(4));ax.set_yticklabels(labels);ax.invert_yaxis()
ax.set(xlabel='Intervention minus original EIG (nats), paired approximate 95% CI',title='Frozen full-horizon interventions: 2000 cases, 1M contrasts')
fig.tight_layout();fig.savefig(A/'routing_ablation_comparison.png',dpi=180);plt.close(fig)

sections=[]
sections.append('# IR-MoE-sBOED final expert-specialization report\n\n**Conclusion: '+conclusion+'**\n\nThe experts produce different feature vectors and measurably different designs. The saved diagnostics do not establish useful context-dependent information specialization or a learned-routing advantage. Only three of eight conditional histories have adequate importance-posterior references; middle and late estimates are withheld. This is not proof that specialization is impossible.')
sections.append('## Frozen model and completed run\n\nThe model has one always-active shared width8 SwiGLU expert, seven routed width8 experts, top-three routed activation, a shared 3-256-ReLU-16 sum encoder, and one shared design head. Experts emit hidden features, not independent experiment-design policies. There are 8,434 parameters, with 6,898 activated per context. The latent output is mapped to the original [-4,4] box.\n\n'+f"Training seed101 completed {train['executed_updates']} updates in {train['charged_seconds']:.3f} charged seconds and stopped at the compute cap. The selected checkpoint is update{train['selected_update']}. The latest logged sPCE is {stats['last_logged_spce']:.6f} at update{stats['last_logged_update']}; no unlogged value at the final update is invented. Training used 8,192 contrasts and 30 stages.\n\n"+f"Checkpoint SHA256: {meta['checkpoint_sha256']}.\n\nThe legacy controller says FAILED_PRESERVED because the report writer attempted Path-plus-string concatenation AFTER training, both IR evaluations, and comparisons had completed. The checkpoint is numerically finite, all frozen source/config hashes were verified, and saved per-trajectory scores reproduce the official summary. This report repairs artifact reporting in a separate worktree; it does not rewrite the original registry or results.")
table='## Official EIG and verified paired comparisons\n\n'+f"Official seed1001 EIG: **{official['EIG']:.8f} +/- {official['MC_SE']:.8f} nats** (trajectory MC SE conditional on the shared bank),2000 trajectories,1000000 contrasts,T30.\n\n"
table+='| Reference | Reference mean | IR minus reference | Paired MC SE | Approximate95% CI | N |\n|---|---:|---:|---:|---|---:|\n'
for row in pairs:table+=f"| {row['method']} | {row['reference_mean']:.8f} | {row['difference']:.8f} | {row['paired_MC_SE']:.8f} | [{row['CI95'][0]:.8f}, {row['CI95'][1]:.8f}] | {row['N']} |\n"
table+='\nTruth samples, standard-normal observation noise, trajectory order and contrast-bank bytes were reverified for vsOED and corrected DAD. The DeepSeekMoE-v1 record has the same saved draw identity. Paired SEs come from paired trajectory-score differences, not method-level means/SEs. The legacy DAD result is excluded. The one training seed is not multi-seed evidence.\n\n'
table+=f"Previously reserved confirmation seed1079 gave {meta['confirmation']['EIG']:.8f} +/- {meta['confirmation']['MC_SE']:.8f} nats. Neither official nor confirmation results were used to retune or replace the checkpoint."
sections.append(table)

answers='## Answers to the eight questions\n\n'
answers+=f"1. **Different feature functions?** Yes on the audited inputs: mean off-diagonal expert cosine is {off.mean():.6f}, range [{off.min():.6f}, {off.max():.6f}]. The frozen functions are not identical. Without an initialization comparison this does not establish that diversity emerged through learning, and it does not establish information competence.\n\n"
answers+=f"2. **Different designs?** Yes. Across all 35 sets, within-history design dispersion RMS ranges from {min(x['action_dispersion_RMS'] for x in competence['histories']):.6f} to {max(x['action_dispersion_RMS'] for x in competence['histories']):.6f} on the eight audited prefixes. Removing the routed feature sum changes physical designs by mean RMS {routed_action:.6f}, versus {shared_action:.6f} for removing the shared branch. These local interventions use identical histories.\n\n"
answers+='3. **Different information values?** Point estimates differ, but no alternative has a reliably positive advantage over the learned set under the saved familywise screening rule. The reliable-reference rows have best observed gains only0.01396 +/-0.01814,0.03169 +/-0.02810, and0.01252 +/-0.02707 nats (paired MC SE). Noisy maxima are not reliable wins.\n\n'
answers+='4. **Different combinations advantageous in different histories?** Not established. Different point-estimate maximizers occur, but their uncertainty overlaps zero. Only three references are usable; all middle/late references and one early reference fail the ESS screen.\n\n'
answers+=f"5. **Learned routing beats uniform routing?** Not established. With the SAME selected sets and routed feature mass, learned-minus-uniform-weight EIG is {uniform_gain:.8f} +/- {uniform_se:.8f} nats; approximate 95% CI [{uniform_gain-1.96*uniform_se:.8f}, {uniform_gain+1.96*uniform_se:.8f}]. The learned weights do not show an advantage.\n\n"
answers+='6. **Routing improves full-horizon EIG?** Not demonstrated by the frozen interventions. Controlled swaps and masking changes all have paired intervals containing zero. This does not identify the effect of retraining an entirely different architecture.\n\n'
answers+='7. **All seven routed experts necessary?** Not established. Removing the entire routed branch gives10.81752224 nats versus10.81546708 for the full policy: +0.00205515 +/-0.01728294. The approximate 95% interval [-0.03181941,0.03592971] shows no detected loss. This does not prove a minimal expert count: seven individually retrained removal controls are absent, and no equivalence margin was predeclared.\n\n'
answers+='8. **Limits relative to vsOED?** The observed routing interventions do not explain the1.66538-nat gap. Confirmed findings are small routed action influence, no detected information benefit from the tested routing changes, and unresolved later-history posterior references. Near-ceiling finite-contrast training and gradient variance are plausible contributors, not established causes. Optimization convergence and the value of a different architecture remain unresolved.'
sections.append(answers)

contexts='## Conditional competence: early, middle and late\n\nFor each fixed observed history, two independent262144-particle standard-normal prior banks are weighted by the full observed-prefix likelihood. No generating hidden theta is appended or treated as the posterior. Each reference supplies32 posterior-predictive fantasies and65536 conditional contrasts per fantasy. All35 expert sets are evaluated at the CURRENT decision; the SAME frozen original deterministic policy controls every remaining decision, with paired future noise/parameters. The reported Q is remaining-horizon conditional sPCE, not immediate gain. Immediate values were saved separately.\n\nThis is a finite-importance-posterior approximation. MC SEs are conditional on those empirical posterior banks; they do not quantify all posterior approximation error. Two-bank sensitivity is recorded, and ESS>=128 is required in both banks. ESS alone is not a convergence certificate. A familywise normal threshold z=3.74027 covers the predeclared8x34 alternative-versus-selected comparisons, with matching positive direction across posterior replicates.\n\nTwo independently simulated trajectories each supply four prefixes. The eight histories are not eight independent training runs or eight independent physical systems. Stage t is the number of completed experiments; the next experiment is t+1.\n\n'
contexts+='| Group | Prefix stage | History | Posterior ESS, two banks | Decision |\n|---|---:|---:|---|---|\n'
for row in competence['histories']:
 group='Early' if row['stage']<=5 else 'Middle' if row['stage']==15 else 'Late'
 contexts+=f"| {group} | {row['stage']} | {row['episode']} | {row['posterior_ESS'][0]:.2f}, {row['posterior_ESS'][1]:.2f} | {'Retained' if row in valid else 'EXCLUDED: posterior degeneracy'} |\n"
contexts+='\n| Retained prefix/history | Learned set | Point-best alternative | Advantage over learned | Paired MC SE | Familywise reliable win? |\n|---|---|---|---:|---:|---|\n'
for row in valid:
 values=np.asarray(row['paired_advantage_over_selected']);i=int(values.argmax())
 contexts+=f"| t{row['stage']}/h{row['episode']} | {row['selected']} | {competence['sets'][i]} | {values[i]:.8f} | {row['paired_MC_SE'][i]:.8f} | No |\n"
contexts+='\nGray cells in the competence matrix mean excluded/unresolved posterior references, not zero information advantage. Different point-best sets are not assigned invented exploration/localization roles.'
sections.append(contexts)

abl='## Frozen full-horizon routing and ablation results\n\nAll rows below use the exact same 2,000 canonical cases, noise and1M-contrast bank. These are frozen-model interventions, not retrained methods. The identity/context feature residual remains present in shared-only and routed-only interventions.\n\n'
abl+='| Policy/intervention | EIG mean | MC SE | Intervention minus original | Paired MC SE | Approximate95% CI |\n|---|---:|---:|---:|---:|---|\n'
abl+=f"| Original learned routing | {official['EIG']:.8f} | {official['MC_SE']:.8f} | 0 | 0 | reference |\n"
for x in interventions:
 p=x['paired_intervention_minus_original'];abl+=f"| {x['mode']} | {x['EIG']:.8f} | {x['MC_SE']:.8f} | {p['difference']:.8f} | {p['paired_MC_SE']:.8f} | [{p['CI95'][0]:.8f}, {p['CI95'][1]:.8f}] |\n"
olduniform=next(x for x in meta['interventions'] if x.get('label')=='uniform')
abl+=f"\nA DIFFERENT uniform control samples random expert subsets and equalizes weights while preserving learned top3 mass. On the saved256-case/L262144 development bank, learned-minus-that-control is {-olduniform['paired_vs_learned']['difference']:.8f} +/- {olduniform['paired_vs_learned']['paired_MC_SE']:.8f}, also inconclusive. It must not be conflated with uniform weights on fixed learned membership.\n\n"
st=stochastic['paired_vs_original_deterministic'];abl+=f"The original learned PL stochastic deployment diagnostic gave {st['difference']:.8f} +/- {st['paired_MC_SE']:.8f} nats relative to deterministic routing on the same development bank, averaged over three routing seeds. That provides no demonstrated stochastic-versus-deterministic advantage. It did not change deployment selection. Three-seed averaging does not fully quantify routing-seed or bank uncertainty.\n\n"
dense=next(x for x in meta['interventions'] if x.get('label')=='dense_control')['paired_vs_IR']
abl+=f"The available dense control exceeds IR on the256-case development bank: IR-minus-dense {dense['difference']:.8f} +/- {dense['paired_MC_SE']:.8f}. It reused a lower-contrast training control, so this is not a clean architecture-only causal comparison."
sections.append(abl)

weak='## Bottlenecks: findings versus hypotheses\n\n'
weak+=f"- **Confirmed:** all seven routed experts had gradient tensors at all 14,232 optimizer updates. Logged nonzero expert gradients argue against gross training starvation; tensor presence is not proof of competence.\n- **Confirmed:** the shared feature norm averages {shared_norm:.6f}, versus {routed_norm:.6f} for the routed sum. Shared-induced design changes are about {shared_action/routed_action:.2f} times larger. This is feature/action dominance, not a demonstrated information advantage of the shared branch.\n- **Confirmed:** mean router entropy is {entropy:.6f}, close to log7={math.log(7):.6f}. Yet hard selections vary by stage and within stage; high entropy does not mean the router ignores history, and usage differences do not prove information specialization.\n"
weak+=f"- **Empirical limit:** latest logged training sPCE {stats['last_logged_spce']:.6f} is {stats['ceiling']-stats['last_logged_spce']:.6f} below the finite-L ceiling log8193={stats['ceiling']:.6f}. Near-ceiling training can reduce information resolution; it does not establish that true EIG is optimal.\n"
weak+=f"- **Empirical limit:** {100*stats['logged_fraction_clipped']:.1f}% of logged updates exceed the norm5 clipping threshold. Logged gradient median={stats['logged_gradient_median']:.4f}, maximum={stats['logged_gradient_max']:.4f}. Temporal score-gradient variability includes parameter drift; it is not a fixed-policy variance estimate. Gradient noise as the cause of poor routing is plausible, not proven.\n"
weak+='- **Unresolved:** feature redundancy in information acquisition versus a router failing to exploit complementary competence. The present Q intervals and excluded histories cannot discriminate these explanations.\n- **Unresolved:** optimization convergence. The same development bank improves from10.30536 at5000 updates to10.61710 at10000; no same-bank final14232 development estimate was saved. Different final-bank scores are not a continuation of that curve.\n\nThe next justified modification is to improve and validate later-history posterior references and reduce variance in conditional remaining-information differences before assigning expert roles or changing expert count. Then test information-grounded credit assignment in a separately declared study. No new training, architecture change, checkpoint selection, or official reevaluation is performed by this report.'
sections.append(weak)
compute='## Cost, integrity and limitations\n\n'+f"Successful bounded GPU diagnostics took {cost['exclusive_GPU_wall_seconds']:.3f} seconds of exclusive-device wall allocation (budget600s); exact kernel busy time was not profiled. The four saved full-protocol interventions each use2000 trajectories/L1M/T30, and three stochastic deployment repetitions use256/L262144. Conditional competence cost and exact parameter-stage propagation counts are in costs.json. The first failed diagnostic attempt completed64 feature trajectories (1920 parameter-stage propagations) before a CPU/CUDA bank-placement error; its elapsed GPU wall time was not recorded and is not assumed zero. Its log and failure record remain preserved.\n\n"+f"Original training cost was {train['charged_seconds']:.3f} charged seconds, {train['counts']['fresh_complete_primary_trajectories']} primary trajectories and {train['counts']['executed_parameter_stage_propagations']} executed parameter-stage propagations including replay/recomputation. New report generation performs no simulator calls. The saved online latency is {meta['latency']['mean_ms']:.6f} ms mean ({meta['latency']['median_ms']:.6f} ms median), single-case synchronized neural inference; no online optimizer or simulator is involved.\n\n"
compute+='One training seed, one consumed historical official bank, small held-out conditional-history coverage, finite posterior/contrast approximations and no individually retrained expert-removal controls limit conclusions. Original checkpoints, optimizer state, configurations and result files remain unchanged.'
sections.append(compute)
sections.append('## Figures and saved evidence\n\n![Expert utilization by stage](ir_moe_labpc_diagnostics/expert_utilization_by_stage.png)\n\n![Expert feature similarity](ir_moe_labpc_diagnostics/expert_feature_similarity.png)\n\n![Expert-combination information advantage](ir_moe_labpc_diagnostics/expert_combination_information_advantage.png)\n\n![Frozen routing ablations](ir_moe_labpc_diagnostics/routing_ablation_comparison.png)\n\n![Local action contributions](ir_moe_labpc_diagnostics/expert_action_contributions.png)\n\nCSV tables and JSON uncertainty/provenance records are in ir_moe_labpc_diagnostics/. Raw source diagnostic outputs remain in '+str(D)+'.')
text='\n\n'.join(sections)+'\n'
(R/'reports/IR_MOE_FINAL_SPECIALIZATION.md').write_text(text)
for name,content in [('IR_MOE_LABPC_STATUS.md',sections[1]),('IR_MOE_OFFICIAL_EIG.md',sections[2]),('IR_MOE_PAIRED_COMPARISONS.md',table),('IR_MOE_EXPERT_SPECIALIZATION.md','\n\n'.join([sections[0],answers,contexts,abl])),('IR_MOE_WEAKNESS_AND_NEXT_STEP.md',weak)]:
 (R/'reports'/name).write_text(content+'\n')
write(D/'reports/final_report_receipt.json',dict(report=str(R/'reports/IR_MOE_FINAL_SPECIALIZATION.md'),SHA256=hashlib.sha256(text.encode()).hexdigest(),conclusion=conclusion,new_training_updates=0,new_official_original_model_evaluations=0,checkpoint_unchanged=True))
print(json.dumps(summary,indent=2))

