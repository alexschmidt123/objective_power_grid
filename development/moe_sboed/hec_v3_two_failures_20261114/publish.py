"""Two-failure root-cause report; frozen diagnostics, exact V2 benchmark."""
import json,pathlib
from diagnostic_common import ROOT,WORK,CANON,CF,sha
out=ROOT/'scratch_runs/hec_v3_two_failures_20261114';fact=ROOT/'scratch_runs/hec_v3_two_failures_20261114_factorial'
r=json.loads((out/'CORRECTED_RESULTS.json').read_text());f=json.loads((fact/'RESULTS.json').read_text());c=json.loads((CANON/'RESULTS.json').read_text())
assert all(v['status']=='COMPLETE' and v['training_updates']==0 for v in [r,f])
for path in [out,fact]:
    assert all(sha(p)==h for p,h in json.loads((path/'MANIFEST.json').read_text()).items())
dest=WORK/'reports/hec_v3_two_failures_20261114';dest.mkdir(parents=True,exist_ok=True)
def where(relative,needle):
    text=(WORK/relative).read_text().splitlines();line=next(i+1 for i,v in enumerate(text) if needle in v)
    return f'`{relative}:{line}`'
source='development/moe_sboed/hec_v3_counterfactual_router_20261114/experiment.py'
policy='development/moe_sboed/hec_v3_v2_pathways_20261113/pathway_policy.py'
refs=dict(value_fit=where(source,'def fit_values'),value_size=where(source,'nn.Linear(x.shape[-1],128)'),
    reset=where(source,'torch.manual_seed(seed)'),train64=where(source,'net(z[:64])'),
    KL=where(source,'loss=-(g*values)'),gate=where(source,"values=pred if arm"),
    head=where(policy,'self.head='),selection=where(policy,"action=proposals.gather"))
lines=['# Why V3 fails in the two requested ways','',
    '**Finding 1: the V2-to-V3 expert decision conversion lost a successful policy function. Finding 2: the latest router fix used an unreliable value fit and a regularization scale that kept routing almost unchanged.** '
    'These are different failures; repairing the second alone has not repaired the first.','',
    'All performance comparisons use the exact original V2 bank and frozen scorer: 2,000 episodes, 1,000,000 shared contrasts, original routing uniforms and physics. '
    'The preceding canonical run reproduced all six historical reference vectors exactly. This diagnosis changed no model weights and performed no optimizer steps.','',
    '## 1. V3 is below V2: verified decision-interface loss','',
    '| Router function | V2 joint feature decoder | V3 standalone decoder + categorical selection |',
    '|---|---:|---:|',
    f"| Inherited V2 router | {f['means']['inherited_router_joint_head']:.8f} | {f['means']['inherited_router_selected_decision']:.8f} |",
    f"| Current V3 router | {f['means']['new_router_joint_head']:.8f} | {f['means']['new_router_selected_decision']:.8f} |",
    f"| Uniform weights | {r['bridge_means']['uniform']:.8f} | {c['means']['uniform']:.8f} |",'',
    'Each cell uses the same frozen final V3 query pathways and shared context. The inherited and current router functions are held fixed across the two decoder columns. '
    'The new bridges are diagnostic assemblies, not independently executable expert solutions.','']
v=f['comparisons']['joint_minus_selected_same_inherited_router']
lines += [f"With the inherited routing function fixed, the joint decoder gains {v['mean']:+.8f} nats over V3 selected decisions (paired 95% CI [{v['CI95'][0]:+.8f}, {v['CI95'][1]:+.8f}], Holm p={v['p_Holm']:.6g})."]
v=r['bridge_comparisons']['uniform_minus_uniform']
lines += [f"With uniform weights fixed, joint decoding gains {v['mean']:+.8f} nats (95% CI [{v['CI95'][0]:+.8f}, {v['CI95'][1]:+.8f}], Holm p={v['p_Holm']:.6g}).",'',
    'This localizes a real weakness in this conversion: V2 combines query contributions before nonlinear decoding; V3 decodes a single pathway and selects its action. '
    'The standalone heads were initialized to V2 one-hot pathways and then fitted to teacher actions; this does not preserve the complete V2 policy. '
    'Prior matched tests already found that low teacher-action MSE did not preserve full-episode EIG. '
    'It is not proof that decision-level MoE is inherently inferior, that all posterior information is irretrievably lost, or that query cores have collapsed.','',
    'The low current-router/joint-head result is important: the learned V3 gates cannot simply be transplanted into V2 fusion. '
    'Router preferences and decoder semantics interact. Restoring the shared head while retaining the current gates is not a demonstrated fix.','',
    'Code: '+refs['head']+' and '+refs['selection']+'.','',
    '## 2. Learned V3 has not beaten uniform: the attempted fix barely changed allocation','',
    'The current matched difference is new-router minus uniform = '
    f"{c['comparisons']['uniform']['mean']:+.8f} nats, 95% CI [{c['comparisons']['uniform']['CI95'][0]:+.8f}, {c['comparisons']['uniform']['CI95'][1]:+.8f}]. There is no demonstrated advantage.",'',
    '### 2a. The fitted routing targets did not reliably learn advantages','',
    'Each round fitted three fresh 91,588-parameter value networks on only 64 distinct training histories, with 16 validation histories and 16 audit histories. '
    'There were only 192 fitting histories across all three rounds, and fitting reset each round rather than accumulating evidence. '
    'Millions of suffix observations therefore did not mean millions of distinct histories for value regression.','',
    'Every one of the nine counterfactual value fits selected the checkpoint after just one AdamW step. '
    'The frozen test reconstructed the original random initializations and their first updates analytically, matching saved weights within 7.46e-9. '
    'The selected ensemble did not reliably improve on its untrained initialization or a zero-advantage baseline.','',
    '| Round | Untrained audit MSE | Selected audit MSE | Zero-advantage audit MSE | Predicted advantage RMS | MC training-label RMS |',
    '|---|---:|---:|---:|---:|---:|']
for row in r['frozen_tests']:
    v=row['value_fit'];lines.append(f"| {row['round']} | {v['initial_independent_audit_MSE']:.8f} | {v['selected_independent_audit_MSE']:.8f} | {v['zero_independent_audit_MSE']:.8f} | {v['predicted_advantage_RMS']:.6f} | {v['training_MC_advantage_RMS']:.6f} |")
lines+=['','The MC labels are noisy estimates, not true advantages. Their larger RMS alone is not proof of bias in the fitted values. '
    'The independent prediction errors and one-step selections establish that this fit did not supply a validated improvement in routing credit. '
    'The code still passed its predictions into router optimization without a gate requiring improvement over the zero baseline. '
    'This is a confirmed workflow/design weakness, not proof that useful conditional expert choices do not exist.','',
    'Code: '+refs['value_fit']+', '+refs['value_size']+', '+refs['train64']+', '+refs['reset']+', '+refs['gate']+'.','',
    '### 2b. The unit KL penalty anchored the router to its old preferences','',
    'The implemented router loss was `-mean(sum(p*A_hat)) + KL(p_old || p)`, plus a separate KL<=0.01 step limit. '
    'The added penalty did not alter the simulator reward, but it did change the router optimization objective. '
    'For fixed history-wise values, its unconstrained optimum is `p_new(k)=p_old(k)/(tau-A_hat(k))`, with `tau>max(A_hat)` chosen so probabilities sum to one. '
    'Small fitted advantages therefore imply a small change even after further optimization.','',
    '| Round | Return/KL gradient cosine | Combined / return gradient norm | Mean probability-mass movement (TV) | Actual KL |',
    '|---|---:|---:|---:|---:|']
for row in r['frozen_tests']:
    v=row['router_update'];lines.append(f"| {row['round']} | {v['gradient_cosine']:.8f} | {v['total_to_reward_gradient_norm']:.6f} | {100*v['mean_actual_TV']:.4f}% | {v['actual_KL']:.9f} |")
lines+=['','At the saved endpoints, the penalty reduced the return-gradient norm by about 89–98% through opposition. '
    'Mean probability-mass movement was only about 0.10–0.19% per round. Actual KL was far below the separate 0.01 cap. '
    'This is expected near a stationary point of the regularized objective; it is not a broken autograd path. '
    'It explains why increasing iteration count alone would not necessarily move this fixed-target update much farther. '
    'The penalty coefficient was not calibrated to the measured advantage scale. That was a flaw in the attempted routing fix.','',
    'Code: '+refs['KL']+'.','',
    '## Classification and limits','',
    '| Conclusion | Classification |',
    '|---|---|',
    '| Current V2-to-standalone conversion fails to preserve the successful decision function under inherited/uniform routing | Supported architectural/design weakness in this implementation |',
    '| New value estimator supplies no reliable demonstrated prediction improvement | Supported empirical weakness; confirmed missing acceptance gate |',
    '| Unit KL term mechanically opposes weak return targets and keeps updates tiny | Verified training-design limitation of the latest router update |',
    '| Missing optimizer participation, disconnected router gradients, or zero expert execution | Not identified in the tested paths |',
    '| No conditional expert advantage exists anywhere | Unresolved; not established by these tests |',
    '| Original native critic misranking alone explains the original 250-update V3 router | Unresolved; latest-fix diagnosis does not establish the original critic cause |','',
    '## Targeted next correction, not another architecture replacement','',
    'For routing: test paired MC advantages directly rather than inserting this unvalidated regression model; remove the uncalibrated additive KL penalty while retaining a measured KL constraint. '
    'Validate return targets and an independent suffix improvement before scaling training. This is a proposed correction, not a claimed EIG gain.','',
    'For the V2 gap: require the standalone expert decision conversion to preserve full-episode V2 performance before another routing campaign. '
    'Pointwise teacher-action MSE and one-hot pathway parity are insufficient acceptance criteria. The query cores alone do not need to be discarded based on current evidence. '
    'Restoring V2 fusion is a diagnostic control, not a solution to the independently executable-expert requirement.','',
    'No fix or new model training was performed in this diagnostic turn.','',
    '## Canonical frozen reference results','', '| Policy | Prior-sPCE (nats) |','|---|---:|']
for k,v in c['means'].items():lines.append(f'| {k} | {v:.8f} |')
lines+=['','All six saved historical reference score vectors reproduced exactly in the canonical evaluation. Training budgets are not matched, '
    'the 1M estimator is ceiling limited, and this is one training lineage. '
    'Value diagnostics use 16 audit histories per round; their uncertainty and approximate posterior labels prevent broad conditional-specialization conclusions.','',
    f'Raw diagnostics: `{out}` and `{fact}`.','']
(dest/'ROOT_CAUSE.md').write_text('\n'.join(lines));(dest/'SUMMARY.json').write_text(json.dumps(dict(frozen=r,factorial=f,canonical=c,code_locations=refs),indent=2)+'\n')
(dest/'VERIFICATION.json').write_text(json.dumps(dict(protocol=r['protocol'],training_updates=0,
    main_results_sha256=sha(out/'RESULTS.json'),corrected_results_sha256=sha(out/'CORRECTED_RESULTS.json'),factorial_results_sha256=sha(fact/'RESULTS.json'),
    manifest_sha256=sha(out/'MANIFEST.json'),factorial_manifest_sha256=sha(fact/'MANIFEST.json'),
    bridge_parity=json.loads((out/'BRIDGE_PARITY.json').read_text()),factorial_reproduction_max_error=f['reproduction_max_error'],
    historical_artifacts_preserved=True),indent=2)+'\n')
print(dest/'ROOT_CAUSE.md')
