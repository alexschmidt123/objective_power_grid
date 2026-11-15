"""Publish small pilot reports; raw datasets and checkpoints stay outside Git."""
import argparse,json,pathlib
from experiment import ROOT,WORK,n
p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);args=p.parse_args()
out=args.output.expanduser();assert ROOT/'scratch_runs' in out.parents
r=json.loads((out/'RESULTS.json').read_text());assert r['status']=='COMPLETE' and not r['smoke']
dest=WORK/'reports/hec_v3_counterfactual_router_20261114';dest.mkdir(exist_ok=True,parents=True)
ev=r['evaluation'];lines=['# Frozen-expert V3 counterfactual router pilot','',
    'Completed router-only pilot. Experts and context remained unchanged; historical artifact hashes passed. '
    'This is one training lineage and does not establish multi-seed superiority.','',
    '| Policy | Prior-sPCE (nats) |','|---|---:|']
for k,v in ev['means'].items():lines.append(f'| {k} | {v:.8f} |')
lines+=['','Evaluation: 1024 paired trajectories,16 million shared contrasts. This is a fresh bank; do not compare these absolute means directly with old banks.','',
    '| Counterfactual minus comparator | Difference | 95% paired CI | Holm p |',
    '|---|---:|---|---:|']
for k,v in ev['comparisons'].items():
    lines.append(f"| {k} | {v['mean']:+.8f} | [{v['CI95'][0]:+.8f}, {v['CI95'][1]:+.8f}] | {v['p_Holm']:.6g} |")
lines+=['',f"All-comparator pilot acceptance: **{ev['accepted']}**.",'',
    '## What changed','',
    'The counterfactual arm trained routing on a fitted ensemble of paired native variational suffix-return advantages. '
    'The control retained frozen original critic advantages. Both collected their own current-policy histories and spent '
    'equal suffix-simulation and estimator-fitting budgets. The control is not a jointly updated native critic/actor baseline. '
    'Neither arm retrained experts, posterior networks or the critic.','',
    '| Arm/round | Accepted router steps | KL | Audit selected minus current [95% CI] |',
    '|---|---:|---:|---|']
for row in r['rounds']:
    v=row['ranking_audit']['selected_minus_current'];up=row['update']
    lines.append(f"| {row['arm']}/{row['round']} | {up['steps']} | {up['KL']:.6f} | {v['mean']:+.6f} [{v['CI95'][0]:+.6f}, {v['CI95'][1]:+.6f}] |")
lines+=['','The audit table tests ensemble-selected actions against the pre-update mixture using16 untouched histories per round '
    'and128 independent suffix draws each. These exploratory intervals are not corrected across rounds and do not establish a root cause.','',
    '## Costs and limitations','',
    f"Runtime: {r['seconds']:.1f} seconds. Training/history/suffix simulator-observation counts: `{r['simulator_observations']}`. "
    'Posterior importance likelihood calculations and final evaluation are additional computation. '
    'Each arm may accept fewer updates because of the same KL stopping rule.','',
    'Counterfactual labels use an approximate importance posterior with ESS/max-weight checks; this is not exact posterior sampling. '
    'Training optimizes the unchanged native variational information return, while final evaluation uses finite-contrast prior-sPCE. '
    'Small history coverage, finite suffix draws, learned value approximation and one training lineage limit conclusions. '
    'No final-bank tuning, deployment-time optimization or expert retraining was performed.','',
    f'Raw authoritative artifacts: `{out}`.','']
diagnostic=dest/'VALUE_DIAGNOSTICS.json'
if diagnostic.exists():
    values=json.loads(diagnostic.read_text())
    lines+=['## Heldout value prediction','',
        '| Counterfactual round | Ensemble advantage MSE | Zero-advantage MSE | Original critic MSE |',
        '|---|---:|---:|---:|']
    for row in values['rounds']:
        if row['round'].startswith('counterfactual'):
            lines.append(f"| {row['round']} | {row['ensemble_advantage_MSE']:.8f} | {row['zero_advantage_MSE']:.8f} | {row['critic_advantage_MSE']:.8f} |")
    lines+=['','These16-history independent-suffix checks show whether the fitted estimator predicts advantages beyond a zero baseline. '
        'They do not establish that conditional routing headroom is absent. Each value network has91,588 parameters, '
        'with only64 fitting histories per round; validation early stopping limits fitting but cannot establish generalization. '
        'Posterior quality and empirical per-history selection diagnostics are in VALUE_DIAGNOSTICS.json.','']
(dest/'RESULTS.md').write_text('\n'.join(lines));(dest/'SUMMARY.json').write_text(json.dumps(r,indent=2)+'\n')
protected=json.loads((out/'PRE_RUN_MANIFEST.json').read_text())
assert all(n.sha(path)==h for path,h in protected.items())
manifest=dict(experiment_source_commit=json.loads((out/'CONFIG.json').read_text())['source_commit'],
    raw_directory=str(out),historical_and_source_hashes_verified=protected,
    new_checkpoint_hashes={str(path.relative_to(out)):n.sha(path) for path in sorted(out.rglob('*.pt'))},
    config_sha256=n.sha(out/'CONFIG.json'),results_sha256=n.sha(out/'RESULTS.json'))
(dest/'ARTIFACTS.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(dest/'RESULTS.md')
