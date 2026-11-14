"""Publish only a complete original-V2 protocol evaluation with reproduced baselines."""
import argparse,json,pathlib
from protocol import ROOT,WORK,LOCK,validate,sha
p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);args=p.parse_args();out=args.output.expanduser()
assert ROOT/'scratch_runs' in out.parents
r=json.loads((out/'RESULTS.json').read_text());assert r['status']=='COMPLETE';validate(r['protocol'])
assert set(r['historical_reproduction'])=={'V2','V2_uniform','vsoed','dad','rl_sboed','scorebed'}
assert all(v.get('max_abs_error',1)<1e-9 for v in r['historical_reproduction'].values())
manifest=json.loads((out/'MANIFEST.json').read_text());assert all(sha(path)==h for path,h in manifest.items())
dest=WORK/'reports/hec_canonical_evaluation_20261114';dest.mkdir(parents=True,exist_ok=True)
lines=['# Mandatory original-V2 evaluation: corrected V3 router comparison','',
    'This is the primary comparable result. The preceding 1,024-case/16M-contrast experiment is supplementary only and must not replace or be compared numerically with this benchmark.','',
    'All policies use the byte-identical saved V2 bank: 2,000 truth/noise cases, 1,000,000 prior contrasts, original routing uniforms, T=30, sigma=0.5, design bounds [-4,4], and the original float64 scorer (chunk 16,384, positive truth included once). '
    'No models were trained or selected by this rerun. Training budgets and checkpoint lineages remain different. Historical references are not newly promoted official benchmark results.','',
    '| Policy | Estimated EIG / prior-sPCE (nats) |','|---|---:|']
for name,value in r['means'].items():lines.append(f'| {name} | {value:.8f} |')
lines+=['','| New router minus comparator | Difference | Paired 95% CI | Holm p |','|---|---:|---|---:|']
for name,v in r['comparisons'].items():lines.append(f"| {name} | {v['mean']:+.8f} | [{v['CI95'][0]:+.8f},{v['CI95'][1]:+.8f}] | {v['p_Holm']:.6g} |")
lines+=['','## Historical reproduction checks','',
    'Each saved baseline score vector was reproduced over all 2,000 cases; acceptance tolerance is 1e-9 maximum absolute error.','',
    '| Reference | Maximum absolute score error |','|---|---:|']
for name,v in r['historical_reproduction'].items():lines.append(f"| {name} | {v['max_abs_error']:.12g} |")
lines+=['','## Protocol identity','',*['- '+key+': `'+str(value)+'`' for key,value in LOCK.items()],
    '',f"Elapsed evaluation time: {r['seconds']:.1f} seconds; training updates: 0. Historical artifacts and evaluated model states were unchanged.",
    '', 'The 1M estimator is ceiling limited and this is one training lineage. Paired intervals condition on this fixed contrast bank; they do not remove finite-contrast bias or establish multi-seed robustness. '
    'The rule prevents silent protocol substitutions; supplementary studies remain separately labeled.','',
    f'Authoritative raw artifacts: `{out}`.','',
    'Rule: `AGENTS.md` and `development/moe_sboed/hec_canonical_evaluation_20261114/RULE.md`.','']
(dest/'RESULTS.md').write_text('\n'.join(lines));(dest/'SUMMARY.json').write_text(json.dumps(r,indent=2)+'\n')
(dest/'VERIFICATION.json').write_text(json.dumps(dict(protocol=LOCK,checkpoint_sha256=r['checkpoint_sha256'],
    historical_reproduction=r['historical_reproduction'],manifest_sha256=sha(out/'MANIFEST.json'),
    config_sha256=sha(out/'CONFIG.json'),raw_results_sha256=sha(out/'RESULTS.json'),guard_tests=json.loads((out/'GUARD_TESTS.json').read_text())),indent=2)+'\n')
print(dest/'RESULTS.md')
