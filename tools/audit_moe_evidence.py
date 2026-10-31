"""Read saved MoE evidence; no training, inference, or checkpoint selection."""
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def paired(a, b):
    assert len(a) == len(b) == 2000
    assert all(x['trajectory_id'] == y['trajectory_id'] and
               x['truth'] == y['truth'] and
               x['measurement_noise_standard'] == y['measurement_noise_standard']
               for x, y in zip(a, b))
    d = np.array([y['spce'] - x['spce'] for x, y in zip(a, b)])
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    return dict(mean=float(d.mean()), paired_mc_se=se,
                paired_mc_95_interval=[float(d.mean()-1.96*se), float(d.mean()+1.96*se)],
                training_seed_variability='unavailable: one training seed')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign', required=True)
    args = p.parse_args()
    root = Path.cwd()
    campaign = Path(args.campaign).resolve()
    out = campaign / 'evidence_audit'
    out.mkdir(exist_ok=True)
    base = root/'experiments/10132026/10132026_bounded_location_restart/pilot'
    sources = dict(original=base, component=root/'experiments/10142026/10142026_moe_component_clipping/pilot',
                   slower=root/'experiments/10142026/10142026_moe_delayed_sharpening/pilot')
    result = dict(updated_utc=datetime.now(timezone.utc).isoformat(),
                  root_cause='unresolved pending completed architecture controls and inference sensitivity',
                  correction_selected=None, historical={}, phases={}, comparisons={}, baselines={},
                  source_hashes={}, ceiling=math.log(1025),
                  limitations=['Evaluation seed1001 already informed development.',
                               'Monte Carlo intervals are conditional on a trained policy.',
                               'Recorded historical minibatch means cannot reconstruct score distributions.',
                               'Expert utilization alone does not establish specialization.',
                               'Pair-order sensitivity alone does not establish that order causes the EIG gap.'])
    def tracked(path):
        result['source_hashes'][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return read(path)
    saved = {}
    for name, source in sources.items():
        f = source/'eval/moe_sboed_seed1001.json'
        if f.exists():
            rows = tracked(f)
            assert len(rows) == 2000
            x = np.array([r['spce'] for r in rows])
            assert np.isfinite(x).all()
            saved[name] = rows
            result['historical'][name] = dict(eig=float(x.mean()), eig_mc_se=float(x.std(ddof=1)/math.sqrt(len(x))),
                mse=float(np.mean([r['inference']['mse'] for r in rows])),
                high_mse_cases=sum(r['inference']['mse'] > 1 for r in rows), status='evaluation complete')
        else:
            result['historical'][name] = dict(status='pending')
        tf = source/'models/moe_sboed_training.json'
        if tf.exists():
            t = tracked(tf)
            result['historical'][name].update(training_seconds=t['training_seconds'],
                primary_episodes=t['generated_training_trajectories'], total_parameters=t['policy_parameters'])
        trace = source/'models/moe_sboed_diagnostics.jsonl'
        if trace.exists():
            result['source_hashes'][str(trace.relative_to(root))] = hashlib.sha256(trace.read_bytes()).hexdigest()
            history = [json.loads(x) for x in trace.read_text().splitlines()]
            phases = []
            for lo, hi in [(0, .2), (.2, .5), (.5, .8), (.8, 1.00001)]:
                rows = [r for r in history if lo <= r['primary_episodes']/1e6 < hi]
                if not rows:
                    continue
                pre = {k:np.array([r['gradient_before'][k] for r in rows]) for k in rows[0]['gradient_before']}
                phases.append(dict(fraction=[lo, hi], logged_updates=len(rows),
                    mean_training_spce=float(np.mean([r['training_eig'] for r in rows])),
                    hard_choice_frequency=np.mean([r['hard_choice_frequency'] for r in rows],axis=0).tolist(),
                    mean_global_clip_scale=float(np.mean([r['clip_scale'] for r in rows])) if rows[0]['clip_scale'] is not None else None,
                    gradients={k:dict(median=float(np.median(v)), p95=float(np.quantile(v,.95)),
                                     zero_fraction=float(np.mean(v==0))) for k,v in pre.items()}))
            result['phases'][name] = phases
    for method in ['random','fixed','dad','rl_sboed','vsoed','scorebed']:
        f = base/'eval'/f'{method}_seed1001.json'
        if f.exists():
            rows = tracked(f)
            result['baselines'][method] = dict(eig=float(np.mean([r['spce'] for r in rows])),
                mse=float(np.mean([r['inference']['mse'] for r in rows])), source=str(base.relative_to(root)), reused=True)
            if method == 'dad':
                saved['dad'] = rows
    for first, second in [('original','component'), ('component','slower'), ('dad','component'), ('dad','slower')]:
        if first in saved and second in saved:
            result['comparisons'][first+'->'+second] = paired(saved[first],saved[second])
    if 'component' in saved and 'dad' in saved:
        failed = [i for i,r in enumerate(saved['component']) if r['inference']['mse']>1]
        delta = np.array([saved['component'][i]['spce']-saved['dad'][i]['spce'] for i in failed])
        result['mse_policy_evidence'] = dict(cases=len(failed),mean_information_difference=float(delta.mean()),
            fraction_moe_lower_information=float(np.mean(delta<0)),
            conclusion='Policy information deficit supported; posterior contribution remains pending sensitivity checks.')
    diag = campaign/'checkpoint_diagnostics/diagnosis.json'
    if diag.exists():
        d = tracked(diag)
        result['checkpoint_probe_distributions'] = d.get('checkpoint_probes',{})
        result['permutation'] = d.get('permutation',{})
        result['mse_inference_sensitivity'] = d.get('mse_audit',{})
        result['diagnostics_complete'] = (campaign/'checkpoint_diagnostics/exit_code.json').exists()
    else:
        prelim = tracked(campaign/'preliminary.json')
        result['checkpoint_probe_distributions'] = prelim['fresh_checkpoint_objective_probes']
        result['probe_status'] = 'preliminary128 fresh checkpoint samples; not historical minibatches'
        result['permutation'] = prelim['permutation']
    controls = {}
    for label,method in dict(A='sum_single',B='attention_single',C='attention_dense_moe',D='attention_sparse_moe',capacity='attention_matched_dense').items():
        f = campaign/'controls/eval'/f'{method}_seed1001.json'
        if f.exists():
            rows = tracked(f)
            controls[label] = rows
    result['ablations'] = {label:dict(status='complete',eig=float(np.mean([r['spce'] for r in rows]))) for label,rows in controls.items()}
    for first,second in [('A','B'),('B','C'),('C','D'),('capacity','C')]:
        if first in controls and second in controls:
            result['comparisons'][first+'->'+second] = paired(controls[first],controls[second])
    result['complete_controls'] = len(controls)==5 and (campaign/'pipeline_exit_code').exists() and (campaign/'pipeline_exit_code').read_text().strip()=='0'
    (out/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# MoE evidence audit','',result['root_cause'],'',
           '| Checkpoint | EIG | MSE | Training seconds |','|---|---:|---:|---:|']
    for name,r in result['historical'].items():
        lines.append(f"| {name} | {r.get('eig','pending')} | {r.get('mse','pending')} | {r.get('training_seconds','pending')} |")
    lines += ['', '| Architecture | EIG |','|---|---:|']
    for label in ['A','B','C','D','capacity']:
        lines.append(f"| {label} | {result['ablations'].get(label,{}).get('eig','pending')} |")
    lines += ['', 'Correction: pending completed controlled evidence. No sparse-MoE advantage established.', '',
              'Historical score distributions were not logged; fresh checkpoint probes are explicitly distinguished.', '',
              'The paired failure analysis supports a policy weakness. Larger-budget inference results are required to assess the secondary MSE contribution.', '',
              'All comparisons use the same existing evaluator, unchanged L and episode budgets.']
    (out/'audit.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(complete_controls=result['complete_controls'],historical=result['historical'],comparisons=result['comparisons']),indent=2))


if __name__ == '__main__':
    main()
