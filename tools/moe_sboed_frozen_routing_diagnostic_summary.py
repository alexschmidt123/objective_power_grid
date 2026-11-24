"""Read-only statistical summary of frozen routing diagnosis; no GPU operations."""
from pathlib import Path
import json, math
import numpy as np
from scipy.stats import t

M = Path.home()/'Documents/objective_power_grid'
O = M/'experiments/location_finding/moe_sboed_uniform_residual_seed101_20261116/frozen_diagnostic/full'
R = M/'reports/moe_sboed_location_benchmark'


def stats(values):
    a = np.asarray(values, dtype=float)
    se = float(a.std(ddof=1)/len(a)**.5)
    q = float(t.ppf(.975, len(a)-1))
    return dict(n=len(a), mean=float(a.mean()), SE=se,
                CI95=[float(a.mean()-q*se), float(a.mean()+q*se)])


def main():
    d = json.loads((O/'RESULTS.json').read_text())
    assert d['status'] == 'COMPLETE' and d['training_launched'] is False
    rows = d['rows']
    stages = [s for s in d['protocol']['stages'] if s > 0]
    complete = [h for h in range(d['protocol']['histories_per_stage'])
                if all(any(r['stage']==s and r['history']==h and r['posterior']['adequate'] for r in rows) for s in stages)]
    keys = ['discovery_selected_validation_gain', 'critic_selected_validation_gain', 'transferred_router_validation_gain']
    clustered = {key: stats([np.mean([r[key]['mean'] for r in rows if r['history']==h and r['stage']>0]) for h in complete]) for key in keys}
    # Guard against inspecting many history/weight cells: report global family correction.
    family = sum(len(r['variants'])-1 for r in rows)
    global_positive = []
    for r in rows:
        if not r['posterior']['adequate']:
            continue
        for label, v in r['variants'].items():
            st = v['validation_paired_gain']
            if label=='uniform' or not st['SE']:
                continue
            lo = st['mean']-float(t.ppf(1-.025/family, st['n']-1))*st['SE']
            if lo > 0:
                global_positive.append(dict(stage=r['stage'], history=r['history'], variant=label, simultaneous_lower=lo))
    correlations = [r['critic_rank_spearman_validation'] for r in rows
                    if r['posterior']['adequate'] and r['critic_rank_spearman_validation'] is not None]
    result = dict(status='COMPLETE', training=False, seed=101, elapsed_seconds=d['seconds'],
        frozen_parent='MoE-sBOED-uniform seed101', histories=len(rows),
        adequate_posterior_histories=d['adequate_posteriors'],
        complete_trajectory_clusters=complete, clustered_validation_gains=clustered,
        individual_history_family_size=family, globally_significant_positive_cells=global_positive,
        critic_rank_correlation_mean=float(np.mean(correlations)),
        critic_rank_correlation_median=float(np.median(correlations)),
        action_spread=dict(minimum=float(min(r['action_spread_max'] for r in rows)),
                          median=float(np.median([r['action_spread_max'] for r in rows])),
                          maximum=float(max(r['action_spread_max'] for r in rows))),
        stage_stratified_summary=d['stage_summaries'],
        resources=json.loads((O/'RESOURCES.json').read_text()),
        conclusions=dict(structural_weight_bypass='Not supported:actions change under fixed-history gate interventions.',
            useful_routing_headroom='Not demonstrated by held-out selected-weight gains;does not establish zero headroom.',
            critic_credit_assignment='Observed rankings weak/noisy;credible hypothesis,not a confirmed causal failure.',
            router_optimization='Not established:no router learning performed in this frozen diagnostic.',
            expert_complementarity='Cannot infer insufficient complementarity merely from absence of a measurable routing gain.'),
        limitations=d['limitations']+['Only7 complete source-trajectory clusters;one late-stage posterior ESS23.69 failed64 threshold,excluding its cluster from complete-case pooled inference. Selection of complete cases can affect generalization.',
            'The exploratory raw report pools23 noninitial histories as though independent. Use this clustered summary instead;histories at three stages share original trajectories.',
            'Estimator ceiling saturation remains in some conditional cases despite1M contrasts;this can hide weight differences.',
            'Native critic estimates the variational actor-critic return,while this diagnostic scores the benchmark prior-contrast EIG estimator. Ranking disagreement can include bound/estimator mismatch;it is not automatically a critic implementation bug.',
            'No independently trained seed101 continuation pair is complete in this new trial. Earlier benchmark results remain unchanged.'])
    (R/'FROZEN_ROUTING_DIAGNOSTIC_CONCLUSIONS.json').write_text(json.dumps(result, indent=2)+'\n')
    text = '# Frozen uniform101 routing diagnosis\n\nCompleted in142.26 seconds. No model training or optimizer updates;all checkpoint files preserved.25 histories at stages1,10,20,30;14 weight choices/history;32 suffix replicates split16 discovery/16 held-out validation. Same exact likelihood,physical bounds,T30 and1M-contrast scorer;fresh diagnostic draws,not the primary benchmark bank.\n\n'
    text += '| Choice at frozen history | Held-out EIG difference from uniform | 95% CI across source-trajectory clusters |\n|---|---:|---:|\n'
    labels = ['Weights selected on discovery simulations', 'Weights selected by native critic', 'Transferred learned101 router weights']
    for label, key in zip(labels, keys):
        s = clustered[key]
        text += f"| {label} | {s['mean']:+.5f} | [{s['CI95'][0]:+.5f},{s['CI95'][1]:+.5f}] |\n"
    text += '\nIntervals use7 complete trajectory clusters,averaging stages10/20/30 within a trajectory. They are descriptive single-seed diagnostic intervals,not training-seed uncertainty or exact-posterior guarantees. Initial stage is reported separately in raw results because its history is identical across episodes.\n\n'
    text += f"Maximum action change across tested weights ranges{result['action_spread']['minimum']:.3f}–{result['action_spread']['maximum']:.3f} physical units per history (median{result['action_spread']['median']:.3f}). Gates affect final designs. Mean critic Spearman rank correlation is{result['critic_rank_correlation_mean']:.3f},median{result['critic_rank_correlation_median']:.3f};ranking measurements themselves have Monte Carlo error.\n\n"
    text += f"{len(global_positive)} positive individual history/weight cells survive a Bonferroni correction across{family} inspected nonuniform cells. Per-history-only intervals in raw rows do not protect the whole search.\n\n"
    text += '## Judgment\n\nThe current test finds action sensitivity but does not demonstrate a reliably exploitable EIG improvement over uniform. It therefore cannot distinguish limited useful reweighting headroom from noisy/inaccurate critic guidance. It does not establish poor expert complementarity,and it does not establish router optimization failure. Large harmful weight changes are not evidence of available positive routing gains.\n\nDo not promote or retrain the prepared continuation pair on the claim that this diagnostic has proved a routing fix. The next evidence-gathering step would be more held-out suffix replication at the same prespecified histories,with posterior/saturation checks. The previously requested seed101 trial implementation is saved and CPU-tested but has not started GPU training;the latest request to diagnose frozen weights takes precedence.\n\n## Limitations\n\n'
    text += '\n'.join('- '+v for v in result['limitations'])+'\n\n'
    text += '## Reproduction and artifacts\n\n`tools/moe_sboed_frozen_routing_diagnostic_seed101.py` executed frozen rollouts;`configs/moe_sboed_frozen_routing_diagnostic_seed101_20261116.json` locks the full diagnostic. Raw per-history results,paired suffix score arrays,parameter/noise draws and prefixes remain under`experiments/location_finding/moe_sboed_uniform_residual_seed101_20261116/frozen_diagnostic/full/`. Source/protocol hashes and executed commit are recorded in`frozen_diagnostic/FULL_LAUNCH.json`. Run`tools/moe_sboed_frozen_routing_diagnostic_summary.py` on the saved outputs to reproduce this clustered report without GPU computation.\n'
    (R/'FROZEN_ROUTING_DIAGNOSTIC_CONCLUSIONS.md').write_text(text)
    print(json.dumps(dict(report=str(R/'FROZEN_ROUTING_DIAGNOSTIC_CONCLUSIONS.md'), clustered=clustered, global_positive=global_positive), indent=2))


if __name__ == '__main__':
    main()
