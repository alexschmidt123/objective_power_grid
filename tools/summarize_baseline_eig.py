"""Report Random/Myopic EIG mean +/- sample SD across evaluation-seed means.

Read one completed run, never replicate baselines across training seeds.
The five-seed conference protocol is checked explicitly; historical partial
results require --allow-incomplete and retain their actual seed count.
"""
import argparse
import json
import math
from pathlib import Path
import statistics


def summarize(rows, method, expected_seeds=range(1001, 1006), allow_incomplete=False):
    if method not in ('random', 'myopic'):
        raise ValueError('This reporter is for untrained Random/Myopic baselines')
    groups, seen = {}, set()
    for row in rows:
        if row['method'] != method:
            continue
        seed = row['evaluation_seed']
        key = (seed, row['system'])
        if key in seen:
            raise ValueError(f'Duplicate evaluation episode: {key}')
        seen.add(key)
        value = float(row['terminal_spce_nats'])
        if not math.isfinite(value):
            raise ValueError('Nonfinite EIG')
        groups.setdefault(seed, []).append(value)
    if not groups:
        raise ValueError(f'No evaluations for {method}')
    expected = set(expected_seeds)
    missing, unexpected = sorted(expected - groups.keys()), sorted(groups.keys() - expected)
    if unexpected or (missing and not allow_incomplete):
        raise ValueError(f'{method}: missing seeds {missing}; unexpected seeds {unexpected}')
    if len({len(v) for v in groups.values()}) != 1:
        raise ValueError('Evaluation seeds have unequal episode counts')
    means = {seed: statistics.mean(values) for seed, values in sorted(groups.items())}
    mean = statistics.mean(means.values())
    sd = statistics.stdev(means.values()) if len(means) > 1 else None
    return dict(method=method, mean_spce_nats=mean, sd_of_seed_means=sd,
                evaluation_seed_means=means, evaluation_seeds=sorted(groups),
                n_evaluation_seeds=len(groups), systems_per_seed=len(next(iter(groups.values()))),
                missing_evaluation_seeds=missing, complete=not missing,
                variability='sample SD across evaluation-seed means; not training variability',
                display=f'{mean:.4f} +/- {sd:.4f}' if sd is not None else f'{mean:.4f} (SD unavailable)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--expected-seeds', default='1001,1002,1003,1004,1005')
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    if (args.run / 'exit_code').read_text().strip() != '0':
        raise ValueError('Run has not completed successfully')
    rows = json.loads((args.run / 'rollouts.json').read_text())
    results = [summarize(rows, method, map(int, args.expected_seeds.split(',')),
                         args.allow_incomplete) for method in ('random', 'myopic')]
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
