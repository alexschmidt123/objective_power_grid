"""Read completed research runs and make one methods-by-horizon table per metric.

First average evaluation seeds within each independent training, then summarize
across training seeds. Never count trajectories or test seeds as trainings.
This diagnostic does not launch, retrain, alter, or prune experiments.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

METRICS=['mean_spce','mean_nmc_upper_estimate','mean_mse','mean_normalized_mse',
         'mean_decision_seconds','training_seconds']


def collect(campaign):
    groups={};provenance=[]
    for config in sorted(campaign.rglob('run_config.json')):
        # Do not descend into scientific snapshots or legacy runs.
        if 'source' in config.relative_to(campaign).parts:continue
        metadata=json.loads(config.read_text())
        if metadata.get('protocol')!='moe_sboed_research_v1':continue
        run=config.parent
        if not (run/'exit_code').exists() or (run/'exit_code').read_text().strip()!='0':
            raise ValueError(f'Incomplete/failed research run: {run}')
        summaries=json.loads((run/'summary.json').read_text())
        expected={(m,s) for m in metadata['methods'] for s in metadata['eval_seeds']}
        actual={(r['method'],r['eval_seed']) for r in summaries}
        if expected!=actual or len(summaries)!=len(expected):raise ValueError(f'Missing/duplicate summaries: {run}')
        identities={}
        for row in summaries:
            path=run/'eval'/f"{row['method']}_seed{row['eval_seed']}.json"
            records=json.loads(path.read_text())
            if len(records)!=row['systems']:raise ValueError(f'Wrong trajectory count: {path}')
            identity=[(r['trajectory_id'],r['truth'],r['measurement_noise_standard']) for r in records]
            key=row['eval_seed']
            if key in identities and identities[key]!=identity:raise ValueError(f'Unpaired evaluation: {run}')
            identities[key]=identity
            if abs(statistics.mean(r['spce'] for r in records)-row['mean_spce'])>1e-9:
                raise ValueError(f'sPCE summary mismatch: {path}')
        physics=metadata['benchmark'].copy();physics.pop('_config_path',None)
        physics['experiment']={k:v for k,v in physics['experiment'].items() if k not in {'methods','step_number'}}
        # No pooling different physics, budgets, source versions or smoke modes.
        source=json.loads((run/'source_manifest.json').read_text())['sha256']
        source={k:v for k,v in source.items() if k.startswith('src/')}
        name=metadata['benchmark']['system']['name']
        # History-dependent architecture dimension legitimately changes with T.
        fingerprint_payload=dict(benchmark=physics,settings=metadata['settings'],source=source,smoke=metadata['smoke'],
                                 evaluation_seeds=metadata['eval_seeds'])
        fingerprint_payload['benchmark'].get('research',{}).pop('plan',None)
        fingerprint=hashlib.sha256(json.dumps(fingerprint_payload,sort_keys=True).encode()).hexdigest()
        for method in metadata['methods']:
            matching=[r for r in summaries if r['method']==method]
            key=(name,fingerprint,method,metadata['horizon'])
            seeds=groups.setdefault(key,{})
            train_seed=metadata['train_seed']
            if train_seed in seeds:raise ValueError(f'Duplicate independent training seed {train_seed}: {run}')
            seeds[train_seed]={metric:statistics.mean(r[metric] for r in matching) for metric in METRICS}
        provenance.append(dict(run=str(run),source_sha256=source,smoke=metadata['smoke']))
    return groups,provenance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',required=True,type=Path)
    args=p.parse_args()
    groups,provenance=collect(args.campaign.resolve())
    if not groups:raise ValueError('No completed research runs')
    output=args.campaign/'combined_results';output.mkdir(exist_ok=False)
    tables=[]
    for network,fingerprint in sorted({(key[0],key[1]) for key in groups}):
        relevant={k:v for k,v in groups.items() if k[:2]==(network,fingerprint)}
        horizons=sorted({k[3] for k in relevant});methods=sorted({k[2] for k in relevant})
        for metric in METRICS:
            path=output/f'{network}_{fingerprint[:8]}_{metric}.csv'
            with path.open('w',newline='') as handle:
                writer=csv.writer(handle);writer.writerow(['method']+[f'T={t}' for t in horizons])
                for method in methods:
                    row=[method]
                    for horizon in horizons:
                        seed_records=relevant.get((network,fingerprint,method,horizon),{})
                        values=[r[metric] for r in seed_records.values()]
                        if not values:row.append('unavailable')
                        elif len(values)==1:row.append(f'{values[0]:.6g}; SD unavailable; n_train=1')
                        else:row.append(f'{statistics.mean(values):.6g} +/- {statistics.stdev(values):.6g}; n_train={len(values)}')
                    writer.writerow(row)
            tables.append(str(path))
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(dict(tables=tables,run_count=len(provenance)),indent=2))


if __name__=='__main__':main()
