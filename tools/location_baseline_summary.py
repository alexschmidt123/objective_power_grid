"""Aggregate completed, hash-matched location baseline cells and show missing seeds."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

METHODS = ('fixed','dad','rl_sboed','vsoed','scorebed','random')


def table(path, rows, fields):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path)
    args=ap.parse_args()
    root=args.directory
    rows=[]
    vectors={}
    identity=None
    stats=[]
    for method,seed in itertools.product(METHODS,range(101,106)):
        row=dict(method=method,seed=seed,status='missing',eig_nats=None,case_mcse=None,path=None)
        pointer=root/f'{method}-{seed}.json'
        if pointer.exists():
            out=Path(json.loads(pointer.read_text())['directory'])
            row.update(path=str(out),status='evaluation_incomplete')
            p=out/'eval/primary.json'
            if p.exists():
                data=json.loads(p.read_text())
                scorefile=p.with_name('scores.npy')
                key=(data['bank_view_sha256'],data['protocol']['scorer_sha256'],data['config_sha256'],data['horizon'])
                if identity is not None and key!=identity:raise ValueError('Mixed evaluation protocols')
                identity=key
                vector=np.load(scorefile)
                if (data['method'],data['seed'])!=(method,seed) or vector.shape!=(2000,) or not np.isfinite(vector).all() or hashlib.sha256(scorefile.read_bytes()).hexdigest()!=data['score_vector_sha256']:
                    raise ValueError(f'Invalid scores in {out}')
                vectors[method,seed]=vector
                row.update(status='complete',eig_nats=vector.mean(),case_mcse=vector.std(ddof=1)/np.sqrt(len(vector)))
        rows.append(row)
    for method in METHODS:
        x=[row['eig_nats'] for row in rows if row['method']==method and row['status']=='complete']
        stats.append(dict(method=method,completed=len(x),mean_nats=np.mean(x) if x else None,seed_sd=np.std(x,ddof=1) if len(x)>1 else None))
    contrasts=[]
    for a,b in itertools.combinations(METHODS,2):
        ds=[vectors[a,s]-vectors[b,s] for s in range(101,106) if (a,s) in vectors and (b,s) in vectors]
        if len(ds)!=5:continue
        d=np.stack(ds)
        mean=float(d.mean())
        half=t.ppf(.975,4)*d.mean(1).std(ddof=1)/np.sqrt(5)
        contrasts.append(dict(method=a,reference=b,difference=mean,paired_seed_ci_low=mean-half,
                              paired_seed_ci_high=mean+half,paired_case_mcse=d.mean(0).std(ddof=1)/np.sqrt(d.shape[1])))
    table(root/'cells.csv',rows,list(rows[0]))
    table(root/'methods.csv',stats,list(stats[0]))
    table(root/'paired.csv',contrasts,['method','reference','difference','paired_seed_ci_low','paired_seed_ci_high','paired_case_mcse'])
    print(f'{len(vectors)}/30 evaluated. Random seeds represent policy randomness.',flush=True)


if __name__=='__main__':
    main()
