#!/usr/bin/env python3
"""Isolated, checkpoint-preserving SIR T4 seed303 crossed reevaluation."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

METHODS={'dad_eig','rl_sboed_eig','step_dad','myopic_delta_h','fixed_open_loop','random'}
SEEDS=[1001,1002,1003,1004,1005]

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def validate(directory, seed):
    data=json.loads((directory/'eval'/'vector_eig_results.json').read_text())
    if data.get('eval_seed') != seed: raise ValueError('Evaluation seed mismatch')
    rows=data['summaries']
    if len(rows)!=6 or {r['method'] for r in rows}!=METHODS:
        raise ValueError('Expected exactly six method summaries')
    for row in rows:
        if row['eval_seed']!=seed or row['n']!=512:
            raise ValueError('Incomplete method/system coverage')
        for key in ('terminal_eig_mean','terminal_eig_std','online_seconds_per_rollout'):
            if not math.isfinite(float(row[key])): raise ValueError('Nonfinite '+key)
    if not (directory/'eval'/'terminal_eig_summary.csv').is_file():
        raise ValueError('Missing CSV summary')
    return data

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','finalize'])
    parser.add_argument('--project-root',type=Path)
    parser.add_argument('--source-run',type=Path)
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    args=parser.parse_args()
    if args.action=='prepare':
        root=args.project_root.resolve(); source=args.source_run.resolve()
        stamp=datetime.datetime.now().strftime('%m%d%Y_%H%M%S')
        run=(root/'experiments'/f'{stamp}_sir_ode_hprc_seed303_rerun_EIG_T4_Nobs1_sigma1').resolve()
        run.mkdir(exist_ok=False)
        shutil.copytree(source/'model',run/'model')
        shutil.copy2(source/'run_config.json',run/'training_run_config.json')
        for name in ('dad_eig.pth','rl_sboed_eig.pth'):
            if not (run/'model'/name).is_file(): raise ValueError('Missing '+name)
        snapshot=run/'source_snapshot'; snapshot.mkdir()
        for name in ('src','tools','scripts','configs'):
            shutil.copytree(root/name,snapshot/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for name in ('GITHUB_COMMIT','AGENTS.md','requirements.txt'):
            if (root/name).is_file(): shutil.copy2(root/name,snapshot/name)
        (snapshot/'data').symlink_to((root/'data').resolve(),target_is_directory=True)
        (snapshot/'experiments').symlink_to((root/'experiments').resolve(),target_is_directory=True)
        hashes={str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*')
                if p.is_file() and not p.is_symlink() and 'data' not in p.relative_to(snapshot).parts
                and 'experiments' not in p.relative_to(snapshot).parts}
        (run/'evaluations').mkdir()
        for seed in SEEDS:
            work=run/'workspaces'/f'{stamp}_sir_ode_eval{seed}_EIG_T4_Nobs1_sigma1'
            work.mkdir(parents=True)
            shutil.copytree(run/'model',work/'model')
        doc={'status':'prepared','training_seed':303,'evaluation_seeds':SEEDS,
             'source_run':str(source),'run_dir':str(run),'source_snapshot':str(snapshot),
             'source_sha256':hashes,'checkpoint_sha256':{n:sha(run/'model'/n) for n in ('dad_eig.pth','rl_sboed_eig.pth')},
             'retrained':False,'methods':sorted(METHODS),'n_test_systems':512}
        (run/'rerun_provenance.json').write_text(json.dumps(doc,indent=2)+'\n')
        print(run)
        return
    run=args.run_dir.resolve()
    doc=json.loads((run/'rerun_provenance.json').read_text())
    if args.action=='run':
        if args.seed is None: parser.error('--seed is required')
        seed=args.seed
        destination=run/'evaluations'/f'seed_{seed}'
        if destination.exists():
            validate(destination,seed)
            print('Already complete:',seed)
            return
        work=next((run/'workspaces').glob(f'*_eval{seed}_EIG_T4_Nobs1_sigma1'))
        for name,expected in doc['checkpoint_sha256'].items():
            if sha(work/'model'/name)!=expected: raise ValueError('Checkpoint mismatch')
        command=[sys.executable,'-m','src.experiment','evaluate','--config','configs/sir_ode.yaml',
                 '--experiment-type','eig_based','--N_obs','1','--noise_sigma','1',
                 '--seed','303','--eval-seed',str(seed),'-T','4','--exp-dir',str(work)]
        subprocess.run(command,cwd=doc['source_snapshot'],check=True)
        validate(work,seed)
        # Each array task owns one destination. Publish only validated outputs.
        staged=run/'evaluations'/f'.seed_{seed}_staging'
        staged.mkdir(exist_ok=False)
        shutil.copytree(work/'eval',staged/'eval')
        shutil.copy2(work/'run_config.json',staged/'run_config.json')
        validate(staged,seed)
        staged.rename(destination)
        print('SEED_COMPLETE',seed,flush=True)
    else:
        for seed in SEEDS: validate(run/'evaluations'/f'seed_{seed}',seed)
        shutil.copy2(run/'evaluations'/'seed_1001'/'run_config.json',run/'run_config.json')
        manifest={'protocol':'crossed_publication_protocol_v1','training_seed':303,
                  'evaluation_seeds':SEEDS,'model_scope':'copied_from_source_run',
                  'shared_models_across_run_folders':True,'source_run':doc['source_run'],
                  'rerun_of_existing_training_seed':True,'status':'complete',
                  'note':'Use this run instead of its incomplete source; it is not another training seed.'}
        (run/'evaluation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        doc['status']='complete'; (run/'rerun_provenance.json').write_text(json.dumps(doc,indent=2)+'\n')
        (run/'completion.md').write_text('Complete: training seed 303, T=4, all six methods, evaluation seeds 1001–1005, 512 systems per method and seed.\nUse this replacement rather than double-counting the source run.\n')
        print('RUN_COMPLETE',run,flush=True)

if __name__=='__main__': main()
