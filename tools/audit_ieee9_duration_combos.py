#!/usr/bin/env python3
"""Sample six-duration IEEE9 catalogs under corrected finite-loss MOCU."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import time
import numpy as np
import torch
from src.config import load_config_for_run
from src.layout import make_experiment_dir_name, write_run_config
from src.objectives.mocu.context import build_context_from_config
from src.objectives.mocu.space_audit import AuditBudget, SpacePlanner, summarize_runs
from tools.bank_sweeps.sweep_ieee9_eig_duration_sets import load_catalog, resolve_pool_actions


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def candidate_sets(durations, baseline, count, seed):
    if count < 2 or len(baseline) != 6:
        raise ValueError('Require >=2 candidates and a six-duration baseline')
    if count > math.comb(len(durations), 6):
        raise ValueError('Candidate count exceeds available combinations')
    rng = np.random.default_rng(seed)
    rows = {tuple(sorted(baseline)), tuple(map(int, np.linspace(0, len(durations)-1, 6).round()))}
    while len(rows) < count:
        rows.add(tuple(sorted(rng.choice(len(durations), 6, replace=False).tolist())))
    return sorted(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/ieee9_mocu.yaml')
    p.add_argument('--bank-dir', default='data/ieee9_duration_dense_0p01')
    p.add_argument('--candidates', type=int, default=32)
    p.add_argument('--finalists', type=int, default=3)
    p.add_argument('--search-seed', type=int, default=8675309)
    p.add_argument('--seeds', default='101,202,303')
    p.add_argument('--horizons', default='2,3,4,5')
    p.add_argument('--device', default='cuda')
    args=p.parse_args()
    horizons=sorted(set(map(int,args.horizons.split(','))))
    seeds=list(map(int,args.seeds.split(',')))
    if min(horizons)<2 or not seeds or args.finalists<1:
        p.error('Use horizons >=2, at least one seed, and positive finalist count')
    cfg=load_config_for_run(args.config, ROOT, step_number=max(horizons))
    observation=cfg.raw.setdefault('observation', {})
    observation.update(observation.get('objective_based', {}))
    control=cfg.raw.get('control') or {}; training=cfg.training_for('objective_based')
    if (control.get('robust_rule')!='quantile' or control.get('alpha')!=.05 or
        training.get('undercontrol_penalty')!=20 or training.get('violation_penalty')!=0 or
        control.get('safety_margin',0)!=0 or not control.get('snap_up',True)):
        p.error('Require finite-loss 95% quantile, penalty20, margin/event penalty0, snap-up')
    cfg.raw['experiment']['allow_trivial_fixed']=True
    folder=make_experiment_dir_name('ieee9_corrected_duration_combo_audit','objective_based',
        max(horizons),n_obs=int(observation['N_obs']),noise_sigma=float(observation['noise_sigma']))
    out=ROOT/'experiments'/folder
    ctx=build_context_from_config(cfg,project_root=ROOT,out_dir=out,smoke=False,experiment_type='objective_based')
    dense=(ROOT/args.bank_dir).resolve()
    catalog=load_catalog(dense); durations=catalog.durations
    current=load_catalog(ctx.data_dir)
    baseline=tuple(durations.index(d) for d in current.durations)
    candidates=candidate_sets(durations,baseline,args.candidates,args.search_seed)
    systems=ctx.train_systems+ctx.validation_systems
    M=np.asarray([s['M'] for s in systems]); K=np.asarray([s['K'] for s in systems])
    # Never take U from the old dense bank: context supplies the verified
    # production control extension. Exact theta identities join the banks.
    for name, expected in [('theta_M',M),('theta_K',K)]:
        actual=np.load(dense/'train'/f'{name}.npy')
        if not np.array_equal(actual,expected):
            raise ValueError(f'Dense/control source theta mismatch: {name}')
    if len(ctx.U_support)!=len(ctx.train_systems):
        raise ValueError('Require the complete production fit support')
    np.testing.assert_array_equal(ctx.M_support,M[:len(ctx.U_support)].mean(axis=1))
    np.testing.assert_array_equal(ctx.K_support,K[:len(ctx.U_support)].mean(axis=1))
    full=np.load(dense/'train'/'delta_f.npy',mmap_mode='r')
    print('[duration] reading dense observation coordinates',flush=True)
    curves=np.asarray(full[:,:,ctx.obs_indices],dtype=np.float64)
    if not np.isfinite(curves).all(): raise ValueError('Nonfinite dense observations')
    # Also verify production observations, not merely latent-row identity.
    def action_key(row): return (round(row[0],10),int(row[1]),round(row[2],10))
    lookup={action_key(row):i for i,row in enumerate(catalog.designs)}
    existing=np.array([lookup[action_key(row)] for row in current.designs])
    np.testing.assert_allclose(curves[:,existing,:],np.stack([s['obs_clean'] for s in systems]),rtol=1e-6,atol=1e-10)
    np.testing.assert_allclose(curves[:len(ctx.U_support),existing,:].transpose(1,0,2),ctx.centres_support,rtol=1e-6,atol=1e-10)
    fit=len(ctx.U_support); half=len(ctx.validation_systems)//2
    if half<32: raise ValueError('Need >=64 off-support validation systems')
    U=np.array([s['u_req'] for s in systems])
    screen_budget=AuditBudget(inner=8,outer=4,first_candidates=6,calibration=128,histories_per_batch=32)
    confirm_budget=AuditBudget(inner=32,outer=16,first_candidates=12,calibration=256,histories_per_batch=16)
    doc={'schema':'finite_loss_duration_combo_v1','status':'running','config':args.config,
        'objective':'candidate-grid bank regret; quantile .95; undercontrol penalty20',
        'continuous_oracle':False,'physical_safety_certified':False,'search_seed':args.search_seed,
        'duration_options':list(durations),'choose':6,'total_combinations':math.comb(len(durations),6),
        'exhaustive':False,'candidates':[list(k) for k in candidates],'baseline':list(baseline),
        'horizons':horizons,'seeds':seeds,'screen_horizon':3,'screen_rows':[fit,fit+half],
        'confirm_rows':[fit+half,len(U)],'n_support':fit,'control_grid':ctx.u_grid.tolist(),
        'noise_sigma':ctx.sigma_y,'observation_indices':ctx.obs_indices.tolist(),
        'dense_observations_sha256':digest(curves),'theta_sha256':digest(np.c_[M,K]),
        'control_values_sha256':digest(U),'screen_budget':asdict(screen_budget),
        'confirm_budget':asdict(confirm_budget),'screen':[],'confirmation':[],
        'limitations':['Approximate Fixed and rolling two-step planner; no no-room theorem.',
            'Validation reused during earlier development; exploratory, not fresh confirmation.',
            'Final test observations excluded. Continuous-oracle validation still required.']}
    write_run_config(out,cfg,ctx.data_dir,experiment_type='objective_based',extra={'duration_audit':doc,'methods':[]})
    started=time.perf_counter()
    def save():
        (out/'duration_summary.json').write_text(json.dumps(doc,indent=2)+'\n')
    save()
    def evaluate(key,phase,horizon,budget):
        ids,_=resolve_pool_actions(catalog,[durations[i] for i in key])
        planner=SpacePlanner(curves[:fit,ids,:].transpose(1,0,2),ctx.U_support,ctx.u_grid,
            sigma=ctx.sigma_y,device=args.device,budget=budget)
        risk=float(planner.risk(planner.prior())[0][0])
        fixed,calibration=planner.fixed_sequence(horizon,seed=104729+horizon)
        begin,end=(fit,fit+half) if phase=='screen' else (fit+half,len(U))
        runs=[]
        stem='_'.join(map(str,key))
        for seed in seeds:
            run=planner.evaluate(curves[begin:end,ids,:],U[begin:end],horizon,fixed,seed)
            arrays={f'{method}_{k}':v for method,row in run.items() for k,v in row.items() if isinstance(v,np.ndarray)}
            np.savez_compressed(out/f'{phase}_{stem}_T{horizon}_s{seed}.npz',**arrays)
            runs.append(run)
        summary=summarize_runs(runs,risk,budget)
        summary.update(indices=list(key),durations_s=[durations[i] for i in key],horizon=horizon,
            fixed_sequence=fixed,fixed_calibration_loss=calibration,prior_bayes_risk=risk)
        return summary,runs
    for i,key in enumerate(candidates,1):
        print(f'[duration] screen {i}/{len(candidates)}: {[durations[j] for j in key]}',flush=True)
        summary,_=evaluate(key,'screen',3,screen_budget)
        doc['screen'].append(summary); save()
    ranked=sorted(doc['screen'],key=lambda row:min(row['adaptive_gain']['mean'],row['nonmyopic_gain']['mean']),reverse=True)
    finalists=[tuple(row['indices']) for row in ranked[:args.finalists]]
    if baseline not in finalists: finalists.append(baseline)
    doc['frozen_finalists']=[list(k) for k in finalists]; save()
    # Freeze all finalists before touching the second validation half.
    family_size=len(finalists)*len(horizons)*3
    for key in finalists:
        for horizon in horizons:
            print(f'[duration] confirm {[durations[j] for j in key]} T={horizon}',flush=True)
            summary,runs=evaluate(key,'confirm',horizon,confirm_budget)
            for metric,base,method in [('adaptive_gain','fixed','lookahead'),
                ('myopic_adaptive_gain','fixed','myopic'),('nonmyopic_gain','myopic','lookahead')]:
                values=np.mean([r[base]['loss']-r[method]['loss'] for r in runs],axis=0)
                rng=np.random.default_rng(493)
                means=values[rng.integers(0,len(values),(20000,len(values)))].mean(1)
                summary[metric]['familywise_interval']=np.quantile(means,[.025/family_size,1-.025/family_size]).tolist()
                summary[metric]['family_size']=family_size
            summary['joint_positive_signal']=all(summary[k]['familywise_interval'][0]>0 for k in ('adaptive_gain','nonmyopic_gain'))
            doc['confirmation'].append(summary); save()
    doc.update(status='complete',elapsed_seconds=time.perf_counter()-started); save()
    lines=['# Corrected IEEE9 duration-combination audit','',
        f'Screened {len(candidates)} of {doc["total_combinations"]:,} six-duration combinations.',
        'Finite-loss 95% quantile; candidate-grid bank regret; exploratory reused validation.',
        'Positive gains favor lookahead. Intervals use theta-cluster bootstrap with Bonferroni tail adjustment across all reported confirmation contrasts.', '']
    for row in doc['confirmation']:
        lines.append(f'## Durations {row["durations_s"]}; T={row["horizon"]}')
        lines.append('')
        for metric in ('myopic_adaptive_gain','adaptive_gain','nonmyopic_gain'):
            g=row[metric]; lines.append(f'- {metric}: {g["mean"]:.6f}; adjusted interval {g["familywise_interval"]}.')
        lines.append('')
    lines+=doc['limitations']
    (out/'duration_report.md').write_text('\n'.join(lines)+'\n')
    print('DURATION_REPORT='+str(out/'duration_report.md'),flush=True)


if __name__=='__main__': main()
