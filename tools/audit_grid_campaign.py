#!/usr/bin/env python3
"""Independent IEEE14/30 duration screening and fresh finite-loss validation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse
from dataclasses import asdict
import hashlib,json,time
import numpy as np
from src.config import load_config_for_run
from src.domains.swing.design import Design,build_simulator
from src.banks.power_grid import _sample_power_grid_prior
from src.observations.compress import obs_indices_for_n_obs
from tools.audit_ieee9_campaign import run_policy,save,fresh
from tools.audits.space import AuditBudget


def config(path):
    cfg=load_config_for_run(path,ROOT,step_number=5)
    obs=cfg.raw['observation'];obs.update(obs['objective_based'])
    c=cfg.raw['control'];tr=cfg.training_for('objective_based')
    if (c['robust_rule']!='quantile' or c['alpha']!=.05 or tr['undercontrol_penalty']!=20
        or tr['violation_penalty']!=0 or c.get('safety_margin',0)!=0 or not c.get('snap_up',True)):
        raise ValueError('Require finite-loss 95% quantile protocol')
    return cfg


def prepare(args,cfg):
    from src.domains.swing.cuda import CudaTrajectoryEngine
    from src.control.cuda_control import CudaControlEngine
    from src.control.u_req import ControlSpec
    from tools.audits.validation import batch_control_oracle
    from src.control.oracle_u_ctrl import compute_u_ctrl_opt
    out=args.campaign
    if (out/'campaign.json').exists():raise ValueError('Do not overwrite an existing audit')
    base=list(map(float,cfg.swing['probe_durations']))
    durations=sorted(set(base+[.25,.4,.6,.8,1.,1.25,1.5,1.75,2.,2.25,2.5,2.75]))
    baseline=tuple(durations.index(d) for d in base)
    rng=np.random.default_rng(9071430)
    candidates=[baseline]
    target=2 if args.quick else 16
    while len(candidates)<target:
        key=tuple(sorted(rng.choice(len(durations),6,replace=False).tolist()))
        if key not in candidates:candidates.append(key)
    buses=cfg.swing['probe_buses']
    designs=[Design(.05,b,d) for d in durations for b in buses]
    fit=16 if args.quick else 512;n_screen=4 if args.quick else 64
    seed=9071400 if cfg.topology=='ieee14' else 9073000
    M,K,_=_sample_power_grid_prior(cfg,fit+n_screen,np.random.default_rng(seed))
    sim=build_simulator(cfg);engine=CudaTrajectoryEngine(sim,designs)
    probe=engine.simulate_delta_f_batch(M[:1],K[:1],np.array([0]),batch_size=128)
    obs=obs_indices_for_n_obs(probe.shape[1],int(cfg.raw['observation']['N_obs']))
    curves=np.empty((len(M),len(designs),len(obs)))
    for i in range(len(M)):
        curves[i]=engine.simulate_delta_f_batch(np.repeat(M[i:i+1],len(designs),0),np.repeat(K[i:i+1],len(designs),0),np.arange(len(designs)),batch_size=128)[:,obs]
        if i%32==0:print(f'[prepare] physical systems {i}/{len(M)}',flush=True)
    if not np.isfinite(curves).all():raise ValueError('Nonfinite observations')
    spec=ControlSpec.from_cfg(cfg);ctrl=CudaControlEngine(build_simulator(cfg),spec)
    oracle=batch_control_oracle(ctrl,M,K,spec)
    if not (oracle['feasible'].all() and oracle['monotonic'].all()):
        np.savez_compressed(out/'failed_oracle.npz',M=M,K=K,**oracle)
        raise ValueError('Infeasible/nonmonotone systems; retained, never resampled')
    for i in range(2):
        ref=compute_u_ctrl_opt(ctrl,M[i],K[i],spec)
        np.testing.assert_allclose(oracle['u_opt'][i],ref['u_ctrl_opt'],atol=1e-10)
    grid=spec.u_grid();safe=oracle['coarse_safe'][:,np.searchsorted(oracle['coarse_grid'],grid)]
    U=grid[safe.argmax(1)]
    inputs=dict(curves=curves,U=U,grid=grid,M=M,K=K,obs_indices=obs,
        excluded_test_M=np.empty((0,cfg.N)),excluded_test_K=np.empty((0,cfg.N)))
    np.savez_compressed(out/'inputs.npz',**inputs)
    metadata=dict(schema='grid_duration_campaign_v1',status='screening',system=cfg.name,quick=args.quick,
        n_support=fit,candidate_count=len(candidates),durations=durations,baseline=list(baseline),
        candidates=candidates,duration_actions={str(i):list(range(i*len(buses),(i+1)*len(buses))) for i in range(len(durations))},
        designs=[list(d.as_tuple()) for d in designs],sigma=float(cfg.raw['observation']['noise_sigma']),
        fresh_systems=4 if args.quick else 256,fresh_theta_seed=seed+1,fresh_noise_seeds=[11001,11002,11003],
        horizons=[3] if args.quick else [3,4,5],screen=[],convergence=[],
        support_seed=seed,screen_systems=n_screen,alpha=.05,undercontrol_penalty=20,
        physical_machine_buses=cfg.swing['dynamic_machine_buses'],latent_dimension=2*cfg.N,
        selection='best minimum(combined,nonmyopic) screening mean plus unchanged baseline',
        claim_scope='synthetic reduced swing model; approximate planning witness, not a no-room theorem',
        existing_banks='not read or modified; independently generated support and validation',
        support_sha256=hashlib.sha256(np.c_[M,K].tobytes()).hexdigest(),
        structure_control=dict(levels=int(len(np.unique(U))),min=float(U.min()),max=float(U.max())))
    save(out/'resolved_config.json',cfg.raw);save(out/'campaign.json',metadata)
    validation=dict(curves=curves[fit:],U=U[fit:])
    budget=AuditBudget(inner=4 if args.quick else 8,outer=2 if args.quick else 4,first_candidates=6,calibration=16 if args.quick else 128,histories_per_batch=16)
    seeds=[101] if args.quick else [101,202,303]
    for i,key in enumerate(candidates):
        print(f'[screen] catalog {i+1}/{len(candidates)}',flush=True)
        metadata['screen'].append(run_policy(out,f'screen_{i}',key,3,budget,inputs,validation,metadata,seeds))
        save(out/'campaign.json',metadata)
    best=max(metadata['screen'],key=lambda r:min(r['adaptive_gain']['mean'],r['nonmyopic_gain']['mean']))['key']
    finalists=[best]
    if list(baseline) not in finalists:finalists.append(list(baseline))
    metadata['finalists']=finalists
    for i,key in enumerate(finalists):
        for tier,inner,outer in [('all_first',32,16),('strong',64,32)]:
            b=AuditBudget(inner=4 if args.quick else inner,outer=2 if args.quick else outer,first_candidates=6*len(buses),calibration=16 if args.quick else 512,fixed_restarts=2 if args.quick else 8,histories_per_batch=8)
            print(f'[convergence] finalist {i} {tier}',flush=True)
            result=run_policy(out,f'convergence_{i}_{tier}',key,3,b,inputs,validation,metadata,seeds)
            metadata['convergence'].append(dict(tier=tier,**result));save(out/'campaign.json',metadata)
    metadata['status']='finalists_frozen';save(out/'campaign.json',metadata)


def confirm(args,cfg):
    out=args.campaign;meta=json.loads((out/'campaign.json').read_text())
    manifest=json.loads((out/'validation_manifest.json').read_text())
    if manifest['finalist_file_sha256']!=hashlib.sha256((out/'campaign.json').read_bytes()).hexdigest():raise ValueError('Frozen protocol changed')
    cells=[(i,h) for i in range(len(meta['finalists'])) for h in meta['horizons']]
    if args.index>=len(cells):print('UNUSED_ARRAY_SLOT');return
    i,h=cells[args.index];q=meta['quick'];started=time.time()
    budget=AuditBudget(inner=4 if q else 64,outer=2 if q else 32,first_candidates=6*len(cfg.swing['probe_buses']),calibration=16 if q else 512,fixed_restarts=2 if q else 8,histories_per_batch=8)
    result=run_policy(out,f'fresh_{i}',meta['finalists'][i],h,budget,np.load(out/'inputs.npz'),np.load(out/'validation.npz'),meta,[11001] if q else meta['fresh_noise_seeds'],continuous=True)
    save(out/f'cell_{args.index}.json',dict(status='complete',results=[result],elapsed_seconds=time.time()-started))


def finalize(args,cfg):
    out=args.campaign;meta=json.loads((out/'campaign.json').read_text());rows=[]
    for i in range(len(meta['finalists'])*len(meta['horizons'])):
        cell=json.loads((out/f'cell_{i}.json').read_text())
        if cell['status']!='complete':raise ValueError('Incomplete cell')
        rows+=cell['results']
    save(out/'campaign_results.json',dict(status='complete',results=rows,n_fresh=meta['fresh_systems']))
    lines=[f'# {cfg.name.upper()} finite-loss MOCU audit','',f"{meta['candidate_count']} catalogs screened; {meta['fresh_systems']} fresh physical validation systems.",'Primary metric: continuous-oracle realized operational regret. Positive gains favor adaptive policies.','Adjusted intervals cover all finalist / horizon / contrast comparisons within this system.','Approximate Fixed and rolling two-step planning cannot prove no room for DAD. Monte Carlo convergence must be assessed separately.','']
    for r in rows:
        lines += [f"## Durations {r['durations_s']}, T={r['horizon']}",'']
        for k in ('myopic_adaptive_gain','adaptive_gain','nonmyopic_gain'):lines.append(f"- {k}: {r[k]['mean']:.6g}; adjusted interval {r[k]['familywise_interval']}")
        for m,s in r['methods'].items():lines.append(f"- {m}: physical safety {s['physical_safety_rate']:.4f}; mean regret {s['mean_loss'] if 'mean_loss' in s else s.get('loss_mean','see JSON')}")
        lines.append('')
    (out/'campaign_report.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','fresh','confirm','finalize'])
    p.add_argument('--config',required=True);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--index',type=int,default=0);p.add_argument('--quick',action='store_true')
    args=p.parse_args();args.campaign=args.campaign.resolve();args.campaign.mkdir(parents=True,exist_ok=True)
    cfg=config(args.config)
    {'prepare':prepare,'fresh':fresh,'confirm':confirm,'finalize':finalize}[args.stage](args,cfg)
if __name__=='__main__':main()
