#!/usr/bin/env python3
"""Planning convergence, 128 duration sets, and fresh physical validation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse
from dataclasses import asdict
import json
import time
import numpy as np
import torch
from src.config import load_config_for_run
from src.layout import write_run_config
from src.objectives.mocu.context import build_context_from_config
from tools.audits.space import AuditBudget, SpacePlanner, summarize_runs
from tools.audit_ieee9_duration_combos import candidate_sets, digest
from tools.bank_sweeps.sweep_ieee9_eig_duration_sets import load_catalog, resolve_pool_actions

SEEDS=[101,202,303]
FRESH_SEEDS=[11001,11002,11003]

def save(path,data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n'); temp.replace(path)

def config():
    cfg=load_config_for_run('configs/ieee9_mocu.yaml',ROOT,step_number=5)
    obs=cfg.raw.setdefault('observation',{}); obs.update(obs.get('objective_based',{}))
    tr=cfg.training_for('objective_based'); c=cfg.raw['control']
    if (c['robust_rule']!='quantile' or c['alpha']!=.05 or tr['undercontrol_penalty']!=20
        or tr['violation_penalty']!=0 or c.get('safety_margin',0)!=0 or not c.get('snap_up',True)):
        raise ValueError('Require aligned finite-loss 95% protocol')
    cfg.raw['experiment']['allow_trivial_fixed']=True
    return cfg

def run_policy(campaign,tag,key,horizon,budget,inputs,validation,metadata,seeds,continuous=False):
    ids=np.concatenate([np.array(metadata['duration_actions'][str(k)],dtype=int) for k in key])
    planner=SpacePlanner(inputs['curves'][:metadata['n_support'],ids,:].transpose(1,0,2),
        inputs['U'][:metadata['n_support']],inputs['grid'],sigma=metadata['sigma'],device='cuda',budget=budget)
    prior=float(planner.risk(planner.prior())[0][0])
    fixed,cal=planner.fixed_sequence(horizon,104729+horizon)
    if continuous:
        lookup={int(a):i for i,a in enumerate(validation['action_ids'])}
        local=np.array([lookup[int(a)] for a in ids])
        centres=validation['curves'][:,local,:]
    else: centres=validation['curves'][:,ids,:]
    runs=[]
    for seed in seeds:
        row=planner.evaluate(centres,validation['U'],horizon,fixed,seed)
        if continuous:
            for method,r in row.items():
                r['bank_loss']=r['loss'].copy()
                u=r['control']; opt=validation['u_opt']
                r['loss']=u+20*np.maximum(opt-u,0)-opt
                where=np.searchsorted(inputs['grid'],u)
                np.testing.assert_allclose(inputs['grid'][where],u,atol=1e-12)
                r['physical_safe']=validation['candidate_safe'][np.arange(len(u)),where]
        arrays={f'{m}_{k}':v for m,r in row.items() for k,v in r.items() if isinstance(v,np.ndarray)}
        np.savez_compressed(campaign/f'{tag}_T{horizon}_s{seed}.npz',**arrays)
        runs.append(row)
    result=summarize_runs(runs,prior,budget)
    result.update(key=list(key),durations_s=[metadata['durations'][k] for k in key],horizon=horizon,
        budget=asdict(budget),fixed_sequence=fixed,fixed_calibration_loss=cal)
    if continuous:
        result['metric']='continuous_oracle_realized_operational_regret'
        for m in result['methods']:
            result['methods'][m]['mean_bank_loss']=float(np.mean([r[m]['bank_loss'] for r in runs]))
            result['methods'][m]['physical_safety_rate']=float(np.mean([r[m]['physical_safe'] for r in runs]))
        family=3*len(metadata['finalists'])*len(metadata['horizons'])
        for metric,base,method in [('adaptive_gain','fixed','lookahead'),
            ('myopic_adaptive_gain','fixed','myopic'),('nonmyopic_gain','myopic','lookahead')]:
            values=np.mean([r[base]['loss']-r[method]['loss'] for r in runs],axis=0)
            rng=np.random.default_rng(493)
            # Generate in chunks to bound bootstrap memory independently of n_theta.
            means=np.concatenate([values[rng.integers(0,len(values),(1000,len(values)))].mean(1) for _ in range(20)])
            result[metric]['familywise_interval']=np.quantile(means,[.025/family,1-.025/family]).tolist()
            result[metric]['family_size']=family
        result['joint_positive_signal']=all(result[k]['familywise_interval'][0]>0 for k in ('adaptive_gain','nonmyopic_gain'))
    return result

def prepare(args):
    campaign=args.campaign; cfg=config()
    ctx=build_context_from_config(cfg,project_root=ROOT,out_dir=campaign,smoke=False,experiment_type='objective_based')
    dense=ROOT/'data/ieee9_duration_dense_0p01'; catalog=load_catalog(dense)
    durations=list(catalog.durations); current=load_catalog(ctx.data_dir)
    baseline=tuple(durations.index(d) for d in current.durations)
    systems=ctx.train_systems+ctx.validation_systems
    M=np.array([s['M'] for s in systems]); K=np.array([s['K'] for s in systems])
    for name,expected in [('theta_M',M),('theta_K',K)]:
        np.testing.assert_array_equal(np.load(dense/'train'/f'{name}.npy'),expected)
    if len(ctx.train_systems)!=len(ctx.U_support): raise ValueError('Require full production fit support')
    print('[campaign] loading dense observation coordinates',flush=True)
    full=np.load(dense/'train/delta_f.npy',mmap_mode='r')
    curves=np.asarray(full[:,:,ctx.obs_indices],dtype=float)
    def key(row): return (round(row[0],10),int(row[1]),round(row[2],10))
    lookup={key(row):i for i,row in enumerate(catalog.designs)}
    active=np.array([lookup[key(row)] for row in current.designs])
    np.testing.assert_allclose(curves[:,active,:],np.stack([s['obs_clean'] for s in systems]),rtol=1e-6,atol=1e-10)
    fit=len(ctx.U_support)
    np.testing.assert_allclose(curves[:fit,active,:].transpose(1,0,2),ctx.centres_support,rtol=1e-6,atol=1e-10)
    U=np.array([s['u_req'] for s in systems]); np.testing.assert_array_equal(U[:fit],ctx.U_support)
    previous=json.loads(args.previous.read_text())
    old=[tuple(k) for k in previous['frozen_finalists']]
    inputs={'curves':curves,'U':U,'grid':ctx.u_grid,'M':M,'K':K,'obs_indices':ctx.obs_indices,
            'excluded_test_M':np.array([s['M'] for s in ctx.test_systems]),
            'excluded_test_K':np.array([s['K'] for s in ctx.test_systems])}
    np.savez_compressed(campaign/'inputs.npz',**inputs)
    count=2 if args.quick else 128
    candidates=candidate_sets(durations,baseline,count,8675309)
    metadata={'schema':'strong_duration_campaign_v1','status':'preparing','quick':args.quick,
        'candidate_count':len(candidates),'durations':durations,'baseline':list(baseline),
        'old_finalists':[list(k) for k in old],'candidates':[list(k) for k in candidates],
        'duration_actions':{str(i):resolve_pool_actions(catalog,[d])[0].tolist() for i,d in enumerate(durations)},
        'designs':[list(d) for d in catalog.designs],'sigma':ctx.sigma_y,'n_support':fit,
        'old_screen_rows':[fit,fit+64],'fresh_systems':4 if args.quick else 256,
        'fresh_theta_seed':906202611,'fresh_noise_seeds':FRESH_SEEDS,
        'horizons':[2] if args.quick else [2,3,4,5],
        'source_previous_audit':str(args.previous),'input_sha256':digest(curves),
        'control_values_sha256':digest(U),'selection':'top three minimum(combined gain, nonmyopic gain), plus baseline',
        'claim_scope':'finite-support approximate planning; fresh conditional-model validation, not model-mismatch certification',
        'convergence':[],'screen':[]}
    write_run_config(campaign,cfg,ctx.data_dir,experiment_type='objective_based',extra={'strong_campaign':metadata,'methods':[]})
    save(campaign/'campaign.json',metadata)
    n=4 if args.quick else 64
    validation={'curves':curves[fit:fit+n],'U':U[fit:fit+n]}
    tiers=[('current',AuditBudget(inner=32,outer=16,first_candidates=12,calibration=256,histories_per_batch=16)),
           ('all_first',AuditBudget(inner=32,outer=16,first_candidates=54,calibration=512,fixed_restarts=8,histories_per_batch=16)),
           ('strong',AuditBudget(inner=64,outer=32,first_candidates=54,calibration=512,fixed_restarts=8,histories_per_batch=16))]
    if args.quick:
        old=old[:1]
        tiers=[(name,AuditBudget(inner=4,outer=2,first_candidates=54,calibration=16,fixed_restarts=2)) for name,_ in tiers]
    for i,k in enumerate(old):
        for name,budget in tiers:
            print(f'[campaign] convergence candidate {i} tier {name}',flush=True)
            result=run_policy(campaign,f'convergence_{i}_{name}',k,3,budget,inputs,validation,metadata,SEEDS[:1] if args.quick else SEEDS)
            metadata['convergence'].append(dict(tier=name,**result)); save(campaign/'campaign.json',metadata)
    screen_budget=AuditBudget(inner=8,outer=4,first_candidates=6,calibration=128,histories_per_batch=32)
    if args.quick: screen_budget=AuditBudget(inner=4,outer=2,first_candidates=6,calibration=16)
    for i,k in enumerate(candidates):
        print(f'[campaign] screen {i+1}/{len(candidates)}',flush=True)
        result=run_policy(campaign,f'screen_{i}',k,3,screen_budget,inputs,validation,metadata,SEEDS[:1] if args.quick else SEEDS)
        metadata['screen'].append(result); save(campaign/'campaign.json',metadata)
    ranked=sorted(metadata['screen'],key=lambda r:min(r['adaptive_gain']['mean'],r['nonmyopic_gain']['mean']),reverse=True)
    finalists=[r['key'] for r in ranked[:1 if args.quick else 3]]
    if list(baseline) not in finalists: finalists.append(list(baseline))
    metadata.update(status='finalists_frozen',finalists=finalists)
    save(campaign/'campaign.json',metadata)
    print('FINALISTS_FROZEN',finalists,flush=True)

def fresh(args, cfg=None):
    from src.banks.power_grid import _sample_power_grid_prior
    from src.domains.swing.design import Design,build_simulator
    from src.domains.swing.cuda import CudaTrajectoryEngine
    from src.control.cuda_control import CudaControlEngine
    from src.control.u_req import ControlSpec
    from tools.audits.validation import batch_control_oracle
    from src.control.oracle_u_ctrl import compute_u_ctrl_opt
    campaign=args.campaign; metadata=json.loads((campaign/'campaign.json').read_text())
    if metadata['status']!='finalists_frozen': raise ValueError('Finalists must be frozen before fresh validation')
    if (campaign/'validation.npz').exists(): raise ValueError('Fresh validation already exists; do not silently replace it')
    inputs=np.load(campaign/'inputs.npz'); cfg=config() if cfg is None else cfg
    count=metadata['fresh_systems']
    M,K,_=_sample_power_grid_prior(cfg,count,np.random.default_rng(metadata['fresh_theta_seed']))
    known=np.r_[np.c_[inputs['M'],inputs['K']],np.c_[inputs['excluded_test_M'],inputs['excluded_test_K']]]
    known_rows={r.tobytes() for r in known}
    if any(r.tobytes() in known_rows for r in np.c_[M,K]): raise ValueError('Fresh validation overlaps existing bank')
    action_ids=np.array(sorted({a for k in metadata['finalists'] for d in k for a in metadata['duration_actions'][str(d)]}))
    designs=[Design(*metadata['designs'][int(a)]) for a in action_ids]
    sim=build_simulator(cfg); engine=CudaTrajectoryEngine(sim,designs)
    obs=inputs['obs_indices'].astype(int)
    # Reproduce known dense-bank entries before simulating new systems.
    for i in (0,1):
        check=engine.simulate_delta_f_batch(np.repeat(inputs['M'][i:i+1],len(designs),0),
            np.repeat(inputs['K'][i:i+1],len(designs),0),np.arange(len(designs)),batch_size=128)
        np.testing.assert_allclose(check[:,obs],inputs['curves'][i,action_ids,:],rtol=1e-6,atol=1e-10)
    curves=np.empty((count,len(designs),len(obs)))
    for i in range(count):
        full=engine.simulate_delta_f_batch(np.repeat(M[i:i+1],len(designs),0),np.repeat(K[i:i+1],len(designs),0),
            np.arange(len(designs)),batch_size=128)
        curves[i]=full[:,obs]
        if i%16==0 or i+1==count: print(f'[fresh] observations {i+1}/{count}',flush=True)
    if not np.isfinite(curves).all(): raise ValueError('Nonfinite fresh observations')
    spec=ControlSpec.from_cfg(cfg); np.testing.assert_array_equal(spec.u_grid(),inputs['grid'])
    control=CudaControlEngine(build_simulator(cfg),spec)
    oracle=batch_control_oracle(control,M,K,spec)
    if not (oracle['feasible'].all() and oracle['monotonic'].all()):
        raise ValueError('Fresh systems contain infeasible/nonmonotone control: retained as failure, never resampled')
    columns=np.searchsorted(oracle['coarse_grid'],inputs['grid'])
    safe=oracle['coarse_safe'][:,columns]
    if not safe.any(1).all(): raise ValueError('No feasible candidate control for fresh system')
    U=inputs['grid'][safe.argmax(1)]
    # Verify the vectorized oracle against the existing scalar implementation.
    for i in range(min(2,count)):
        ref=compute_u_ctrl_opt(control,M[i],K[i],spec)
        np.testing.assert_allclose(oracle['u_opt'][i],ref['u_ctrl_opt'],atol=1e-10)
    temp=campaign/'validation_staging.npz'
    np.savez_compressed(temp,curves=curves,U=U,u_opt=oracle['u_opt'],M=M,K=K,
        candidate_safe=safe,action_ids=action_ids,bracket_lower=oracle['bracket_lower'],bracket_upper=oracle['bracket_upper'])
    temp.replace(campaign/'validation.npz')
    save(campaign/'validation_manifest.json',{'status':'complete','n':count,'theta_seed':metadata['fresh_theta_seed'],
        'theta_sha256':digest(np.c_[M,K]),'curves_sha256':digest(curves),'u_opt_sha256':digest(oracle['u_opt']),
        'finalist_file_sha256':__import__('hashlib').sha256((campaign/'campaign.json').read_bytes()).hexdigest(),
        'tolerance':1e-4,'scalar_oracle_checks':2,'old_probe_checks':2,'no_resampling':True})
    print('FRESH_VALIDATION_COMPLETE',flush=True)

def confirm(args):
    campaign=args.campaign; metadata=json.loads((campaign/'campaign.json').read_text())
    validation_manifest=json.loads((campaign/'validation_manifest.json').read_text())
    if validation_manifest['finalist_file_sha256']!=__import__('hashlib').sha256((campaign/'campaign.json').read_bytes()).hexdigest():
        raise ValueError('Finalist protocol changed after validation generation')
    if args.index>=len(metadata['finalists']): print('UNUSED_ARRAY_SLOT'); return
    inputs=np.load(campaign/'inputs.npz'); validation=np.load(campaign/'validation.npz')
    budget=AuditBudget(inner=64,outer=32,first_candidates=54,calibration=512,fixed_restarts=8,histories_per_batch=16)
    if metadata['quick']: budget=AuditBudget(inner=4,outer=2,first_candidates=54,calibration=16,fixed_restarts=2)
    results=[]; started=time.time()
    for horizon in metadata['horizons']:
        print(f'[fresh-confirm] candidate={args.index} T={horizon}',flush=True)
        result=run_policy(campaign,f'fresh_{args.index}',metadata['finalists'][args.index],horizon,budget,
            inputs,validation,metadata,FRESH_SEEDS[:1] if metadata['quick'] else FRESH_SEEDS,continuous=True)
        results.append(result)
        save(campaign/f'confirmation_{args.index}.json',{'status':'running','results':results})
    save(campaign/f'confirmation_{args.index}.json',{'status':'complete','results':results,'elapsed_seconds':time.time()-started})
    print('CANDIDATE_COMPLETE',args.index,flush=True)

def finalize(args):
    campaign=args.campaign; metadata=json.loads((campaign/'campaign.json').read_text())
    rows=[]
    for i in range(len(metadata['finalists'])):
        d=json.loads((campaign/f'confirmation_{i}.json').read_text())
        if d['status']!='complete' or len(d['results'])!=len(metadata['horizons']): raise ValueError('Incomplete candidate')
        rows+=d['results']
    save(campaign/'campaign_results.json',{'status':'complete','n_fresh':metadata['fresh_systems'],'results':rows})
    lines=['# Stronger IEEE9 duration audit','',
        f'{metadata["candidate_count"]} duration sets screened; {metadata["fresh_systems"]} fresh validation systems.',
        'Primary metric: finite-loss realized regret against the continuous-control oracle.',
        'Positive gains favor the named adaptive policy. Adjusted intervals include all finalist/horizon/contrast comparisons.',
        'Approximate Fixed and rolling two-step planning remain limitations; validation assumes the specified physical/prior/noise model.','']
    for r in rows:
        lines += [f'## {r["durations_s"]}; T={r["horizon"]}','']
        for k in ('myopic_adaptive_gain','adaptive_gain','nonmyopic_gain'):
            lines.append(f'- {k}: {r[k]["mean"]:.6f}; adjusted interval {r[k]["familywise_interval"]}.')
        lines.append('')
    (campaign/'campaign_report.md').write_text('\n'.join(lines)+'\n')
    print('CAMPAIGN_COMPLETE',campaign,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','fresh','confirm','finalize'])
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--previous',type=Path)
    p.add_argument('--index',type=int,default=0)
    p.add_argument('--quick',action='store_true')
    args=p.parse_args(); args.campaign=args.campaign.resolve()
    {'prepare':prepare,'fresh':fresh,'confirm':confirm,'finalize':finalize}[args.stage](args)

if __name__=='__main__': main()
