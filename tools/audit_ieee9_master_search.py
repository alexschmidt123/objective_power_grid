#!/usr/bin/env python3
"""Global/local six-duration search over the complete IEEE9 master catalog."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse,json,math,time
import numpy as np
from src.layout import write_run_config
from src.objectives.mocu.context import build_context_from_config
from tools.audit_ieee9_campaign import config,run_policy,save,fresh,confirm,finalize as finalize_campaign
from tools.audit_ieee9_duration_combos import digest
from tools.audits.space import AuditBudget
from tools.bank_sweeps.sweep_ieee9_eig_duration_sets import load_catalog,resolve_pool_actions
from tools.bank_sweeps.search_ieee9_eig_duration_sets_full import proxy_scores,proxy_value


def select_finalists(rows,baseline,gap_count=2,loss_count=2):
    # Two scientific objectives: space for adaptivity and low operational loss.
    gap=sorted(rows,key=lambda r:min(r['adaptive_gain']['mean'],r['nonmyopic_gain']['mean']),reverse=True)
    loss=sorted(rows,key=lambda r:r['methods']['lookahead']['mean_loss'])
    keys=[]
    for r in gap[:gap_count]+loss[:loss_count]:
        if r['key'] not in keys:keys.append(r['key'])
    if list(baseline) not in keys:keys.append(list(baseline))
    return keys


def prepare(args):
    campaign=args.campaign; cfg=config()
    if (campaign/"campaign.json").exists(): raise ValueError("Refuse to overwrite a search")
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
    inputs={'curves':curves,'U':U,'grid':ctx.u_grid,'M':M,'K':K,'obs_indices':ctx.obs_indices,
        'excluded_test_M':np.array([s['M'] for s in ctx.test_systems]),
        'excluded_test_K':np.array([s['K'] for s in ctx.test_systems])}
    if len(durations)!=281 or len(catalog.designs)!=2529:raise ValueError('Require complete IEEE9 master bank')
    np.savez_compressed(campaign/'inputs.npz',**inputs)
    if not np.isfinite(curves).all():raise ValueError('Nonfinite master observations')
    q=args.quick;rng=np.random.default_rng(9070917)
    keys={baseline}
    # Each duration must participate in a directly scored global candidate.
    for d in range(len(durations)):
        others=rng.choice(np.delete(np.arange(len(durations)),d),5,replace=False)
        keys.add(tuple(sorted([d]+others.tolist())))
    previous=json.loads(args.previous.read_text())
    keys.update(tuple(k) for k in previous['candidates'])
    while len(keys)<1024:keys.add(tuple(sorted(rng.choice(281,6,replace=False).tolist())))
    keys=sorted(keys)
    if set(k for row in keys for k in row)!=set(range(281)):raise ValueError('Incomplete duration coverage')
    if q:keys=[baseline,next(k for k in keys if k!=baseline)]
    meta={'schema':'ieee9_master_global_local_v1','status':'global_search','quick':q,
        'candidate_count':len(keys),'durations':durations,'baseline':list(baseline),
        'candidates':[list(k) for k in keys], 'total_combinations':math.comb(281,6),'exhaustive':False,
        'combo_definition':'six durations crossed with all nine physical probe buses; 54 actions',
        'master_bank':str(dense),'master_action_count':2529,'duration_options':281,
        'duration_actions':{str(i):resolve_pool_actions(catalog,[d])[0].tolist() for i,d in enumerate(durations)},
        'designs':[list(d) for d in catalog.designs],'sigma':ctx.sigma_y,'n_support':fit,
        'old_screen_rows':[fit,fit+(4 if q else 64)],'fresh_systems':4 if q else 512,
        'fresh_theta_seed':907091701,'fresh_noise_seeds':[11001,11002,11003],
        'horizons':[2] if q else [2,3,4,5],
        'input_sha256':digest(curves),'control_values_sha256':digest(U),
        'theta_sha256':digest(np.c_[M,K]),'previous_search':str(args.previous),
        'selection':'two strongest min(combined,nonmyopic) mean gains plus two lowest lookahead losses plus original baseline',
        'claim_scope':'best found in sampled/local search; not global optimum; screen is exploratory',
        'screen':[],'refinement':[],'convergence':[]}
    write_run_config(campaign,cfg,ctx.data_dir,experiment_type='objective_based',extra={'master_search':meta,'methods':[]})
    save(campaign/'campaign.json',meta)
    n=4 if q else 64
    validation={'curves':curves[fit:fit+n],'U':U[fit:fit+n]}
    cheap=AuditBudget(inner=4 if q else 8,outer=2 if q else 4,first_candidates=6,
        calibration=16 if q else 128,histories_per_batch=32)
    def score(candidates,phase,budget,seeds):
        for i,key in enumerate(candidates):
            print(f'[{phase}] {i+1}/{len(candidates)}: {[durations[k] for k in key]}',flush=True)
            r=run_policy(campaign,f'{phase}_{i}',key,3,budget,inputs,validation,meta,seeds)
            meta[phase].append(r);save(campaign/'campaign.json',meta)
    score(keys,'screen',cheap,[101] if q else [101,202])
    parents=select_finalists(meta['screen'],baseline,1 if q else 8,1 if q else 8)
    tested=set(keys);local=set()
    for parent in parents:
        for pos in range(6):
            for replacement in range(281):
                if replacement in parent:continue
                k=tuple(sorted(parent[:pos]+[replacement]+parent[pos+1:]))
                if k not in tested:local.add(k)
    info,distance=proxy_scores(curves[:fit].transpose(1,0,2),281,9)
    order=sorted(local,key=lambda k:(-proxy_value(k,info,distance),k))
    chosen=order[:1 if q else 128]
    remaining=sorted(local-set(chosen))
    if remaining:
        pick=rng.choice(len(remaining),min(1 if q else 128,len(remaining)),replace=False)
        chosen += [remaining[int(i)] for i in pick]
    meta.update(status='local_search',local_neighborhood_count=len(local),local_candidates=[list(k) for k in chosen],
        local_proxy='response diversity only for proposals; final scores use aligned finite-loss MOCU')
    save(campaign/'campaign.json',meta)
    # Append local results to the same exploratory screen with distinct raw names.
    meta['local_screen']=[];score(chosen,'local_screen',cheap,[101] if q else [101,202])
    all_rows=meta['screen']+meta['local_screen']
    meta['candidate_count']=len(all_rows)
    eligible=select_finalists(all_rows,baseline,1 if q else 6,1 if q else 6)
    stronger=AuditBudget(inner=4 if q else 32,outer=2 if q else 16,first_candidates=54,
        calibration=16 if q else 512,fixed_restarts=2 if q else 8,histories_per_batch=16)
    meta['status']='refining';score(eligible,'refinement',stronger,[101] if q else [101,202,303])
    meta['finalists']=select_finalists(meta['refinement'],baseline,1 if q else 2,1 if q else 2)
    strong=AuditBudget(inner=4 if q else 64,outer=2 if q else 32,first_candidates=54,
        calibration=16 if q else 512,fixed_restarts=2 if q else 8,histories_per_batch=16)
    for i,key in enumerate(meta['finalists']):
        r=run_policy(campaign,f'convergence_{i}',key,3,strong,inputs,validation,meta,[101] if q else [101,202,303])
        meta['convergence'].append(dict(tier='strong',**r));save(campaign/'campaign.json',meta)
    # Convergence results are reported, not used for another unregistered selection.
    meta['status']='finalists_frozen';save(campaign/'campaign.json',meta)
    print('FINALISTS_FROZEN',meta['finalists'],flush=True)


def finalize(args):
    finalize_campaign(args)
    meta=json.loads((args.campaign/'campaign.json').read_text())
    results=json.loads((args.campaign/'campaign_results.json').read_text())['results']
    ranks={}
    for horizon in meta['horizons']:
        rows=[r for r in results if r['horizon']==horizon]
        ranks[str(horizon)]=[
            {'durations_s':r['durations_s'],'lookahead_regret':r['methods']['lookahead']['mean_loss'],
             'physical_safety_rate':r['methods']['lookahead']['physical_safety_rate'],
             'joint_positive_signal':r['joint_positive_signal']}
            for r in sorted(rows,key=lambda r:r['methods']['lookahead']['mean_loss'])]
    save(args.campaign/'best_combos.json',{'status':'complete','rankings':ranks,
        'scope':'descriptive ordering of frozen finalists; no global-optimality or post-selection superiority claim',
        'searched_combinations':meta['candidate_count'],'total_combinations':meta['total_combinations']})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','fresh','confirm','finalize'])
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--previous',type=Path)
    p.add_argument('--index',type=int,default=0);p.add_argument('--quick',action='store_true')
    args=p.parse_args();args.campaign=args.campaign.resolve();args.campaign.mkdir(parents=True,exist_ok=True)
    {'prepare':prepare,'fresh':fresh,'confirm':confirm,'finalize':finalize}[args.stage](args)
if __name__=='__main__':main()
