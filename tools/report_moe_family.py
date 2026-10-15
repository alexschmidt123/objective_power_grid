"""Frozen-history strategy diagnostics. Never trains or substitutes for EIG.

All proposal and observation interventions are counterfactual diagnostics.
Forced-mode EIG is imported only from existing provenance-preserved evaluations.
"""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from src.research.benchmarks import make_benchmark
from src.research.policies import history_features
from src.research.moe_family import load_policy,EXPERT_IDS


def write(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def describe(x):
    a=np.asarray(x)
    return dict(count=len(a),mean=float(a.mean()),median=float(np.median(a)),maximum=float(a.max()),
        distribution={str(k):int(v) for k,v in zip(*np.unique(a,return_counts=True))}) if len(a) else None


def summarize_rows(rows):
    routes=[r['routing'] for r in rows];a=np.array([[r['selected_expert'] for r in rs] for rs in routes])
    n,t=a.shape;durations=[[] for _ in EXPERT_IDS]
    for row in a:
        start=0
        for end in range(1,t+1):
            if end==t or row[end]!=row[start]:durations[row[start]].append(end-start);start=end
    changes=a[:,1:]!=a[:,:-1]
    request=np.array([[r['strategy']['reroute_requested'] for r in rs] for rs in routes])
    stage=np.stack([(a==j).mean(0) for j in range(4)],-1)
    episode=np.stack([(a==j).mean(1) for j in range(4)],-1)
    return dict(expert_ids=EXPERT_IDS,first_measurement_excluded=True,
        stage_numbers=list(range(2,t+2)),selection_frequency=stage.mean(0).tolist(),
        hard_state=bool(routes[0][0]['strategy']['hard_state']),
        interpretation='Hard expert decisions' if routes[0][0]['strategy']['hard_state'] else 'Relaxed argmax proxies; not hard deployed strategy durations',
        stage_selection_frequency=stage.tolist(),duration_by_expert={e:describe(v) for e,v in zip(EXPERT_IDS,durations)},
        durations_include_end_censored_terminal_runs=True,switches_per_episode=describe(changes.sum(1)),
        reroute_fraction_excluding_initial=float(request[:,1:].mean()),
        stage_reroute_fraction=request.mean(0).tolist(),stage_actual_change_fraction=np.r_[0.,changes.mean(0)].tolist(),
        single_expert_all_routed_decisions_fraction=float((episode.max(1)==1).mean()),
        single_expert_at_least_80percent_fraction=float((episode.max(1)>=.8).mean()),
        episode_usage=episode.tolist())


@torch.no_grad()
def counterfactuals(b,p,rows,limit):
    pairdist=[];proposals=[];selected=[];ys=[];stages=[];cf=[];regions=[];corrections=[]
    rng=torch.Generator().manual_seed(552019)
    for row in rows[:limit]:
        actions=[torch.tensor([a],dtype=torch.float64) for a in row['actions']]
        obs=[torch.tensor([y],dtype=torch.float64) for y in row['observations']]
        for stage in range(1,p.horizon):
            f=history_features(b,actions[:stage],obs[:stage],1,p.horizon)
            context=p.context(f);prev=row['routing'][stage-2]['selected_expert'] if stage>1 else None
            raw=torch.stack([e(context[:,j]) for j,e in enumerate(p.experts)],1)
            base=p.core.base(f,stage) if hasattr(p.core,'base') else torch.zeros_like(raw[:,0])
            alpha=p.core.alpha if hasattr(p.core,'alpha') else 1.
            candidates=torch.stack([b.feasible(base+alpha*raw[:,j],actions[:stage],p.horizon) for j in range(4)],1)[0]
            dist=torch.cdist(candidates,candidates)
            pairdist.append(dist.numpy());proposals.append(candidates.numpy());stages.append(stage+1)
            selected.append(row['routing'][stage-1]['selected_expert']);ys.append(float(obs[stage-1][0,0]))
            regions.append((candidates>0).long().numpy())
            if hasattr(p.core,'base'):
                physical_base=b.feasible(base,actions[:stage],p.horizon)
                physical_delta=candidates-physical_base
                corrections.append(dict(stage=stage+1,prebounded_norm=raw[0].norm(dim=-1).tolist(),
                    prebounded_direction=raw[0].tolist(),physical_delta=physical_delta.tolist(),
                    physical_delta_norm=physical_delta.norm(dim=-1).tolist()))
            if p.manager is not None and prev is not None:
                previous=torch.nn.functional.one_hot(torch.tensor([prev]),4).double()
                original=float(p.manager.switching_gate(context.mean(1),previous)[0,0])
                theta=torch.tensor([row['truth']],dtype=torch.float64)
                # Location: exact log-signal mean at the latest completed design.
                # Never supply theta to policy/gate; only simulator generates y.
                if b.name=='location_finding':
                    mean,_=b.forward(theta,b.initial((1,)),actions[stage-1])
                    bs=[]
                    for _ in range(4):
                        sampled=mean+b.sigma*torch.randn(mean.shape,generator=rng,dtype=torch.float64)
                        changed=f.clone();slots=changed[:,1:].reshape(1,p.horizon,b.action_dim+b.obs_dim+1)
                        slots[:,stage-1,b.action_dim:b.action_dim+b.obs_dim]=sampled/b.obs_scale
                        c=p.context(changed)
                        bs.append(float(p.manager.switching_gate(c.mean(1),previous)[0,0]))
                    cf.append(dict(stage=stage+1,previous_expert=EXPERT_IDS[prev],original_probability=original,
                        resampled_probabilities=bs,range=float(np.ptp(bs)),std=float(np.std(bs)),
                        hard_request_flip=any((v>=.5)!=(original>=.5) for v in bs)))
    ds=np.asarray(pairdist);ps=np.asarray(proposals);ss=np.asarray(stages);selection=np.asarray(selected);y=np.asarray(ys)
    pairs=[(i,j) for i in range(4) for j in range(i+1,4)]
    bystage={str(s):{f'{EXPERT_IDS[i]}-{EXPERT_IDS[j]}':float(ds[ss==s,i,j].mean()) for i,j in pairs} for s in np.unique(ss)}
    conditional=[]
    for s in np.unique(ss):
        mask=ss==s;edges=np.quantile(y[mask],[0.,.25,.5,.75,1.])
        bins=np.searchsorted(edges[1:-1],y[mask],side='right')
        conditional.append(dict(stage=int(s),observable='most_recent_log_signal',quartile_edges=edges.tolist(),
            selection_frequency_by_quartile=[(np.bincount(selection[mask][bins==k],minlength=4)/max(1,int((bins==k).sum()))).tolist() for k in range(4)],
            counts_by_quartile=[int((bins==k).sum()) for k in range(4)],
            note='observational association; not causal specialization evidence'))
    cfreport=dict(label='counterfactual routing diagnostics, not rollout EIG',cases=len(cf),
        changed_observation='Four independent likelihood draws at the same completed design, using held-out truth only inside simulator',
        held_fixed='all designs, other observations, stage and previous expert identity',
        mean_probability_range=float(np.mean([r['range'] for r in cf])) if cf else None,
        maximum_probability_range=max((r['range'] for r in cf),default=None),
        hard_decision_flip_fraction=float(np.mean([r['hard_request_flip'] for r in cf])) if cf else None,records=cf)
    return dict(label='matched-history counterfactual expert proposals; not rollout EIG',histories=len(ds),
        pairwise_physical_distance_mean={f'{EXPERT_IDS[i]}-{EXPERT_IDS[j]}':float(ds[:,i,j].mean()) for i,j in pairs},
        pairwise_distance_by_stage=bystage,mean_design_by_expert=ps.mean(0).tolist(),
        positive_coordinate_frequency_by_expert=(ps>0).mean(0).tolist(),
        conditional_observable_usage=conditional,residual_corrections=corrections),cfreport


def training_report(path):
    groups={};phase_mass={};scores=[];updates=0
    for line in Path(path).open():
        r=json.loads(line);updates+=1
        phase=min(4,int((r['primary_episodes']-1)/200000))
        phase_mass.setdefault(phase,[]).append(r['router_probabilities'])
        for k,v in r['gradient_before'].items():
            g=groups.setdefault(k,dict(raw=[],post=[],last20=[]));g['raw'].append(v);g['post'].append(r['gradient_after'][k])
            if r['primary_episodes']>800000:g['last20'].append(r['gradient_after'][k])
        if 'training_score_distribution' in r:scores.append(r['training_score_distribution'])
    result={}
    for k,g in groups.items():
        a=np.asarray(g['post']);last=np.asarray(g['last20'])
        result[k]=dict(raw_quantiles_p50_p95_p99_max=np.quantile(g['raw'],[.5,.95,.99,1]).tolist(),
            fraction_nonzero=float((a>0).mean()),fraction_above_1e_8=float((a>1e-8).mean()),fraction_above_1e_3=float((a>1e-3).mean()),
            final20_fraction_above_1e_3=float((last>1e-3).mean()) if len(last) else None)
    return dict(updates=updates,gradient_exposure=result,gate_mass_by_training_fifth={str(k):np.mean(v,axis=0).tolist() for k,v in phase_mass.items()},
        training_score_distributions_recorded=len(scores),training_scores_not_comparable_to_high_precision_eval=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--rows',required=True);ap.add_argument('--output',required=True);ap.add_argument('--max-cases',type=int,default=64)
    ap.add_argument('--training-trace');args=ap.parse_args();torch.set_num_threads(1)
    run=Path(args.run);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    b=make_benchmark(json.loads((run/'run_config.json').read_text())['benchmark'])
    p=load_policy(args.checkpoint,b);rows=json.loads(Path(args.rows).read_text())
    persistence=summarize_rows(rows);proposals,obs=counterfactuals(b,p,rows,args.max_cases)
    write(out/'persistence.json',persistence);write(out/'matched_history_proposals.json',proposals);write(out/'observation_counterfactuals.json',obs)
    result=dict(variant=p.variant,canonical_name=p.canonical_name,expert_ids=EXPERT_IDS,
        persistence=persistence,counterfactual_summary={k:v for k,v in obs.items() if k!='records'},
        proposal_summary={k:v for k,v in proposals.items() if k not in {'conditional_observable_usage','residual_corrections'}},
        training=training_report(args.training_trace) if args.training_trace else None,
        limitations=['No semantic expert roles are assigned','Distinct proposals and usage do not establish useful specialization',
            'Forced-mode degradation may reflect out-of-distribution histories','Single training seed; development seed1001 is not untouched'])
    write(out/'strategy_report.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in {'persistence','training','proposal_summary'}},indent=2))

if __name__=='__main__':main()
