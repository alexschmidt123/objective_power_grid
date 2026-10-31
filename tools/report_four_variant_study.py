"""Reproducible four-variant study report from immutable run artifacts.

No training or EIG estimation occurs here. Missing runs stay explicitly pending.
Trajectory-level intervals are conditional on the one trained checkpoint.
"""
if __name__ == '__main__':
    raise SystemExit('LEGACY_EXCLUDED_FROM_CURRENT_PROJECT: historical result generation is archived')
import argparse,csv,hashlib,html,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES=dict(moe='MoE-sBOED',resmoe='ResMoE-sBOED',switchmoe='SwitchMoE-sBOED',switchresmoe='SwitchResMoE-sBOED')
COLORS=['#3569a8','#e1812c','#32956c','#a057a8']
EXPERTS=['E1','E2','E3','E4']

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def stats(a):
    a=np.asarray(a);se=float(a.std(ddof=1)/math.sqrt(len(a))) if len(a)>1 else None
    return dict(n=len(a),mean=float(a.mean()) if len(a) else None,mc_standard_error=se,
        conditional_95_interval=[float(a.mean()-1.96*se),float(a.mean()+1.96*se)] if se is not None else None)

def case_identity(rows):
    return hashlib.sha256(json.dumps([[r['truth'],r['measurement_noise_standard']] for r in rows],separators=(',',':')).encode()).hexdigest()

def normalize_routes(rows):
    for row in rows:
        old=-1;duration=changes=0
        for k,r in enumerate(row.get('routing',[])):
            if r.get('selected_expert') is None:r['selected_expert']=int(np.argmax(r['weights']))
            a=r['selected_expert'];changed=k>0 and a!=old
            duration=duration+1 if a==old else 1;changes+=int(changed)
            if 'strategy' not in r:
                r['strategy']=dict(reroute_requested=True,actual_strategy_change=changed,
                    reroute_probability=1.,consecutive_duration=duration,hard_state=True,
                    initialized=k==0,active_strategy=a,previous_strategy=old,strategy_changes=changes,
                    reroute_requests=k+1)
            old=a
    return rows

def training_diagnostics(path,budget):
    points=[];groups={};phase={};logged=[];near=0.;total=0;last_count=0;window_peak={}
    if not path.exists():return None
    with path.open() as f:
        for line in f:
            r=json.loads(line);count=r['primary_episodes'];n=count-last_count;last_count=count
            frac=count/budget;part='early' if frac<=.2 else 'middle' if frac<=.8 else 'late'
            if 'router_probabilities' in r:phase.setdefault(part,[]).append(r['router_probabilities'])
            dist=r.get('training_score_distribution',{})
            if dist:
                near+=n*dist.get('fraction_within_01_of_ceiling',0.);total+=n
            for key,value in r.get('gradient_before',{}).items():
                window_peak[key]=max(window_peak.get(key,0.),value)
                g=groups.setdefault(key,dict(raw=[],post=[],late=[]));g['raw'].append(value);g['post'].append(r['gradient_after'][key])
                if frac>.8:g['late'].append(r['gradient_after'][key])
            if r.get('training_scores'):logged.extend(r['training_scores'])
            if r['update']%50==0 or count==budget:
                points.append(dict(episodes=count,eig=r['training_eig'],gamma=r.get('gamma'),
                    usage=r.get('router_probabilities'),hard_usage=r.get('hard_choice_frequency'),
                    gradient=r.get('gradient_after'),raw_gradient=r.get('gradient_before'),
                    raw_gradient_window_max=window_peak.copy(),routing_entropy=r.get('routing_entropy'),
                    score_quantiles=dist.get('quantiles'),near=dist.get('fraction_within_01_of_ceiling'),
                    reroute_probability=r.get('strategy_reroute_probability_mean')))
                window_peak.clear()
    exposure={}
    for key,g in groups.items():
        a=np.asarray(g['post']);late=np.asarray(g['late']);raw=np.asarray(g['raw'])
        exposure[key]=dict(updates=len(a),nonzero_fraction=float((a>0).mean()),
            fraction_above_1e_8=float((a>1e-8).mean()),fraction_above_1e_3=float((a>1e-3).mean()),
            final20_fraction_above_1e_3=float((late>1e-3).mean()) if len(late) else None,
            raw_p50_p95_p99_max=np.quantile(raw,[.5,.95,.99,1]).tolist(),
            post_clip_max=float(a.max()),raw_fraction_above1000=float((raw>1000).mean()))
    return dict(points=points,gradient_exposure=exposure,
        phase_gate_mass={k:np.mean(v,axis=0).tolist() for k,v in phase.items()},
        fraction_within_point1_of_training_ceiling=near/total if total else None,
        logged_trajectory_score_statistics=dict(n=len(logged),mean=float(np.mean(logged)) if logged else None,std=float(np.std(logged)) if logged else None,interpretation='Descriptive evolving-training distribution, not iid held-out uncertainty'),logged_trajectory_score_quantiles=np.quantile(logged,[0,.1,.5,.9,.99,1]).tolist() if logged else None,
        logged_trajectory_scores=logged,
        score_note='Raw per-trajectory scores logged at predeclared update intervals, not all simulated episodes; ceiling fraction aggregates every batch')

def load_run(root,run,method,label,kind):
    run=Path(run);p=run/'summary.json'
    item=dict(label=label,kind=kind,method=method,run=str(run),status='pending')
    if not p.exists() or not (run/'exit_code').exists() or (run/'exit_code').read_text().strip()!='0':return item
    summary=next(x for x in read(p) if x['method']==method and x['eval_seed']==1001)
    rows=normalize_routes(read(run/'eval'/f'{method}_seed1001.json'));meta=read(run/'run_config.json')
    training=read(run/'models'/f'{method}_training.json')
    score=stats([r['spce'] for r in rows]);bank=run/'eval/contrast_bank_seed1001.npy'
    item.update(status='complete',eig=score,summary=summary,training=training,metadata=meta,
        cases_sha256=case_identity(rows),bank_sha256=hashlib.sha256(bank.read_bytes()).hexdigest(),rows=rows,
        additional_episodes=training.get('generated_training_trajectories',0),
        total_episodes=training.get('total_training_episodes',training.get('generated_training_trajectories',0)),
        training_seconds=training.get('training_seconds',0.),base_training_seconds=training.get('base_training_seconds',0.),
        evaluation_seconds=summary['evaluation_seconds'],online_seconds=summary['mean_decision_seconds'],
        parameter_count=training.get('policy_parameters'),measured_parameters=training.get('measured_parameters'))
    if kind in set(NAMES)|{'single','continued','matched','contrast_control','intervention'}:
        n=item['additional_episodes'];L=meta['settings']['contrasts'];h=meta['horizon']
        item['likelihood_cost']=dict(additional_contrast_trajectories=n*L,
            additional_truth_inclusive_trajectories=n*(L+1),additional_particle_stages=n*(L+1)*h)
        pre=training.get('base_checkpoint') or training.get('base_pretraining') or {}
        if pre.get('path'):
            config=read(Path(pre['path']).parent.parent/'run_config.json');count=pre['training_episodes'];baseL=config['settings']['contrasts']
            item['likelihood_cost'].update(base_primary_episodes=count,base_contrasts=baseL,
                base_truth_inclusive_trajectories=count*(baseL+1),
                total_truth_inclusive_trajectories=count*(baseL+1)+n*(L+1))
    if kind in NAMES or kind=='intervention':
        trace=run/'models'/f'{method}_diagnostics.jsonl'
        item['diagnostics']=training_diagnostics(trace,item['additional_episodes'])
        from tools.report_moe_family import summarize_rows
        item['management']=summarize_rows(rows)
    elif kind in {'single','continued','matched','contrast_control'}:
        item['diagnostics']=training_diagnostics(run/'models'/f'{method}_diagnostics.jsonl',item['additional_episodes'])
    return item

def paired_runs(a,b):
    if a['status']!='complete' or b['status']!='complete':return None
    if a['cases_sha256']!=b['cases_sha256'] or a['bank_sha256']!=b['bank_sha256']:
        return dict(status='not paired; no paired interval reported')
    return stats(np.array([r['spce'] for r in a['rows']])-np.array([r['spce'] for r in b['rows']]))

def savefig(fig,out,name):
    fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=170);fig.savefig(out/(name+'.svg'));plt.close(fig)


def plots(items,out):
    done=[x for x in items if x['status']=='complete' and x['kind'] in NAMES]
    if not done:return
    fig,axes=plt.subplots(len(done),2,figsize=(12,3.2*len(done)),squeeze=False)
    for i,x in enumerate(done):
        d=x.get('diagnostics');ax=axes[i,0];bx=axes[i,1]
        if not d:continue
        points=d['points'];t=np.array([p['episodes'] for p in points])/1e6
        usage=np.array([p['usage'] for p in points]);ax.stackplot(t,usage.T,colors=COLORS,labels=EXPERTS)
        ax.set(title=x['label'],xlabel='Training episodes (millions)',ylabel='Relaxed expert weight',ylim=(0,1));ax.legend(ncol=4,fontsize=8)
        exposure=d['gradient_exposure'];full=[exposure[f'expert_{j}']['fraction_above_1e_3'] for j in range(4)]
        late=[exposure[f'expert_{j}']['final20_fraction_above_1e_3'] for j in range(4)]
        k=np.arange(4);bx.bar(k-.18,np.array(full)*100,.36,label='All updates');bx.bar(k+.18,np.array(late)*100,.36,label='Final20%')
        bx.set(xticks=k,xticklabels=EXPERTS,ylabel='Updates with gradient norm >0.001 (%)',ylim=(0,105));bx.legend(fontsize=8)
    savefig(fig,out,'expert_utilization_gradient_exposure')
    fig,axes=plt.subplots(len(done),2,figsize=(12,3.2*len(done)),squeeze=False)
    for i,x in enumerate(done):
        m=x['management'];a=axes[i,0];b=axes[i,1]
        im=a.imshow(np.array(m['stage_selection_frequency']).T,aspect='auto',origin='lower',vmin=0,vmax=1,cmap='viridis',extent=[1.5,30.5,-.5,3.5])
        a.set(title=x['label'],xlabel='Measurement (first excluded)',yticks=np.arange(4),yticklabels=EXPERTS);fig.colorbar(im,ax=a,label='Selection frequency')
        b.plot(m['stage_numbers'],m['stage_reroute_fraction'],label='Rerouting request')
        b.plot(m['stage_numbers'],m['stage_actual_change_fraction'],label='Actual expert change')
        b.set(xlabel='Measurement',ylabel='Fraction of trajectories',ylim=(-.03,1.03));b.legend(fontsize=8)
    savefig(fig,out,'stage_utilization_switching')
    fig,axes=plt.subplots(len(done),2,figsize=(12,3.2*len(done)),squeeze=False)
    for i,x in enumerate(done):
        d=x['diagnostics'];p=d['points'];ep=np.array([v['episodes'] for v in p])/1e6
        ax=axes[i,0];ax.plot(ep,[v['eig'] for v in p],lw=.8,label='Batch sPCE mean')
        q=np.asarray([v['score_quantiles'] for v in p]);ax.fill_between(ep,q[:,1],q[:,3],alpha=.2,label='Trajectory q10–q90')
        ax.axhline(math.log(1025),ls='--',color='black',lw=.8,label='log1025 ceiling');ax.set(title=x['label'],xlabel='Training episodes (millions)',ylabel='Training sPCE (nats)');ax.legend(fontsize=7)
        bx=axes[i,1];bx.hist(d['logged_trajectory_scores'],bins=60,density=True,color=COLORS[list(NAMES).index(x['kind'])]);bx.axvline(math.log(1025),ls='--',color='black')
        bx.set(title='Logged trajectory training scores',xlabel='sPCE (nats)',ylabel='Density')
    savefig(fig,out,'training_curves_score_distributions')
    fig,axes=plt.subplots(len(done),3,figsize=(15,3*len(done)),squeeze=False)
    for x,(a,b,c) in zip(done,axes):
        points=x['diagnostics']['points'];ep=np.array([p['episodes'] for p in points])/1e6
        for group,color in [('router','#3569a8'),('switching_gate','#e1812c')]:
            if group not in x['diagnostics']['gradient_exposure']:continue
            a.semilogy(ep,[max(p['raw_gradient_window_max'].get(group,0.),1e-12) for p in points],label=group,color=color,lw=.8)
        a.axhline(5,ls='--',color='black',lw=.7,label='Component clip limit');a.set(title=x['label'],xlabel='Training episodes (millions)',ylabel='Raw norm: max per50 updates');a.legend(fontsize=7)
        for j in range(4):b.semilogy(ep,[max(p['gradient'].get(f'expert_{j}',0.),1e-12) for p in points],label=EXPERTS[j],color=COLORS[j],lw=.7)
        b.set(xlabel='Training episodes (millions)',ylabel='Expert norm after clipping (sampled)');b.legend(ncol=4,fontsize=7)
        valid=[(p['episodes']/1e6,p['routing_entropy']) for p in points if p['routing_entropy'] is not None]
        if valid:c.plot(*zip(*valid),label='Mean routing entropy',color='#32956c')
        c.set(xlabel='Training episodes (millions)',ylabel='Routing entropy (nats)');d=c.twinx();d.plot(ep,[p['gamma'] for p in points],color='black',ls='--',lw=.8);d.set(ylabel='DSelect gamma',ylim=(0,1.05))
    savefig(fig,out,'routing_gradient_entropy_sharpening')
    switches=[x for x in done if x['kind'].startswith('switch')]
    if switches:
        fig,axes=plt.subplots(len(switches),1,figsize=(8,3*len(switches)),squeeze=False)
        for ax,x in zip(axes[:,0],switches):
            m=x['management'];means=[m['duration_by_expert'][e]['mean'] if m['duration_by_expert'][e] else 0 for e in EXPERTS];medians=[m['duration_by_expert'][e]['median'] if m['duration_by_expert'][e] else 0 for e in EXPERTS]
            k=np.arange(4);ax.bar(k-.18,means,.36,label='Mean');ax.bar(k+.18,medians,.36,label='Median');ax.set(title=x['label']+' (end-censored runs included)',xticks=k,xticklabels=EXPERTS,ylabel='Consecutive routed measurements');ax.legend()
        savefig(fig,out,'strategy_durations')
    fig,ax=plt.subplots(figsize=(9,4));ax.bar([x['label'] for x in done],[x['eig']['mean'] for x in done],yerr=[1.96*x['eig']['mc_standard_error'] for x in done],color=[COLORS[list(NAMES).index(x['kind'])] for x in done],capsize=4)
    ax.axhline(10.66260,label='Historical DAD-MLP',color='black',ls='--');ax.set(ylabel='Held-out EIG (nats)',title='Development seed1001; intervals are trajectory MC only');ax.legend();plt.setp(ax.get_xticklabels(),rotation=15,ha='right');savefig(fig,out,'four_variant_eig')


def parameter_table(family,root):
    path=root/'experiments/10152026/10152026_four_variant_validation/measured_parameters.json'
    verified=read(path) if path.exists() else {}
    body='<table><tr><th>Variant</th><th>Total parameters</th><th>Pretrained</th><th>Newly initialized</th><th>Active first routed</th><th>Active later routed</th></tr>'
    for x in family:
        d=x.get('measured_parameters') or verified.get(x['kind'])
        if not d:continue
        first=d['initial_routed'];later=d['later_routed'];meta=d['first']
        body+=f"<tr><td>{html.escape(x['label'])}</td><td>{meta['total']:,}</td><td>{meta['pretrained']:,}</td><td>{meta['newly_initialized']:,}</td><td>{first['active']:,}</td><td>{later['active']:,}</td></tr>"
    return body+'</table><p>Measured unique parameters touched by a batch-one hard forward. Counts include frozen base parameters when executed. Later counts depend on whether rerouting occurs; these values are the measured checkpoint decision, not estimates.</p>'


def route_consistency(root,residual,family,out):
    sources={}
    old=root/'experiments/10142026/10142026_223452_sum_moe_slower_sharpening/diagnostics/slower/relaxed_scores.json'
    if old.exists():sources['moe']=read(old)
    old=root/'experiments/10142026/10142026_232132_strategy_moe_pilot/diagnostics/learned_relaxed/scores.json'
    if old.exists():sources['switchmoe']=read(old)
    for v in ['resmoe','switchresmoe']:
        path=residual/(v+'_relaxed')/'scores.json'
        if path.exists():sources[v]=[r['spce'] for r in read(path)]
    result={}
    for x in family:
        if x['kind'] not in sources or x['status']!='complete':continue
        soft=sources[x['kind']]
        if isinstance(soft,dict):
            soft=soft.get('scores',soft.get('spce'))
        if soft and isinstance(soft[0],dict):soft=[r['spce'] for r in soft]
        hard=np.asarray([r['spce'] for r in x['rows']]);soft=np.asarray(soft)
        if soft.shape!=hard.shape:raise ValueError('Hard/relaxed case-count mismatch')
        result[x['kind']]=dict(hard=stats(hard),relaxed=stats(soft),hard_minus_relaxed=stats(hard-soft))
    if result:
        fig,ax=plt.subplots(figsize=(9,4));keys=[x['kind'] for x in family if x['kind'] in result];k=np.arange(len(keys))
        ax.bar(k-.18,[result[v]['hard']['mean'] for v in keys],.36,label='Hard')
        ax.bar(k+.18,[result[v]['relaxed']['mean'] for v in keys],.36,label='Relaxed')
        ax.set(xticks=k,xticklabels=[NAMES[v] for v in keys],ylabel='Held-out EIG (nats)',title='Identical checkpoint and paired development cases');ax.legend();plt.setp(ax.get_xticklabels(),rotation=15,ha='right');savefig(fig,out,'hard_relaxed_eig')
    return result


def table(items):
    body='<table><tr><th>Model</th><th>EIG</th><th>95% trajectory MC interval</th><th>Additional / total training episodes</th><th>Training L</th><th>Training + base time (min)</th><th>Status</th></tr>'
    for x in items:
        if x['status']!='complete':body+='<tr><td>'+html.escape(x['label'])+'</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>Pending</td></tr>';continue
        L=x['training'].get('contrasts',x['metadata']['settings']['contrasts'])
        if x['kind']=='external' and x['method'] not in {'dad','fixed'}:L='method-specific'
        e=x['eig'];ci=e['conditional_95_interval'];body+=f"<tr><td>{html.escape(x['label'])}</td><td>{e['mean']:.5f}</td><td>[{ci[0]:.5f}, {ci[1]:.5f}]</td><td>{x['additional_episodes']:,} / {x['total_episodes']:,}</td><td>{L}</td><td>{x['training_seconds']/60:.2f} + {x['base_training_seconds']/60:.2f}</td><td>Complete</td></tr>"
    return body+'</table>'


def mechanism_evidence(root,out):
    """Collect frozen diagnostics without treating interventions as new variants."""
    research=root/'experiments/10152026/10152026_moe_four_variant_research'
    evidence={}
    for v in NAMES:
        item={}
        for key,relative in [('geometry',f'geometry/{v}/geometry.json'),
            ('observation_routing',f'observation_routing/{v}/summary.json'),
            ('routing_interventions',f'{v}_router_contribution/report.json'),
            ('one_step_interventions',f'{v}_expert_interventions/report.json'),
            ('contrast_resolution',f'spce_resolution/{v}/report.json')]:
            p=research/relative
            if p.exists():item[key]=read(p)
        forced={}
        for mode in ['always_reroute','always_continue']:
            path=research/(v+'_'+mode)/'summary.json'
            if v=='switchmoe':path=root/'experiments/10142026/10142026_232132_strategy_moe_pilot/diagnostics'/mode/'summary.json'
            if path.exists():forced[mode]=read(path)
        if forced:item['frozen_switching_modes']=forced
        if item:evidence[v]=item
    baseline_probe=research/'spce_resolution/dad_mlp/report.json'
    if baseline_probe.exists():evidence['dad_mlp']={'contrast_resolution':read(baseline_probe)}
    geometries=[(v,d['geometry']) for v,d in evidence.items() if 'geometry' in d]
    if geometries:
        fig,axes=plt.subplots(len(geometries),2,figsize=(11,3.6*len(geometries)),squeeze=False)
        for (v,g),(a,b) in zip(geometries,axes):
            mat=np.zeros((4,4))
            for pair,value in g['proposals']['pairwise_physical_distance_mean'].items():
                i,j=[EXPERTS.index(e) for e in pair.split('-')];mat[i,j]=mat[j,i]=value
            im=a.imshow(mat,cmap='magma');a.set(title=NAMES[v]+' proposal distances',xticks=range(4),yticks=range(4),xticklabels=EXPERTS,yticklabels=EXPERTS);fig.colorbar(im,ax=a,label='Physical design distance')
            for i in range(4):
                for j in range(4):a.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',color='white' if mat[i,j]<mat.max()*.6 else 'black',fontsize=8)
            obs=evidence[v].get('observation_routing',{}).get('by_stage',{})
            if obs:
                stages=sorted(obs,key=int);b.plot([int(s) for s in stages],[100*obs[s]['selected_flip_fraction'] for s in stages],label='Selected expert')
                b.plot([int(s) for s in stages],[100*obs[s]['gate_flip_fraction'] for s in stages],label='Rerouting request');b.legend(fontsize=8)
            b.set(title='Likelihood-resampled observation diagnostic',xlabel='Measurement',ylabel='Changed hard decisions (%)')
        savefig(fig,out,'expert_disagreement_observation_sensitivity')
        fig,axes=plt.subplots(len(geometries),4,figsize=(15,3.1*len(geometries)),squeeze=False)
        for (v,g),row in zip(geometries,axes):
            p=g['proposals'];region=np.asarray(p['positive_coordinate_frequency_by_expert'])
            im=row[0].imshow(region,vmin=0,vmax=1,cmap='viridis',aspect='auto');row[0].set(title=NAMES[v]+' proposed regions',xticks=[0,1],xticklabels=['x>0','y>0'],yticks=range(4),yticklabels=EXPERTS)
            fig.colorbar(im,ax=row[0],label='Frequency')
            for stage,ax in zip([5,15,25],row[1:]):
                entries=[x for x in p['conditional_observable_usage'] if x['stage']==stage]
                if not entries:continue
                values=np.asarray(entries[0]['selection_frequency_by_quartile']).T
                ax.imshow(values,vmin=0,vmax=1,cmap='viridis',aspect='auto');ax.set(title=f'Measurement {stage}: observable association',xticks=range(4),xticklabels=['Q1','Q2','Q3','Q4'],yticks=range(4),yticklabels=EXPERTS,xlabel='Latest log-signal quartile (within stage)')
        savefig(fig,out,'expert_regions_observable_usage')
    residuals=[(v,d['geometry']['residual_relative_to_base']) for v,d in evidence.items() if d.get('geometry',{}).get('residual_relative_to_base')]
    if residuals:
        fig,axes=plt.subplots(len(residuals),2,figsize=(11,3.2*len(residuals)),squeeze=False)
        for (v,records),(a,b) in zip(residuals,axes):
            stages=sorted(set(r['stage'] for r in records))
            for j,e in enumerate(EXPERTS):
                a.plot(stages,[np.mean([r['correction_logit_norm'][j] for r in records if r['stage']==s]) for s in stages],label=e,color=COLORS[j])
                b.plot(stages,[np.mean([r['correction_cosine_to_base'][j] for r in records if r['stage']==s]) for s in stages],label=e,color=COLORS[j])
            a.set(title=NAMES[v],xlabel='Measurement',ylabel='Mean correction logit norm');a.legend(ncol=4,fontsize=8)
            b.set(xlabel='Measurement',ylabel='Mean cosine to base logit',ylim=(-1.05,1.05))
        savefig(fig,out,'residual_magnitude_direction')
    probes=[(v,d['contrast_resolution']) for v,d in evidence.items() if 'contrast_resolution' in d]
    if probes:
        fig,axes=plt.subplots(len(probes),2,figsize=(11,3.4*len(probes)),squeeze=False)
        for (v,probe),(a,b) in zip(probes,axes):
            for number,part in probe['counts'].items():
                gaps=part['ceiling']-np.asarray(part['scores']);a.plot(np.sort(gaps),np.linspace(0,1,len(gaps)),label='L='+number)
                norms=np.asarray(part['gradient_norms']);b.plot(np.sort(norms).clip(1e-20),np.linspace(0,1,len(norms)),label='L='+number)
            a.set(title=NAMES.get(v,'DAD-MLP')+' frozen training-policy checkpoint',xlabel='Gap below each training-bound ceiling (nats)',ylabel='Empirical cumulative fraction');a.legend()
            b.set(xscale='log',xlabel='Trajectory sensitivity to post-initial action logits',ylabel='Empirical cumulative fraction');b.legend()
        savefig(fig,out,'contrast_resolution_diagnostic')
    write(out/'mechanism_evidence.json',evidence)
    return evidence


def conclusions(family,items):
    complete=[x for x in family if x['status']=='complete'];dad=next(x for x in items if x['method']=='dad');single=next(x for x in items if x['kind']=='single');continued=next(x for x in items if x['kind']=='continued')
    best=max(complete,key=lambda x:x['eig']['mean']) if complete else None
    if not best:return ['Results are pending.']
    result=[f"The strongest completed MoE variant is {best['label']} at {best['eig']['mean']:.5f} nats. This is a one-training-seed development result."]
    if dad['status']=='complete':result.append(f"Compared with the original DAD reference, the observed difference is {best['eig']['mean']-dad['eig']['mean']:+.5f} nats; residual methods include an extra pretrained training budget.")
    result.append('Historical SwitchMoE improves numerically on ordinary MoE, but also used a different sharpening schedule. Its hard learned gate always requests rerouting; always-reroute EIG is identical. The historical improvement does not demonstrate useful temporal persistence.')
    result.append(f"Compared with the original DAD-MLP checkpoint, the observed difference is {best['eig']['mean']-single['eig']['mean']:+.5f} nats.")
    if continued['status']=='complete':
        d=paired_runs(best,continued);ci=d.get('conditional_95_interval')
        result.append(f"Against equal-additional-budget continued DAD-MLP, the paired development difference is {d['mean']:+.5f} nats, conditional trajectory 95% interval {ci}. This excludes training-seed variability.")
    else:result.append('Equal-additional-budget continued DAD-MLP is pending, so extra training remains an unresolved explanation.')
    result.append('ResMoE learned routing and forced dominant-expert routing currently match within paired uncertainty. Proposal disagreement or higher overall EIG alone does not establish beneficial expert specialization. Forced routing may induce unfamiliar histories.')
    matched=next(x for x in items if x['kind']=='matched')
    if matched['status']=='complete':
        result.append(f"The matched193-input single policy achieves {matched['eig']['mean']:.5f} nats with one million training episodes. Ordinary MoE and historical SwitchMoE remain below it, so the full-expert deficit cannot be explained solely by the original772-versus193 input convention. Residual variants include an additional pretrained policy budget and greater model capacity.")
    return result


def intervention_diagnostics(best,improved,out):
    if not improved or improved['status']!='complete':return None
    pairs=[best,improved];fig,axes=plt.subplots(2,2,figsize=(12,7))
    for col,x in enumerate(pairs):
        d=x['diagnostics'];ep=np.array([p['episodes'] for p in d['points']])/1e6;L=x['metadata']['settings']['contrasts'];ceiling=math.log(L+1)
        axes[0,col].plot(ep,[ceiling-p['eig'] for p in d['points']],lw=.7)
        axes[0,col].set(title=x['label']+f' (training L={L})',xlabel='Additional training episodes (millions)',ylabel='Gap below own training ceiling (nats)')
        for j in range(4):axes[1,col].plot(ep,[p['usage'][j] for p in d['points']],label=EXPERTS[j],color=COLORS[j],lw=.8)
        axes[1,col].set(xlabel='Additional training episodes (millions)',ylabel='Relaxed expert mass',ylim=(-.02,1.02));axes[1,col].legend(ncol=4,fontsize=8)
    savefig(fig,out,'controlled_intervention_training')
    result={x['label']:dict(training_contrasts=x['metadata']['settings']['contrasts'],
        near_ceiling_fraction=x['diagnostics']['fraction_within_point1_of_training_ceiling'],
        gradient_exposure=x['diagnostics']['gradient_exposure'],management=x['management']) for x in pairs}
    path=Path(improved['run']).parent/(improved['method']+'_relaxed')
    if (path/'summary.json').exists():
        summary=read(path/'summary.json')
        assert summary['case_identity_sha256']==improved['cases_sha256']
        assert summary['contrast_bank_sha256']==improved['bank_sha256']
        soft=np.array([r['spce'] for r in read(path/'scores.json')]);hard=np.array([r['spce'] for r in improved['rows']])
        result['intervention_hard_relaxed']=dict(hard=stats(hard),relaxed=stats(soft),hard_minus_relaxed=stats(hard-soft))
    routing=Path(improved['run']).parent/(improved['method']+'_router_contribution')/'report.json'
    if routing.exists():result['intervention_routing_contribution']=read(routing)
    geometry=Path(improved['run']).parent/(improved['method']+'_geometry')/'geometry.json'
    if geometry.exists():result['intervention_geometry']=read(geometry)
    write(out/'controlled_intervention_diagnostics.json',result)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--residual-campaign',required=True);ap.add_argument('--controls-campaign',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--improvement-run');ap.add_argument('--improvement-variant',choices=list(NAMES));ap.add_argument('--improvement-control-run');args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    prior=root/'experiments/10132026/10132026_bounded_location_restart/pilot';controls=root/'experiments/10142026/10142026_moe_controlled_diagnosis/controls'
    residual=Path(args.residual_campaign);fair=Path(args.controls_campaign)
    specs=[(prior,m,label,'external') for m,label in [('random','Random'),('fixed','Optimized Fixed'),('dad','DAD reference'),('rl_sboed','RL-sBOED'),('scorebed','ScoreBED')]]
    specs += [(root/'experiments/10142026/10142026_vsoed_incremental_discount_correction/pilot_retry','vsoed','vsOED (latest corrected)','external'),(controls,'sum_single','DAD-MLP','single')]
    specs += [(root/'experiments/10142026/10142026_223452_sum_moe_slower_sharpening/pilot','moe_sboed',NAMES['moe'],'moe'),
        (residual/'resmoe','resmoe',NAMES['resmoe'],'resmoe'),
        (root/'experiments/10142026/10142026_232132_strategy_moe_pilot/pilot','strategy_moe_sboed',NAMES['switchmoe'],'switchmoe'),
        (residual/'switchresmoe','switchresmoe',NAMES['switchresmoe'],'switchresmoe')]
    specs += [(fair/'continued_dad_mlp','sum_single','DAD-MLP continued training','continued'),(fair/'matched_sum_single_193','sum_single_193','Matched193-input single policy','matched')]
    if args.improvement_control_run:specs.append((Path(args.improvement_control_run),'sum_single','DAD-MLP continued training, matched intervention L','contrast_control'))
    items=[load_run(root,*spec) for spec in specs];family=[x for x in items if x['kind'] in NAMES];complete=[x for x in family if x['status']=='complete']
    best=max(complete,key=lambda x:x['eig']['mean']) if complete else None
    improvement=load_run(root,Path(args.improvement_run),args.improvement_variant,NAMES[args.improvement_variant]+' controlled intervention','intervention') if args.improvement_run else None
    comparisons={}
    for a in family:
        for b in items:
            if b['kind'] in {'single','continued','matched'}:comparisons[a['kind']+'_minus_'+b['kind']]=paired_runs(a,b)
    comparisons['switchmoe_minus_moe']=paired_runs(next(x for x in family if x['kind']=='switchmoe'),next(x for x in family if x['kind']=='moe'))
    comparisons['switchresmoe_minus_resmoe']=paired_runs(next(x for x in family if x['kind']=='switchresmoe'),next(x for x in family if x['kind']=='resmoe'))
    if improvement and improvement['status']=='complete':
        for b in ([best] if best else [])+[x for x in items if x['kind'] in {'single','continued','contrast_control','matched'}]:comparisons['intervention_minus_'+b['kind']]=paired_runs(improvement,b)
    plots(items,out)
    consistency=route_consistency(root,residual,family,out)
    improved_diagnostics=intervention_diagnostics(best,improvement,out)
    evidence=mechanism_evidence(root,out);answers=conclusions(family,items)
    if improvement and improvement['status']=='complete':
        answers.append(f"The controlled intervention gives {improvement['eig']['mean']:.5f} nats. Its paired differences against the original variant and all single-policy controls are reported; this remains the same canonical variant.")
        intervention=read(root/'experiments/10152026/10152026_moe_four_variant_research/controlled_intervention_plan.json')
        evidence['controlled_intervention_plan']=intervention
    report=dict(controlled_intervention_diagnostics=improved_diagnostics,mechanism_evidence=evidence,research_conclusions=answers,analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),hard_relaxed_eig=consistency,primary_metric='Common held-out EIG, nats;2000 trajectories/L1000000',training_objective='Full-horizon sPCE; initial comparisons L1024, controlled contrast sensitivity separately records L4096',
        training_seed=101,evaluation_seed=1001,seed_role='development, not untouched final confirmation',
        strongest_completed_initial_variant=best['kind'] if best else None,paired_comparisons=comparisons,
        limitations=['One independent training seed; intervals condition on trained checkpoints and the shared contrast bank, not training-seed or contrast-bank variability',
            'Historical SwitchMoE uses legacy gamma schedule; original slower MoE and residual variants use30/95 cosine',
            'Residual models include pretrained base effort; continued DAD-MLP is required for attribution',
            'DAD-MLP input772 versus full-MoE input193; matched193 control reported separately',
            'Repeated seed101 training uses the established step-seed convention; generated episode counts do not assert unique samples across pretraining and additional-training phases',
            'GPU posterior inference overlaps parts of the fairness training campaign; actual elapsed times are not isolated speed benchmarks',
            'External RL-sBOED is an existing documented two-Q adaptation; exact published reproduction is not established',
            'Forced-routing failures may reflect out-of-distribution histories'],items=items,improvement=improvement)
    all_items=items+([improvement] if improvement else [])
    reference=next(x for x in items if x['kind']=='single')
    paired_items=[x for x in all_items if x['status']=='complete' and x['cases_sha256']==reference['cases_sha256'] and x['bank_sha256']==reference['bank_sha256']]
    with (out/'paired_scores.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['trajectory_id']+[x['label'] for x in paired_items])
        for i in range(len(reference['rows'])):writer.writerow([i]+[x['rows'][i]['spce'] for x in paired_items])
    report['paired_score_export']=dict(file='paired_scores.csv',models=[x['label'] for x in paired_items],cases_sha256=reference['cases_sha256'],bank_sha256=reference['bank_sha256'])
    for x in all_items:
        x.pop('rows',None)
        if x.get('diagnostics'):
            write(out/(x['kind']+'_training_diagnostics.json'),x['diagnostics'])
            x['diagnostics'].pop('logged_trajectory_scores',None)
    write(out/'research_report.json',report)
    with (out/'results.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['model','status','eig','mc_se','additional_training_episodes','total_training_episodes','training_seconds','base_training_seconds','online_seconds','parameters'])
        for x in all_items:w.writerow([x['label'],x['status'],x.get('eig',{}).get('mean'),x.get('eig',{}).get('mc_standard_error'),x.get('additional_episodes'),x.get('total_episodes'),x.get('training_seconds'),x.get('base_training_seconds'),x.get('online_seconds'),x.get('parameter_count')])
    body='<!doctype html><meta charset="utf-8"><title>MoE-sBOED controlled research report</title><style>body{font:16px system-ui;max-width:1200px;margin:35px auto;padding:0 25px;color:#182431}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:left}img{max-width:100%}pre{white-space:pre-wrap;background:#f3f5f7;padding:16px}</style>'
    body+='<h1>MoE-sBOED: four-variant research report</h1><p>Development study. Full results are reported only for completed runs. No statistical superiority across training seeds is claimed.</p>'
    body+='<h2>A. Exactly four MoE-sBOED variants</h2>'+table(family)
    body+=parameter_table(family,root)
    body+='<h2>B. External algorithms and strong single-policy controls</h2>'+table([x for x in items if x['kind'] not in NAMES])
    body+='<h2>C. Mechanism diagnostics</h2><p>The first learned measurement is excluded from expert usage. Gradient exposure thresholds and all stage-wise data are retained in machine-readable outputs. Training sPCE values are not held-out EIG.</p>'
    for name in ['four_variant_eig','expert_utilization_gradient_exposure','stage_utilization_switching','strategy_durations','training_curves_score_distributions','routing_gradient_entropy_sharpening','hard_relaxed_eig','expert_disagreement_observation_sensitivity','expert_regions_observable_usage','residual_magnitude_direction','contrast_resolution_diagnostic']:
        if (out/(name+'.png')).exists():body+=f'<p><a href="{name}.svg"><img src="{name}.png" alt="{name}"></a></p>'
    body+='<h2>D. Controlled improvement</h2>'+(table([best,improvement]+[x for x in items if x['kind']=='contrast_control']) if improvement else '<p>Not run yet. Selection and cost preflight are recorded separately; no performance is assumed.</p>')
    if (out/'controlled_intervention_training.png').exists():body+='<p><a href="controlled_intervention_training.svg"><img src="controlled_intervention_training.png" alt="Controlled intervention training diagnostics"></a></p>'
    body+='<h2>E. Reproducibility and paired differences</h2><p><a href="research_report.json">Complete configurations, source paths, costs, diagnostics and limitations (JSON)</a> · <a href="results.csv">Results CSV</a> · <a href="paired_scores.csv">All paired trajectory scores</a></p><pre>'+html.escape(json.dumps(comparisons,indent=2))+'</pre>'
    body+='<h2>F. Research conclusions and limits</h2><ol>'+''.join('<li>'+html.escape(x)+'</li>' for x in answers)+'</ol><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in report['limitations'])+'</ul><p>Incomplete experiments and the one controlled intervention remain pending where indicated. Expert proposal differences alone do not establish useful specialization.</p>'
    (out/'report.html').write_text(body)
    print(json.dumps(dict(output=str(out),strongest_completed_initial_variant=report['strongest_completed_initial_variant'],pending=[x['label'] for x in items if x['status']!='complete']),indent=2))

if __name__=='__main__':main()
