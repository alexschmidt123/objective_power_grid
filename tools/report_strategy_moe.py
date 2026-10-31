"""Frozen policy interventions and quantitative strategy diagnostics, no training."""
import csv,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout
from src.research.location_evaluation import contrast_scores
from src.research.strategy_moe import StrategyMoEPolicy,restore_strategy
from tools.diagnose_frozen_sum_routing import restore,exposure,digest

ROOT=Path('/home/grads/g/g.lin/Documents/objective_power_grid')
C=ROOT/'experiments/10142026/10142026_232132_strategy_moe_pilot'
RUN=C/'pilot'
OLD=ROOT/'experiments/10142026/10142026_moe_controlled_diagnosis/encoder_correction/corrected_seed101'
SLOW=ROOT/'experiments/10142026/10142026_223452_sum_moe_slower_sharpening/pilot'
DAD=ROOT/'experiments/10132026/10132026_bounded_location_restart/pilot'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def paired(a,b):
    x=np.asarray(a)-np.asarray(b);se=float(x.std(ddof=1)/math.sqrt(len(x)))
    return dict(mean=float(x.mean()),standard_error=se,conditional_95_interval=[float(x.mean()-1.96*se),float(x.mean()+1.96*se)])

def evaluation_identity(rows,mode,relaxed):
    cases=json.dumps([[r['truth'],r['measurement_noise_standard']] for r in rows],separators=(',',':'))
    return dict(checkpoint_sha256=digest(RUN/'models/strategy_moe_sboed.pt'),bank_sha256=digest(RUN/'eval/contrast_bank_seed1001.npy'),cases_sha256=hashlib.sha256(cases.encode()).hexdigest(),systems=len(rows),mode=mode,relaxed=relaxed)

def management(active,records):
    n,t=active.shape;counts=np.stack([np.bincount(active[:,s],minlength=4) for s in range(t)])
    durations=[[] for _ in range(4)]
    for row in active:
        start=0
        for end in range(1,t+1):
            if end==t or row[end]!=row[start]:
                durations[row[start]].append(end-start);start=end
    request=np.array([[r['reroute_requested'] for r in trajectory] for trajectory in records])
    changed=np.array([[r['actual_strategy_change'] for r in trajectory] for trajectory in records])
    probability=np.array([[r['reroute_probability'] for r in trajectory] for trajectory in records])
    assert np.array_equal(changed[:,1:],active[:,1:]!=active[:,:-1])
    return dict(first_measurement_excluded=True,initial_routing_request_excluded_from_switch_rate=True,
        available_strategies=['S0','S1','S2','S3'],expert_counts=counts.sum(0).tolist(),
        expert_frequencies=(counts.sum(0)/(n*t)).tolist(),stage_expert_frequencies=(counts/n).tolist(),
        average_consecutive_duration_by_expert=[float(np.mean(x)) if x else None for x in durations],
        runs_by_expert=[len(x) for x in durations],
        average_strategy_changes_per_trajectory=float(changed[:,1:].sum(1).mean()),
        rerouting_request_fraction=float(request[:,1:].mean()),actual_switch_fraction=float(changed[:,1:].mean()),
        rerouting_request_fraction_by_stage=request.mean(0).tolist(),actual_switch_fraction_by_stage=changed.mean(0).tolist(),
        reroute_probability_by_stage=probability.mean(0).tolist(),
        duration_note='Consecutive same-expert runs, including end-censored terminal runs; rerouting to the same expert does not end a run.')

def training(out):
    path=RUN/'models/strategy_moe_sboed_diagnostics.jsonl'
    gradient=exposure(path);write(out/'gradient_exposure.json',gradient)
    rs=[json.loads(s) for s in path.open()];phases={}
    for name,lo,hi in [('first25',0,250000),('second25',250000,500000),('third25',500000,750000),('last25',750000,1000000),('final5',950000,1000000)]:
        rows=[r for r in rs if lo<r['primary_episodes']<=hi]
        phases[name]=dict(updates=len(rows),gate_mass=np.mean([r['router_probabilities'] for r in rows],axis=0).tolist(),argmax_frequency=np.mean([r['hard_choice_frequency'] for r in rows],axis=0).tolist(),gate_mass_by_stage=np.mean([r['router_probabilities_by_stage'] for r in rows],axis=0).tolist(),reroute_probability_by_stage=np.mean([r['strategy_reroute_probability_by_stage'] for r in rows],axis=0).tolist(),expert_gradient_fraction_above_1e_3=[float(np.mean([r['gradient_after'][f'expert_{j}']>1e-3 for r in rows])) for j in range(4)])
    norms={}
    for group in ['encoder','router','switching_gate','first_design']+[f'expert_{j}' for j in range(4)]:
        pre=np.array([r['gradient_before'][group] for r in rs]);post=np.array([r['gradient_after'][group] for r in rs])
        norms[group]=dict(raw_p50_p95_p99_max=np.quantile(pre,[.5,.95,.99,1]).tolist(),clipped_max=float(post.max()),fraction_above_1e_8=float(np.mean(post>1e-8)),fraction_above_1e_3=float(np.mean(post>1e-3)),raw_fraction_above_1000=float(np.mean(pre>1000)))
    write(out/'training_phases.json',phases);write(out/'gradient_norms.json',norms)
    return dict(phases=phases,gradient_norms=norms,exposure=gradient)

@torch.no_grad()
def evaluate_variant(b,saved,rows,bank,label,mode='learned',relaxed=False,diagnose=False):
    out=C/'diagnostics'/label;out.mkdir(parents=True,exist_ok=True)
    identity=evaluation_identity(rows,mode,relaxed)
    if not diagnose and (out/'summary.json').exists():
        cached=read(out/'summary.json')
        if cached.get('identity')!=identity:raise RuntimeError('Cached evaluation provenance differs')
        return cached,read(out/'scores.json')
    p=restore_strategy(saved,mode,relaxed);p.eval()
    truth=torch.tensor([r['truth'] for r in rows],dtype=torch.float64)
    noise=torch.tensor([r['measurement_noise_standard'] for r in rows],dtype=torch.float64)
    scores=[];active=[];records=[];patterns=[];counter_proposals=[];sensitivity=[]
    started=time.perf_counter()
    systems=len(rows)
    for start in range(0,systems,16):
        n=min(16,systems-start);routes=[];executed=[0]*4
        def capture(module,args,value):
            if module.last_route is not None:
                routes.append({k:v.detach().cpu().tolist() for k,v in module.last_route['strategy'].items()})
        handle=p.register_forward_hook(capture)
        hooks=[expert.register_forward_pre_hook(lambda module,args,j=j:executed.__setitem__(j,executed[j]+len(args[0]))) for j,expert in enumerate(p.experts)]
        try:r=rollout(b,p,30,n,1001+start*97451,0,truth=truth[start:start+n],noise=noise[start:start+n],method='strategy_moe_sboed')
        finally:
            handle.remove()
            for hook in hooks:hook.remove()
        if not relaxed:assert sum(executed)==n*29
        case_records=[[{k:v[i] for k,v in d.items()} for d in routes] for i in range(n)]
        records.extend(case_records);active.extend([[d['active_strategy'] for d in t] for t in case_records])
        if diagnose:
            # The native hard rollout must exactly reproduce the saved common run.
            np.testing.assert_allclose(torch.stack(r['actions'],1).numpy(),np.array([x['actions'] for x in rows[start:start+n]]),rtol=0,atol=1e-10)
            np.testing.assert_allclose(torch.stack(r['observations'],1).numpy(),np.array([x['observations'] for x in rows[start:start+n]]),rtol=0,atol=1e-10)
            scores.extend([x['spce'] for x in rows[start:start+n]])
            if start==0:
                info,_=contrast_scores(b,r,bank,16384)
                np.testing.assert_allclose(info[:,-1].numpy(),scores[:n],rtol=0,atol=1e-9)
            # Separate posthoc proposal diagnostics: outside policy inference.
            for stage in range(1,30):
                f=r['features'][stage];context=p.context(f)
                proposals=torch.stack([b.feasible(expert(context[:,j]),r['actions'][:stage],30) for j,expert in enumerate(p.experts)],1)
                for i in range(n):counter_proposals.append(dict(trajectory_id=start+i,measurement_stage=stage+1,proposals=proposals[i].tolist()))
                previous=torch.stack(r['actions'][:stage],1)
                chosen=r['actions'][stage]
                novelty=(chosen[:,None]-previous).square().sum(-1).sqrt().min(-1).values
                proximity=(chosen[:,None]-truth[start:start+n].reshape(n,2,2)).square().sum(-1).sqrt().min(-1).values
                proposal_novelty=(proposals[:,:,None]-previous[:,None]).square().sum(-1).sqrt().min(-1).values
                proposal_proximity=(proposals[:,:,None]-truth[start:start+n].reshape(n,1,2,2)).square().sum(-1).sqrt().min(-1).values
                for i in range(n):patterns.append(dict(trajectory_id=start+i,measurement_stage=stage+1,expert=routes[stage-1]['active_strategy'][i],radius=float(chosen[i].norm()),step_distance=float((chosen[i]-r['actions'][stage-1][i]).norm()),distance_to_previous_measurement=float(novelty[i]),nearest_true_source_distance=float(proximity[i]),matched_proposal_radius=proposals[i].norm(dim=-1).tolist(),matched_proposal_novelty=proposal_novelty[i].tolist(),matched_proposal_nearest_source_distance=proposal_proximity[i].tolist()))
                if stage>1:
                    prev=F.one_hot(torch.tensor(routes[stage-1]['previous_strategy']),4).double()
                    original=p.manager.switching_gate(context.mean(1),prev)[:,0]
                    torch.testing.assert_close(original,torch.tensor(routes[stage-1]['reroute_probability'],dtype=torch.float64),rtol=1e-12,atol=1e-12)
                    vals=[]
                    for sign in [-1,1]:
                        cf=f.clone();cf[:,1+4*(stage-1)+2]+=sign*b.sigma/b.obs_scale
                        vals.append(p.manager.switching_gate(p.context(cf).mean(1),prev)[:,0])
                    for i in range(n):sensitivity.append(dict(trajectory_id=start+i,measurement_stage=stage+1,b_actual=float(original[i]),b_minus_one_noise_sd=float(vals[0][i]),b_plus_one_noise_sd=float(vals[1][i]),threshold_flip=bool((vals[0][i]>=.5)!=(vals[1][i]>=.5))))
        else:
            info,_=contrast_scores(b,r,bank,16384);scores.extend(info[:,-1].tolist())
        if start%400==0:print(label,start+n,flush=True)
    a=np.array(active);mgmt=management(a,records)
    # Relaxed counts/durations are explicitly argmax proxies, not hard execution.
    mgmt['state_interpretation']='relaxed argmax proxy' if relaxed else 'actual hard execution'
    write(out/'management.json',mgmt);write(out/'scores.json',scores)
    with (out/'stage_management.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['measurement_stage','S0','S1','S2','S3','reroute_request_fraction','actual_switch_fraction','mean_reroute_probability'])
        for stage in range(29):w.writerow([stage+2]+mgmt['stage_expert_frequencies'][stage]+[mgmt[k][stage] for k in ['rerouting_request_fraction_by_stage','actual_switch_fraction_by_stage','reroute_probability_by_stage']])
    if diagnose:
        write(out/'observation_sensitivity.json',sensitivity)
        write(out/'measurement_patterns.json',patterns)
        write(out/'matched_expert_proposals.json',counter_proposals)
        mgmt['observation_sensitivity']=dict(cases=len(sensitivity),mean_absolute_probability_change_plus_vs_minus=float(np.mean([abs(x['b_plus_one_noise_sd']-x['b_minus_one_noise_sd']) for x in sensitivity])),fraction_threshold_flips=float(np.mean([x['threshold_flip'] for x in sensitivity])),perturbation='Latest observed log signal +/- one noise SD; stage, preceding active expert, all other observations and designs fixed. No trajectory altered.')
        mgmt['measurement_geometry_by_expert']=[]
        for j in range(4):
            selected=[x for x in patterns if x['expert']==j]
            mgmt['measurement_geometry_by_expert'].append(dict(strategy=f'S{j}',selected_decisions=len(selected),selected_metrics={k:float(np.mean([x[k] for x in selected])) if selected else None for k in ['radius','step_distance','distance_to_previous_measurement','nearest_true_source_distance']},matched_history_metrics={k:float(np.mean([x[k][j] for x in patterns])) for k in ['matched_proposal_radius','matched_proposal_novelty','matched_proposal_nearest_source_distance']}))
        write(out/'management.json',mgmt)
    summary=dict(eig=float(np.mean(scores)),mc_standard_error=float(np.std(scores,ddof=1)/math.sqrt(systems)),management=mgmt,evaluation_seconds=time.perf_counter()-started,identity=identity)
    write(out/'summary.tmp',summary);(out/'summary.tmp').replace(out/'summary.json')
    return summary,scores

def main():
    torch.set_num_threads(2);out=C/'diagnostics';out.mkdir(exist_ok=True)
    assert (RUN/'exit_code').read_text().strip()=='0'
    frozen=read(RUN/'source_manifest.json')['sha256']
    assert not [k for k,v in frozen.items() if k.startswith('src/') and digest(ROOT/k)!=v]
    originals={str(p):digest(p) for p in [RUN/'models/strategy_moe_sboed.pt',OLD/'models/moe_sboed.pt',SLOW/'models/moe_sboed.pt',DAD/'models/dad.pt']}
    config=read(RUN/'run_config.json');assert config['train_seed']==101
    rows=read(RUN/'eval/strategy_moe_sboed_seed1001.json');assert len(rows)==2000
    early=C/'diagnostics/early_learned_hard/scores.json'
    if early.exists():np.testing.assert_allclose(read(early),[r['spce'] for r in rows],rtol=0,atol=1e-9)
    bank=torch.from_numpy(np.load(RUN/'eval/contrast_bank_seed1001.npy'));assert bank.shape==(1000000,4)
    b=make_benchmark(config['benchmark']);saved=torch.load(RUN/'models/strategy_moe_sboed.pt',weights_only=False,map_location='cpu')
    assert saved['history_encoder_revision']==2 and saved['routing_mode']=='learned'
    result=dict(training=training(out),architecture=dict(total_parameters=sum(v.numel() for v in restore_strategy(saved).parameters()),new_switch_parameters=198,hard_threshold=.5,minimum_duration=None,switch_schedule=None,semantic_roles='Unassigned: S0-S3 are neutral identifiers'),evaluation_seed_role='development, not untouched test',training_seeds=[101],simulation_budget=read(RUN/'models/strategy_moe_sboed_training.json'))
    result['native_evaluation_summary']=read(RUN/'summary.json')[0]
    references={}
    for label,path,name in [('DAD',DAD,'dad'),('sum_moe_original',OLD,'moe_sboed'),('sum_moe_slower',SLOW,'moe_sboed')]:
        rr=read(path/f'eval/{name}_seed1001.json');assert len(rr)==2000
        for key in ['truth','measurement_noise_standard']:np.testing.assert_array_equal([r[key] for r in rr],[r[key] for r in rows])
        assert digest(path/'eval/contrast_bank_seed1001.npy')==digest(RUN/'eval/contrast_bank_seed1001.npy')
        references[label]=np.array([r['spce'] for r in rr]);result[label]=dict(eig=float(references[label].mean()),reused=True,common_evaluation_summary=next(x for x in read(path/'summary.json') if x['method']==name))
    # Required equivalence additionally checked on a real saved source checkpoint.
    old_saved=torch.load(OLD/'models/moe_sboed.pt',map_location='cpu',weights_only=False)
    wrapped=StrategyMoEPolicy(30,mode='always_reroute',reference=restore(old_saved)).eval()
    rr=read(OLD/'eval/moe_sboed_seed1001.json')[:16]
    with torch.no_grad():
        replay=rollout(b,wrapped,30,16,1001,0,truth=torch.tensor([x['truth'] for x in rr],dtype=torch.float64),noise=torch.tensor([x['measurement_noise_standard'] for x in rr],dtype=torch.float64))
        np.testing.assert_allclose(torch.stack(replay['actions'],1).numpy(),np.array([x['actions'] for x in rr]),rtol=0,atol=1e-10)
        v,_=contrast_scores(b,replay,bank,16384);np.testing.assert_allclose(v[:,-1].numpy(),[x['spce'] for x in rr],rtol=0,atol=1e-9)
    allscores={}
    for label,mode,soft,diag in [('learned_hard','learned',False,True),('always_reroute','always_reroute',False,False),('always_continue','always_continue',False,False),('learned_relaxed','learned',True,False)]:
        result[label],allscores[label]=evaluate_variant(b,saved,rows,bank,label,mode,soft,diag)
    result['paired_comparisons']={k:paired(allscores['learned_hard'],v) for k,v in references.items()}
    result['paired_comparisons'].update({k:paired(allscores['learned_hard'],allscores[k]) for k in ['always_reroute','always_continue','learned_relaxed']})
    result['same_checkpoint_intervention_note']='Three strategy routing modes share one learned-switching checkpoint; not three independently trained methods. Forced histories can leave usual training distribution.'
    result['limitations']=['One independent training seed; no across-seed superiority claim.','Evaluation seed1001 is development data.','Intervals condition on checkpoints and common contrast bank.','Geometry proxies do not establish disambiguation or verification roles.','State durations alone cannot distinguish learned persistence from a naturally stable per-step router.']
    assert originals=={p:digest(Path(p)) for p in originals};result['preserved_checkpoint_hashes']=originals
    write(out/'report.json',result)
    lines=['# Strategy-MoE-sBOED: one-seed controlled pilot','','Separate sum encoder -> StrategyManager (previous soft state, stage, SwitchingGate) -> existing DSelect1 -> four independent adaptive experts. Hard inference executes only the active expert.','','27 correctness/regression tests passed, including exact always-reroute rollouts and matching gradients. Original MoE entrypoint smoke and Strategy smoke both passed.','','| Policy | EIG | Training |','|---|---:|---|']
    for k in ['DAD','sum_moe_original','sum_moe_slower','always_reroute','always_continue','learned_hard','learned_relaxed']:
        lines.append(f"| {k} | {result[k]['eig']:.6f} | {'reused reference' if k in references else 'same learned Strategy checkpoint'} |")
    lines+=['','The controlled architecture reference is original sum-MoE with the same legacy gamma schedule. The successful slower-schedule result is an additional reference; its routing schedule differs. No second Strategy training was launched.','',result['same_checkpoint_intervention_note'],'',f"Paired learned-hard differences and conditional Monte Carlo intervals: {result['paired_comparisons']}",'','| Mode | S0 % | S1 % | S2 % | S3 % | Reroute requests % | Actual switches % |','|---|---:|---:|---:|---:|---:|---:|']
    for k in ['learned_hard','always_reroute','always_continue']:
        m=result[k]['management'];lines.append('| '+k+' | '+' | '.join(f'{100*x:.3f}' for x in m['expert_frequencies']+[m['rerouting_request_fraction'],m['actual_switch_fraction']])+' |')
    m=result['learned_hard']['management']
    lines+=['',f"Learned hard average run duration S0-S3: {m['average_consecutive_duration_by_expert']}",f"Observation sensitivity: {m['observation_sensitivity']}",'','Per-mode stage_management.csv reports all29 routed stages. gradient_exposure.json, gradient_norms.json and training_phases.json cover the whole training budget. measurement_patterns.json and matched_expert_proposals.json report quantitative geometry under selected and matched histories. These are posthoc diagnostics; unselected experts were not executed by hard policy inference.','',f"Geometry summary: {m['measurement_geometry_by_expert']}",'','Semantic interpretations remain unassigned: radius, spatial novelty and nearest-source distance are proxies; the current diagnostics do not identify hypothesis-disambiguation or verification mechanisms. Frequent selection alone is not specialization.','','All training used 1million primary episodes, L1024, seed101; common evaluation used2000 paired cases, L1million, development seed1001. Final checkpoint chosen in advance; MSE not optimized or used for selection.','',f"Training seconds: {result['simulation_budget']['training_seconds']:.1f}. Native evaluation: {read(RUN/'summary.json')[0]['evaluation_seconds']:.1f} seconds. Frozen-mode EIG diagnostics exclude additional posterior inference.",'','Implementation: src/research/strategy_moe.py, strategy_training.py, optional registration and strategy-record serialization in runner.py, tools/tests/test_strategy_moe.py. Original sparse_moe.py, history implementation, simulator, feasible map, likelihood and common evaluator retain their original hashes. Config: learned.yaml; switching init0.99, no duration/switch schedule, separate component norm5 switching-gate clipping.','']
    (out/'report.md').write_text('\n'.join(lines))
    write(C/'completion.json',dict(status='complete',report=str(out/'report.json'),preserved_checkpoint_hashes=originals))
    print(json.dumps({k:result[k]['eig'] for k in ['DAD','sum_moe_original','sum_moe_slower','always_reroute','always_continue','learned_hard','learned_relaxed']}),flush=True)
if __name__=='__main__':main()
