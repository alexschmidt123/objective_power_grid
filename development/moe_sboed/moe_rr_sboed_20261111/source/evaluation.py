"""Common-bank EIG and specialist interventions: complete resimulations."""
import gc,subprocess
from common import *
from rr_policy import RRPolicy
from frozen_evaluator import contrast_scores
from parent_evaluation import stats,holm
E=O/'evaluation'

def load_rr(arm):
    from evaluate_v2 import actor_load
    parent,_=actor_load('candidate');policy=RRPolicy(parent,CFG)
    cp=torch.load(O/f'arms/{arm}/actor.pt',map_location='cuda',weights_only=False)
    policy.load_state_dict(cp['actor_state']);return policy.eval().requires_grad_(False)

@torch.no_grad()
def simulate(policy,truth,noise,u,uniform=False,disable=None,collect=False,prefix=None,intervene_once=False):
    n=len(truth);ds=truth.new_zeros(n,30,2,dtype=torch.float32);ys=truth.new_zeros(n,30,1,dtype=torch.float32)
    actions=[];observations=[];start=0
    if prefix is not None:
        previous_actions,previous_observations=prefix;start=len(previous_actions)
        for t in range(start):
            actions.append(previous_actions[t]);observations.append(previous_observations[t])
            ds[:,t]=previous_actions[t].float();ys[:,t]=previous_observations[t].float()
    records={k:[] for k in ['X','probabilities','chosen_pair','selected','delta','base_action']}
    for t in range(start,30):
        x=prefix_inputs(ds,ys,torch.full((n,),t,device=truth.device,dtype=torch.long))
        action,info=policy.select(x,u=u[:,t].float(),uniform=uniform,epsilon=CFG['joint_pair_exploration'],
                                disable=disable if not intervene_once or t==start else None,collect=True)
        action=action.double();distance=(truth.reshape(n,2,2)-action[:,None]).square().sum(-1)
        y=(.1+(distance+1e-4).reciprocal().sum(-1)).log()+.5*noise[:,t]
        assert torch.isfinite(action).all() and bool((action.abs()<=4).all())
        ds[:,t]=action.float();ys[:,t,0]=y.float();actions.append(action);observations.append(y[:,None])
        if collect:
            records['X'].append(x)
            for key in records:
                if key!='X':records[key].append(info[key])
    result=dict(actions=actions,observations=observations,truth=truth)
    if collect:result.update({k:torch.stack(v,1) for k,v in records.items()})
    return result

@torch.no_grad()
def evaluate_one(name,bank,negative,u,deadline):
    rr=name in ['full','control','uniform_deployment'] or name.startswith('remove')
    if rr:model=load_rr('control' if name=='control' else 'full')
    else:
        from evaluate_v2 import actor_load
        model,_=actor_load('candidate' if name in ['hec_v2','hec_uniform'] else 'ordinary' if name=='conventional_moe' else name)
    tick=time.monotonic();scores=[];curves=[];records={};bench=type('Bench',(),{'sigma':.5,'calls':0})()
    for lo in range(0,len(bank['truth']),16):
        if time.monotonic()>deadline:raise TimeoutError('Locked evaluation limit')
        sl=slice(lo,lo+16);truth=torch.tensor(bank['truth'][sl],device='cuda');noise=torch.tensor(bank['noise'][sl],device='cuda');uu=torch.tensor(u[sl],device='cuda')
        if rr:
            data=simulate(model,truth,noise,uu,uniform=name in ['control','uniform_deployment'],disable=int(name[-1])-1 if name.startswith('remove') else None,collect=name=='full')
        else:
            from evaluate_v2 import simulate as legacy_simulate
            extra=dict(gates=torch.full((30,4),.25,device='cuda')) if name=='hec_uniform' else {}
            data=legacy_simulate(model,truth,noise,uu,'candidate' if name in ['hec_v2','hec_uniform'] else 'ordinary' if name=='conventional_moe' else name,**extra)
        values,_=contrast_scores(bench,data,negative,16384,'cuda');scores.extend(values[:,-1].numpy());curves.append(values.numpy())
        if name=='full':
            for k in ['X','probabilities','chosen_pair','selected','delta','base_action']:records.setdefault(k,[]).append(data[k].cpu().numpy())
            records.setdefault('actions',[]).append(torch.stack(data['actions'],1).cpu().numpy())
            records.setdefault('observations',[]).append(torch.stack(data['observations'],1).cpu().numpy())
    scores=np.array(scores);np.save(E/f'{name}_scores.npy',scores);np.save(E/f'{name}_curves.npy',np.concatenate(curves))
    if records:np.savez_compressed(E/'full_histories.npz',**{k:np.concatenate(v) for k,v in records.items()})
    receipt=dict(**stats(scores),seconds=time.monotonic()-tick,simulator_observations=len(scores)*30,
                 negative_density_terms=bench.calls,positive_density_terms=len(scores)*30)
    put(E/f'{name}_receipt.json',receipt);print('EVALUATED',name,json.dumps(receipt),flush=True)
    return scores,receipt

@torch.no_grad()
def diagnostics(bank,negative,deadline):
    model=load_rr('full');hist=np.load(E/'full_histories.npz');p=hist['probabilities'];pairs=hist['chosen_pair'];selected=hist['selected']
    npairs=np.bincount(pairs.flatten(),minlength=6);nexpert=np.bincount(selected.flatten(),minlength=4)
    x=torch.tensor(hist['X'][:256,12],device='cuda');features=model.frozen(x);acts,delta=model.all_actions(features)
    changes=(acts[:,:,None]-acts[:,None,:]).norm(dim=-1)
    result=dict(pair_selection_counts=npairs.tolist(),expert_selection_counts=nexpert.tolist(),pair_means=p.mean((0,1)).tolist(),
        expert_inclusion_fractions=(nexpert/(len(pairs)*30)).tolist(),within_stage_pair_probability_SD=p.std(0).mean(0).tolist(),
        stage_pair_counts=[np.bincount(pairs[:,t],minlength=6).tolist() for t in range(30)],
        mean_correction_norm=float(np.linalg.norm(hist['delta'],axis=-1).mean()),
        fixed_history_mean_pair_action_distance=float(changes[:,torch.triu(torch.ones(6,6,device='cuda'),diagonal=1).bool()].mean()),
        fixed_history_max_pair_action_distance=float(changes.max()),
        intelligent_routing_requires_resimulated_EIG_evidence=True)
    # A held-out prefix panel retains the actual joint latent draw. It is never
    # supplied to a policy. New suffix noise/choices replace ALL saved futures.
    rng=np.random.default_rng(CFG['diagnostic_seed']);n=128;replicas=2;conditional={};raw={};competencies={}
    truth0=torch.tensor(bank['truth'][:n],device='cuda');truth=truth0.repeat_interleave(replicas,0)
    for stage in [3,12,24]:
        if time.monotonic()>deadline:raise TimeoutError('Conditional panel deadline')
        prefix=([torch.tensor(hist['actions'][:n,t],device='cuda').repeat_interleave(replicas,0) for t in range(stage)],
                [torch.tensor(hist['observations'][:n,t],device='cuda').repeat_interleave(replicas,0) for t in range(stage)])
        noise=torch.tensor(rng.standard_normal((n*replicas,30)),device='cuda');u=torch.tensor(rng.random((n*replicas,30)),device='cuda')
        scores={}
        for label in ['normal','remove1','remove2','remove3','remove4']:
            pieces=[]
            for lo in range(0,len(truth),16):
                sl=slice(lo,lo+16);pr=([v[sl] for v in prefix[0]],[v[sl] for v in prefix[1]])
                data=simulate(model,truth[sl],noise[sl],u[sl],disable=None if label=='normal' else int(label[-1])-1,
                              prefix=pr,intervene_once=True)
                bench=type('Bench',(),{'sigma':.5,'calls':0})();v,_=contrast_scores(bench,data,negative,16384,'cuda');pieces.append(v[:,-1].numpy())
            scores[label]=np.concatenate(pieces).reshape(n,replicas).mean(1)
        xx=torch.tensor(hist['X'][:n,stage],device='cuda');w,mu,sd=model.base.belief.params(xx);m=(w[...,None]*mu).sum(1)
        variance=((w[...,None]*(mu.square()+sd.square())).sum(1)-m.square()).sum(-1).cpu().numpy()
        threshold=np.median(variance[:64]);tests={}
        for label in ['lower','higher']:
            mask=(variance[64:]<=threshold) if label=='lower' else (variance[64:]>threshold)
            for k in range(1,5):
                diff=(scores['normal']-scores[f'remove{k}'])[64:][mask]
                if len(diff)>=16:tests[f'{label}_expert{k}']=stats(diff)
            if int(mask.sum())>=16:
                for k in range(1,5):
                    for ell in range(k+1,5):
                        relative=(scores[f'remove{ell}']-scores[f'remove{k}'])[64:][mask]
                        competencies[f'stage{stage}_{label}_{k}_{ell}']=stats(relative)
        conditional[str(stage)]=tests;raw.update({f'stage{stage}_{k}':v for k,v in scores.items()})
        raw[f'stage{stage}_posterior_variance']=variance
    alltests={f'stage{s}_{key}':v for s,tests in conditional.items() for key,v in tests.items()}
    if alltests:holm(alltests,'Holm_conditional_p')
    if competencies:holm(competencies,'Holm36_p')
    crossovers=[]
    for stage in [3,12,24]:
        for k in range(1,5):
            for ell in range(k+1,5):
                lo=competencies.get(f'stage{stage}_lower_{k}_{ell}');hi=competencies.get(f'stage{stage}_higher_{k}_{ell}')
                if lo and hi and lo['mean']*hi['mean']<0 and max(lo['Holm36_p'],hi['Holm36_p'])<.05:
                    crossovers.append(dict(stage=stage,experts=[k,ell],lower=lo,higher=hi))
    result.update(conditional_interventions=alltests,conditional_cases=128,conditional_noise_replicates=2,
                  conditional_pair_competencies=competencies,conditional_crossovers=crossovers,
                  conditional_suffix_observations=128*2*5*(27+18+6),
                  conditional_panel='Same observed prefixes,retained joint theta,new future noise and routing;remove one adaptation at next decision only,then normal policy;paired finite-prior-contrast differences;replicas clustered by history',
                  limitation='Single lineage and noisy conditional panel;probability variation or action variation alone does not establish useful routing')
    np.savez_compressed(E/'conditional_interventions.npz',**raw);put(E/'ROUTING_RESULTS.json',result)
    latency={}
    for arm in ['full','control']:
        model=load_rr(arm);xx=x[:1]
        for _ in range(10):model.select(xx,uniform=arm=='control')
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(100):model.select(xx,uniform=arm=='control')
        torch.cuda.synchronize();latency[arm]=(time.perf_counter()-start)/100
    put(E/'LATENCY.json',dict(seconds_per_design=latency,repeats=100,stage=12,includes_full_frozen_HEC_and_two_selected_adaptations=True))

def run_evaluation():
    E.mkdir();tick=time.monotonic();deadline=tick+CFG['evaluation_max_seconds']
    assert all(json.loads((O/f'arms/{arm}/RESULT.json').read_text())['status']=='COMPLETE' for arm in CFG['arms'])
    rng=np.random.default_rng(CFG['evaluation_seed']);bank=dict(truth=rng.standard_normal((2000,4)),noise=rng.standard_normal((2000,30)),contrasts=rng.standard_normal((1000000,4)))
    np.savez_compressed(E/'development_validation_bank.npz',**bank);u=rng.random((2000,30));np.save(E/'routing_uniforms.npy',u)
    protected={arm:sha(O/f'arms/{arm}/actor.pt') for arm in CFG['arms']};negative=torch.tensor(bank['contrasts'],device='cuda');scores={};methods={};status='INCOMPLETE'
    try:
        with torch.no_grad(),torch.device('cuda'):
            for name in ['full','control','hec_v2','hec_uniform','sg','conventional_moe','vsoed','uniform_deployment','remove1','remove2','remove3','remove4']:
                scores[name],methods[name]=evaluate_one(name,bank,negative,u,deadline);gc.collect();torch.cuda.empty_cache()
        for name in ['dad','rl_sboed','scorebed']:
            with (E/f'{name}_worker.log').open('wb') as log:
                subprocess.run(['/home/grads/g/g.lin/miniconda3/envs/mocu_optimized/bin/python',str(O/'source/historical_worker.py'),name],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,deadline-time.monotonic()),env={**os.environ,'MOE_V2_RUN_ROOT':str(O),'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1'})
            scores[name]=np.load(E/f'{name}_scores.npy');methods[name]=json.loads((E/f'{name}_receipt.json').read_text())
        primary={name:stats(scores['full']-scores[name]) for name in CFG['primary_comparisons']};holm(primary,'Holm3_p')
        mechanisms={name:stats(scores['full']-scores[name]) for name in CFG['mechanistic_comparisons']};holm(mechanisms,'Holm5_p')
        external={name:stats(scores['full']-scores[name]) for name in ['sg','conventional_moe','vsoed','dad','rl_sboed','scorebed']};holm(external,'Holm6_p')
        put(E/'EIG_RESULTS.json',dict(methods=methods,primary=primary,mechanisms=mechanisms,external=external,bank_sha256=sha(E/'development_validation_bank.npz'),single_training_lineage=True,finite_prior_contrast_lower_bound=True))
        with torch.no_grad(),torch.device('cuda'):diagnostics(bank,negative,deadline)
        status='COMPLETE'
    finally:
        assert all(sha(O/f'arms/{arm}/actor.pt')==v for arm,v in protected.items())
        put(E/'COMPLETION.json',dict(status=status,seconds=time.monotonic()-tick,policies_completed=list(scores),checkpoints_unchanged=True))
