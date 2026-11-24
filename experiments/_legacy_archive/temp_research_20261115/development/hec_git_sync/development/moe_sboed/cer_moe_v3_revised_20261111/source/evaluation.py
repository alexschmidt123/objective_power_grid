"""New common-bank prior-contrast EIG with complete policy resimulation."""
import gc,subprocess
from common import *
from integration import make_policy
from frozen_evaluator import contrast_scores
from parent_evaluation import stats,holm
E=O/'evaluation'

def load_cer(arm):
    # Only a small template is needed; never load the 1M-episode runtime for eval.
    from evaluate_v2 import actor_load
    parent,_=actor_load('candidate')
    class Holder:pass
    holder=Holder();holder.actor_net=parent
    policy=make_policy(holder)
    from src.research.vendor.vsoed.utils import Net
    critic=Net([119,256,256,256,1],torch.nn.ReLU,torch.tensor([[-float('inf'),float('inf')]],device='cuda'),'critic',None,2).cuda()
    path=O/f'arms/{arm}/actor.pt';cp=torch.load(path,map_location='cuda',weights_only=False)
    policy.load_state_dict(cp['actor_state']);critic.load_state_dict(cp['critic_state'])
    policy.epsilon=CFG['final_epsilon'];return policy.eval().requires_grad_(False),critic.eval().requires_grad_(False)

@torch.no_grad()
def simulate(policy,critic,truth,noise,u,uniform=False,expert=None,collect=False):
    n=len(truth);ds=truth.new_zeros(n,30,2,dtype=torch.float32);ys=truth.new_zeros(n,30,1,dtype=torch.float32)
    actions=[];observations=[];record={k:[] for k in ['X','probabilities','selected','values','proposals']}
    for t in range(30):
        x=prefix_inputs(ds,ys,torch.full((n,),t,device=truth.device,dtype=torch.long))
        action,info=policy.select(x,critic,uniform=uniform,u=u[:,t].float(),expert=expert)
        action=action.double();distance=(truth.reshape(n,2,2)-action[:,None]).square().sum(-1)
        y=(.1+(distance+1e-4).reciprocal().sum(-1)).log()+.5*noise[:,t]
        assert torch.isfinite(action).all() and bool((action.abs()<=4).all())
        ds[:,t]=action.float();ys[:,t,0]=y.float();actions.append(action);observations.append(y[:,None])
        if collect:
            record['X'].append(x)
            for k in record:
                if k!='X':record[k].append(info[k])
    data=dict(actions=actions,observations=observations,truth=truth)
    if collect:data.update({k:torch.stack(v,1) for k,v in record.items()})
    return data

@torch.no_grad()
def evaluate_one(name,bank,neg,u,deadline):
    if name in ['full','control','uniform'] or name.startswith('expert'):
        policy,critic=load_cer('control' if name=='control' else 'full')
    else:
        from evaluate_v2 import actor_load
        policy,_=actor_load('candidate' if name=='hec_v2' else name);critic=None
    tick=time.monotonic();scores=[];curves=[];records={};bench=type('Bench',(),{'sigma':.5,'calls':0})()
    for lo in range(0,len(bank['truth']),16):
        if time.monotonic()>deadline:raise TimeoutError('Predeclared evaluation limit')
        sl=slice(lo,lo+16);truth=torch.tensor(bank['truth'][sl],device='cuda');noise=torch.tensor(bank['noise'][sl],device='cuda')
        if critic is not None:
            data=simulate(policy,critic,truth,noise,torch.tensor(u[sl],device='cuda'),uniform=name in ['uniform','control'],
                          expert=int(name[-1])-1 if name.startswith('expert') else None,collect=name=='full')
        else:
            from evaluate_v2 import simulate as old_simulate
            data=old_simulate(policy,truth,noise,torch.tensor(u[sl],device='cuda'),'candidate' if name=='hec_v2' else 'vsoed')
        v,_=contrast_scores(bench,data,neg,16384,'cuda');scores.extend(v[:,-1].numpy());curves.append(v.numpy())
        if name=='full':
            for k in ['X','probabilities','selected','values','proposals']:records.setdefault(k,[]).append(data[k].cpu().numpy())
    scores=np.array(scores);np.save(E/f'{name}_scores.npy',scores);np.save(E/f'{name}_curves.npy',np.concatenate(curves))
    if records:np.savez_compressed(E/'full_histories.npz',**{k:np.concatenate(v) for k,v in records.items()})
    receipt=dict(**stats(scores),seconds=time.monotonic()-tick,simulator_observations=len(scores)*30,
                 negative_density_terms=bench.calls,positive_density_terms=len(scores)*30)
    put(E/f'{name}_receipt.json',receipt);print('EVALUATED',name,json.dumps(receipt),flush=True)
    return scores,receipt

class Deployment:
    def __init__(self,policy,critic):self.policy=policy;self.critic=critic;self.belief=policy.belief
    def __call__(self,x):return self.policy.select(x,self.critic)[0]

@torch.no_grad()
def diagnostics(deadline):
    from value_signatures import conditional_particles,suffix_values
    policy,critic=load_cer('full');rng=np.random.default_rng(CFG['diagnostic_seed']);generator=torch.Generator(device='cuda').manual_seed(CFG['diagnostic_seed'])
    torch.manual_seed(CFG['diagnostic_seed']);n=CFG['diagnostic_cases']
    truth=torch.tensor(rng.standard_normal((n,4)),device='cuda');noise=torch.tensor(rng.standard_normal((n,30)),device='cuda');u=torch.tensor(rng.random((n,30)),device='cuda')
    hist=simulate(policy,critic,truth,noise,u,collect=True)
    deployment=Deployment(policy,critic);rows=[];Q=[];P=[];ESS=[];variance=[]
    for stage in CFG['diagnostic_stages']:
        stage_rows=[];stage_ess=[];x=hist['X'][:,stage];acts,q,p=policy.distribution(x,critic)
        w,mu,sd=policy.belief.params(x);m=(w[...,None]*mu).sum(1);vv=((w[...,None]*(mu.square()+sd.square())).sum(1)-m.square()).sum(-1)
        for lo in range(0,n,8):
            if time.monotonic()>deadline:raise TimeoutError('Conditional diagnostics deadline')
            xx=x[lo:lo+8];th,pw,ess=conditional_particles(policy.belief,xx,4096,generator)
            stage_rows.append(suffix_values(deployment,xx,acts[lo:lo+8],th,pw,CFG['diagnostic_replicates'],generator).cpu().numpy());stage_ess.append(ess.cpu().numpy())
        rows.append(np.concatenate(stage_rows));ESS.append(np.concatenate(stage_ess));Q.append(q.cpu().numpy());P.append(p.cpu().numpy());variance.append(vv.cpu().numpy())
    values=np.stack(rows,1);means=values.mean(-1);q=np.stack(Q,1);p=np.stack(P,1);vv=np.stack(variance,1);ess=np.stack(ESS,1)
    expected=(p*means).sum(-1);uniform=means.mean(-1);best_q=q.argmax(-1);best_mc=means.argmax(-1)
    strengths={};half=n//2
    for j,stage in enumerate(CFG['diagnostic_stages']):
        threshold=np.median(vv[:half,j])
        for label,mask in [('lower_uncertainty',vv[:,j]<=threshold),('higher_uncertainty',vv[:,j]>threshold)]:
            cal=np.flatnonzero(mask[:half]);val=np.flatnonzero(mask[half:])+half
            if len(cal)<8 or len(val)<8:continue
            chosen=int(means[cal,j].mean(0).argmax())
            strengths[f'stage{stage}_{label}']=dict(calibration_best_expert=chosen+1,validation_cases=len(val),
                expert_minus_uniform=[stats(means[val,j,k]-uniform[val,j]) for k in range(4)],
                calibrated_expert_minus_others=stats(means[val,j,chosen]-np.delete(means[val,j],chosen,axis=1).mean(-1)))
    align=dict(expected_routing_minus_uniform_MC=stats((expected-uniform).mean(1)),
               critic_argmax_matches_MC_fraction=float((best_q==best_mc).mean()),
               MC_estimated_expert_means=means.mean((0,1)).tolist(),conditional_strengths=strengths,
               mean_conditional_MCSE=float((values.std(-1,ddof=1)/math.sqrt(CFG['diagnostic_replicates'])).mean()),
               diagnostic_initial_observations=n*30,
               diagnostic_suffix_observations=n*4*CFG['diagnostic_replicates']*sum(30-t for t in CFG['diagnostic_stages']),
               posterior_ESS_quantiles=np.quantile(ess,[0,.1,.5,.9,1]).tolist(),
               limitations='SNIS and frozen variational posterior approximate conditional information; noisy MC maxima; one trained lineage; descriptive subgroup diagnostics')
    np.savez_compressed(E/'conditional_value_diagnostics.npz',values=values,critic_values=q,probabilities=p,posterior_variance=vv,ess=ess)
    full=np.load(E/'full_histories.npz');g=full['probabilities'];selected=full['selected']
    align.update(selection_frequency=np.bincount(selected.flatten(),minlength=4).tolist(),
                 stage_selection_frequency=[np.bincount(selected[:,t],minlength=4).tolist() for t in range(30)],
                 within_stage_probability_SD=g.std(0).mean(0).tolist(),mean_probability=g.mean((0,1)).tolist())
    put(E/'COMPETENCY_AND_ROUTING.json',align)
    latency={}
    for arm in ['full','control']:
        model,value=load_cer(arm);x=hist['X'][:1,12]
        for _ in range(10):model.select(x,value,uniform=arm=='control')
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(100):model.select(x,value,uniform=arm=='control')
        torch.cuda.synchronize();latency[arm]=(time.perf_counter()-start)/100
    put(E/'LATENCY.json',dict(seconds_per_design=latency,repeats=100,history_stage=12,includes_belief_four_proposals_critic_and_selection=True))

def run_evaluation():
    E.mkdir();tick=time.monotonic();deadline=tick+CFG['evaluation_max_seconds']
    assert all(json.loads((O/f'arms/{arm}/RESULT.json').read_text())['status']=='COMPLETE' for arm in CFG['arms'])
    rng=np.random.default_rng(CFG['evaluation_seed']);bank=dict(truth=rng.standard_normal((CFG['main_cases'],4)),noise=rng.standard_normal((CFG['main_cases'],30)),contrasts=rng.standard_normal((CFG['negative_contrasts'],4)))
    np.savez_compressed(E/'development_validation_bank.npz',**bank);u=rng.random((CFG['main_cases'],30));np.save(E/'routing_uniforms.npy',u)
    protected={arm:sha(O/f'arms/{arm}/actor.pt') for arm in CFG['arms']};neg=torch.tensor(bank['contrasts'],device='cuda');scores={};methods={};status='INCOMPLETE'
    try:
        with torch.no_grad(),torch.device('cuda'):
            for name in ['full','control','uniform','expert1','expert2','expert3','expert4','hec_v2','vsoed']:
                scores[name],methods[name]=evaluate_one(name,bank,neg,u,deadline);gc.collect();torch.cuda.empty_cache()
        for name in ['dad','rl_sboed','scorebed']:
            with (E/f'{name}_worker.log').open('wb') as f:
                subprocess.run(['/home/grads/g/g.lin/miniconda3/envs/mocu_optimized/bin/python',str(O/'source/historical_worker.py'),name],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=max(1,deadline-time.monotonic()),env={**os.environ,'MOE_V2_RUN_ROOT':str(O),'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1'})
            scores[name]=np.load(E/f'{name}_scores.npy');methods[name]=json.loads((E/f'{name}_receipt.json').read_text())
        primary={name:stats(scores['full']-scores[name]) for name in CFG['primary_comparisons']};holm(primary,'Holm7_p')
        external={name:stats(scores['full']-scores[name]) for name in ['vsoed','dad','rl_sboed','scorebed']};holm(external,'Holm4_p')
        put(E/'EIG_RESULTS.json',dict(methods=methods,primary=primary,external=external,bank_sha256=sha(E/'development_validation_bank.npz'),single_training_lineage=True,finite_contrast_lower_bound=True))
        with torch.no_grad(),torch.device('cuda'):diagnostics(deadline)
        status='COMPLETE'
    finally:
        assert all(sha(O/f'arms/{arm}/actor.pt')==v for arm,v in protected.items())
        put(E/'COMPLETION.json',dict(status=status,seconds=time.monotonic()-tick,policies_completed=list(scores),checkpoints_unchanged=True))
