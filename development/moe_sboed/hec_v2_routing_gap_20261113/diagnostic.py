"""Frozen V2 single-decision routing gap, with independent selection/test banks."""
import argparse,gc,hashlib,json,math,os,pathlib,subprocess,sys,time
sys.dont_write_bytecode=True
import numpy as np
import torch
ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp'
WORK=ROOT/'development/hec_git_sync'
BASE=ROOT/'scratch_runs/moe_hec_v2_icml_20261109'
sys.path.insert(0,str(WORK/'development/moe_sboed/hec_v2_resolution_20261112'))
from run_resolution import load_library,sha,put
from resolution import nested_scores,paired,signal

NAMES=['learned','uniform','favor_E1','favor_E2','favor_E3','favor_E4']

def mixtures(g):
    uniform=torch.full_like(g,.25)
    favored=.5*torch.eye(4,device=g.device,dtype=g.dtype)+.125
    return torch.cat([g[:,None],uniform[:,None],favored[None].expand(len(g),-1,-1)],1)

def prior_logprob(theta):
    return -.5*theta.square().sum(-1)-2*math.log(2*math.pi)

def gaussian_mix_logprob(theta,w,mu,sd):
    terms=-.5*((theta[:,None]-mu)/sd).square()-sd.log()-.5*math.log(2*math.pi)
    return torch.logsumexp(w.log()+terms.sum(-1),-1)

@torch.no_grad()
def posterior(actor,x,ds,ys,stage,count,particles,seed):
    """Self-normalized importance posterior: 10% prior, 90% symmetrized GMM.

    Finite particle approximation, NOT exact posterior sampling. All densities
    use float64 and exact prefix likelihood. Truth is not an argument.
    """
    rng=np.random.default_rng(seed)
    w,mu,sd=actor.belief.params(x)
    w=w[0].double(); w=w/w.sum(); mu=mu[0].double(); sd=sd[0].double()
    ids=torch.tensor(rng.choice(len(w),particles,p=w.cpu().numpy()),device=x.device)
    theta=mu[ids]+sd[ids]*torch.tensor(rng.standard_normal((particles,4)),device=x.device)
    kind=rng.random(particles)
    swap=torch.tensor(kind<.45,device=x.device)
    theta[swap]=theta[swap][:,[2,3,0,1]]
    use_prior=torch.tensor(kind>=.9,device=x.device)
    theta[use_prior]=torch.tensor(rng.standard_normal((int(use_prior.sum()),4)),device=x.device)
    lp=prior_logprob(theta)
    q1=gaussian_mix_logprob(theta,w,mu,sd)
    q2=gaussian_mix_logprob(theta[:,[2,3,0,1]],w,mu,sd)
    lq=torch.logsumexp(torch.stack([lp+math.log(.1),q1+math.log(.45),q2+math.log(.45)]),0)
    likelihood=torch.zeros(particles,device=x.device,dtype=torch.float64)
    for t in range(stage):
        prediction=signal(theta.reshape(particles,2,2),ds[0,t].double())
        likelihood+=-2*(ys[0,t]-prediction).square()
    weight=(lp+likelihood-lq).softmax(0)
    assert torch.isfinite(weight).all() and abs(float(weight.sum())-1)<1e-10
    ess=float(1/weight.square().sum()); maximum=float(weight.max())
    selected=torch.tensor(rng.choice(particles,count,p=weight.cpu().numpy()),device=x.device)
    draw=theta[selected]
    return draw,dict(seed=seed,particles=particles,ESS=ess,max_weight=maximum,
                     selected_unique=int(selected.unique().numel()),
                     posterior_mean=(weight[:,None]*theta).sum(0).cpu().tolist(),
                     approximate=True,proposal='0.1 prior + 0.45 GMM + 0.45 source-permuted GMM')

@torch.no_grad()
def branches(actor,lib,ds,ys,stage,theta,noise,deadline):
    """One intervention at stage; native learned V2 thereafter, branch-local y."""
    m=len(theta); k=len(NAMES)
    actions=ds.expand(m,k,30,2).clone().reshape(m*k,30,2)
    obs=ys.expand(m,k,30).clone().reshape(m*k,30)
    actions[:,stage:]=0; obs[:,stage:]=0
    latent=theta[:,None].expand(m,k,4).reshape(m*k,4)
    eps=noise[:,None].expand(m,k,30).reshape(m*k,30)
    for t in range(stage,30):
        if time.monotonic()>deadline:raise TimeoutError('Fixed wall budget')
        x=lib.packed(actions,obs,t)
        if t==stage:
            _,info=actor(x,collect=True)
            gates=mixtures(info['gates'])
            choice=torch.arange(k,device=x.device).repeat(m)
            weights=gates[torch.arange(m*k,device=x.device),choice]
            # Learned branch uses unmodified forward to avoid override roundoff.
            action=actor(x,routing_override=weights)
            native=actor(x)
            action[choice==0]=native[choice==0]
        else:action=actor(x)
        assert torch.isfinite(action).all() and bool((action.abs()<=4.000001).all())
        actions[:,t]=action
        obs[:,t]=signal(latent.reshape(-1,2,2),action.double())+.5*eps[:,t]
    assert torch.equal(actions[:,:stage],ds[:,:stage].expand(m*k,-1,-1))
    assert torch.equal(obs[:,:stage],ys[:,:stage].expand(m*k,-1))
    return actions,obs,latent

def heldout_summary(selection,test,stages):
    """Independent histories are inference units, not branch replicates."""
    selected=selection.mean(1).argmax(-1)
    global_choice=int(selection.mean((0,1)).argmax())
    stage_choice={int(t):int(selection[stages==t].mean((0,1)).argmax()) for t in np.unique(stages)}
    testmean=test.mean(1); rows=np.arange(len(test))
    chosen=testmean[rows,selected]
    fixed=testmean[:,global_choice]
    staged=testmean[rows,[stage_choice[int(t)] for t in stages]]
    return dict(histories=len(test),selected_choices=selected.tolist(),
                fixed_choice=global_choice,stage_choices=stage_choice,
                choices_selected_using_test=False,
                primary_selected_minus_learned=paired(chosen,testmean[:,0]),
                secondary_selected_minus_fixed=paired(chosen,fixed),
                secondary_selected_minus_stage=paired(chosen,staged),
                secondary_selected_minus_uniform=paired(chosen,testmean[:,1]),
                heldout_mean_by_mixture=dict(zip(NAMES,testmean.mean(0).tolist())),
                heldout_mean_selected=float(chosen.mean()))

def tests(actor,lib,out):
    g=torch.tensor([[.1,.2,.3,.4],[.4,.3,.2,.1]],device='cuda')
    options=mixtures(g)
    assert options.shape==(2,6,4) and torch.allclose(options.sum(-1),torch.ones(2,6,device='cuda'))
    assert torch.equal(options[:,0],g)
    theta=torch.tensor([[.2,-.4,.8,1.],[-.3,.7,-.5,.1]],device='cuda',dtype=torch.float64)
    noise=torch.zeros(2,30,device='cuda',dtype=torch.float64)
    ds,ys,_=lib.rollout(actor,theta,noise)
    for stage in [0,14,29]:
        a,y,_=branches(actor,lib,ds[:1],ys[:1],stage,theta[:1],noise[:1],float('inf'))
        torch.testing.assert_close(a[0],ds[0],atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(y[0],ys[0],atol=2e-4,rtol=2e-5)
        # All branches use the same theta and noise; only first intervention differs.
        torch.testing.assert_close(y[:,stage:],signal(theta[:1].reshape(1,1,2,2),a[:,stage:].double()),atol=1e-12,rtol=1e-12)
    zero=ds.new_zeros(1,30,2); zobs=ys.new_zeros(1,30)
    x=lib.packed(zero,zobs,0)
    _,p=posterior(actor,x,zero,zobs,0,64,8192,171)
    assert abs(p['ESS']-8192)<1e-5 and abs(p['max_weight']-1/8192)<1e-10
    assert max(abs(v) for v in p['posterior_mean'])<.06
    # Deliberately make holdout rankings disagree with selection rankings.
    s=np.zeros((4,3,6)); s[:,:,2]=5
    v=np.zeros((4,5,6)); v[:,:,4]=10
    h=heldout_summary(s,v,np.array([0,0,14,14]))
    assert h['selected_choices']==[2]*4 and h['fixed_choice']==2
    put(out/'VALIDATION.json',dict(status='PASS',mixture_normalization=True,
         learned_branch_matches_original_stages=[0,14,29],prefix_preserved=True,
         exact_signal_noise_chronology=True,prior_posterior_identity=True,
         selection_ignores_test_rankings=True,optimizer_steps=0))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=pathlib.Path,required=True)
    p.add_argument('--run',action='store_true');p.add_argument('--smoke',action='store_true');args=p.parse_args()
    if not args.run:p.error('Explicit --run required')
    out=args.output.expanduser().resolve();assert ROOT/'scratch_runs' in out.parents and not out.exists()
    out.mkdir(parents=True)
    cfg=dict(smoke=args.smoke,stages=[0,14,29] if args.smoke else [3,14,26],
             histories_per_stage=1 if args.smoke else 8,selection_trials=2 if args.smoke else 32,
             test_trials=3 if args.smoke else 64,particles=8192 if args.smoke else 131072,
             selection_contrasts=31 if args.smoke else 4000000,test_contrasts=127 if args.smoke else 16000000,
             minimum_ESS=1 if args.smoke else 256,maximum_weight=1 if args.smoke else .05,
             wall_seconds=300 if args.smoke else 14400,seed=202611130701,mixtures=NAMES,
             primary='selection-chosen mixture minus learned, heldout per-history means; paired95% t interval',
             source_commit=subprocess.check_output(['git','-C',str(WORK),'rev-parse','HEAD'],text=True).strip())
    put(out/'CONFIG.json',cfg)
    sources=[pathlib.Path(__file__).resolve(),*sorted((BASE/'source').glob('*.py')),
             *sorted((WORK/'development/moe_sboed/hec_v2_resolution_20261112').glob('*.py')),
             *sorted((WORK/'development/moe_sboed/hec_v2_counterfactual_routing_20261112').glob('*.py')),
             BASE/'arms/candidate/actor.pt',BASE/'arms/candidate/runtime.pkl']
    manifest={str(f):sha(f) for f in sources};put(out/'PRE_RUN_MANIFEST.json',manifest)
    os.environ['MOE_V2_RUN_ROOT']=str(BASE);sys.path.insert(0,str(BASE/'source'))
    from native_common import original
    original.setup()
    while True:
        try:lease=original.acquire();break
        except BlockingIOError:put(out/'STATUS.json',dict(status='WAITING_FOR_IDLE_GPU'));time.sleep(30)
    start=time.monotonic();deadline=start+cfg['wall_seconds']
    try:
        lib=load_library();actor=lib.load_frozen('cuda');before={k:v.cpu().clone() for k,v in actor.state_dict().items()}
        tests(actor,lib,out)
        rng=np.random.default_rng(cfg['seed']);h=cfg['histories_per_stage']*len(cfg['stages'])
        truth=torch.tensor(rng.standard_normal((h,4)),device='cuda'); noise=torch.tensor(rng.standard_normal((h,30)),device='cuda')
        ds,ys,_=lib.rollout(actor,truth,noise,deadline=deadline)
        stages=np.repeat(cfg['stages'],cfg['histories_per_stage'])
        np.savez_compressed(out/'OBSERVED_HISTORIES.npz',actions=ds.cpu().numpy(),observations=ys.cpu().numpy(),stages=stages)
        # Original hidden truths are never passed to posterior or branching.
        del truth,noise
        quality=[];draws_all=[]
        for i,t in enumerate(stages):
            if time.monotonic()>deadline:raise TimeoutError('Fixed wall budget')
            x=lib.packed(ds[i:i+1],ys[i:i+1],int(t));row={};draws={}
            for split in ['selection','test']:
                seed=cfg['seed']+10000+100*i+(0 if split=='selection' else 1)
                draws[split],row[split]=posterior(actor,x,ds[i:i+1],ys[i:i+1],int(t),cfg[split+'_trials'],cfg['particles'],seed)
            good=all(v['ESS']>=cfg['minimum_ESS'] and v['max_weight']<=cfg['maximum_weight'] for v in row.values())
            quality.append(dict(history=i,stage=int(t),accepted=good,**row));draws_all.append(draws)
            put(out/'POSTERIOR_QUALITY.json',quality)
            put(out/'STATUS.json',dict(status='POSTERIOR_PREFLIGHT',completed=i+1,total=h))
        failed=[q['history'] for q in quality if not q['accepted']]
        if failed:
            assert all(torch.equal(before[k],v.cpu()) for k,v in actor.state_dict().items())
            assert all(sha(f)==v for f,v in manifest.items())
            put(out/'STATUS.json',dict(status='INCONCLUSIVE_POSTERIOR_QUALITY',failed_histories=failed,
                 historical_artifacts_preserved=True,optimizer_steps=0));return
        banks={}
        for split in ['selection','test']:
            bank=np.random.default_rng(cfg['seed']+(1 if split=='selection' else 2)).standard_normal((cfg[split+'_contrasts'],4))
            put(out/(split+'_BANK.json'),dict(seed=cfg['seed']+(1 if split=='selection' else 2),sha256=hashlib.sha256(bank.view(np.uint8)).hexdigest()))
            banks[split]=torch.tensor(bank,device='cuda');del bank
        values={'selection':[],'test':[]}
        for i,t in enumerate(stages):
            draws=draws_all[i]
            for split in ['selection','test']:
                count=cfg[split+'_trials'];seed=cfg['seed']+50000+100*i+(0 if split=='selection' else 1)
                eps=torch.tensor(np.random.default_rng(seed).standard_normal((count,30)),device='cuda')
                a,y,theta=branches(actor,lib,ds[i:i+1],ys[i:i+1],int(t),draws[split],eps,deadline)
                v=[]
                for first in range(0,len(a),16):
                    v.append(nested_scores(a[first:first+16],y[first:first+16],theta[first:first+16],banks[split],
                             [cfg[split+'_contrasts']],deadline=deadline)[cfg[split+'_contrasts']].cpu().numpy())
                scores=np.concatenate(v).reshape(count,len(NAMES));values[split].append(scores)
                np.save(out/f'{split}_history{i:02d}.npy',scores)
                put(out/'STATUS.json',dict(status='RUNNING',history=i+1,total=h,stage=int(t),split=split,seconds=time.monotonic()-start))
                print(json.dumps(json.loads((out/'STATUS.json').read_text())),flush=True)
        result=heldout_summary(np.array(values['selection']),np.array(values['test']),stages)
        result.update(status='COMPLETE',smoke=args.smoke,seconds=time.monotonic()-start,
                      limitations='Finite importance posterior approximation and finite prior-sPCE; 24 histories, one checkpoint, six mixtures, one-decision interventions only. No deployable selector or repeated-routing gain established. Secondary intervals exploratory, unadjusted.')
        assert all(torch.equal(before[k],v.cpu()) for k,v in actor.state_dict().items())
        assert all(sha(f)==v for f,v in manifest.items())
        put(out/'RESULTS.json',result);put(out/'STATUS.json',dict(status='COMPLETE',smoke=args.smoke,seconds=time.monotonic()-start,historical_artifacts_preserved=True,optimizer_steps=0))
    except Exception as exc:
        put(out/'STATUS.json',dict(status='FAILED',error=str(exc)));raise
    finally:
        for f in lease:f.close()

if __name__=='__main__':main()
