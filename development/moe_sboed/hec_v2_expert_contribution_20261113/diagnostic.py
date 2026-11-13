"""Frozen V2 all-consumer expert removal and independent-history regime test."""
import argparse,hashlib,importlib.util,json,os,pathlib,subprocess,sys,time
sys.dont_write_bytecode=True
import numpy as np
import torch
ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp'
WORK=ROOT/'development/hec_git_sync'
PREVIOUS=WORK/'development/moe_sboed/hec_v2_routing_gap_20261113'
spec=importlib.util.spec_from_file_location('routing_gap_reference',PREVIOUS/'diagnostic.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
from run_resolution import knockout_checks
from resolution import pathway_knockout
BASE=old.BASE;put=old.put;sha=old.sha
NAMES=['original']+[f'single_remove_E{k}' for k in range(1,5)]+[f'persistent_remove_E{k}' for k in range(1,5)]

@torch.no_grad()
def knockout_actions(actor,x):
    native,info=actor(x,collect=True);a0=actor.base(x)
    actions=[native]
    for k in range(4):
        a,g=pathway_knockout(actor,info['history'],info['slots'],info['legacy_features'],info['responses'],a0,k)
        assert bool((g[:,k]==0).all())
        actions.append(a)
    return torch.stack(actions,1),info

@torch.no_grad()
def branches(actor,lib,ds,ys,stage,theta,noise,deadline):
    m=len(theta);k=9
    actions=ds.expand(m,k,30,2).clone().reshape(-1,30,2)
    obs=ys.expand(m,k,30).clone().reshape(-1,30)
    actions[:,stage:]=0;obs[:,stage:]=0
    latent=theta[:,None].expand(m,k,4).reshape(-1,4)
    eps=noise[:,None].expand(m,k,30).reshape(-1,30)
    branch=torch.arange(k,device=ds.device).repeat(m)
    for t in range(stage,30):
        if time.monotonic()>deadline:raise TimeoutError('Fixed wall budget')
        x=lib.packed(actions,obs,t);action=actor(x)
        for expert in range(4):
            mask=(branch==expert+5)
            if t==stage:mask=mask|(branch==expert+1)
            if bool(mask.any()):
                xx=x[mask];_,info=actor(xx,collect=True)
                action[mask],_=pathway_knockout(actor,info['history'],info['slots'],info['legacy_features'],info['responses'],actor.base(xx),expert)
        assert torch.isfinite(action).all() and bool((action.abs()<=4.000001).all())
        actions[:,t]=action
        obs[:,t]=old.signal(latent.reshape(-1,2,2),action.double())+.5*eps[:,t]
    assert torch.equal(actions[:,:stage],ds[:,:stage].expand(m*k,-1,-1))
    assert torch.equal(obs[:,:stage],ys[:,:stage].expand(m*k,-1))
    return actions,obs,latent

def features(info,stage):
    g=info['gates'][0].double()
    return np.array([stage/29,float(info['uncertainty_trace'][0]),float(-(g*g.clamp_min(1e-30).log()).sum())])

def fit_regime(x,damage):
    """Choose one stump and leaf expert labels using selection histories only."""
    candidates=[]
    for column in range(x.shape[1]):
        for q in [.33,.5,.67]:
            threshold=float(np.quantile(x[:,column],q));side=x[:,column]>threshold
            if min(int(side.sum()),int((~side).sum()))<max(1,len(x)//6):continue
            labels=[int(damage[side==leaf].mean(0).argmax()) for leaf in [False,True]]
            choices=np.where(side,labels[1],labels[0])
            value=float(damage[np.arange(len(x)),choices].mean())
            candidates.append((value,column,threshold,labels))
    assert candidates
    value,column,threshold,labels=max(candidates,key=lambda v:v[0])
    return dict(feature=column,threshold=threshold,labels=labels,selection_mean_damage=value,
                global_expert=int(damage.mean(0).argmax()))

def choose(regime,x):
    return np.where(x[:,regime['feature']]>regime['threshold'],regime['labels'][1],regime['labels'][0])

def results(selection,test,train_x,test_x):
    # Positive damage means removing this expert reduces terminal utility.
    train_damage=selection[:,:,:1]-selection[:,:,5:9]
    test_damage=test[:,:,:1]-test[:,:,5:9]
    regime=fit_regime(train_x,train_damage.mean(1));chosen=choose(regime,test_x)
    means=test_damage.mean(1);rows=np.arange(len(test_x))
    primary=old.paired(means[rows,chosen],means[:,regime['global_expert']])
    comparisons={}
    for mode,start in [('single',1),('persistent',5)]:
        for k in range(4):
            comparisons[f'{mode}_E{k+1}']=old.paired(test[:,:,0].mean(1),test[:,:,start+k].mean(1))
    # One family of eight secondary removal-effect tests, Holm adjusted.
    ordered=sorted(comparisons,key=lambda k:comparisons[k]['p_two_sided']);previous=0
    for rank,key in enumerate(ordered):
        previous=max(previous,min(1,(len(ordered)-rank)*comparisons[key]['p_two_sided']))
        comparisons[key]['p_Holm8']=previous
    leaves={}
    for leaf in [0,1]:
        idx=(test_x[:,regime['feature']]>regime['threshold'])==bool(leaf)
        leaves[str(leaf)]=dict(histories=int(idx.sum()),selected_expert=regime['labels'][leaf]+1,
            mean_persistent_damage=means[idx].mean(0).tolist() if idx.any() else None)
    return dict(primary_conditional_minus_global_damage=primary,regime=regime,
                feature_names=['stage_fraction','belief_variance_trace','gate_entropy'],
                test_selected_experts=(chosen+1).tolist(),test_regimes=leaves,
                expert_removal_effects=comparisons,
                interpretation='Positive primary difference supports history-dependent differences in expert necessity under persistent masking. It does not establish optimal routing, sparse execution, or an improved deployable policy.')

def tests(actor,lib,out):
    check=knockout_checks(actor)
    theta=torch.tensor([[.2,-.4,.8,1.]],device='cuda',dtype=torch.float64)
    eps=torch.zeros(1,30,device='cuda',dtype=torch.float64)
    ds,ys,_=lib.rollout(actor,theta,eps)
    for t in [0,14,29]:
        a,y,_=branches(actor,lib,ds,ys,t,theta,eps,float('inf'))
        torch.testing.assert_close(a[0],ds[0],atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(y[0],ys[0],atol=2e-4,rtol=2e-5)
        torch.testing.assert_close(y[:,t:],old.signal(theta.reshape(1,1,2,2),a[:,t:].double()),atol=1e-12,rtol=1e-12)
        for k in range(4):torch.testing.assert_close(a[k+1,t],a[k+5,t],atol=0,rtol=0)
    x=np.array([[0,0,0],[0,1,0],[1,2,0],[1,3,0]],float)
    damage=np.array([[3,0,0,0],[3,0,0,0],[0,3,0,0],[0,3,0,0]],float)
    regime=fit_regime(x,damage);assert choose(regime,x).tolist()==[0,0,1,1]
    put(out/'VALIDATION.json',dict(status='PASS',all_consumer_mask=check,
         observed_prefix_preserved=True,native_control_matches=True,exact_signal_chronology=True,
         single_persistent_first_action_equal=True,regime_fit_selection_only=True,optimizer_steps=0))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=pathlib.Path,required=True)
    p.add_argument('--run',action='store_true');p.add_argument('--smoke',action='store_true');args=p.parse_args()
    if not args.run:p.error('Explicit --run required')
    out=args.output.expanduser().resolve();assert ROOT/'scratch_runs' in out.parents and not out.exists();out.mkdir(parents=True)
    cfg=dict(smoke=args.smoke,stages=[0,14,29] if args.smoke else [3,14,26],histories_per_stage=2 if args.smoke else 8,
             selection_trials=2 if args.smoke else 32,test_trials=3 if args.smoke else 64,
             selection_contrasts=31 if args.smoke else 4000000,test_contrasts=127 if args.smoke else 16000000,
             particles=8192 if args.smoke else 131072,minimum_ESS=1 if args.smoke else 256,
             maximum_weight=1 if args.smoke else .05,seed=202611131901,wall_seconds=300 if args.smoke else 14400,
             primary='Heldout persistent-removal damage of selection-trained regime expert minus selection-chosen global expert; paired95% t over independent test histories',
             regime_features=['stage_fraction','belief_variance_trace','gate_entropy'],
             source_commit=subprocess.check_output(['git','-C',str(WORK),'rev-parse','HEAD'],text=True).strip())
    put(out/'CONFIG.json',cfg)
    sources=[*sorted(pathlib.Path(__file__).parent.glob('*.py')),*sorted(PREVIOUS.glob('*.py')),
             *sorted((BASE/'source').glob('*.py')),*sorted((WORK/'development/moe_sboed/hec_v2_resolution_20261112').glob('*.py')),
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
        lib=old.load_library();actor=lib.load_frozen('cuda');before={k:v.cpu().clone() for k,v in actor.state_dict().items()}
        tests(actor,lib,out);records={};quality=[];h=len(cfg['stages'])*cfg['histories_per_stage']
        stages=np.repeat(cfg['stages'],cfg['histories_per_stage'])
        for split_index,split in enumerate(['selection','test']):
            seed=cfg['seed']+split_index*100000;rng=np.random.default_rng(seed)
            truth=torch.tensor(rng.standard_normal((h,4)),device='cuda');noise=torch.tensor(rng.standard_normal((h,30)),device='cuda')
            ds,ys,_=lib.rollout(actor,truth,noise,deadline=deadline);del truth,noise
            np.savez_compressed(out/f'{split}_HISTORIES.npz',actions=ds.cpu().numpy(),observations=ys.cpu().numpy(),stages=stages)
            draws=[];contexts=[];action_stats=[]
            for i,t in enumerate(stages):
                x=lib.packed(ds[i:i+1],ys[i:i+1],int(t));alternatives,info=knockout_actions(actor,x)
                contexts.append(features(info,int(t)))
                change=alternatives[0,1:]-alternatives[0,:1]
                action_stats.append(dict(history=i,stage=int(t),gates=info['gates'][0].cpu().tolist(),
                     removal_action_L2=change.norm(dim=-1).cpu().tolist(),
                     removal_action_absmax=change.abs().amax(-1).cpu().tolist()))
                draw,q=old.posterior(actor,x,ds[i:i+1],ys[i:i+1],int(t),cfg[split+'_trials'],cfg['particles'],seed+10000+i)
                accepted=q['ESS']>=cfg['minimum_ESS'] and q['max_weight']<=cfg['maximum_weight']
                quality.append(dict(split=split,history=i,stage=int(t),accepted=accepted,**q));draws.append(draw)
                put(out/'POSTERIOR_QUALITY.json',quality)
            records[split]=dict(ds=ds,ys=ys,draws=draws,x=np.array(contexts),seed=seed)
            put(out/f'{split}_ACTION_CONTRIBUTIONS.json',action_stats)
            np.save(out/f'{split}_CONTEXT.npy',np.array(contexts))
        failed=[dict(split=q['split'],history=q['history']) for q in quality if not q['accepted']]
        if failed:
            assert all(torch.equal(before[k],v.cpu()) for k,v in actor.state_dict().items())
            assert all(sha(f)==v for f,v in manifest.items())
            put(out/'STATUS.json',dict(status='INCONCLUSIVE_POSTERIOR_QUALITY',failed_histories=failed,historical_artifacts_preserved=True,optimizer_steps=0));return
        scores={}
        for split in ['selection','test']:
            rec=records[split];rng=np.random.default_rng(rec['seed']+20000)
            bank=rng.standard_normal((cfg[split+'_contrasts'],4))
            put(out/f'{split}_BANK.json',dict(seed=rec['seed']+20000,sha256=hashlib.sha256(bank.view(np.uint8)).hexdigest()))
            contrasts=torch.tensor(bank,device='cuda');del bank;values=[]
            for i,t in enumerate(stages):
                theta=rec['draws'][i];eps=torch.tensor(np.random.default_rng(rec['seed']+30000+i).standard_normal((len(theta),30)),device='cuda')
                a,y,latent=branches(actor,lib,rec['ds'][i:i+1],rec['ys'][i:i+1],int(t),theta,eps,deadline)
                v=[]
                for first in range(0,len(a),16):
                    v.append(old.nested_scores(a[first:first+16],y[first:first+16],latent[first:first+16],contrasts,[cfg[split+'_contrasts']],deadline=deadline)[cfg[split+'_contrasts']].cpu().numpy())
                score=np.concatenate(v).reshape(len(theta),9);values.append(score);np.save(out/f'{split}_history{i:02d}.npy',score)
                put(out/'STATUS.json',dict(status='RUNNING',split=split,history=i+1,total=h,seconds=time.monotonic()-start));print(json.dumps(json.loads((out/'STATUS.json').read_text())),flush=True)
            scores[split]=np.array(values);del contrasts
            if split=='selection':put(out/'LOCKED_REGIME.json',fit_regime(rec['x'],(scores[split][:,:,:1]-scores[split][:,:,5:9]).mean(1)))
        result=results(scores['selection'],scores['test'],records['selection']['x'],records['test']['x'])
        assert result['regime']==json.loads((out/'LOCKED_REGIME.json').read_text())
        result.update(status='COMPLETE',smoke=args.smoke,seconds=time.monotonic()-start,
            limitations='One checkpoint,24 selection and24 independent test histories,approximate importance posterior,finite prior-sPCE. Masking is an out-of-distribution intervention and can expose shared-head coadaptation; damage is not proof of intrinsic scientific specialization. Shared history/SG paths remain. Experts still execute; no compute saving. Secondary tests Holm8; regime leaf summaries descriptive.')
        assert all(torch.equal(before[k],v.cpu()) for k,v in actor.state_dict().items())
        assert all(sha(f)==v for f,v in manifest.items())
        put(out/'RESULTS.json',result);put(out/'STATUS.json',dict(status='COMPLETE',smoke=args.smoke,seconds=time.monotonic()-start,historical_artifacts_preserved=True,optimizer_steps=0))
    except Exception as exc:put(out/'STATUS.json',dict(status='FAILED',error=str(exc)));raise
    finally:
        for f in lease:f.close()

if __name__=='__main__':main()
