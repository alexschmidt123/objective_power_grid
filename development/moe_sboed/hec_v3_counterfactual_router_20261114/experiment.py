"""Frozen V3 experts; iterative paired-suffix router learning. No online optimization."""
import argparse,copy,gc,json,pathlib,sys,time,subprocess
sys.dont_write_bytecode=True
import numpy as np
import torch
from torch import nn
ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp'
WORK=ROOT/'development/hec_git_sync'
sys.path.insert(0,str(WORK/'development/moe_sboed/hec_v3_troubleshoot_20261113'))
import diagnose as d
from compare import n,PILOT,BASE
from run import rollout
from resolution import paired,nested_scores,signal

def frozen_digest(model):
    return n.digest({k:v for k,v in model.state_dict().items() if k.startswith(('context.','experts.'))})

@torch.no_grad()
def features(model,x):
    proposals,g,legacy,response=model.components(x)
    return torch.cat([model.context(x)['s'],legacy.flatten(1),response.flatten(1),proposals.flatten(1)],-1)

def advantage(y):return y-y.mean(-1,keepdim=True)

def fit_values(x,y,out,seed,steps):
    """Train histories 0:64, validation 64:80; audit 80:96 never used for fitting."""
    torch.manual_seed(seed);mu=x[:64].mean(0);sd=x[:64].std(0).clamp_min(.01)
    z=((x-mu)/sd).clamp(-20,20);target=advantage(y);scale=target[:64].std().clamp_min(.01)
    nets=[];receipts=[]
    for member in range(3):
        net=nn.Sequential(nn.Linear(x.shape[-1],128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,4)).cuda()
        opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
        best=float('inf');saved=None;best_step=0
        for step in range(steps):
            opt.zero_grad(set_to_none=True);loss=(advantage(net(z[:64]))-target[:64]/scale).square().mean()
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
            with torch.no_grad():val=float((advantage(net(z[64:80]))-target[64:80]/scale).square().mean())
            if val<best:best=val;saved=copy.deepcopy(net.state_dict());best_step=step+1
        net.load_state_dict(saved);net.requires_grad_(False);nets.append(net)
        receipts.append(dict(validation_MSE=best,best_step=best_step))
    with torch.no_grad():pred=torch.stack([advantage(net(z))*scale for net in nets]).mean(0)
    torch.save(dict(models=[net.state_dict() for net in nets],mean=mu,std=sd,scale=scale,receipts=receipts),out/'value_ensemble.pt')
    return pred,receipts

def update(model,x,values,steps):
    params=[p for p in model.parameters() if p.requires_grad]
    assert params and all(p.requires_grad is False for p in model.experts.parameters())
    before=frozen_digest(model);weights=[p.detach().clone() for p in params]
    with torch.no_grad():old=model.components(x)[1].detach()
    values=advantage(values.detach());opt=torch.optim.Adam(params,lr=1e-4)
    used=0;grad_norm=None
    for step in range(steps):
        saved=[p.detach().clone() for p in params];opt.zero_grad(set_to_none=True)
        g=model.components(x)[1];kl=(old*(old.clamp_min(1e-12).log()-g.clamp_min(1e-12).log())).sum(-1).mean()
        loss=-(g*values).sum(-1).mean()+kl
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        grad_norm=float(torch.nn.utils.clip_grad_norm_(params,1));opt.step()
        with torch.no_grad():
            new=model.components(x)[1];newkl=(old*(old.clamp_min(1e-12).log()-new.clamp_min(1e-12).log())).sum(-1).mean()
        if newkl>.01:
            with torch.no_grad():
                for p,v in zip(params,saved):p.copy_(v)
            break
        used+=1
    assert frozen_digest(model)==before
    changed=any(not torch.equal(p,v) for p,v in zip(params,weights))
    return dict(steps=used,router_changed=changed,gradient_norm=grad_norm,experts_context_unchanged=True,
                KL=float((old*(old.clamp_min(1e-12).log()-model.components(x)[1].detach().clamp_min(1e-12).log())).sum(-1).mean()))

@torch.no_grad()
def collect(model,a,lib,out,seed,deadline,smoke=False):
    rng=np.random.default_rng(seed);count=4 if smoke else 96
    th=torch.tensor(rng.standard_normal((count,4)),device='cuda');eps=torch.tensor(rng.standard_normal((count,30)),device='cuda')
    uu=torch.tensor(rng.random((count,30)),device='cuda',dtype=torch.float32).clamp_max(1-torch.finfo(torch.float32).eps)
    ds,ys,_=rollout(model,lib,th,eps,uu);del th,eps,uu
    stages=np.array([0,3,14,29]) if smoke else rng.permutation(np.tile(np.arange(30),4))[:count]
    x=torch.cat([lib.packed(ds[i:i+1],ys[i:i+1],int(t)) for i,t in enumerate(stages)])
    f=features(model,x);props,g,_,_=model.components(x)
    q=torch.stack([a.critic_net(torch.cat([x,props[:,k]],-1)).squeeze(-1) for k in range(4)],-1)
    train=[];test=[];quality=[];calls=count*30
    for i,t0 in enumerate(stages):
        t=int(t0)
        if time.monotonic()>deadline:raise TimeoutError('Wall cap')
        draws=8 if smoke else 192
        if t==0:
            theta=torch.tensor(rng.standard_normal((draws,4)),device='cuda');qr=dict(ESS=None,max_weight=None,exact_prior=True)
        else:
            for attempt,particles in enumerate([2097152,8388608]):
                theta,qr=d.post.posterior(model.context,x[i:i+1],ds[i:i+1],ys[i:i+1],t,draws,particles,seed+1000+i*10+attempt)
                if qr['ESS']>=256 and qr['max_weight']<=.05:break
            if qr['ESS']<256 or qr['max_weight']>.05:raise RuntimeError('Posterior quality failed; do not train on unreliable labels')
        noise=torch.tensor(rng.standard_normal((draws,30)),device='cuda')
        u=torch.tensor(rng.random((draws,30)),device='cuda',dtype=torch.float32).clamp_max(1-torch.finfo(torch.float32).eps)
        acts,obs,latent=d.futures(model,lib,ds[i:i+1],ys[i:i+1],t,theta,noise,u)
        paired_noise=noise[:,None].expand(draws,4,30).reshape(-1,30)
        expected=signal(latent.reshape(-1,1,2,2),acts[:,t:].double())+.5*paired_noise[:,t:]
        torch.testing.assert_close(obs[:,t:],expected,atol=1e-12,rtol=1e-12)
        # Native discounted=1 posterior increments telescope exactly. No true theta in inputs.
        with torch.device('cuda'):
            terminal=a.post_approx.log_poi_post(29,acts.float(),obs.float()[...,None],latent.float())
            prefix=0 if t==0 else a.post_approx.log_poi_post(t-1,acts[:,:t].float(),obs[:,:t].float()[...,None],latent.float())
        if t==0:
            # Standard normal prior used by native location-finding objective.
            prefix=(-.5*latent.float().square()- .5*np.log(2*np.pi)).sum(-1)
        values=(terminal-prefix).reshape(draws,4);assert torch.isfinite(values).all()
        cut=4 if smoke else 64;train.append(values[:cut].cpu().numpy());test.append(values[cut:].cpu().numpy())
        calls+=draws*4*(30-t);quality.append(dict(history=i,stage=t,**qr))
        n.put(out/'PROGRESS.json',dict(histories=i+1,total=count,simulator_observations=calls))
    np.savez_compressed(out/'DATA.npz',history=x.cpu().numpy(),features=f.cpu().numpy(),selection=np.array(train),
        test=np.array(test),critic=q.cpu().numpy(),weights=g.cpu().numpy(),stages=stages)
    n.put(out/'POSTERIOR_QUALITY.json',quality)
    return x,f,torch.tensor(np.array(train).mean(1),device='cuda'),np.array(test),q,calls

def rankings(pred,test,q,g):
    mean=test.mean(1);idx=np.arange(len(mean));choose=pred.argmax(-1)
    return dict(histories=len(mean),predicted_choice=choose.tolist(),
        selected_minus_critic=paired(mean[idx,choose],mean[idx,q.argmax(-1)]),
        selected_minus_current=paired(mean[idx,choose],(g*mean).sum(-1)),
        selected_minus_E3=paired(mean[idx,choose],mean[:,2]),
        note='Independent suffix draws, approximate conditional posterior; histories are statistical units, MC argmax agreement is descriptive.')

@torch.no_grad()
def evaluate(models,lib,out,deadline,smoke):
    rng=np.random.default_rng(202611145901);cases=8 if smoke else 1024;negatives=4096 if smoke else 16000000
    truth=torch.tensor(rng.standard_normal((cases,4)),device='cuda');noise=torch.tensor(rng.standard_normal((cases,30)),device='cuda')
    u=torch.tensor(rng.random((cases,30)),device='cuda',dtype=torch.float32).clamp_max(1-torch.finfo(torch.float32).eps)
    neg=torch.tensor(np.random.default_rng(202611145902).standard_normal((negatives,4)),device='cuda')
    np.savez_compressed(out/'FINAL_CASES.npz',truth=truth.cpu().numpy(),noise=noise.cpu().numpy(),uniforms=u.cpu().numpy())
    scores={};counts={}
    original=models['original'];v2=lib.load_frozen('cuda')
    names=list(models)+['uniform','expert1','expert2','expert3','expert4','V2']
    for name in names:
        vals=[];cc=torch.zeros(4,device='cuda',dtype=torch.long)
        for j in range(0,cases,16):
            model=models.get(name,original)
            if name=='V2':ds,ys,_=lib.rollout(v2,truth[j:j+16],noise[j:j+16],'original',deadline=deadline)
            else:
                ds,ys,c=rollout(model,lib,truth[j:j+16],noise[j:j+16],u[j:j+16],
                    'expert' if name.startswith('expert') else 'uniform' if name=='uniform' else 'learned',
                    int(name[-1])-1 if name.startswith('expert') else None);cc+=c
            vals.append(nested_scores(ds,ys,truth[j:j+16],neg,[negatives],deadline=deadline)[negatives].cpu().numpy())
            n.put(out/'STATUS.json',dict(status='EVALUATING',method=name,completed=min(j+16,cases)))
        scores[name]=np.concatenate(vals);counts[name]=cc.cpu().tolist();np.save(out/f'{name}_scores.npy',scores[name])
    comp={k:paired(scores['counterfactual'],v) for k,v in scores.items() if k!='counterfactual'}
    largest=0
    for i,k in enumerate(sorted(comp,key=lambda k:comp[k]['p_two_sided'])):
        largest=max(largest,min(1,(len(comp)-i)*comp[k]['p_two_sided']));comp[k]['p_Holm']=largest
    return dict(means={k:float(v.mean()) for k,v in scores.items()},comparisons=comp,counts=counts,cases=cases,contrasts=negatives,
       accepted=not smoke and all(comp[k]['mean']>0 and comp[k]['p_Holm']<.05 for k in ['uniform','expert1','expert2','expert3','expert4','original','critic_control','V2']))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    out=args.output.expanduser();assert ROOT/'scratch_runs' in out.parents and not out.exists();out.mkdir(parents=True)
    cfg=dict(rounds=3,histories_per_round=96,train_histories=64,validation_histories=16,audit_histories=16,
        selection_futures=64,test_futures=128,value_ensemble=3,value_steps=300,router_steps=50,router_lr=.0001,KL_limit=.01,
        control='Frozen original critic advantages; same suffix collection budget and router update cap, no native critic/posterior retraining',
        objective='Unchanged native variational terminal-minus-prefix return; approximate posterior for conditional counterfactuals',
        evaluation='Fresh fixed 1024 cases,16M contrasts; one lineage,not definitive multi-seed evidence',smoke=args.smoke,
        source_commit=subprocess.check_output(['git','-C',str(WORK),'rev-parse','HEAD'],text=True).strip())
    n.put(out/'CONFIG.json',cfg);n.original.setup();start=time.monotonic();deadline=start+7200
    protected={str(p):n.sha(p) for p in [PILOT/'selector/actor.pt',PILOT/'selector/runtime.pkl',BASE/'arms/candidate/actor.pt',BASE/'arms/candidate/runtime.pkl',pathlib.Path(__file__)]}
    n.put(out/'PRE_RUN_MANIFEST.json',protected)
    lease=None
    while lease is None:
        try:lease=n.original.acquire()
        except BlockingIOError:n.put(out/'STATUS.json',dict(status='WAITING_FOR_IDLE_GPU'));time.sleep(30)
    try:
        a=d.load_algorithm();a.critic_net.eval().requires_grad_(False)
        original=a.actor_net.eval();original.freeze_experts();digest=frozen_digest(original)
        critic_digest=n.digest(a.critic_net.state_dict())
        # Centering alone cannot change the categorical expected-value gradient.
        z=torch.tensor([[1.,-1.,.3,.1]],device='cuda',requires_grad=True)
        v=torch.tensor([[3.,2.,4.,1.]],device='cuda')
        raw_grad=torch.autograd.grad(-(z.softmax(-1)*v).sum(),z,retain_graph=True)[0]
        centered_grad=torch.autograd.grad(-(z.softmax(-1)*advantage(v)).sum(),z)[0]
        torch.testing.assert_close(raw_grad,centered_grad,atol=1e-6,rtol=1e-6)
        models=dict(original=original,counterfactual=copy.deepcopy(original),critic_control=copy.deepcopy(original));lib=n.load_library()
        totals={};reports=[]
        for arm in ['counterfactual','critic_control']:
            model=models[arm];totals[arm]=0
            for r in range(1 if args.smoke else 3):
                target=out/f'{arm}_round{r+1}';target.mkdir();n.put(out/'STATUS.json',dict(status='COLLECTING',arm=arm,round=r+1))
                x,f,y,test,q,calls=collect(model,a,lib,target,202611146000+10000*r,deadline,args.smoke);totals[arm]+=calls
                if args.smoke:
                    # Small test exercises value fitting with replicated rows; no scientific inference.
                    xx=x.repeat(24,1);ff=f.repeat(24,1);yy=y.repeat(24,1)
                    pred,receipt=fit_values(ff,yy,target,71,5);pred=pred[:4]
                else:pred,receipt=fit_values(f,y,target,202611147000+r,300)
                with torch.no_grad():g=model.components(x)[1].cpu().numpy()
                sl=slice(None) if args.smoke else slice(80,96)
                audit=rankings(pred.cpu().numpy()[sl],test[sl],q.cpu().numpy()[sl],g[sl])
                values=pred if arm=='counterfactual' else q
                update_receipt=update(model,x[:64],values[:64],3 if args.smoke else 50)
                assert frozen_digest(model)==digest
                result=dict(arm=arm,round=r+1,ranking_audit=audit,value_fit=receipt,update=update_receipt,simulator_observations=calls)
                n.put(target/'RESULT.json',result);reports.append(result)
                torch.save(dict(actor_state=model.state_dict()),target/'actor.pt')
        # Counts must match despite distinct on-policy histories.
        assert len(set(totals.values()))==1
        evaluation=evaluate(models,lib,out,deadline,args.smoke)
        assert all(frozen_digest(m)==digest for m in models.values()) and all(n.sha(path)==h for path,h in protected.items())
        assert n.digest(a.critic_net.state_dict())==critic_digest
        result=dict(status='COMPLETE',smoke=args.smoke,rounds=reports,evaluation=evaluation,simulator_observations=totals,
            historical_artifacts_preserved=True,experts_unchanged=True,seconds=time.monotonic()-start)
        n.put(out/'RESULTS.json',result);n.put(out/'STATUS.json',dict(status='COMPLETE',accepted=evaluation['accepted'],seconds=result['seconds']))
    except Exception as exc:n.put(out/'STATUS.json',dict(status='FAILED',error=repr(exc)));raise
    finally:
        for f in lease:f.close()
if __name__=='__main__':main()
