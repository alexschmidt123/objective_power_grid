"""Frozen diagnosis of V3<V2 and absence of V3 routing gain. No optimizer steps."""
import argparse,copy,json,pathlib,subprocess,sys,time
sys.dont_write_bytecode=True
import numpy as np
import torch
from torch import nn
ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp';WORK=ROOT/'development/hec_git_sync'
sys.path.insert(0,str(WORK/'development/moe_sboed/hec_canonical_evaluation_20261114'))
from protocol import load,LOCK,sha
sys.path.insert(0,str(WORK/'development/moe_sboed/hec_v3_counterfactual_router_20261114'))
from experiment import n,d,PILOT,BASE,advantage,frozen_digest
from resolution import paired,signal
CF=ROOT/'scratch_runs/hec_v3_counterfactual_router_20261114_pilot02'
CANON=ROOT/'scratch_runs/hec_canonical_evaluation_20261114'

def ensemble(ck,features):
    z=((features-ck['mean'])/ck['std']).clamp(-20,20);models=[]
    for state in ck['models']:
        net=nn.Sequential(nn.Linear(648,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,4)).cuda()
        net.load_state_dict(state);models.append(net)
    with torch.no_grad():return torch.stack([advantage(net(z))*ck['scale'] for net in models]).mean(0)

def fit_diagnosis(round_id,bank,ck):
    f=torch.tensor(bank['features'],device='cuda');z=((f-ck['mean'])/ck['std']).clamp(-20,20)
    y=advantage(torch.tensor(bank['selection'].mean(1),device='cuda'))/ck['scale']
    test=advantage(torch.tensor(bank['test'].mean(1),device='cuda'));initial=[];selected=[];checks=[]
    torch.manual_seed(202611147000+round_id-1)
    for member,state in enumerate(ck['models']):
        # Original constructors draw CPU weights; optimizer/backward consume no RNG.
        with torch.device('cpu'):
            net=nn.Sequential(nn.Linear(648,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,4)).cuda()
        with torch.no_grad():initial.append(advantage(net(z))*ck['scale'])
        loss=(advantage(net(z[:64]))-y[:64]).square().mean();params=list(net.parameters())
        grads=torch.autograd.grad(loss,params);norm=torch.linalg.vector_norm(torch.stack([g.norm() for g in grads]));clip=(5/(norm+1e-6)).clamp_max(1)
        # Closed-form AdamW t=1 reconstruction, without stepping or saving a model.
        manual={}
        for (name,p),g in zip(net.named_parameters(),grads):
            g=g*clip;manual[name]=p.detach()*(1-.001*.001)-.001*g/(g.abs()+1e-8)
        error=max(float((manual[k]-state[k]).abs().max()) for k in state)
        assert ck['receipts'][member]['best_step']==1 and error<2e-6
        checks.append(error);net.load_state_dict(state)
        with torch.no_grad():selected.append(advantage(net(z))*ck['scale'])
    initial=torch.stack(initial).mean(0);selected=torch.stack(selected).mean(0)
    def mse(p,target,sl):return float((p[sl]-target[sl]).square().mean())
    train=slice(0,64);val=slice(64,80);audit=slice(80,96);selection=y*ck['scale']
    return selected,dict(train_histories=64,validation_histories=16,audit_histories=16,
        parameters_per_value_model=sum(p.numel() for p in net.parameters()),ensemble_members=3,
        first_step_reconstruction_max_errors=checks,best_steps=[v['best_step'] for v in ck['receipts']],
        initial_validation_MSE=mse(initial,selection,val),selected_validation_MSE=mse(selected,selection,val),
        initial_independent_audit_MSE=mse(initial,test,audit),selected_independent_audit_MSE=mse(selected,test,audit),
        zero_independent_audit_MSE=float(test[audit].square().mean()),
        predicted_advantage_RMS=float(selected[train].square().mean().sqrt()),
        training_MC_advantage_RMS=float(selection[train].square().mean().sqrt()),
        independent_audit_advantage_RMS=float(test[audit].square().mean().sqrt()),
        note='Step0 comparison is retrospective diagnosis only; no model selection or optimizer steps performed.')

def gradients(before,after,x,values):
    after.freeze_experts();params=[p for p in after.parameters() if p.requires_grad]
    with torch.no_grad():g0=before.components(x)[1]
    p=after.components(x)[1];values=advantage(values.detach())
    reward=-(p*values).sum(-1).mean();kl=(g0*(g0.clamp_min(1e-12).log()-p.clamp_min(1e-12).log())).sum(-1).mean()
    gr=torch.cat([v.flatten() for v in torch.autograd.grad(reward,params,retain_graph=True)])
    gk=torch.cat([v.flatten() for v in torch.autograd.grad(kl,params)])
    with torch.no_grad():
        optimum=(g0.clamp_min(1e-12).log()+values).softmax(-1)
        def tv(q):return float((q-g0).abs().sum(-1).mean()/2)
        total=gr+gk;den=(gr.norm()*gk.norm()).clamp_min(1e-30)
        result=dict(reward_gradient_norm=float(gr.norm()),KL_gradient_norm=float(gk.norm()),
            gradient_cosine=float(gr.dot(gk)/den),total_to_reward_gradient_norm=float(total.norm()/gr.norm().clamp_min(1e-30)),
            actual_KL=float(kl),mean_actual_TV=tv(p),mean_unrestricted_KL_penalized_optimum_TV=tv(optimum),
            mean_probability_before=g0.mean(0).cpu().tolist(),mean_probability_after=p.mean(0).cpu().tolist(),
            max_actual_probability_change=float((p-g0).abs().max()),
            explanation='For fixed per-history advantages and KL coefficient1, unconstrained optimal p is proportional to old_p*exp(A). This is a local analytic diagnostic, not an EIG-winning policy.')
    after.requires_grad_(False);return result

@torch.no_grad()
def bridge(model,parent,x,kind):
    c=model.context(x);_,g,legacy,response=model.components(x)
    if kind=='inherited':
        raw=parent.coordinator(torch.cat([c['s'],legacy.flatten(1)],-1))+parent.response_gate(response.flatten(1))
        g=(10*(raw/10).tanh()).softmax(-1)
    elif kind=='uniform':g=torch.full_like(g,.25)
    elif kind!='new_router':raise ValueError(kind)
    first=torch.stack([e.packet(c)[0] for e in model.experts],1)
    pre=parent.fusion[0](torch.cat([c['s'],(g[...,None]*first).flatten(1)],-1))+parent.response_fusion((g[...,None]*response).flatten(1))
    delta=parent.fusion[2](parent.fusion[1](pre)).tanh()
    return c['a0']+4*delta-c['a0']*delta.abs()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path,required=True);args=ap.parse_args();out=args.output.expanduser()
    assert ROOT/'scratch_runs' in out.parents and not out.exists();out.mkdir(parents=True)
    bank,u,scorer=load();n.put(out/'CONFIG.json',dict(protocol=LOCK,training_updates=0,
        source_commit=subprocess.check_output(['git','-C',str(WORK),'rev-parse','HEAD'],text=True).strip(),
        purpose=['V3_below_V2','V3_not_above_uniform'],variants=['inherited','new_router','uniform']))
    files=[pathlib.Path(__file__),CANON/'RESULTS.json',CANON/'MANIFEST.json',PILOT/'selector/actor.pt',PILOT/'INITIAL_DISTILLED.pt',
        *[CF/f'counterfactual_round{k}'/name for k in [1,2,3] for name in ['actor.pt','DATA.npz','value_ensemble.pt']]]
    files+=sorted(CANON.glob('*_scores.npy'))
    certified=json.loads((WORK/'reports/hec_canonical_evaluation_20261114/VERIFICATION.json').read_text())
    assert certified['protocol']==LOCK and len(certified['historical_reproduction'])==6
    assert all(v['max_abs_error']==0 for v in certified['historical_reproduction'].values())
    assert sha(CF/'counterfactual_round3/actor.pt')==certified['checkpoint_sha256']['counterfactual']
    protected={str(p):sha(p) for p in files};n.put(out/'MANIFEST.json',protected);n.original.setup();lease=None
    while lease is None:
        try:lease=n.original.acquire()
        except BlockingIOError:n.put(out/'STATUS.json',dict(status='WAITING_FOR_IDLE_GPU'));time.sleep(30)
    start=time.monotonic();deadline=start+1200
    try:
        lib=n.load_library();parent=lib.load_frozen('cuda');original=d.PathwayMoE(parent).cuda()
        original.load_state_dict(torch.load(PILOT/'selector/actor.pt',map_location='cuda',weights_only=True)['actor_state']);original.eval().requires_grad_(False)
        previous=copy.deepcopy(original);reports=[]
        for r in [1,2,3]:
            n.put(out/'STATUS.json',dict(status='FROZEN_GRADIENT_DIAGNOSIS',round=r))
            path=CF/f'counterfactual_round{r}';data=np.load(path/'DATA.npz');ck=torch.load(path/'value_ensemble.pt',map_location='cuda',weights_only=True)
            pred,fit=fit_diagnosis(r,data,ck)
            after=copy.deepcopy(original);after.load_state_dict(torch.load(path/'actor.pt',map_location='cuda',weights_only=True)['actor_state'])
            assert frozen_digest(after)==frozen_digest(original)
            x=torch.tensor(data['history'][:64],device='cuda');before_digest=n.digest(after.state_dict())
            result=dict(round=r,value_fit=fit,router_update=gradients(previous,after,x,pred[:64]))
            assert n.digest(after.state_dict())==before_digest
            reports.append(result);n.put(out/'GRADIENT_AND_VALUE.json',reports);previous=after
        model=previous.eval().requires_grad_(False)
        prepared=d.PathwayMoE(parent).cuda();prepared.load_state_dict(torch.load(PILOT/'INITIAL_DISTILLED.pt',map_location='cuda',weights_only=True)['actor_state']);prepared.requires_grad_(False)
        hist=np.load(BASE/'evaluation/diagnostic_candidate_histories.npz')['X'][:2,[0,14,29]].copy()
        x=torch.tensor(hist,device='cuda').reshape(-1,117)
        torch.testing.assert_close(bridge(prepared,parent,x,'inherited'),parent(x),atol=2e-6,rtol=2e-5)
        n.put(out/'BRIDGE_PARITY.json',dict(status='PASS',stages=[0,14,29],B=2,original_cores_bridge_matches_V2=True))
        theta=torch.tensor(bank['truth'],device='cuda');noise=torch.tensor(bank['noise'],device='cuda');neg=torch.tensor(bank['contrasts'],device='cuda')
        uniforms=torch.tensor(u,device='cuda',dtype=torch.float32).clamp_max(1-torch.finfo(torch.float32).eps)
        scores={}
        with torch.no_grad(),torch.device('cuda'):
            for kind in ['inherited','new_router','uniform']:
                values=[]
                for i in range(0,2000,16):
                    if time.monotonic()>deadline:raise TimeoutError('Frozen diagnostic cap')
                    th=theta[i:i+16];ds=torch.zeros(len(th),30,2,device='cuda');ys=torch.zeros(len(th),30,device='cuda',dtype=torch.float64)
                    for t in range(30):
                        action=bridge(model,parent,lib.packed(ds,ys,t),kind);assert torch.isfinite(action).all() and (action.abs()<=4.000001).all()
                        ds[:,t]=action;ys[:,t]=signal(th.reshape(-1,2,2),action.double())+.5*noise[i:i+16,t]
                    result,_=scorer(type('Benchmark',(),{'sigma':.5,'calls':0})(),dict(actions=list(ds.unbind(1)),observations=list(ys[...,None].unbind(1)),truth=th),neg,16384,'cuda')
                    values.extend(result[:,-1].numpy());n.put(out/'STATUS.json',dict(status='EVALUATING_FROZEN_BRIDGE',variant=kind,completed=min(i+16,2000),total=2000))
                scores[kind]=np.array(values);np.save(out/f'{kind}_scores.npy',scores[kind])
        comparisons={}
        for kind,v in scores.items():
            for reference in ['V2','counterfactual','original_V3','uniform']:
                comparisons[kind+'_minus_'+reference]=paired(v,np.load(CANON/f'{reference}_scores.npy'))
        largest=0
        for j,k in enumerate(sorted(comparisons,key=lambda k:comparisons[k]['p_two_sided'])):
            largest=max(largest,min(1,(len(comparisons)-j)*comparisons[k]['p_two_sided']));comparisons[k]['p_Holm']=largest
        assert all(sha(path)==h for path,h in protected.items())
        result=dict(status='COMPLETE',protocol=LOCK,training_updates=0,frozen_tests=reports,
            bridge_means={k:float(v.mean()) for k,v in scores.items()},bridge_comparisons=comparisons,
            seconds=time.monotonic()-start,historical_artifacts_preserved=True,
            limitations='Bridges are frozen mechanism tests, not standalone-expert solutions. Gradient/value checks use saved training/diagnostic data, not an alternative performance benchmark. No retraining; no new accepted policy.')
        n.put(out/'RESULTS.json',result);n.put(out/'STATUS.json',dict(status='COMPLETE',training_updates=0,seconds=result['seconds']))
    except Exception as exc:n.put(out/'STATUS.json',dict(status='FAILED',error=repr(exc)));raise
    finally:
        for f in lease:f.close()
if __name__=='__main__':main()
