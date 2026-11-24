"""Frozen four-pathway V3 routing credit audit;no optimization steps."""
import argparse,copy,importlib.util,io,json,pathlib,sys,time
sys.dont_write_bytecode=True
import numpy as np
import torch
from compare import ROOT,WORK,V3SOURCE,BASE,PILOT,n,audit
from pathway_policy import PathwayMoE
from run import rollout
from resolution import paired,signal
POST=WORK/'development/moe_sboed/hec_v2_routing_gap_20261113/diagnostic.py'
spec=importlib.util.spec_from_file_location('conditional_posterior',POST);post=importlib.util.module_from_spec(spec);spec.loader.exec_module(post)

def load_algorithm():
    p=PILOT/'selector/runtime.pkl';receipt=json.loads((PILOT/'selector/RESULT.json').read_text());assert n.sha(p)==receipt['runtime_sha256']
    old=torch.storage._load_from_bytes
    torch.storage._load_from_bytes=lambda b:torch.load(io.BytesIO(b),map_location=lambda s,l:s.cuda(0) if l.startswith('cuda') else s,weights_only=False)
    try:
        with p.open('rb') as f:state=n.cloudpickle.load(f)
    finally:torch.storage._load_from_bytes=old
    return state['algorithm']

@torch.no_grad()
def futures(model,lib,ds,ys,t,theta,noise,u):
    m=len(theta);actions=ds.expand(m,4,30,2).clone().reshape(-1,30,2);obs=ys.expand(m,4,30).clone().reshape(-1,30)
    actions[:,t:]=0;obs[:,t:]=0
    th=theta[:,None].expand(m,4,4).reshape(-1,4);eps=noise[:,None].expand(m,4,30).reshape(-1,30);uu=u[:,None].expand(m,4,30).reshape(-1,30)
    for step in range(t,30):
        x=lib.packed(actions,obs,step)
        if step==t:
            proposals,_,_,_=model.components(x);action=proposals[torch.arange(len(x),device='cuda'),torch.arange(4,device='cuda').repeat(m)]
        else:action=model(x,uniforms=uu[:,step])
        actions[:,step]=action;obs[:,step]=signal(th.reshape(-1,2,2),action.double())+.5*eps[:,step]
    assert torch.equal(actions[:,:t],ds[:,:t].expand(m*4,-1,-1))
    assert torch.equal(obs[:,:t],ys[:,:t].expand(m*4,-1))
    return actions,obs,th

def analyze(selection,test,w,q):
    selected=selection.mean(1).argmax(-1);fixed=int(selection.mean((0,1)).argmax());critic=q.argmax(-1)
    means=test.mean(1);row=np.arange(len(means));learned=(w*means).sum(-1)
    return dict(selection_chosen=(selected+1).tolist(),critic_chosen=(critic+1).tolist(),fixed_choice=fixed+1,
       independent_choice_minus_learned=paired(means[row,selected],learned),
       independent_choice_minus_fixed=paired(means[row,selected],means[:,fixed]),
       independent_choice_minus_critic=paired(means[row,selected],means[row,critic]),
       critic_choice_minus_learned=paired(means[row,critic],learned),
       average_native_return_by_expert=means.mean(0).tolist(),
       critic_selection_agreement=float((selected==critic).mean()),
       note='Native variational return,not16M prior-sPCE. Independent banks for choice/test;24 histories,approximate conditional posterior;secondary comparisons descriptive.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--output',type=pathlib.Path,required=True);args=p.parse_args()
    if not args.run:p.error('Explicit --run required')
    out=args.output.expanduser();assert ROOT/'scratch_runs' in out.parents and not out.exists();out.mkdir(parents=True)
    protected_files=[PILOT/'selector/runtime.pkl',PILOT/'selector/actor.pt',PILOT/'INITIAL_DISTILLED.pt',
                     BASE/'arms/candidate/actor.pt',BASE/'arms/candidate/runtime.pkl',*sorted(V3SOURCE.glob('*.py')),
                     *sorted(pathlib.Path(__file__).parent.glob('*.py'))]
    protected={str(p):n.sha(p) for p in protected_files};n.put(out/'PRE_RUN_MANIFEST.json',protected)
    n.put(out/'CONFIG.json',dict(training_steps=0,histories=24,selection_futures=32,test_futures=128,seed=202611134501,
        particles=[2097152,8388608],minimum_ESS=256,max_weight=.05,criterion='Native terminal-minus-prefix logposterior,matching critic return target',wall_seconds=1200))
    n.original.setup()
    while True:
        try:lease=n.original.acquire();break
        except BlockingIOError:n.put(out/'STATUS.json',dict(status='WAITING_FOR_IDLE_GPU'));time.sleep(30)
    start=time.monotonic()
    try:
        a=load_algorithm();model=a.actor_net.eval();a.critic_net.eval().requires_grad_(False);lib=n.load_library()
        before=n.digest(model.state_dict());state=n.load_parent();parent=state['algorithm'].actor_net
        initial=PathwayMoE(parent).cuda();initial.load_state_dict(torch.load(PILOT/'INITIAL_DISTILLED.pt',map_location='cuda',weights_only=True)['actor_state'])
        for k in range(4):
            cp=torch.load(PILOT/f'expert{k+1}/actor.pt',map_location='cuda',weights_only=True)['actor_state']
            initial.experts[k].load_state_dict({key[7:]:v for key,v in cp.items() if key.startswith('expert.')})
            assert all(torch.equal(v,model.experts[k].state_dict()[key[7:]]) for key,v in cp.items() if key.startswith('expert.'))
        expected=json.loads((PILOT/'selector/STARTED.json').read_text())['initial_actor_digest'];assert n.digest(initial.state_dict())==expected
        rng=np.random.default_rng(202611134501);truth=torch.tensor(rng.standard_normal((24,4)),device='cuda');noise=torch.tensor(rng.standard_normal((24,30)),device='cuda')
        u=torch.tensor(rng.random((24,30)),device='cuda',dtype=torch.float32).clamp_max(1-torch.finfo(torch.float32).eps)
        ds,ys,_=rollout(model,lib,truth,noise,u);del truth,noise,u
        stages=np.repeat([3,14,26],8);x=torch.cat([lib.packed(ds[i:i+1],ys[i:i+1],int(t)) for i,t in enumerate(stages)])
        with torch.no_grad():
            proposals,g,legacy,response=model.components(x);_,g0,_,_=initial.components(x);c=model.context(x)
            raw=model.coordinator(torch.cat([c['s'],legacy.flatten(1)],-1))+model.response_gate(response.flatten(1))+model.proposal_gate(proposals.flatten(1))
            q=torch.stack([a.critic_net(torch.cat([x,proposals[:,k]],-1)).squeeze(-1) for k in range(4)],-1)
        z=raw.detach().requires_grad_(True);gg=(10*(z/10).tanh()).softmax(-1);manual=-(gg*q).sum(-1).mean()
        grad=torch.autograd.grad(manual,z)[0]
        analytic=-gg*(q-(gg*q).sum(-1,keepdim=True))*(1-(z/10).tanh().square())/len(z)
        torch.testing.assert_close(grad,analytic,atol=1e-7,rtol=1e-5)
        actual=model.native_actor_loss(a.critic_net,x);torch.testing.assert_close(actual,manual,atol=1e-5,rtol=1e-5)
        for p in model.parameters():p.grad=None
        actual.backward()
        trainable={name:p for name,p in model.named_parameters() if p.requires_grad}
        opt_ids={id(p) for group in a.actor_optimizer.param_groups for p in group['params']}
        assert opt_ids=={id(p) for p in trainable.values()}
        steps={name:float(a.actor_optimizer.state[p]['step']) for name,p in trainable.items()}
        assert set(steps.values())=={250.}
        gradients={name:float(p.grad.norm()) for name,p in trainable.items()}
        assert all(torch.isfinite(p.grad).all() for p in trainable.values())
        report=dict(status='PASS',initial_selector_exact_digest=True,frozen_experts_match_saved_standalones=True,
           router_optimizer_steps=steps,router_gradient_norms=gradients,exact_actor_loss_and_logit_gradient=True,
           initial_mean_weights=g0.mean(0).cpu().tolist(),final_mean_weights=g.mean(0).cpu().tolist(),
           critic_mean_Q=q.mean(0).cpu().tolist(),critic_argmax_counts=torch.bincount(q.argmax(-1),minlength=4).cpu().tolist(),
           logit_cap_mean_derivative=float((1-(raw/10).tanh().square()).mean()),
           categories=['routing_quality','expert_design'])
        n.put(out/'IMPLEMENTATION_CHECKS.json',report)
        weights=g.detach().cpu().numpy();qvalues=q.cpu().numpy();values={'selection':[],'test':[]};quality=[]
        for i,t in enumerate(stages):
            for j,split in enumerate(['selection','test']):
                count=32 if split=='selection' else 128
                for attempt,particles in enumerate([2097152,8388608]):
                    theta,qr=post.posterior(model.context,x[i:i+1],ds[i:i+1],ys[i:i+1],int(t),count,particles,202611134600+100*i+10*j+attempt)
                    if qr['ESS']>=256 and qr['max_weight']<=.05:break
                quality.append(dict(history=i,split=split,**qr));n.put(out/'POSTERIOR_QUALITY.json',quality)
                if qr['ESS']<256 or qr['max_weight']>.05:
                    n.put(out/'STATUS.json',dict(status='INCONCLUSIVE_POSTERIOR_QUALITY',implementation_checks=report));return
                rr=np.random.default_rng(202611134700+100*i+j)
                eps=torch.tensor(rr.standard_normal((count,30)),device='cuda');uu=torch.tensor(rr.random((count,30)),device='cuda',dtype=torch.float32).clamp_max(1-torch.finfo(torch.float32).eps)
                with torch.no_grad(),torch.device('cuda'):
                    acts,obs,th=futures(model,lib,ds[i:i+1],ys[i:i+1],int(t),theta,eps,uu)
                    terminal=a.post_approx.log_poi_post(29,acts.float(),obs.float()[...,None],th.float())
                    prefix=a.post_approx.log_poi_post(int(t)-1,acts[:,:int(t)].float(),obs[:,:int(t)].float()[...,None],th.float())
                    scores=(terminal-prefix).reshape(count,4).cpu().numpy()
                values[split].append(scores);np.save(out/f'{split}_{i:02d}.npy',scores)
            n.put(out/'STATUS.json',dict(status='RUNNING',history=i+1,total=24,seconds=time.monotonic()-start));assert time.monotonic()-start<1200
        result=dict(status='COMPLETE',implementation=report,native_return_audit=analyze(np.array(values['selection']),np.array(values['test']),weights,qvalues),
                    seconds=time.monotonic()-start,training_steps=0)
        assert n.digest(model.state_dict())==before
        assert all(n.sha(p)==h for p,h in protected.items())
        n.put(out/'RESULTS.json',result);n.put(out/'STATUS.json',dict(status='COMPLETE',historical_artifacts_preserved=True,training_steps=0,seconds=time.monotonic()-start))
        print(json.dumps(result),flush=True)
    except Exception as exc:n.put(out/'STATUS.json',dict(status='FAILED',error=str(exc)));raise
    finally:
        for f in lease:f.close()
if __name__=='__main__':main()
