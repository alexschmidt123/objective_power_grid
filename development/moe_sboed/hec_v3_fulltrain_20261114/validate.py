"""Functional checks and exact canonical initialization preservation; no long training."""
import argparse,copy,json,pathlib,sys,time
sys.dont_write_bytecode=True
import numpy as np
import torch
from v3_native import n,ROOT,WORK
from feature_policy import FeatureMoE,observed_prefix
sys.path.insert(0,str(WORK/'development/moe_sboed/hec_canonical_evaluation_20261114'))
from protocol import load,LOCK,sha
from resolution import signal

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);args=p.parse_args();out=args.output.expanduser()
    assert ROOT/'scratch_runs' in out.parents and not out.exists();out.mkdir(parents=True);n.original.setup();lease=n.original.acquire();start=time.monotonic()
    protected={str(p):sha(p) for p in [n.BASE/'arms/candidate/actor.pt',n.BASE/'arms/candidate/runtime.pkl',*sorted(pathlib.Path(__file__).parent.glob('*.py'))]}
    n.put(out/'MANIFEST.json',protected)
    try:
        lib=n.load_library();parent=lib.load_frozen('cuda');model=FeatureMoE(parent).cuda().eval()
        original_state=n.digest(model.state_dict());data=np.load(n.BASE/'evaluation/diagnostic_candidate_histories.npz')
        x=torch.tensor(data['X'][:2,[0,14,29]].copy(),device='cuda').reshape(-1,117)
        with torch.no_grad():
            expected,pi=parent(x,collect=True);actual,info=model(x,collect=True)
            assert torch.equal(expected,actual),'Mixed initialization must preserve exact V2 actions'
            torch.testing.assert_close(info['gates'],pi['gates'],atol=0,rtol=0)
            torch.testing.assert_close(info['first'],pi['slots'][...,:64],atol=0,rtol=0)
            torch.testing.assert_close(info['responses'],pi['responses'],atol=0,rtol=0)
            assert info['router_input'].shape==(6,147) and info['innovation'].shape==(6,4,128)
            assert all(v.shape[-1]==width for v,width in zip(info['private'],[13,28,20,26]))
            torch.testing.assert_close(model(x,mode='uniform'),parent(x,routing_override=x.new_full((4,),.25)),atol=0,rtol=0)
        calls=[0]*4;hooks=[]
        for k,e in enumerate(model.experts):
            def count(*unused,k=k):calls[k]+=1
            hooks.append(e.register_forward_hook(count))
        solos=[]
        for k in range(4):
            calls[:]=[0]*4
            with torch.no_grad():
                action=model(x,mode='expert',expert=k);g=x.new_zeros(4);g[k]=1
                assert calls==[int(i==k) for i in range(4)],calls
                torch.testing.assert_close(action,parent(x,routing_override=g),atol=0,rtol=0)
                single=model.standalone(k).cuda().eval();torch.testing.assert_close(single(x),action,atol=0,rtol=0)
            assert not any(name.startswith(('experts.','router.')) for name in single.state_dict())
            solos.append(dict(expert=k+1,parameters=sum(p.numel() for p in single.parameters()),private_width=info['private'][k].shape[-1],only_own_query_executed=True));del single
        for hook in hooks:hook.remove()
        poisoned=x.clone();stage=x[:,:30].argmax(-1);mask=torch.arange(29,device='cuda')[None]>=stage[:,None]
        poisoned[:,30:88].reshape(-1,29,2)[mask]=float('nan');poisoned[:,88:][mask]=float('nan')
        with torch.no_grad():torch.testing.assert_close(model(poisoned),actual,atol=0,rtol=0)
        # Frozen native critic gradients; no optimizer step or stored model modification.
        runtime=n.load_parent();a=runtime['algorithm'];a.critic_net.requires_grad_(False)
        loss=-a.critic_net(torch.cat([x,model(x)],-1)).mean();loss.backward()
        norms={}
        for name,p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and torch.isfinite(p.grad).all(),name
                norms[name]=float(p.grad.norm())
            else:assert p.grad is None,name
        assert norms['router.2.weight']>0 and norms['router.2.bias']>0
        assert all(norms[f'experts.{k}.innovation.4.weight']>0 for k in range(4))
        for p in model.parameters():p.grad=None
        # Check hidden-layer connectivity on a throwaway perturbation, not a trained checkpoint.
        clone=copy.deepcopy(model)
        with torch.no_grad():
            for e in clone.experts:e.innovation[-1].weight.fill_(.001)
            clone.router[-1].weight.fill_(.001)
        value=-a.critic_net(torch.cat([x,clone(x)],-1)).mean();value.backward()
        assert all(torch.isfinite(p.grad).all() for p in clone.parameters() if p.requires_grad)
        assert all(clone.experts[k].innovation[0].weight.grad.norm()>0 for k in range(4))
        del clone,a,runtime
        n.put(out/'FUNCTIONAL.json',dict(status='PASS',exact_V2_mixed=True,exact_V2_uniform=True,exact_V2_single_pathways=True,
            standalone_exports=solos,unobserved_prefix_NaNs_ignored=True,all_trainable_gradients_finite=True,
            initial_output_layers_nonzero_gradients=True,hidden_layers_connected_after_nonzero_output=True,
            initial_gradient_norms=norms,shapes=dict(history=[2,117],encoded=[2,128],commands=[2,20],
                first=[2,4,64],response=[2,4,64],innovation=[2,4,128],router_input=[2,147],gates=[2,4],output=[2,2]),
            stages=[0,14,29],parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
        bank,u,scorer=load();neg=torch.tensor(bank['contrasts'],device='cuda');scores=[];max_action_error=0.;max_observation_error=0.
        with torch.no_grad(),torch.device('cuda'):
            for i in range(0,2000,16):
                th=torch.tensor(bank['truth'][i:i+16],device='cuda');noise=torch.tensor(bank['noise'][i:i+16],device='cuda')
                ds=torch.zeros(len(th),30,2,device='cuda');ys=torch.zeros(len(th),30,device='cuda',dtype=torch.float64)
                for t in range(30):
                    xx=lib.packed(ds,ys,t);action=model(xx);ref=parent(xx);assert torch.equal(action,ref)
                    max_action_error=max(max_action_error,float((action-ref).abs().max()))
                    ds[:,t]=action;ys[:,t]=signal(th.reshape(-1,2,2),action.double())+.5*noise[:,t]
                values,_=scorer(type('Benchmark',(),{'sigma':.5,'calls':0})(),dict(actions=list(ds.unbind(1)),observations=list(ys[...,None].unbind(1)),truth=th),neg,16384,'cuda')
                scores.extend(values[:,-1].numpy());n.put(out/'STATUS.json',dict(status='CANONICAL_INITIALIZATION_CHECK',completed=min(i+16,2000),total=2000))
        scores=np.array(scores);saved=np.load(n.BASE/'evaluation/candidate_scores.npy');error=float(np.abs(scores-saved).max());assert error<1e-9
        np.save(out/'INITIAL_scores.npy',scores);torch.save(dict(actor_state=model.state_dict()),out/'INITIAL_actor.pt')
        assert n.digest(model.state_dict())==original_state and all(sha(p)==h for p,h in protected.items())
        result=dict(status='COMPLETE',protocol=LOCK,canonical_cases=2000,mean=float(scores.mean()),reference_mean=float(saved.mean()),
            maximum_score_error=error,maximum_action_error=max_action_error,training_updates=0,
            historical_artifacts_preserved=True,seconds=time.monotonic()-start,
            claim='Exact V2 initialization preservation,not an EIG improvement. Standalone policies initially match V2 one-hot pathways,not full V2.')
        n.put(out/'RESULTS.json',result);n.put(out/'STATUS.json',dict(status='COMPLETE',training_updates=0,seconds=result['seconds']))
    except Exception as exc:n.put(out/'STATUS.json',dict(status='FAILED',error=repr(exc)));raise
    finally:
        for f in lease:f.close()
if __name__=='__main__':main()
