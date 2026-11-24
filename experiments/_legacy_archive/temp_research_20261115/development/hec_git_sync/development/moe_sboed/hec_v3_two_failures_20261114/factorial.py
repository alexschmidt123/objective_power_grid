"""Complete frozen inherited/new-router x joint/selected-decision comparison."""
from diagnostic_common import *
def main():
    out=ROOT/'scratch_runs/hec_v3_two_failures_20261114_factorial';assert not out.exists();out.mkdir()
    bank,u,scorer=load();n.original.setup();lease=n.original.acquire();start=time.monotonic()
    old=ROOT/'scratch_runs/hec_v3_two_failures_20261114'
    files=[pathlib.Path(__file__),CF/'counterfactual_round3/actor.pt',*sorted(old.glob('*_scores.npy')),
        CANON/'counterfactual_scores.npy',ROOT/'scratch_runs/hec_v3_causal_troubleshoot_20261113/initial_router_scores.npy']
    protected={str(p):sha(p) for p in files};n.put(out/'MANIFEST.json',protected)
    try:
        lib=n.load_library();parent=lib.load_frozen('cuda');model=d.PathwayMoE(parent).cuda()
        model.load_state_dict(torch.load(CF/'counterfactual_round3/actor.pt',map_location='cuda',weights_only=True)['actor_state']);model.eval().requires_grad_(False)
        theta=torch.tensor(bank['truth'],device='cuda');noise=torch.tensor(bank['noise'],device='cuda');neg=torch.tensor(bank['contrasts'],device='cuda')
        uniforms=torch.tensor(u,device='cuda',dtype=torch.float32).clamp_max(1-torch.finfo(torch.float32).eps);values=[]
        with torch.no_grad(),torch.device('cuda'):
            for i in range(0,2000,16):
                th=theta[i:i+16];ds=torch.zeros(len(th),30,2,device='cuda');ys=torch.zeros(len(th),30,device='cuda',dtype=torch.float64)
                for t in range(30):
                    x=lib.packed(ds,ys,t);c=model.context(x);props,_,legacy,response=model.components(x)
                    raw=parent.coordinator(torch.cat([c['s'],legacy.flatten(1)],-1))+parent.response_gate(response.flatten(1))
                    g=(10*(raw/10).tanh()).softmax(-1)
                    chosen=(uniforms[i:i+16,t,None]>=g.cumsum(-1)).sum(-1).clamp_max(3)
                    action=props[torch.arange(len(th),device='cuda'),chosen]
                    ds[:,t]=action;ys[:,t]=signal(th.reshape(-1,2,2),action.double())+.5*noise[i:i+16,t]
                result,_=scorer(type('Benchmark',(),{'sigma':.5,'calls':0})(),dict(actions=list(ds.unbind(1)),observations=list(ys[...,None].unbind(1)),truth=th),neg,16384,'cuda')
                values.extend(result[:,-1].numpy());n.put(out/'STATUS.json',dict(status='EVALUATING',completed=min(i+16,2000),total=2000))
        scores=np.array(values);np.save(out/'inherited_selected_scores.npy',scores)
        reference=np.load(ROOT/'scratch_runs/hec_v3_causal_troubleshoot_20261113/initial_router_scores.npy')
        error=float(np.abs(scores-reference).max());assert error<1e-9
        ji=np.load(old/'inherited_scores.npy');jn=np.load(old/'new_router_scores.npy');cn=np.load(CANON/'counterfactual_scores.npy')
        contrasts=dict(joint_minus_selected_same_inherited_router=paired(ji,scores),
            new_minus_inherited_router_same_selected_decisions=paired(cn,scores),
            new_minus_inherited_router_same_joint_head=paired(jn,ji),
            router_decoder_interaction=paired(jn-cn,ji-scores))
        maximum=0
        for j,k in enumerate(sorted(contrasts,key=lambda k:contrasts[k]['p_two_sided'])):
            maximum=max(maximum,min(1,(len(contrasts)-j)*contrasts[k]['p_two_sided']));contrasts[k]['p_Holm']=maximum
        assert all(sha(path)==h for path,h in protected.items())
        result=dict(status='COMPLETE',protocol=LOCK,training_updates=0,reproduction_max_error=error,
            means=dict(inherited_router_joint_head=float(ji.mean()),inherited_router_selected_decision=float(scores.mean()),
                new_router_joint_head=float(jn.mean()),new_router_selected_decision=float(cn.mean())),
            comparisons=contrasts,seconds=time.monotonic()-start,historical_artifacts_preserved=True)
        n.put(out/'RESULTS.json',result);n.put(out/'STATUS.json',dict(status='COMPLETE',training_updates=0))
    except Exception as exc:n.put(out/'STATUS.json',dict(status='FAILED',error=repr(exc)));raise
    finally:
        for f in lease:f.close()
if __name__=='__main__':main()
