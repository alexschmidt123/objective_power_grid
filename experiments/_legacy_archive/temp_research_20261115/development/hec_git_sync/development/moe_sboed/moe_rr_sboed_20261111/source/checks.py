"""Independent Top2 calculus, exact base identity and full native batch."""
from common import *
from rr_policy import RRPolicy,pair_probabilities,PAIRS
from integration import install,bind,loss_for
from train import Logger,instrument

def probability_checks():
    logits=torch.tensor([[.3,-.7,.2,1.1],[-5.,8.,2.,-3.]],dtype=torch.float64,requires_grad=True)
    p=pair_probabilities(logits,.1);w=logits.softmax(-1)
    literal=torch.stack([w[:,i]*w[:,j]/(1-w[:,i])+w[:,j]*w[:,i]/(1-w[:,j]) for i,j in PAIRS],-1)*.9+.1/6
    torch.testing.assert_close(p,literal,rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(p.sum(-1),torch.ones(2,dtype=p.dtype),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(pair_probabilities(torch.zeros(3,4,dtype=p.dtype),0),torch.full((3,6),1/6,dtype=p.dtype))
    values=torch.tensor([-.3,.2,1.1,-.7,.8,.4],dtype=p.dtype)
    direct=torch.autograd.grad((p*values).sum(),logits,retain_graph=True)[0];score=torch.zeros_like(logits)
    for i in range(2):
        for j in range(6):score+=(p[i,j]*values[j]).detach()*torch.autograd.grad(p[i,j].log(),logits,retain_graph=True)[0]
    torch.testing.assert_close(direct,score,rtol=1e-11,atol=1e-11)
    assert torch.autograd.gradcheck(lambda x: (pair_probabilities(x,.1)*values).sum(),(logits,),eps=1e-6,atol=1e-6)
    gen=torch.Generator().manual_seed(101);draw=torch.multinomial(p[0].detach().expand(200000,-1),1,generator=gen).flatten()
    torch.testing.assert_close(torch.bincount(draw,minlength=6).double()/len(draw),p[0].detach(),rtol=0,atol=.004)
    extreme=pair_probabilities(torch.tensor([[1000.,-1000.,0.,500.]],dtype=p.dtype),.1)
    assert torch.isfinite(extreme).all() and float(extreme.min())>=.1/6

def main():
    probability_checks();original.setup();lease=original.acquire();torch.manual_seed(CFG['validation_seed'])
    state=restore();a=state['algorithm'];parent=copy.deepcopy(a.actor_net).eval();src=install(a);model=a.actor_net
    before=digest(dict(model.base.named_buffers()));original_sha=sha(OLD/'arms/candidate/actor.pt')
    with torch.no_grad(),torch.device('cuda'):
        for t in range(30):
            ds=torch.randn(3,30,2);ys=torch.randn(3,30,1);stages=torch.full((3,),t,dtype=torch.long)
            x=prefix_inputs(ds,ys,stages);torch.testing.assert_close(x,a.form_actor_input(t,ds[:,:t],ys[:,:t]),rtol=0,atol=0)
            dd=ds.clone();yy=ys.clone();dd[:,t:]=999;yy[:,t:]=-999
            torch.testing.assert_close(x,prefix_inputs(dd,yy,stages),rtol=0,atol=0)
            base=parent(x,routing_override=x.new_full((4,),.25));features=model.frozen(x)
            actions,p,delta=model.distribution(features);assert torch.count_nonzero(delta)==0
            torch.testing.assert_close(actions,base[:,None].expand(-1,6,-1),rtol=0,atol=0)
            for j in range(6):
                action,info=model.select(x,u=x.new_full((3,),(j+.5)/6),uniform=True,collect=True)
                torch.testing.assert_close(action,base,rtol=0,atol=0)
                assert bool((info['selected'][:,0]!=info['selected'][:,1]).all())
    x=x[:2];features=model.frozen(x);a.critic_net.requires_grad_(False)
    model.zero_grad();loss_for(a,x,features).backward()
    assert norm([p.grad for p in model.correction.parameters()])>0
    assert all(norm([p.grad for p in m.parameters()])==0 for m in model.adaptations)
    assert norm([p.grad for p in model.router.parameters()])==0
    with torch.no_grad():model.correction.weight.normal_(0,.01)
    a._rr_uniform=False;model.zero_grad();loss_for(a,x,features).backward()
    assert all(norm([p.grad for p in m.parameters()])>0 for m in model.adaptations)
    assert norm([p.grad for p in model.router.parameters()])>0
    with torch.no_grad():
        all_actions,_=model.all_actions(features)
        assert float((all_actions-all_actions[:,:1]).abs().max())>0
        zero=torch.zeros_like(model.tokens(features))
        assert torch.count_nonzero(model.radius*model.correction(zero.sum(-2)).tanh())==0
        for j in range(6):
            selected=model.select(x,u=features.new_full((2,),(j+.5)/6),uniform=True)
            torch.testing.assert_close(selected,all_actions[:,j],rtol=2e-5,atol=2e-6)
    a.critic_net.requires_grad_(True)
    with torch.no_grad():model.correction.weight.zero_()
    a.actor_target_net.load_state_dict(model.state_dict());a._rr_uniform=True
    a.dowel=l=Logger(a,'fixture',time.monotonic(),16001,fixture=True);a.post_approx.dowel=l;originals=instrument(a,l)
    original.restore(state['rng']);tick=time.monotonic();profile=json.loads((P/'frozen_protocol.json').read_text())['profile'];profile['discount']=1.
    with torch.device('cuda'):original.train_native(a,l,profile,target=16002)
    assert a.update==16002 and l.counts==dict(actor=1,critic=5,posterior=150)
    assert l.sim==dict(episodes=1000,observations=30000,batch_calls=30)
    assert a._rr_last['actor_rows']==300000
    assert digest(dict(a.actor_net.base.named_buffers()))==before==digest(dict(a.actor_target_net.base.named_buffers()))
    with torch.no_grad(),torch.device('cuda'):
        for case,t in [(0,0),(17,12),(9999,29)]:
            xx=a.form_actor_input(t,a.ds_hist[case:case+1,:t],a.ys_hist[case:case+1,:t])
            torch.testing.assert_close(a._rr_cache[case,t],model.frozen(xx)[0],rtol=2e-4,atol=1e-4)
    a.m_f,a.asses=originals;a._rr_cache=None
    for opt in allopts(a):opt._optimizer_step_pre_hooks.clear();opt._optimizer_step_post_hooks.clear()
    fixture=O/'evidence/fixture_runtime.pkl'
    with fixture.open('wb') as f:cloudpickle.dump(dict(algorithm=a,rng=original.rng()),f,protocol=5)
    with fixture.open('rb') as f:reloaded=cloudpickle.load(f)
    assert digest(reloaded['algorithm'].actor_net.state_dict())==digest(a.actor_net.state_dict())
    assert sha(OLD/'arms/candidate/actor.pt')==original_sha
    put(O/'evidence/IMPLEMENTATION_CHECKS.json',dict(status='PASS',native_batch=10000,actor_rows=300000,all30stages=True,
        initial_uniform_HEC_bitwise_identity=True,unordered_Top2_probabilities=True,finite_difference_and_score_identity=True,
        zero_head_blocks_router_and_adaptations=True,nonzero_head_unblocks_router_and_all_adaptations=True,
        no_independent_correction_bypass=True,actual_selection_changes_correction=True,base_and_target_frozen_buffers_unchanged=True,
        cache_prefix_alignment=True,future_mutation_invariance=True,full_native_optimizer_counts=l.counts,
        simulator_counts=l.sim,fixture_update_seconds=time.monotonic()-tick,checkpoint_roundtrip=True,
        original_checkpoint_unchanged=True,modified_native_source_sha256=hashlib.sha256(src.encode()).hexdigest(),scientific_updates=0))
    (O/'evidence/native_top2_expectations.py').write_text(src)
    print('RR_IMPLEMENTATION_CHECKS_PASS',flush=True)

if __name__=='__main__':main()
