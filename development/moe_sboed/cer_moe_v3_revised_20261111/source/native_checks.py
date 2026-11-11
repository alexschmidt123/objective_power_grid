"""Disposable native full-batch continuation, plus independent policy identities."""
from common import *
from integration import make_policy,install,actor_update,target_value,exposure
from train import Logger,instrument

def main():
    original.setup();leases=original.acquire()
    torch.manual_seed(CFG['validation_seed']);state=restore();a=state['algorithm'];old_actor_sha=sha(OLD/'arms/candidate/actor.pt')
    policy=make_policy(a)
    with torch.no_grad():
        for k,head in enumerate(policy.experts):head[-1].bias.add_(.005*k)
    src=install(a,policy)
    assert exposure(a)==1
    for n,expected in [(200,1.),(350,.55),(500,.1),(1999,.1)]:
        a.update=16001+n;assert abs(exposure(a)-expected)<1e-12
    a.update=16001
    # Check gradient accumulation against one complete expectation (small fixture).
    from policy import expected_loss
    x=torch.randn(17,117,device='cuda');x[:,:30]=0;x[:,12]=1
    a.actor_net.epsilon=.37;a.critic_net.requires_grad_(False)
    a.actor_net.zero_grad();expected_loss(policy,a.critic_net,x).backward()
    expected=[None if p.grad is None else p.grad.clone() for p in policy.parameters()]
    a.actor_net.zero_grad()
    for xx in x.split(7):(expected_loss(policy,a.critic_net,xx)*len(xx)/len(x)).backward()
    for p,g in zip(policy.parameters(),expected):
        if g is not None:torch.testing.assert_close(p.grad,g,rtol=2e-3,atol=2e-5)
    a.critic_net.requires_grad_(True)
    a.dowel=l=Logger(a,'fixture',time.monotonic(),fixture=True);a.post_approx.dowel=l
    originals=instrument(a,l);original.restore(state['rng'])
    tick=time.monotonic();profile=json.loads((P/'frozen_protocol.json').read_text())['profile'];profile['discount']=1.
    with torch.device('cuda'):original.train_native(a,l,profile,target=16002)
    assert a.update==16002 and l.counts==dict(actor=1,critic=5,posterior=150)
    assert l.sim==dict(episodes=1000,observations=30000,batch_calls=30)
    assert a._cer_last['actor_rows']==300000
    assert all(g>0 for g in a._cer_last['expert_gradients'])
    assert sha(OLD/'arms/candidate/actor.pt')==old_actor_sha
    a.m_f,a.asses=originals
    put(O/'evidence/NATIVE_FULL_BATCH_CHECKS.json',dict(status='PASS',seconds=time.monotonic()-tick,
         replay_batch=10000,actor_rows=300000,native_optimizer_steps=l.counts,simulator_counts=l.sim,
         chunked_gradient_equivalence=True,exposure_schedule=True,all_expert_gradients=a._cer_last['expert_gradients'],
         modified_native_source_sha256=hashlib.sha256(src.encode()).hexdigest(),original_actor_unchanged=True,
         disposable_fixture_not_scientific_training=True))
    (O/'evidence/native_policy_expectations.py').write_text(src)
    print('CER_NATIVE_FULL_BATCH_PASS',flush=True)

if __name__=='__main__':main()
