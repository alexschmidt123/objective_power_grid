"""CPU semantics tests plus actual frozen-checkpoint and native-step validation."""
import argparse,copy,gc,json,pathlib,sys,time
import torch
from torch import nn
from policy import DecisionMoE,observed_prefix

class ToyPolicy(nn.Module):
    def __init__(self):
        super().__init__();self.head=nn.Linear(117,2)
    def forward(self,x):return 4*self.head(x).tanh()

def fixture(batch=7,stage=14):
    x=torch.zeros(batch,117);x[:,stage]=1
    return x

def cpu_tests():
    torch.manual_seed(9);parent=ToyPolicy();v3=DecisionMoE(parent);x=fixture()
    for expert in v3.experts:assert torch.equal(expert(x),parent(x))
    ids=[{p.data_ptr() for p in e.parameters()} for e in v3.experts]
    assert all(not ids[i]&ids[j] for i in range(3) for j in range(i))
    with torch.no_grad():
        for k,expert in enumerate(v3.experts):expert.core.head.bias.add_(k*.4)
    u=torch.tensor([0.,.1,.3334,.5,.6667,.8,.999])
    action,info=v3(x,mode='uniform',uniforms=u,collect=True)
    assert info['selected_expert'].tolist()==[0,0,1,1,2,2,2]
    assert torch.equal(action,info['proposals'][torch.arange(len(x)),info['selected_expert']])
    assert not torch.allclose(action,info['proposals'].mean(1))
    hook=v3.router.register_forward_pre_hook(lambda *args:(_ for _ in ()).throw(AssertionError('router used in standalone mode')))
    assert torch.equal(v3(x,mode='expert',expert=1),v3.experts[1](x));hook.remove()
    poison=x.clone();poison[:,30+28:88]=73;poison[:,88+14:]=99
    assert torch.equal(observed_prefix(poison),observed_prefix(x))
    assert torch.equal(v3(poison,uniforms=u),v3(x,uniforms=u))
    critic=nn.Linear(119,1)
    v3.set_phase('router');loss=v3.native_actor_loss(critic,x)
    _,proposals,_,weights=v3.proposals_and_weights(x)
    manual=-sum(weights[:,k]*critic(torch.cat([x,proposals[:,k]],-1)).squeeze(-1) for k in range(3))
    torch.testing.assert_close(loss,manual.mean())
    loss.backward();assert all(p.grad is None for expert in v3.experts for p in expert.parameters())
    assert sum(float(p.grad.square().sum()) for p in v3.router.parameters() if p.grad is not None)>0
    v3.zero_grad(set_to_none=True);v3.set_phase('joint');v3.native_actor_loss(critic,x).backward()
    assert all(sum(float(p.grad.square().sum()) for p in e.parameters() if p.grad is not None)>0 for e in v3.experts)
    for name,p in v3.named_parameters():
        if p.grad is not None:assert torch.isfinite(p.grad).all(),name
    return dict(standalone_independence=True,independent_parameter_storage=True,hard_action_selection=True,
       uniform_same_execution=True,no_future_padding_leak=True,exact_categorical_actor_expectation=True,
       router_gradients=True,joint_expert_gradients=True)

def checkpoint_tests(output,native_step=False):
    import numpy as np
    from common import n,audit
    from native_adapter import build_native_class
    n.original.setup();lease=n.original.acquire();output.mkdir(parents=True,exist_ok=False)
    try:
        state=n.load_parent();a=state['algorithm'];parent=a.actor_net
        data=np.load(audit.BASE/'evaluation/diagnostic_candidate_histories.npz')
        x=torch.tensor(data['X'][:2,[0,14,29]].copy(),device='cuda').reshape(-1,117)
        original=parent(x).detach();v3=DecisionMoE(parent).cuda()
        with torch.no_grad():
            for k in range(3):assert torch.equal(v3(x,mode='expert',expert=k),original)
            action,info=v3(x,uniforms=torch.linspace(.01,.99,len(x),device='cuda'),collect=True)
            assert torch.equal(action,original)
        # Three copied policies have identical Q, so zero router gradient is correct at initialization.
        cls,patch=build_native_class(type(a));a.__class__=cls
        a.actor_net=v3;a.actor_target_net=copy.deepcopy(v3)
        lr=a.actor_optimizer.param_groups[0]['lr']
        a.actor_optimizer=torch.optim.Adam(v3.router.parameters(),lr=lr)
        a.actor_lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(a.actor_optimizer,gamma=.9999)
        result=dict(status='PASS',cpu=cpu_tests(),exact_V2_initialization=True,stages=[0,14,29],batch=2,
                    native_patch=patch,parameters=sum(p.numel() for p in v3.parameters()),
                    router_parameters=sum(p.numel() for p in v3.router.parameters()),native_updates=0)
        from train import rollout
        lib=n.load_library();theta=torch.tensor([[.2,-.4,.8,1.],[-.3,.7,-.5,.1]],device='cuda',dtype=torch.float64)
        noise=torch.zeros(2,30,device='cuda',dtype=torch.float64);u=torch.full((2,30),.45,device='cuda')
        ds,ys,_=lib.rollout(parent,theta,noise)
        actual,observed,counts=rollout(v3,lib,theta,noise,u,'learned')
        torch.testing.assert_close(actual,ds,atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(observed,ys,atol=2e-4,rtol=2e-5)
        assert int(counts.sum())==60
        result['full_episode_exact_simulator_chronology']=True
        if native_step:
            from train import Logger,all_optimizers
            cfg=dict(updates=1,wall_seconds=300);logger=Logger(a,'validation_router',output,cfg,time.monotonic())
            a.dowel=logger;a.post_approx.dowel=logger;a.save_path=None
            for optimizer,key in all_optimizers(a):
                def count(*unused,key=key):logger.counts[key]+=1
                optimizer.register_step_post_hook(count)
            before={k:v.detach().clone() for k,v in v3.state_dict().items() if k.startswith('experts.')}
            profile=json.loads((n.P/'frozen_protocol.json').read_text())['profile'];profile['discount']=1.
            torch.manual_seed(202611133099)
            n.original.train_native(a,logger,profile,target=16002)
            assert logger.counts==dict(actor=1,critic=5,posterior=150)
            assert all(torch.equal(v,v3.state_dict()[k]) for k,v in before.items())
            result.update(native_updates=1,native_optimizer_counts=logger.counts,frozen_experts_preserved=True)
            for optimizer,_ in all_optimizers(a):
                optimizer._optimizer_step_pre_hooks.clear();optimizer._optimizer_step_post_hooks.clear()
            with (output/'VALIDATION_RUNTIME.pkl').open('wb') as f:n.cloudpickle.dump(dict(algorithm=a,rng=n.original.rng()),f,protocol=5)
            old=torch.storage._load_from_bytes
            import io
            torch.storage._load_from_bytes=lambda b:torch.load(io.BytesIO(b),map_location='cpu',weights_only=False)
            try:
                with (output/'VALIDATION_RUNTIME.pkl').open('rb') as f:restored=n.cloudpickle.load(f)
            finally:torch.storage._load_from_bytes=old
            assert n.digest(restored['algorithm'].actor_net.state_dict())==n.digest(a.actor_net.state_dict())
            assert callable(restored['algorithm'].actor_net.native_actor_loss)
            result['native_runtime_roundtrip']=True
        torch.save(dict(actor_state=v3.state_dict(),experts=3,trained_selector=False),output/'INITIAL_OR_VALIDATION_ACTOR.pt')
        n.put(output/'VALIDATION.json',result);print(json.dumps(result),flush=True)
    finally:
        for f in lease:f.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=pathlib.Path);p.add_argument('--native-step',action='store_true');args=p.parse_args()
    if args.output:checkpoint_tests(args.output.expanduser(),args.native_step)
    else:print(json.dumps(cpu_tests()))
