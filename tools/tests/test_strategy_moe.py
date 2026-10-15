import copy
import pytest
import torch
from src.research.sparse_moe import SparseMoESBOED,stabilize_gradients
from src.research.strategy_moe import StrategyMoEPolicy,StrategyManager,component_groups,restore_strategy
from src.research.benchmarks import LocationFinding
from src.research.engine import rollout
from src.research.moe_architecture_controls import grad_norm

torch.set_num_threads(1)
def benchmark():return LocationFinding({'location_finding':{'design_bounds':[-4.,4.]},'observation':{'noise_sigma':.5}})
def base(h=30):
    torch.manual_seed(101)
    b=benchmark()
    return SparseMoESBOED(2,1,h,64,k=1,first_action=[0.,0.],history_encoder_revision=2,action_scale=b.action_scale,obs_scale=b.obs_scale)

@pytest.mark.parametrize('training',[True,False])
def test_always_reroute_reproduces_original_full_rollout_and_gradients(training):
    a=base();b=StrategyMoEPolicy(30,mode='always_reroute',reference=copy.deepcopy(a))
    a.train(training);b.train(training)
    x=rollout(benchmark(),a,30,4,13001,16);y=rollout(benchmark(),b,30,4,13001,16)
    for name in ['actions','observations','latents']:
        for av,bv in zip(x[name],y[name]):torch.testing.assert_close(av,bv,rtol=0,atol=0)
    torch.testing.assert_close(x['info'],y['info'],rtol=0,atol=0)
    if training:
        x['info'][:,-1].mean().backward();y['info'][:,-1].mean().backward()
        for name,value in a.named_parameters():
            other=dict(b.named_parameters())[('history_encoder.'+name) if name.startswith('sum_encoder.') else name]
            assert (value.grad is None)==(other.grad is None)
            if value.grad is not None:torch.testing.assert_close(value.grad,other.grad,rtol=1e-12,atol=1e-12)

def test_hard_continue_executes_only_active_expert_but_adapts_actions():
    p=StrategyMoEPolicy(30,mode='always_continue',reference=base()).eval()
    counts=[0]*4;router_calls=[];states=[]
    hooks=[e.register_forward_pre_hook(lambda m,args,j=j:counts.__setitem__(j,counts[j]+len(args[0]))) for j,e in enumerate(p.experts)]
    hooks.append(p.router.register_forward_pre_hook(lambda m,args:router_calls.append(len(args[0]))))
    hooks.append(p.register_forward_hook(lambda m,args,out:states.append({k:v.clone() for k,v in m.last_route['strategy'].items()}) if m.last_route is not None else None))
    with torch.no_grad():r=rollout(benchmark(),p,30,6,41001,0)
    for hook in hooks:hook.remove()
    assert sum(counts)==6*29 and sum(router_calls)==6
    for s in states[1:]:
        assert not s['reroute_requested'].any() and not s['actual_strategy_change'].any()
        assert (s['active_strategy']==states[0]['active_strategy']).all()
    assert (states[-1]['consecutive_duration']==29).all()
    assert any(not torch.allclose(r['actions'][1],a) for a in r['actions'][2:])

def test_reroute_request_is_not_an_expert_change():
    p=StrategyMoEPolicy(5,mode='always_reroute',reference=base(5)).eval()
    with torch.no_grad():
        p.router.bits.weight.zero_();p.router.bits.bias.fill_(1.)
        rollout(benchmark(),p,5,3,41,0)
    d=p.manager.last_diagnostics
    assert d['reroute_requested'].all() and not d['actual_strategy_change'].any()
    assert (d['reroute_requests']==4).all() and (d['strategy_changes']==0).all()
    assert (d['consecutive_duration']==4).all()

def test_learned_hard_switch_is_independent_for_each_batch_row():
    p=StrategyMoEPolicy(5,reference=base(5)).eval();m=p.manager
    calls=[];hook=p.router.register_forward_pre_hook(lambda model,args:calls.append(len(args[0])))
    with torch.no_grad():
        p.router.bits.weight.zero_();p.router.bits.bias.fill_(1.)
        m(torch.zeros(2,193,dtype=torch.float64),p.router,hard=True)
        p.router.bits.bias.fill_(-1.)
        m.switching_gate.logit.weight.zero_();m.switching_gate.logit.weight[0,0]=10.;m.switching_gate.logit.bias.zero_()
        context=torch.zeros(2,193,dtype=torch.float64);context[:,0]=torch.tensor([-1.,1.])
        q,_,_=m(context,p.router,hard=True)
    hook.remove()
    assert calls==[2,1]
    assert q.argmax(-1).tolist()==[3,0]
    assert m.last_diagnostics['reroute_requested'].tolist()==[False,True]
    assert m.last_diagnostics['actual_strategy_change'].tolist()==[False,True]
    assert m.last_diagnostics['consecutive_duration'].tolist()==[2,1]

def test_relaxed_state_gradients_reach_previous_state_and_switch():
    p=StrategyMoEPolicy(8,reference=base(8),initial_reroute_probability=.8)
    m=p.manager;context=torch.randn(4,193,dtype=torch.float64,requires_grad=True)
    q,_,_=m(context,p.router);q.retain_grad()
    next_context=(context+.2)
    q2,_,_=m(next_context,p.router)
    torch.testing.assert_close(q2.sum(-1),torch.ones(4,dtype=torch.float64))
    assert (q2>=0).all() and (q2<=1).all()
    q2[:,0].sum().backward()
    assert q.grad is not None and q.grad.abs().sum()>0
    assert grad_norm(m.switching_gate.parameters())>0
    assert context.grad.abs().sum()>0

def test_full_bounded_spce_gradients_all_components_and_clipping():
    p=StrategyMoEPolicy(12,reference=base(12))
    r=rollout(benchmark(),p,12,6,41001,16)
    (-r['info'][:,-1].mean()).backward()
    groups=component_groups(p)
    assert set(id(v) for vs in groups.values() for v in vs)==set(id(v) for v in p.parameters())
    assert sum(len(v) for v in groups.values())==len(list(p.parameters()))
    for name,group in groups.items():
        assert grad_norm(group)>0,name
        assert all(torch.isfinite(v.grad).all() for v in group if v.grad is not None)
    stabilize_gradients(p,groups,'component',5.)
    assert all(grad_norm(g)<=5.000001 for g in groups.values())
    assert all((a>=-4).all() and (a<=4).all() for a in r['actions'])

def test_episode_state_reset_and_checkpoint_round_trip():
    p=StrategyMoEPolicy(8,reference=base(8)).eval()
    with torch.no_grad():
        a=rollout(benchmark(),p,8,3,1001,0)
        b=rollout(benchmark(),p,8,3,1001,0)
    for x,y in zip(a['actions'],b['actions']):torch.testing.assert_close(x,y,rtol=0,atol=0)
    saved=dict(horizon=8,hidden=64,routing_mode='learned',initial_reroute_probability=.99,action_scale=4.,obs_scale=3.,gamma=p.router.gamma,state_dict=p.state_dict())
    q=restore_strategy(saved)
    assert q.manager.state is None
    with torch.no_grad():c=rollout(benchmark(),q,8,3,1001,0)
    for x,y in zip(a['actions'],c['actions']):torch.testing.assert_close(x,y,rtol=0,atol=0)

def test_sum_encoder_permutation_invariance_and_switch_observation_input():
    p=StrategyMoEPolicy(8,reference=base(8))
    f=torch.zeros(3,33,dtype=torch.float64);f[:,0]=.5
    slots=f[:,1:].reshape(3,8,4);slots[:,:4,:3]=torch.randn(3,4,3,dtype=torch.float64);slots[:,:4,3]=1
    g=f.clone();g[:,1:].reshape(3,8,4)[:,:4]=slots[:,[2,0,3,1]]
    torch.testing.assert_close(p.context(f),p.context(g),rtol=1e-12,atol=1e-12)
    with torch.no_grad():p.manager.switching_gate.logit.weight[:,:16].fill_(.1)
    previous=torch.full((3,4),.25,dtype=torch.float64)
    v=p.manager.switching_gate(p.context(f).mean(1),previous)
    altered=f.clone();altered[:,1:].reshape(3,8,4)[:,3,2]+=1
    w=p.manager.switching_gate(p.context(altered).mean(1),previous)
    assert not torch.equal(v,w)

def test_high_rerouting_initialization_and_added_parameter_count():
    ref=base();p=StrategyMoEPolicy(30,reference=copy.deepcopy(ref))
    b=p.manager.switching_gate(torch.randn(7,193,dtype=torch.float64),torch.full((7,4),.25,dtype=torch.float64))
    torch.testing.assert_close(b,torch.full((7,1),.99,dtype=torch.float64),rtol=1e-14,atol=1e-14)
    assert p.manager.available_strategies==('S0','S1','S2','S3')
    assert sum(v.numel() for v in p.parameters())-sum(v.numel() for v in ref.parameters())==198
