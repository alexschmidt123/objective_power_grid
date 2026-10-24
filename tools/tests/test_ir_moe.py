import copy
import itertools
import math
from unittest.mock import patch
import pytest
import torch
from src.research.ir_moe import IRDeepSeekPolicy, ordered_log_probability, sample_ordered, reward_to_go, integrated_spce
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout

RAW={'system':{'name':'location_finding'},'observation':{'noise_sigma':.5},'location_finding':{'design_bounds':[-4.,4.]},'research':{'paper_protocol':{'case':'two_source_common_bounded_box'}}}


def test_pl_distribution_and_logprob():
    logits=torch.tensor([[.3,-.4,.7]],dtype=torch.float64,requires_grad=True)
    probabilities=[]
    for selection in itertools.permutations(range(3),2):
        indices=torch.tensor([selection]);p=ordered_log_probability(logits,indices).exp()
        a,b=selection;expected=logits.softmax(-1)[0,a]*logits[:,[j for j in range(3) if j!=a]].softmax(-1)[0,[j for j in range(3) if j!=a].index(b)]
        torch.testing.assert_close(p[0],expected);probabilities.append(p)
    torch.testing.assert_close(sum(probabilities),torch.ones(1,dtype=torch.float64))
    with pytest.raises(ValueError):ordered_log_probability(logits,torch.tensor([[0,0]]))


def test_sampling_frequencies_and_no_starvation():
    logits=torch.tensor([[.3,-.4,.7]],dtype=torch.float64).expand(60000,-1)
    ids=sample_ordered(logits,2,torch.Generator().manual_seed(9))
    assert (ids[:,0]!=ids[:,1]).all()
    frequency=torch.bincount(ids[:,0],minlength=3).double()/len(ids)
    torch.testing.assert_close(frequency,logits[0].softmax(-1),atol=.007,rtol=0)
    assert (torch.bincount(ids.flatten(),minlength=3)>1000).all()


def test_complete_estimator_analytic_and_finite_difference():
    # Two sequential stochastic selections, history depends on first action.
    # Includes direct selected gate pathwise derivative, and encoder dependence.
    theta=torch.tensor(.23,dtype=torch.float64,requires_grad=True)
    def objective(value, surrogate=False):
        total=value*0
        for a,b in itertools.product(range(2),repeat=2):
            history=(a+1)*value.sin()
            l1=torch.stack([value,-value])[None]
            l2=torch.stack([value+history,-value-history])[None]
            p1=l1.softmax(-1)[0,a];p2=l2.softmax(-1)[0,b]
            u1=history.square()
            reward=((b+1)*value.cos()+history).square()
            if not surrogate:total=total+p1*p2*reward;continue
            # Conditional l2 derivative fixes history, but pathwise reward does not.
            conditional=torch.stack([value+history.detach(),-value-history.detach()])[None]
            # History contributes to probability of later draw through earlier
            # continuous actions: pathwise derivative of future expected reward
            # needs a stochastic-node downstream score-history correction.
            log1=ordered_log_probability(l1,torch.tensor([[a]]))[0]
            log2=ordered_log_probability(conditional,torch.tensor([[b]]))[0]
            history_correction=ordered_log_probability(l2,torch.tensor([[b]]))[0]-log2
            total=total+(p1*p2).detach()*(reward+reward.detach()*log1+(reward-u1).detach()*(log2+history_correction))
        return total
    exact=torch.autograd.grad(objective(theta),theta)[0]
    estimate=torch.autograd.grad(objective(theta,True),theta)[0]
    # This test deliberately exposes the history-score term. For endogenous
    # differentiable history, detached conditional scores alone are incomplete.
    torch.testing.assert_close(exact,estimate,atol=1e-12,rtol=1e-12)
    eps=1e-5
    fd=(objective(theta.detach()+eps)-objective(theta.detach()-eps))/(2*eps)
    torch.testing.assert_close(exact,fd,atol=1e-8,rtol=1e-7)


def test_prefix_indexing():
    x=torch.tensor([[1.,3.,6.]])
    torch.testing.assert_close(reward_to_go(x),torch.tensor([[6.,5.,3.]]))


def test_spce_prefix_and_pathwise_gradients():
    torch.manual_seed(101);p=IRDeepSeekPolicy(horizon=3).eval();q=copy.deepcopy(p)
    a=rollout(make_benchmark(RAW),p,3,4,123,19)
    b=integrated_spce(make_benchmark(RAW),q,3,4,123,19,7)
    torch.testing.assert_close(a['info'],b['prefix_information'],atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(a['info'][:,-1],b['info'][:,-1],atol=1e-10,rtol=1e-10)
    (-a['info'][:,-1].mean()).backward();(-b['info'][:,-1].mean()).backward()
    for x,y in zip(p.parameters(),q.parameters()):
        assert (x.grad is None)==(y.grad is None)
        if x.grad is not None:torch.testing.assert_close(x.grad,y.grad,atol=1e-9,rtol=1e-8)


def test_components_and_updates():
    torch.manual_seed(101);p=IRDeepSeekPolicy(horizon=3).train()
    r=integrated_spce(make_benchmark(RAW),p,3,64,123,7,4)
    loss=-r['info'].mean()+p.routing_loss(r['prefix_information'],torch.zeros(3))[0]+p.balance_loss()
    loss.backward();p.mask_unused_gradients()
    for module in [p.shared,p.router,p.encoder,p.emitter,*p.routed]:
        grads=[x.grad for x in module.parameters() if x.grad is not None]
        assert grads and all(torch.isfinite(x).all() for x in grads)
        assert sum(float(x.abs().sum()) for x in grads)>0
    before=p.shared.up.weight.detach().clone();torch.nn.utils.clip_grad_norm_(p.parameters(),5,error_if_nonfinite=True)
    torch.optim.Adam(p.parameters(),lr=5e-5).step()
    assert not torch.equal(before,p.shared.up.weight)
    for scores,ids in p._gate_cache:
        torch.testing.assert_close(scores.sum(-1),torch.ones(len(scores),dtype=torch.float64))
        assert (scores.gather(1,ids).sum(-1)<=1+1e-12).all()
    assert all((x.abs()<=4).all() for x in r['actions'])


def test_direct_online_no_forbidden_calls():
    p=IRDeepSeekPolicy().eval();f=torch.zeros(2,121,dtype=torch.float64)
    def forbidden(*a,**k):raise AssertionError('Forbidden online routine called')
    with patch('src.research.engine.rollout',forbidden),patch('src.research.engine.replay_likelihood',forbidden),patch('torch.optim.Adam.step',forbidden),patch('torch.autograd.backward',forbidden),patch('src.research.ir_moe.integrated_spce',forbidden),patch('src.research.benchmarks.make_benchmark',forbidden):
        with torch.no_grad():
            for stage in range(30):
                assert p(f,stage).shape==(2,2)
    assert p._score_cache==[]


def test_score_reward_baseline_stop_gradient_and_history_derivative():
    p=IRDeepSeekPolicy(horizon=2).train()
    f=torch.randn(16,9,dtype=torch.float64,requires_grad=True)
    p(f,0);p(f,1)
    # Direct conditional score cannot differentiate physical history.
    direct=torch.stack(p._score_cache,1).sum()
    assert torch.autograd.grad(direct,f,retain_graph=True,allow_unused=True)[0] is None
    rewards=torch.randn(16,2,dtype=torch.float64,requires_grad=True)
    baseline=torch.randn(2,dtype=torch.float64,requires_grad=True)
    loss,_=p.routing_loss(rewards,baseline)
    reward_grad,baseline_grad=torch.autograd.grad(loss,[rewards,baseline],retain_graph=True,allow_unused=True)
    assert reward_grad is None and baseline_grad is None
    assert torch.isfinite(torch.autograd.grad(loss,p.router.weight)[0]).all()


def test_complete_pl_score_pathwise_estimator():
    # Exact expectation over ordered top2 of three feature experts. The selected
    # gates retain their original full-softmax mass. Complete feature/action loss.
    theta=torch.tensor(.31,dtype=torch.float64,requires_grad=True)
    sets=list(itertools.permutations(range(3),2))
    def expected(x,surrogate=False):
        logits=torch.stack([x,2*x,-x])[None]
        score=logits.softmax(-1)[0]
        expert=torch.stack([x.sin(),x.square(),x.cos()])
        result=x*0
        for selection in sets:
            index=torch.tensor([selection]);lp=ordered_log_probability(logits,index)[0]
            action=x+(expert[index[0]]*score[index[0]]).sum()
            value=(action-.4).square()
            result=result+(lp.exp().detach()*(value+value.detach()*lp) if surrogate else lp.exp()*value)
        return result
    true=torch.autograd.grad(expected(theta),theta)[0]
    hybrid=torch.autograd.grad(expected(theta,True),theta)[0]
    eps=1e-5;fd=(expected(theta.detach()+eps)-expected(theta.detach()-eps))/(2*eps)
    torch.testing.assert_close(true,hybrid,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(true,fd,atol=1e-8,rtol=1e-8)
