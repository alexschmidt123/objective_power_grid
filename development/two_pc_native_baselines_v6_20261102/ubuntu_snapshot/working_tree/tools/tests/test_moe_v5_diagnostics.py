import copy,math
import pytest,torch
from src.research.moe_v5_diagnostics import prefix_scores,captured_rollout,Work,state_hash,diagnose
from src.research.deepseek_sboed import DeepSeekPolicy
from src.research.benchmarks import LocationFinding
from src.research.engine import rollout

def benchmark():return LocationFinding({'location_finding':{'design_bounds':[-4,4]},'observation':{'noise_sigma':.5}})

def test_telescoping_signed_information_and_gradients():
    x=torch.randn(2,4,5,dtype=torch.float64,generator=torch.Generator().manual_seed(61))
    x[:,1,0]-=10;x.requires_grad_(True);d=prefix_scores(x)
    expected=x.sum(1)[:,0]-torch.logsumexp(x.sum(1),-1)+math.log(5)
    torch.testing.assert_close(d['increments'].sum(1),expected)
    a=torch.autograd.grad(d['increments'].sum(),x,retain_graph=True)[0]
    b=torch.autograd.grad(expected.sum(),x)[0];torch.testing.assert_close(a,b)
    assert (d['increments']<0).any()

def test_truth_index_ceiling_and_stable_extreme_loglik():
    x=torch.tensor([[[1e6,-1e6,-1e6],[-1e6,1e6,1e6]]],dtype=torch.float64)
    d=prefix_scores(x);assert torch.isfinite(d['terminal']).all()
    assert float(d['prefix'].max())<=math.log(3)+1e-9
    with pytest.raises(ValueError):prefix_scores(torch.zeros(2,3))

def test_capture_is_behavior_and_gradient_preserving():
    torch.manual_seed(67);p=DeepSeekPolicy().train();q=copy.deepcopy(p)
    b=benchmark();a=captured_rollout(b,p,30,2,596771,16,Work())
    other=rollout(benchmark(),q,30,2,596771,16)
    for x,y in zip(a['actions'],other['actions']):torch.testing.assert_close(x,y,rtol=0,atol=0)
    torch.testing.assert_close(prefix_scores(a['step_loglik'])['prefix'],other['info'])
    assert 'log_likelihood' not in b.__dict__
    ga=torch.autograd.grad(-prefix_scores(a['step_loglik'])['terminal'].mean(),tuple(p.parameters()),allow_unused=True)
    gb=torch.autograd.grad(-other['info'][:,-1].mean(),tuple(q.parameters()),allow_unused=True)
    for x,y in zip(ga,gb):
        assert (x is None)==(y is None)
        if x is not None:torch.testing.assert_close(x,y,atol=1e-10,rtol=1e-8)

def test_diagnostics_preserve_original_rng_and_parameters():
    torch.manual_seed(68);p=DeepSeekPolicy();before=state_hash(p);rng=torch.get_rng_state().clone()
    d=diagnose(benchmark(),p,batch=2,negatives=16,seed=596772)
    assert state_hash(p)==before and torch.equal(rng,torch.get_rng_state())
    assert d['training_resolution_saturation']=='NOT_MEASURED' and d['not_a_performance_evaluation']
    assert d['work']['trajectories']==2 and len(d['stage_routing'])==30

def test_resource_and_reserved_seed_rejection():
    work=Work()
    with pytest.raises(ValueError):work.rollout(17,30,16)
    with pytest.raises(ValueError):work.rollout(1,30,33)
    with pytest.raises(ValueError):captured_rollout(benchmark(),DeepSeekPolicy(),30,1,8831001,8,work)

def test_no_reset_state_carry_in_capture():
    class Stateful:
        name='stateful_toy';obs_dim=1;param_dim=1;sigma=.1;action_dim=2;action_scale=4.;obs_scale=3.
        def sample(self,shape,generator):return torch.randn(*shape,1,generator=generator,dtype=torch.float64)
        def initial(self,shape):self.states=[];return torch.zeros(*shape,1,dtype=torch.float64)
        def encode_action(self,a):return a/4
        def feasible(self,z,actions,h):return z.sigmoid()
        def forward(self,theta,state,action):
            self.states.append(state.detach().clone());new=state+theta+action[...,:1];return new,new
        def log_likelihood(self,y,mean):return (-.5*((y-mean)/self.sigma).square()).sum(-1)
    b=Stateful();captured_rollout(b,DeepSeekPolicy(horizon=3),3,2,596773,4,Work())
    assert len(b.states)==3 and b.states[0].abs().sum()==0 and b.states[1].abs().sum()>0
