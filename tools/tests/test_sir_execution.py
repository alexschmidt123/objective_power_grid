"""Execution switches must leave the benchmark contract unchanged."""
import copy
import pytest
import torch
from src.research.benchmarks import SIR
from src.research.sir_execution import rk4_steps
from src.research.journal_validation import CommonValidation


@pytest.mark.parametrize('duration', [.11, 24., 99.])
def test_eight_step_blocks_match_reference(monkeypatch,duration):
    import src.research.sir_execution as execution
    # Exercise decomposition without requiring a compiler/GPU in the unit suite.
    monkeypatch.setattr(execution,'compiled_steps',lambda n:
        lambda *args: rk4_steps(*args,steps=n))
    theta=torch.tensor([[.5,.1],[1.7,.4]],dtype=torch.float64)
    results=[]
    for backend in ['eager','compiled_rk4']:
        b=SIR({'sir_ode':{'execution_backend':backend}})
        action=torch.full((2,1),duration,dtype=torch.float64,requires_grad=True)
        mean,state=b.propagate(theta,b.initial((2,)),action)
        grad=torch.autograd.grad(mean.sum(),action)[0]
        results.append((state,grad))
    for a,b in zip(*results):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)


def test_execution_backend_is_opt_in():
    assert SIR({}).execution_backend=='eager'
    with pytest.raises(ValueError,match='execution backend'):
        SIR({'sir_ode':{'execution_backend':'approximate'}})


def test_validation_draw_identity_unchanged():
    raw={'system':{'name':'sir_ode'},'sir_ode':{},'observation':{'noise_sigma':1.}}
    reference=CommonValidation(raw,5,772001,4,8,2,4)
    accelerated=copy.deepcopy(raw)
    accelerated['sir_ode']['execution_backend']='compiled_rk4'
    candidate=CommonValidation(accelerated,5,772001,4,8,2,4,device='cpu')
    assert reference.identity==candidate.identity
    assert reference.device==candidate.device=='cpu'


def test_reserved_seed_remains_protected():
    with pytest.raises(ValueError,match='Reserved'):
        CommonValidation({},5,8831001,4,8)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA numerical equivalence')
def test_fused_random_states_and_all_input_vjps():
    with torch.device('cuda:0'):
        torch.manual_seed(8101)
        theta=(torch.randn(32,2,dtype=torch.float64)*.6+
               torch.tensor([-.7,-2.3],dtype=torch.float64)).exp()
        initial=torch.stack([torch.full((32,),470.),torch.full((32,),20.),
                             torch.full((32,),10.),torch.full((32,),1.)],-1).double()
        times=1.1+torch.rand(32,1,dtype=torch.float64)*98.
        cotangent=torch.randn(32,4,dtype=torch.float64)
        results=[]
        for backend in ['eager','fused_rk4']:
            b=SIR({'sir_ode':{'execution_backend':backend}})
            th=theta.clone().requires_grad_();st=initial.clone().requires_grad_()
            a=times.clone().requires_grad_()
            _,state=b.propagate(th,st,a)
            grads=torch.autograd.grad((state*cotangent).sum(),(th,st,a))
            results.append((state.detach(),*[x.detach() for x in grads]))
        for a,b in zip(*results):torch.testing.assert_close(a,b,rtol=1e-7,atol=1e-8)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA finite differences')
def test_fused_gradcheck():
    from src.research.sir_fused import fused_rk4
    with torch.device('cuda:0'):
        theta=torch.tensor([[.5,.1]],dtype=torch.float64,requires_grad=True)
        initial=torch.tensor([[450.,20.,30.]],dtype=torch.float64,requires_grad=True)
        duration=torch.tensor([3.2],dtype=torch.float64,requires_grad=True)
        counts=torch.tensor([512],dtype=torch.long)
        assert torch.autograd.gradcheck(
            lambda th,st,du: fused_rk4(th,st,du,counts,500.,512),
            (theta,initial,duration),eps=1e-5,rtol=1e-5,atol=1e-6)
