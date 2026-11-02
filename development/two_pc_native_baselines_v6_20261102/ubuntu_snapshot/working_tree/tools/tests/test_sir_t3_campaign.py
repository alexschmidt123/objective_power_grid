import math,inspect
import numpy as np,torch,pytest
from scipy.integrate import solve_ivp
from src.research.sir_t3_common import T3SIR,fixed_shared_spce
from src.research.sir_dad import SIRSumDAD
from src.research.sir_t3_native_vsoed import T3InputUnits
from src.research.engine import rollout,replay_likelihood
from src.research.policies import Policy

torch.set_num_threads(2)
RAW={'system':{'name':'sir_ode'},'sir_ode':{'execution_backend':'eager'},'observation':{'noise_sigma':1.}}

def test_physical_prior_and_three_actions():
 b=T3SIR(RAW);assert b.population==500 and b.i0==2 and b.end==100 and b.gap==.1 and b.steps==512
 torch.testing.assert_close(b.initial((1,)),torch.tensor([[498.,2.,0.,0.]],dtype=torch.float64))
 g=torch.Generator().manual_seed(2026102835);th=b.sample((5000,),g);z=b.transform(th)
 assert z.mean().abs()<.04 and (z.std(0)-1).abs().max()<.04
 p=SIRSumDAD(3);r=rollout(b,p,3,3,2026102835,4);a=torch.stack(r['actions'],1)[:,:,0]
 assert a.shape==(3,3) and (a[:,1:]-a[:,:-1]>=.1-1e-10).all() and a.min()>=.1 and a.max()<=100
 assert list(inspect.signature(p.forward).parameters)==['features','stage']

def test_independent_high_accuracy_ode_reference():
 b=T3SIR(RAW);theta=torch.tensor([[.5,.1],[1.7,.4],[.08,.03],[2.7,.005]],dtype=torch.float64);state=b.initial((4,));errors=[]
 for time in [1.,20.,99.9]:
  y,state=b.forward(theta,state,torch.full((4,1),time,dtype=torch.float64))
  for j,(beta,gamma) in enumerate(theta.numpy()):
   def rhs(t,s):
    rate=beta*s[0]*s[1]/500;recovery=gamma*s[1];return [-rate,rate-recovery,recovery]
   ref=solve_ivp(rhs,(0,time),[498.,2.,0.],method='DOP853',rtol=1e-12,atol=1e-13).y[:,-1]
   errors.append(np.max(abs(state[j,:3].numpy()-ref)))
 assert max(errors)<.001
 assert (state[:,:3].sum(-1)-500).abs().max()<1e-8

def test_shared_fixed_is_identical_to_replayed_shared_bank():
 torch.manual_seed(101);b=T3SIR(RAW);p=Policy(10,1,3,'fixed');v,d=fixed_shared_spce(b,p,4,8,2026102835,return_details=True)
 r=d['result'];bank=d['bank'][None].expand(4,-1,-1);ll,_,_=replay_likelihood(b,bank,r['actions'],r['observations']);truth,_,_=replay_likelihood(b,r['truth'][:,None],r['actions'],r['observations']);ref=truth[:,0]-torch.logaddexp(truth[:,0],torch.logsumexp(ll,-1))+math.log(9)
 torch.testing.assert_close(v,ref,rtol=1e-10,atol=1e-10)
 g1=torch.autograd.grad(v.mean(),p.sequence,retain_graph=True)[0];g2=torch.autograd.grad(ref.mean(),p.sequence)[0];torch.testing.assert_close(g1,g2,rtol=1e-7,atol=1e-8)

@pytest.mark.parametrize('critic,posterior',[(False,False),(True,False),(False,True)])
def test_normalization_invertible_and_differentiable(critic,posterior):
 n=6 if posterior else 7+int(critic);x=torch.arange(1,n+1,dtype=torch.float64)[None].requires_grad_();h=T3InputUnits(3,critic=critic,posterior=posterior);y=h(None,(x,))[0]
 scale=x/y;torch.testing.assert_close(y*scale,x);g=torch.autograd.grad(y.sum(),x)[0];torch.testing.assert_close(g,1/scale)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='GPU exact discrete RK4 equivalence')
def test_optimized_per_block_counts_all_input_vjps():
 from src.research.benchmarks import SIR
 with torch.device('cuda:0'):
  torch.manual_seed(2026102835);theta=torch.exp(torch.randn(257,2,dtype=torch.float64)*.5+torch.tensor([math.log(.5),math.log(.1)],dtype=torch.float64));theta[-1]=torch.tensor([7.,.01]);state=SIR(RAW).initial((257,));state[:,3]=.1;action=torch.linspace(.10001,99.9,257,dtype=torch.float64)[:,None];w=torch.randn(257,4,dtype=torch.float64);values=[]
  for cls in [SIR,T3SIR]:
   b=cls(dict(RAW,sir_ode={'execution_backend':'fused_rk4'}));th=theta.clone().requires_grad_();st=state.clone().requires_grad_();a=action.clone().requires_grad_();mean,out=b.forward(th,st,a);grads=torch.autograd.grad((out*w).sum(),(th,st,a));values.append([out.detach(),*[x.detach() for x in grads]])
  for x,y in zip(*values):torch.testing.assert_close(x,y,rtol=1e-9,atol=1e-8)
