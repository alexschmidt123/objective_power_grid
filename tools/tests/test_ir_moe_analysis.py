import math
import torch
from torch import nn
from tools.ir_moe_frozen_diagnostics import posterior_reference,remaining_values

class Gaussian:
    param_dim=1;action_dim=2;obs_dim=1;action_scale=1.;obs_scale=1.;sigma=.5
    def __init__(self):self.calls=0
    def sample(self,shape,rng):return torch.randn(*shape,1,generator=rng,dtype=torch.float64)
    def initial(self,shape):return torch.zeros(*shape,1,dtype=torch.float64)
    def encode_action(self,a):return a
    def feasible(self,a,history,horizon):return a
    def forward(self,theta,state,action):
        self.calls+=theta.numel();return theta*action[...,:1],state
    def log_likelihood(self,y,mean):return (-.5*((y-mean)/self.sigma).square()-math.log(self.sigma*math.sqrt(2*math.pi))).sum(-1)

class Constant(nn.Module):
    def forward(self,f,t):return torch.tensor([[.7,0.]],dtype=torch.float64).expand(len(f),-1)

def test_posterior_conditioning_matches_analytic_gaussian():
    b=Gaussian();h=dict(actions=[torch.tensor([[.8,0.]],dtype=torch.float64)],observations=[torch.tensor([[.4]],dtype=torch.float64)],stage=1,horizon=3)
    theta,w,ess=posterior_reference(b,h,100000,112)
    v=1/(1+(.8/.5)**2);m=v*.8*.4/.5**2
    mu=float((theta[:,0]*w).sum());variance=float(((theta[:,0]-mu).square()*w).sum())
    assert abs(mu-m)<.012 and abs(variance-v)<.012 and ess>10000

def test_remaining_horizon_is_not_immediate_utility():
    b=Gaussian();h=dict(actions=[torch.tensor([[.8,0.]],dtype=torch.float64)],observations=[torch.tensor([[.4]],dtype=torch.float64)],stage=1,horizon=3)
    truth=torch.tensor([[.5]],dtype=torch.float64)
    contrasts=torch.tensor([[[-1.],[0.],[1.]]],dtype=torch.float64)
    noise=torch.tensor([[[.2],[-.3]]],dtype=torch.float64);first=torch.tensor([[1.1,0.]],dtype=torch.float64)
    result,immediate=remaining_values(b,Constant(),h,first,truth,contrasts,noise)
    parameters=torch.cat([truth[:,None],contrasts],1)
    y0=truth*1.1+.5*noise[:,0];y1=truth*.7+.5*noise[:,1]
    l0=b.log_likelihood(y0[:,None],parameters*1.1);l1=b.log_likelihood(y1[:,None],parameters*.7)
    expected=(l0+l1)[:,0]-(l0+l1).logsumexp(-1)+math.log(4)
    torch.testing.assert_close(result,expected)
    torch.testing.assert_close(immediate,l0[:,0]-l0.logsumexp(-1)+math.log(4))
    assert not torch.equal(result,immediate)

def test_conditional_values_reproduce_same_history_and_noise():
    b=Gaussian();h=dict(actions=[],observations=[],stage=0,horizon=1)
    theta=torch.tensor([[.5]],dtype=torch.float64);c=torch.tensor([[[-1.],[1.]]],dtype=torch.float64);n=torch.zeros(1,1,1,dtype=torch.float64)
    a=remaining_values(b,None,h,torch.tensor([[1.,0.]],dtype=torch.float64),theta,c,n)
    q=remaining_values(b,None,h,torch.tensor([[1.,0.]],dtype=torch.float64),theta,c,n)
    torch.testing.assert_close(a[0],q[0])

