import math,torch
from torch import nn
from src.research.benchmarks import SIR
from src.research.sir_upstream_vsoed import SIRNativeModel,UpstreamSIRActor,physical_time
from src.research.policies import history_features
from src.research.vendor.vsoed import PGvsOED

def test_native_sir_prior_simulator_history_and_actor():
 torch.manual_seed(101);torch.set_num_threads(2);b=SIR({});m=SIRNativeModel(b,5)
 alg=PGvsOED(5,2,1,1,m,m,[(0.,100.)],phys_state_info=(1,[0.],m.xp_f),actor_dimns=[256]*3,critic_dimns=[256]*3,activate=nn.ReLU)
 wrapped=UpstreamSIRActor(alg.actor_net,5,b);z=torch.tensor([[.2,-.3],[-.4,.5]]);theta=b.inverse(z.double());state=b.initial((2,));previous=torch.zeros(2,1,dtype=torch.float64);actions=[];observations=[];designs=[]
 torch.testing.assert_close(m.log_poi_prior(z),(-z.square()/2-.5*math.log(2*math.pi)).sum(-1))
 for stage in range(5):
  ds=torch.stack(designs,1) if designs else torch.empty(2,0,1);ys=torch.stack(observations,1).float() if observations else torch.empty(2,0,1)
  percent=alg.actor_net(alg.form_actor_input(stage,ds,ys))
  f=history_features(b,actions,observations,2,5);actual=wrapped(f,stage);expected=physical_time(stage,percent,previous,5,100,.1)
  torch.testing.assert_close(actual,expected,rtol=2e-6,atol=2e-5)
  mean=m.deterministic(stage,z,percent);reference,state=b.forward(theta,state,expected)
  torch.testing.assert_close(mean,reference,rtol=1e-10,atol=1e-10)
  torch.testing.assert_close(m.xp_f(stage,None,None,None).double(),expected,rtol=1e-6,atol=4e-6)
  y=mean+.2;torch.testing.assert_close(b.log_likelihood(y,mean),torch.full((2,),-.02-.5*math.log(2*math.pi),dtype=torch.float64))
  assert torch.all(actual>=previous+.1-1e-9)
  designs.append(percent.detach());actions.append(expected.detach());observations.append(y.detach());previous=expected.detach()


def test_original_SIR_log_coordinates_and_reward_prior_convention():
 b=SIR({});m=SIRNativeModel(b,5,latent_coordinates='log_parameters')
 torch.manual_seed(101);params=m.rvs(8)
 torch.manual_seed(101);expected=b.log_mean.float()+b.log_std*torch.randn(8,2)
 torch.testing.assert_close(params,expected);assert m.log_poi_prior(params)==0
 design=torch.full((8,1),10.);times=physical_time(0,design,torch.zeros(8,1),5,100,.1)
 mean=m.deterministic(0,params,design);reference,_=b.forward(params.double().exp(),b.initial((8,)),times)
 torch.testing.assert_close(mean,reference)
