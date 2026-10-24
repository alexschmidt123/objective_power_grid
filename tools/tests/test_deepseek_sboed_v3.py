import copy, math
import pytest, torch
from src.research.benchmarks import LocationFinding, Benchmark
from src.research.engine import rollout
from src.research.dad_location_policy import LocationDAD
from src.research.deepseek_sboed_v3 import DeepSeekV3Policy, streamed_spce, load_v3

def benchmark(): return LocationFinding({'location_finding':{'design_bounds':[-4,4]}})

@pytest.mark.parametrize('policy_type',[LocationDAD,DeepSeekV3Policy])
@pytest.mark.parametrize('chunk',[1,7,64])
def test_terminal_objective_and_all_gradients_match(policy_type,chunk):
 torch.manual_seed(101);p=policy_type(30);q=copy.deepcopy(p)
 b=benchmark();c=benchmark();r=rollout(b,p,30,3,400001,31);s=streamed_spce(c,q,30,3,400001,31,chunk)
 torch.testing.assert_close(r['info'][:,-1],s['info'][:,-1],atol=2e-10,rtol=1e-10)
 for x,y in zip(r['actions'],s['actions']):torch.testing.assert_close(x,y,atol=0,rtol=0)
 (-r['info'][:,-1].mean()).backward();(-s['info'][:,-1].mean()).backward()
 for (name,x),(n,y) in zip(p.named_parameters(),q.named_parameters()):
  assert name==n and (x.grad is None)==(y.grad is None)
  if x.grad is not None:torch.testing.assert_close(x.grad,y.grad,atol=2e-8,rtol=1e-7)
 assert s['work_counts']['executed_parameter_stage_propagations']==3*30*(1+2*31)
 assert s['work_counts']['executed_scalar_likelihood_evaluations']==3*30*(1+2*31)
 assert c.calls==s['work_counts']['executed_parameter_stage_propagations']

def test_native_contrast_count_finite_and_within_ceiling():
 torch.manual_seed(101);p=DeepSeekV3Policy();r=streamed_spce(benchmark(),p,30,2,400002,8192,1024)
 loss=-r['info'].mean()+p.balance_loss();loss.backward();p.mask_unused_gradients()
 assert torch.isfinite(loss) and (r['info']<=math.log(8193)+1e-10).all()
 assert all(torch.isfinite(x.grad).all() for x in p.parameters() if x.grad is not None)
 assert all((x.abs()<=4).all() for x in r['actions'])
 assert any(x.grad is not None and x.grad.abs().sum()>0 for x in p.router.parameters())
 assert any(x.grad is not None and x.grad.abs().sum()>0 for x in p.shared.parameters())
 assert sum(any(x.grad is not None and x.grad.abs().sum()>0 for x in e.parameters()) for e in p.routed)>=p.top_k

class AnalyticGaussian(Benchmark):
 name='location_finding';param_dim=1;action_dim=1;sigma=.5;action_scale=1.;obs_scale=1.
 def __init__(self):self.calls=0
 def sample(self,shape,generator):return torch.randn(*shape,1,generator=generator,dtype=torch.float64)
 def feasible(self,latent,actions,horizon):return latent
 def propagate(self,theta,state,action):return theta*action,state
class ScalarPolicy(torch.nn.Module):
 kind='test';outputs_design=True
 def __init__(self):super().__init__();self.action=torch.nn.Parameter(torch.tensor(.1,dtype=torch.float64))
 def forward(self,f,stage):return self.action.expand(len(f),1)

def test_controlled_gaussian_information_learning_improves():
 p=ScalarPolicy();o=torch.optim.Adam(p.parameters(),lr=.04)
 def objective():return streamed_spce(AnalyticGaussian(),p,1,128,400003,64,32)['info'].mean()
 initial=float(objective().detach())
 for _ in range(20):o.zero_grad();(-objective()).backward();o.step()
 assert float(objective().detach())>initial+.2

def test_deployment_has_no_information_search_or_simulator_calls(monkeypatch):
 p=DeepSeekV3Policy().eval();f=torch.zeros(1,121,dtype=torch.float64)
 def forbidden(*a,**k):raise AssertionError('Forbidden online training/simulation/information search')
 import src.research.deepseek_sboed_v3 as v
 monkeypatch.setattr(v,'streamed_spce',forbidden);monkeypatch.setattr(v,'rollout',forbidden)
 monkeypatch.setattr(LocationFinding,'forward',forbidden)
 monkeypatch.setattr(torch.optim.Adam,'step',forbidden)
 calls=[0]*7;hooks=[e.register_forward_hook(lambda a,b,c,i=i:calls.__setitem__(i,calls[i]+1)) for i,e in enumerate(p.routed)]
 with torch.no_grad():a=p(f,0);b=p(f,0)
 torch.testing.assert_close(a,b,atol=0,rtol=0);assert sum(calls)==6
 for h in hooks:h.remove()

def test_v3_checkpoint_roundtrip(tmp_path):
 from src.research.deepseek_sboed_experiment import save
 p=DeepSeekV3Policy().eval();path=tmp_path/'v3.pt';save(p,path,0,{'test':True});q=load_v3(path)
 f=torch.zeros(2,121,dtype=torch.float64)
 torch.testing.assert_close(p(f,0),q(f,0),atol=0,rtol=0)


def test_every_expert_has_a_valid_selected_gradient_path():
 p=DeepSeekV3Policy();z=torch.randn(7,16,dtype=torch.float64)
 ids=torch.stack([torch.arange(7),(torch.arange(7)+1)%7,(torch.arange(7)+2)%7],1)
 weights=p.router(z).softmax(-1).gather(1,ids)
 p.routed_feature_sum(z,weights,ids).square().sum().backward()
 assert all(any(x.grad is not None and x.grad.abs().sum()>0 for x in e.parameters()) for e in p.routed)
 assert p.router.weight.grad.abs().sum()>0
