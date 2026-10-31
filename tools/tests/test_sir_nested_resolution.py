import torch
from tools.e1_sir_resolution import nested
from src.research.sir_evaluation import contrast_scores
from src.research.benchmarks import make_benchmark

def test_nested_scores_match_common_scorer():
 raw={'system':{'name':'sir_ode'},'sir_ode':{'integration_steps_per_stage':8},'observation':{'noise_sigma':1.}}
 b=make_benchmark(raw);g=torch.Generator().manual_seed(101);truth=b.sample((2,),g);bank=b.sample((8,),g)
 actions=[torch.tensor([[2.],[3.]],dtype=torch.float64),torch.tensor([[5.],[7.]],dtype=torch.float64)];state=b.initial((2,));obs=[]
 for x in actions:
  y,state=b.forward(truth,state,x);obs.append(y+.2)
 expected=nested(b,truth,actions,obs,bank,[4,8],chunk=3,device='cpu')
 result={'truth':truth,'actions':actions,'observations':obs}
 for k in [4,8]:
  ref=contrast_scores(b,result,bank[:k],chunk=2,device='cpu')[0][:,-1]
  torch.testing.assert_close(torch.from_numpy(expected[k]['scores']),ref,atol=1e-12,rtol=1e-12)
