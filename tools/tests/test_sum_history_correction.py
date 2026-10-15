"""Independent checks for a staged encoder-only MoE candidate."""
import importlib.util
from pathlib import Path
import torch
from src.research.sparse_moe import SparseMoESBOED as Original
from src.research.benchmarks import LocationFinding
from src.research.engine import rollout

import src.research.sparse_moe as candidate
torch.set_num_threads(1)

def test_default_is_identical():
 torch.manual_seed(101);a=Original(2,1,30,64,first_action=[0.,0.])
 torch.manual_seed(101);b=candidate.SparseMoESBOED(2,1,30,64,first_action=[0.,0.])
 assert list(a.state_dict())==list(b.state_dict())
 for n,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[n],rtol=0,atol=0)
 bench=LocationFinding({'location_finding':{'design_bounds':[-4.,4.]},'observation':{'noise_sigma':.5}})
 for mode in [True,False]:
  a.train(mode);b.train(mode)
  ar=rollout(bench,a,30,2,13001,8);br=rollout(bench,b,30,2,13001,8)
  torch.testing.assert_close(ar['info'],br['info'],rtol=0,atol=0)
  for x,y in zip(ar['actions'],br['actions']):torch.testing.assert_close(x,y,rtol=0,atol=0)

def test_sum_context_exact_reference_invariant_and_masked():
 torch.manual_seed(101);p=candidate.SparseMoESBOED(2,1,30,64,first_action=[0.,0.],history_encoder_revision=2)
 f=torch.zeros(4,121,dtype=torch.float64);f[:,0]=.5
 slots=f[:,1:].reshape(4,30,4);slots[:,:15,:3]=torch.randn(4,15,3,dtype=torch.float64);slots[:,:15,3]=1
 expected=p.sum_encoder(torch.cat((slots[:,:15,:2],slots[:,:15,2:3]*3),-1)).sum(1)
 torch.testing.assert_close(p.context(f)[:,0,:16],expected)
 copy=f.clone();permuted=copy[:,1:].reshape(4,30,4);permuted[:,:15]=slots[:,torch.randperm(15)]
 for mode in [True,False]:
  p.train(mode);torch.testing.assert_close(p(f,15),p(copy,15),rtol=1e-12,atol=1e-12)
 copy=f.clone();copy[:,1:].reshape(4,30,4)[:,15:,:3]=999
 torch.testing.assert_close(p.context(f),p.context(copy),rtol=0,atol=0)
 assert len(p.experts)==4 and p.router.k==1
 assert sum(v.numel() for v in p.parameters())==72544
 assert abs(72544/72208-1)<.005

def test_bounded_full_horizon_gradients_and_selected_only_execution():
 torch.manual_seed(101);p=candidate.SparseMoESBOED(2,1,30,64,first_action=[0.,0.],history_encoder_revision=2)
 bench=LocationFinding({'location_finding':{'design_bounds':[-4.,4.]},'observation':{'noise_sigma':.5}})
 r=rollout(bench,p,30,4,41001,8);r['info'][:,-1].mean().backward()
 assert torch.isfinite(r['info']).all()
 assert all((x>=-4).all() and (x<=4).all() for x in r['actions'])
 assert p.first_design.grad.abs().sum()>0
 assert all(any(v.grad is not None and v.grad.abs().sum()>0 for v in group.parameters()) for group in [p.sum_encoder,p.router,*p.experts])
 p.eval();counts=[]
 handles=[expert.register_forward_pre_hook(lambda m,args:counts.append(len(args[0]))) for expert in p.experts]
 rollout(bench,p,30,4,41001,0)
 for h in handles:h.remove()
 assert sum(counts)==4*29

if __name__=='__main__':
 for fn in [test_default_is_identical,test_sum_context_exact_reference_invariant_and_masked,test_bounded_full_horizon_gradients_and_selected_only_execution]:
  fn();print(fn.__name__,'PASS',flush=True)
