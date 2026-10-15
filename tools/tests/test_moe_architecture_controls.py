import unittest
import torch
from src.research.moe_architecture_controls import ArchitectureControl,CONTROL_METHODS,component_groups
from src.research.sparse_moe import SparseMoESBOED
from src.research.benchmarks import LocationFinding
from src.research.engine import rollout
class Tests(unittest.TestCase):
 def setUp(self):torch.manual_seed(101);torch.set_num_threads(1)
 def test_same_head_for_encoder_comparison(self):
  a=ArchitectureControl('sum_single',30);b=ArchitectureControl('attention_single',30)
  self.assertEqual([(n,tuple(p.shape)) for n,p in a.head.named_parameters()],[(n,tuple(p.shape)) for n,p in b.head.named_parameters()])
 def test_parameter_matched_and_full_information(self):
  for suffix in ['dense','sparse']:
   p=ArchitectureControl('attention_matched_'+suffix,30)
   n=sum(x.numel() for x in p.parameters())
   self.assertLess(abs(n-p.matched_target_parameters)/p.matched_target_parameters,.01)
   self.assertEqual(p.head[0].in_features,772)
 def test_original_sparse_exact_identity(self):
  torch.manual_seed(101);a=SparseMoESBOED(2,1,30,64,first_action=[0.,0.])
  torch.manual_seed(101);b=ArchitectureControl('attention_sparse_moe',30)
  self.assertEqual(list(a.state_dict()),list(b.state_dict()))
  for n,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[n],rtol=0,atol=0)
  f=torch.randn(4,121,dtype=torch.float64);f[:,4::4]=1.
  for training in [True,False]:
   a.train(training);b.train(training)
   torch.testing.assert_close(a(f,12),b(f,12),rtol=0,atol=0)
 def test_all_control_end_to_end_gradients(self):
  b=LocationFinding({'location_finding':{'design_bounds':None}})
  for method in CONTROL_METHODS:
   p=ArchitectureControl(method,30,hidden=8)
   val=rollout(b,p,30,2,101,8)['info'][:,-1].mean();val.backward()
   self.assertTrue(torch.isfinite(val))
   groups=component_groups(p)
   for name,params in groups.items():
    # Single selectors' mixing logits are redundant k=1 but bits must learn.
    self.assertTrue(any(v.grad is not None and v.grad.abs().sum()>0 for v in params),method+name)
 def test_sum_order_invariance_and_mask(self):
  p=ArchitectureControl('sum_single',30);f=torch.zeros(2,121,dtype=torch.float64)
  slots=f[:,1:].reshape(2,30,4);slots[:,:3,:3]=torch.randn(2,3,3,dtype=torch.float64);slots[:,:3,3]=1.
  out=p(f,3);copy=f.clone();copy[:,1:].reshape(2,30,4)[:,:3]=slots[:,[2,0,1]]
  torch.testing.assert_close(out,p(copy,3))
  copy=f.clone();copy[:,1:].reshape(2,30,4)[:,3:,:3]=999.
  torch.testing.assert_close(out,p(copy,3),rtol=0,atol=0)
 def test_checkpoint_reconstruction(self):
  for method in CONTROL_METHODS:
   a=ArchitectureControl(method,30,8);b=ArchitectureControl(method,30,8);b.load_state_dict(a.state_dict())
   f=torch.randn(2,121,dtype=torch.float64);f[:,4::4]=1.
   a.eval();b.eval();torch.testing.assert_close(a(f,9),b(f,9),rtol=0,atol=0)
if __name__=='__main__':unittest.main()
