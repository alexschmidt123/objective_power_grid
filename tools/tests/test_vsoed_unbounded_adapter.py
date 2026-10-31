import unittest,torch,io
from torch import nn
from src.research.vendor.vsoed.utils import Net
from src.research.upstream_vsoed import compress_input,finite_gradient_step
class Tests(unittest.TestCase):
 def test_unbounded_actor_reload_retains_compression(self):
  bounds=torch.tensor([[-float('inf'),float('inf')]]*2)
  a=Net([5,16,2],nn.ReLU,bounds,'actor');a.register_forward_pre_hook(compress_input)
  x=torch.ones(3,5)*1e18;y=a(x)
  b=Net([5,16,2],nn.ReLU,bounds,'actor');b.load_state_dict(a.state_dict());b.register_forward_pre_hook(compress_input)
  torch.testing.assert_close(y,b(x),rtol=0,atol=0)
  self.assertTrue(torch.isfinite(y).all())
  with torch.no_grad():a.net[-1].weight.zero_();a.net[-1].bias.fill_(6.)
  self.assertTrue((a(x)>4).all())
 def test_optimizer_rejects_nonfinite_gradient(self):
  p=nn.Parameter(torch.ones(1));o=torch.optim.Adam([p]);o.register_step_pre_hook(finite_gradient_step);p.grad=torch.full_like(p,float('nan'))
  with self.assertRaises(RuntimeError):o.step()
  torch.testing.assert_close(p,torch.ones_like(p))
if __name__=='__main__':unittest.main()
