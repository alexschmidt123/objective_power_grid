import unittest,torch
from src.research.upstream_vsoed import ActionTrustRegionAdam
class Tests(unittest.TestCase):
 def test_update_limited_without_bounding_action(self):
  p=torch.nn.Linear(1,1).double()
  with torch.no_grad():p.weight.zero_();p.bias.fill_(10.)
  x=torch.ones(8,1,dtype=torch.float64);before=p(x).detach()
  o=ActionTrustRegionAdam(p.parameters(),p,lambda:x,radius=.05,lr=100.)
  (-p(x).mean()).backward();o.step()
  self.assertLessEqual(float((p(x)-before).abs().max()),.05)
  self.assertGreater(float(p(x).min()),4.)
  self.assertLess(o.last_scale,1.)
 def test_gradient_descent_direction_retained(self):
  p=torch.nn.Linear(1,1).double();x=torch.ones(3,1,dtype=torch.float64)
  before=p(x).detach();o=ActionTrustRegionAdam(p.parameters(),p,lambda:x,radius=.05,lr=.001)
  (-p(x).mean()).backward();o.step();self.assertTrue((p(x)>before).all())
if __name__=='__main__':unittest.main()
