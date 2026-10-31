import unittest,torch
from src.research.benchmarks import LocationFinding
from src.research.research_redq import RLPolicy,checked_step
class UnboundedTests(unittest.TestCase):
 def test_support_and_extreme_history(self):
  p=RLPolicy(LocationFinding({}),30);f=torch.zeros(2,121,dtype=torch.float64)
  with torch.no_grad():
   for x in p.head.parameters():x.zero_()
   p.head[-1].bias[:2].fill_(10.)
  a,lp=p.sample(f,deterministic=True)
  self.assertTrue(bool((a>4).all()));self.assertTrue(bool(torch.isfinite(lp).all()))
  f[:,1:]=1e25
  self.assertTrue(bool(torch.isfinite(p.summary(f)).all()))
 def test_invalid_loss_does_not_update(self):
  x=torch.nn.Parameter(torch.tensor(1.));opt=torch.optim.Adam([x])
  with self.assertRaises(FloatingPointError):checked_step(x*float('nan'),opt,[x],'test')
  self.assertEqual(float(x),1.)
if __name__=='__main__':unittest.main()
