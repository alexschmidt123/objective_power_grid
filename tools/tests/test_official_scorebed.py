import unittest,numpy as np,torch
from pathlib import Path
from src.research.scorebed import OfficialScorePolicy,verify_bridge
class Tests(unittest.TestCase):
 def setUp(self):
  p=Path(__file__).resolve().parent/'fixtures/official_scorebed_bridge_t30.npz'
  self.arrays=np.load(p);self.p=OfficialScorePolicy(self.arrays,30)
 def test_all_stages_match_official_JAX(self):self.assertLess(verify_bridge(self.p,self.arrays),2e-5)
 def test_future_history_not_used(self):
  a=torch.randn(3,121,dtype=torch.float64);b=a.clone();b[:,1+4*5:]=1e8
  torch.testing.assert_close(self.p(a,5),self.p(b,5),rtol=0,atol=0)
 def test_unbounded_first_design(self):
  with torch.no_grad():self.p.layers['emitter_net_layers_0_bias'].fill_(6.);self.p.layers['emitter_net_layers_0_kernel'].zero_()
  self.assertTrue((self.p(torch.zeros(3,121,dtype=torch.float64),0)>4).all())
if __name__=='__main__':unittest.main()
