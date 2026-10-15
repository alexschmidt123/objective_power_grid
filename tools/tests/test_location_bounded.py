import unittest,torch,numpy as np
from pathlib import Path
from src.research.benchmarks import LocationFinding
from src.research.engine import generator
from src.research.scorebed import OfficialScorePolicy,verify_bridge
from src.research.moe_architecture_controls import ArchitectureControl
class Tests(unittest.TestCase):
 def test_truncated_normal_random_has_correct_scale_and_support(self):
  b=LocationFinding({'location_finding':{'design_bounds':[-4.,4.],'random_design_distribution':'truncated_standard_normal'}})
  a=b.feasible(b.random_latent(100000,generator(101)),[],30)
  self.assertLess(float(a.abs().max()),4.)
  self.assertLess(abs(float(a.mean())),.01)
  self.assertLess(abs(float(a.var())-.99892929),.02)
 def test_scorebed_bounded_bridge(self):
  a=dict(np.load(Path(__file__).parent/'fixtures/official_scorebed_bridge_t30.npz'))
  a['common_design_bounds']=np.array([-4.,4.])
  a['reference_actions']=(-4+8/(1+np.exp(-a['reference_actions']))).astype(np.float32)
  p=OfficialScorePolicy(a,30,4.,3.)
  self.assertLess(verify_bridge(p,a),2e-5)
  self.assertTrue(p.outputs_design)
 def test_common_feasible_map_gradients(self):
  b=LocationFinding({'location_finding':{'design_bounds':[-4.,4.]}})
  x=torch.tensor([[0.,1.]],dtype=torch.float64,requires_grad=True)
  self.assertTrue(torch.autograd.gradcheck(lambda z:b.feasible(z,[],30),(x,)))
 def test_sum_encoder_recovers_raw_bounded_history(self):
  p=ArchitectureControl('sum_single',30,8,4.,3.)
  f=torch.zeros(2,121,dtype=torch.float64)
  slots=f[:,1:].reshape(2,30,4);slots[:,:3,:3]=torch.randn(2,3,3,dtype=torch.float64);slots[:,:3,3]=1.
  raw=torch.cat((slots[:,:3,:2]*4.,slots[:,:3,2:3]*3.),-1)
  expected=p.head(torch.nn.functional.pad(p.sum_encoder(raw).sum(1),(0,4*25-16)))
  torch.testing.assert_close(expected,p(f,3),rtol=0,atol=0)
if __name__=='__main__':unittest.main()
