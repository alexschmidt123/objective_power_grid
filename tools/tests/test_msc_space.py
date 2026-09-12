import unittest
import numpy as np
import torch
from tools.audits.space import SpacePlanner,AuditBudget

class MSCSpaceTests(unittest.TestCase):
 def test_msc_quantile_and_evaluation_are_control_not_regret(self):
  p=SpacePlanner(np.zeros((3,4,1)),[.35,.35,.41,.41],[.35,.41],sigma=1,alpha=.1,penalty=10,objective='msc',budget=AuditBudget(inner=4,outer=4,first_candidates=3,calibration=8))
  cost,u=p.risk(p.prior());self.assertAlmostEqual(float(cost[0]),.41);self.assertAlmostEqual(float(u[0]),.41)
  r=p.evaluate(np.zeros((4,3,1)),[.35,.35,.41,.41],2,[0,1],101)
  self.assertEqual(set(r),{'no_probe','random','fixed','myopic','lookahead'})
  for row in r.values():np.testing.assert_allclose(row['loss'],.41)
 def test_msc_lookahead_detects_complementary_probes(self):
  x=np.array([[[0],[0],[0],[0]],[[-8],[-8],[8],[8]],[[-8],[8],[-8],[8]]])
  p=SpacePlanner(x,[.35,.41,.41,.35],[.35,.41],sigma=1,alpha=.1,penalty=10,objective='msc',budget=AuditBudget(inner=32,outer=32,first_candidates=3))
  choice=p.choose(p.prior(),torch.zeros((1,3),dtype=torch.bool),depth=2,seed=177)
  self.assertIn(int(choice[0]),[1,2])
