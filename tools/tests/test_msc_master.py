import unittest,tempfile
from pathlib import Path
import numpy as np
import torch
from tools.audits.campaign import run_policy
from tools.audits.space import AuditBudget

class MSCMasterTests(unittest.TestCase):
 @unittest.skipUnless(torch.cuda.is_available(),'CUDA required by master planner')
 def test_fresh_validation_retains_msc_not_shortfall_regret(self):
  inputs={'curves':np.zeros((4,6,1)),'U':np.array([.35,.35,.41,.41]),'grid':np.array([.35,.41])}
  validation={'curves':inputs['curves'],'U':inputs['U'],'u_opt':np.array([.34,.34,.40,.40]),'action_ids':np.arange(6),'candidate_safe':np.ones((4,2),dtype=bool)}
  meta={'n_support':4,'sigma':1.,'durations':list(range(6)),'duration_actions':{str(k):[k] for k in range(6)},'objective':'msc','posterior_coverage':.9,'finalists':[list(range(6))],'horizons':[3]}
  with tempfile.TemporaryDirectory() as d:
   result=run_policy(Path(d),'test',list(range(6)),3,AuditBudget(inner=4,outer=2,first_candidates=6,calibration=8,fixed_restarts=2),inputs,validation,meta,[101],continuous=True)
   self.assertEqual(result['metric'],'selected_msc_with_physical_safety')
   for method in result['methods'].values():
    self.assertAlmostEqual(method['mean_loss'],.41)
    self.assertEqual(method['physical_safety_rate'],1.)
