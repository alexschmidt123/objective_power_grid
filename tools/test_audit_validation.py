from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
import unittest
import numpy as np
from src.objectives.mocu.audit_validation import batch_control_oracle
from src.control.oracle_u_ctrl import compute_u_ctrl_opt

class FakeEngine:
    def simulate_metrics_batch(self,M,K,u,batch_size=512):
        safe=(u>=M[:,0]) & ((K[:,0]<0)|(u<=K[:,0]))
        return np.where(safe,0.,2.),np.zeros(len(u))
    def evaluate_one(self,M,K,u):
        r,_=self.simulate_metrics_batch(np.array([M]),np.array([K]),np.array([u]))
        return {'safe_total':float(r[0]<=1)}

class Tests(unittest.TestCase):
    def test_batched_matches_scalar_monotone_zero_island_and_infeasible(self):
        spec=SimpleNamespace(u_candidates=(0.,.1,.2,.3,.4,.5),rocof_limit_hz_s=1.,delta_f_nadir_hz=-1.)
        M=np.array([[.213],[0.],[.151],[.7]]); K=np.array([[-1.],[-1.],[.33],[-1.]])
        engine=FakeEngine(); batch=batch_control_oracle(engine,M,K,spec)
        for i in range(len(M)):
            scalar=compute_u_ctrl_opt(engine,M[i],K[i],spec)
            self.assertAlmostEqual(batch['u_opt'][i],scalar['u_ctrl_opt'],places=12)
            self.assertEqual(bool(batch['feasible'][i]),scalar['feasible'])
            self.assertEqual(bool(batch['monotonic'][i]),scalar['monotonic'])

if __name__=='__main__': unittest.main(verbosity=2)
