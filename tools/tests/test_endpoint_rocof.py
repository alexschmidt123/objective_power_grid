"""Production endpoint sensor and full/conditional EIG gradient checks."""
import unittest
from pathlib import Path
import numpy as np
import torch
from src.config import load_config
from tools.tests.test_rocof_pathwise import RocofPathwiseTests

@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class EndpointRocofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.domains.swing.continuous_rocof import EndpointRocofObserver
        cls.cfg=load_config(Path(__file__).resolve().parents[2]/'configs/ieee9_eig.yaml')
        cls.kw=dict(duration_bounds=(.2,3.),injection_bus=1,amplitude=.05,window=3.5)
        cls.observer=EndpointRocofObserver(cls.cfg,**cls.kw)

    def test_signed_endpoint_and_state_carry_against_independent_cpu(self):
        from src.domains.swing.continuous import ContinuousSwingObserver
        cpu=ContinuousSwingObserver(self.cfg,n_obs=2,**self.kw);cpu.times=np.array([3.475,3.5])
        lo=np.r_[self.cfg.swing['M_lower_nodes'],self.cfg.swing['K_lower_nodes']]
        hi=np.r_[self.cfg.swing['M_upper_nodes'],self.cfg.swing['K_upper_nodes']]
        theta=np.vstack([(lo+hi)/2,np.random.default_rng(17).uniform(lo,hi,(2,6))])
        gs=self.observer.initial_state(3);cs=cpu.initial_state(3)
        ys=[]
        for d in (.3,.6,1.):
            g=self.observer.propagate(theta,gs,d);c=cpu.propagate(theta,cs,d)
            expected=np.diff(c.observations,axis=1)/.025
            np.testing.assert_allclose(g.observations,expected,rtol=1e-4,atol=1e-6)
            np.testing.assert_allclose(g.terminal_state,c.terminal_state,rtol=1e-4,atol=1e-6)
            gs,cs=g.terminal_state,c.terminal_state;ys.append(g.observations)
        self.assertTrue(np.any(np.asarray(ys)<0))
        reset=self.observer.propagate(theta,self.observer.initial_state(3),1.)
        self.assertGreater(float(np.max(abs(reset.observations-ys[-1]))),1e-4)
        self.assertEqual(self.observer.n_obs,1)
        self.assertEqual(self.observer.times.tolist(),[3.5])

    def test_full_t3_fixed_and_dad_gradients(self):
        for fixed in (True,False):
            with self.subTest(fixed=fixed):RocofPathwiseTests.check_direction(self,3,fixed)

    def test_conditional_t3_stepdad_gradients(self):
        for prefix in (1,2):
            with self.subTest(prefix=prefix):RocofPathwiseTests.check_direction(self,3,False,prefix)

if __name__=='__main__':unittest.main()
