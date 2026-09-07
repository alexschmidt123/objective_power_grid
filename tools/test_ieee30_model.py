"""Topology, latent identity, independent Kron solve and CPU/CUDA checks."""
from pathlib import Path
import sys,copy,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.integrate import solve_ivp
from src.config import load_config_for_run
from src.domains.swing.design import build_simulator,Design
from src.domains.swing.simulator import IEEE30_BRANCHES,IEEE30_MACHINE_BUSES
from src.banks.power_grid import _sample_power_grid_prior
ROOT=Path(__file__).resolve().parents[1]
class Tests(unittest.TestCase):
    def setUp(self):
        self.cfg=load_config_for_run('configs/ieee30_mocu.yaml',ROOT)
        self.sim=build_simulator(self.cfg)
    def test_latent_and_physical_dimensions(self):
        M,K,_=_sample_power_grid_prior(self.cfg,16,np.random.default_rng(10))
        self.assertEqual(M.shape,(16,6));self.assertEqual(K.shape,(16,6))
        self.assertEqual(self.sim.B.shape,(6,6));self.assertEqual(self.sim.physical_input_map.shape,(30,6))
        self.assertEqual(len(IEEE30_BRANCHES),41)
        np.testing.assert_allclose(self.sim.physical_input_map.sum(1),1,atol=1e-12)
        np.testing.assert_allclose(self.sim.physical_input_map[np.array(IEEE30_MACHINE_BUSES)-1],np.eye(6),atol=1e-12)
        np.testing.assert_allclose(self.sim.B,self.sim.B.T,atol=1e-12)
        L=np.diag(self.sim.B.sum(1))-self.sim.B
        self.assertEqual(np.count_nonzero(np.linalg.eigvalsh(L)>1e-8),5)
    def test_reject_load_bus_latents_and_wrong_map(self):
        for field,value in [('N',30),('dynamic_machine_buses',[1,2,3,4,5,6]),('physical_bus_count',29)]:
            c=copy.deepcopy(self.cfg)
            if field=='N':c.raw['N']=value
            else:c.raw['swing_equation'][field]=value
            with self.assertRaises(ValueError):build_simulator(c)
    def test_full_network_matches_reduced_response(self):
        L=np.zeros((30,30))
        for i,j,x in IEEE30_BRANCHES:
            e=np.zeros(30);e[i]=1;e[j]=-1;L+=np.outer(e,e)/x
        inj=np.random.default_rng(12).normal(size=30);inj-=inj.mean()
        phase=np.r_[0,np.linalg.solve(L[1:,1:],inj[1:])]
        g=np.array(IEEE30_MACHINE_BUSES)-1
        reduced=np.diag(self.sim.B.sum(1))-self.sim.B
        np.testing.assert_allclose(reduced@phase[g],self.sim.physical_input_map.T@inj,atol=1e-11)
    def test_cuda_against_independent_adaptive_ode(self):
        from src.domains.swing.cuda import CudaTrajectoryEngine
        M,K,_=_sample_power_grid_prior(self.cfg,2,np.random.default_rng(15))
        designs=[Design(.05,0,.25),Design(.05,29,2.75)]
        engine=CudaTrajectoryEngine(self.sim,designs)
        gpu=engine.simulate_delta_f_batch(M,K,np.arange(2))
        ts=(np.arange(gpu.shape[1])+1)*self.sim.ode_dt
        for i,d in enumerate(designs):
            sol=solve_ivp(lambda t,y:self.sim._rhs(t,y,M[i],K[i],d),(0,float(ts[-1])),np.zeros(12),t_eval=ts,rtol=1e-10,atol=1e-12)
            self.assertTrue(sol.success)
            cpu=sol.y[6+self.sim.observation_bus]/(2*np.pi)
            np.testing.assert_allclose(gpu[i],cpu,atol=2e-7,rtol=2e-4)
if __name__=='__main__':unittest.main(verbosity=2)
