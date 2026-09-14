import unittest
import tempfile
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
from src.config import load_config
from src.objectives.eig.continuous_eig import feasible_duration,OnlineEIG,train,evaluate
from src.objectives.eig.continuous_pathwise import feasible_duration_torch

class IncreasingHorizonTests(unittest.TestCase):
    def test_map_and_gradient_all_horizons(self):
        for T in [1,3,4,5]:
            for value in [0.,.43,1.]:
                history=[];thistory=[]
                for stage in range(T):
                    unit=np.array([value]);tunit=torch.tensor(unit,requires_grad=True)
                    a=feasible_duration(unit,history,(.2,3.),.01,T)
                    ta=feasible_duration_torch(tunit,thistory,(.2,3.),.01,T)
                    np.testing.assert_allclose(a,ta.detach().numpy(),atol=1e-12)
                    self.assertLessEqual(a[0],3.-(T-stage-1)*.01+1e-12)
                    if history:self.assertGreaterEqual(a[0]-history[-1][0],.01)
                    history.append(a);thistory.append(ta)
                self.assertTrue(torch.isfinite(torch.autograd.grad(ta.sum(),tunit)[0]).all())

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_rocof_all_six_methods_t4_t5(self):
        from src.domains.swing.continuous_rocof import MaxRocofObserver
        cfg=load_config('configs/ieee9_eig.yaml')
        observer=MaxRocofObserver(cfg,window=3.,duration_bounds=(.2,3.),injection_bus=1,amplitude=.05)
        for T in [4,5]:
            engine=OnlineEIG(cfg,observer,horizon=T,sigma=.005,contrasts=4)
            args=SimpleNamespace(T=T,N_obs=0,seed=101,eval_seed=1001,learning_rate=.001,
                updates=1,batch_size=2,validate_every=1,validation_systems=2,
                methods='dad,rl_sboed,step_dad,myopic,fixed,random',eval_systems=1,
                planner_particles=4,noise_sigma=.005,contrasts=4,refinement_updates=1,
                search_rounds=1,search_candidates=2,fantasies=2)
            with tempfile.TemporaryDirectory() as directory:
                policies={m:train(engine,m,args,Path(directory))[0] for m in ['fixed','dad','rl_sboed']}
                rows=evaluate(engine,policies,args)
            self.assertEqual(len(rows),6)
            for row in rows:
                self.assertTrue(np.isfinite(row['terminal_spce_nats']))
                d=np.array(row['duration_sequence_s'])
                self.assertEqual(len(d),T)
                self.assertTrue(np.all(np.diff(d)>=.01))
                self.assertLessEqual(d[-1],3.)
                self.assertEqual(np.asarray(row['observations_rocof_hz_s']).shape,(T,1))

if __name__=='__main__':unittest.main()
