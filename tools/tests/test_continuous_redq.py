"""Regression checks for normalized-action REDQ entropy and replay semantics."""
import math
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from src.objectives.eig.continuous_redq import REDQPolicy, REDQTrainer

class REDQTests(unittest.TestCase):
    def trainer(self):
        torch.manual_seed(1)
        return REDQTrainer(REDQPolicy(3,1),SimpleNamespace(T=3,N_obs=0,seed=101))

    def test_temperature_decreases_above_target_and_increases_below(self):
        t=self.trainer()
        self.assertEqual(t.target_entropy,-1.)
        # A uniform normalized action has entropy log(2), already above -1.
        loss=t.temperature_loss(torch.full((32,),-math.log(2)))
        g=torch.autograd.grad(loss,t.log_alpha)[0]
        self.assertGreater(float(g),0.)  # gradient descent decreases temperature
        loss=t.temperature_loss(torch.full((32,),2.)) # entropy=-2, too low
        g=torch.autograd.grad(loss,t.log_alpha)[0]
        self.assertLess(float(g),0.)

    def test_log_density_matches_torch_tanh_transform(self):
        p=self.trainer().policy.double()
        x=torch.zeros((20,10),dtype=torch.float64)
        torch.manual_seed(8)
        unit,logp=p.sample_action(x)
        dist,_=p(x)
        transformed=torch.distributions.TransformedDistribution(
            dist,[torch.distributions.TanhTransform(cache_size=1)])
        torch.testing.assert_close(logp,transformed.log_prob(2*unit-1),atol=1e-10,rtol=1e-10)
        self.assertTrue(torch.isfinite(logp).all())

    def test_scalar_rocof_replay_rewards_telescope_and_terminate(self):
        t=self.trainer()
        class Engine:
            horizon=3
            def features(self,actions,observations,batch,stage):
                x=torch.zeros((batch,10));x[:,0]=stage/3
                return x
        result={'info':np.array([[.2,.8,1.1],[.1,.5,.9]]),
                'actions':[np.zeros(2)]*3,'observations':[np.zeros((2,1))]*3,
                'unit_actions':[np.full(2,.3)]*3}
        t.observe(Engine(),result,2)
        data=t.replay[:6].reshape(3,2,-1)
        np.testing.assert_allclose(data[:,:,11].sum(0),[1.1,.9],atol=1e-7)
        np.testing.assert_array_equal(data[:,:,12],[[0,0],[0,0],[1,1]])
        self.assertEqual(t.dimension,10)

if __name__=='__main__':unittest.main()
