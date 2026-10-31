"""Scientific invariants for the common EIG suite (no publication experiments)."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from scipy.integrate import solve_ivp
from src.research.benchmarks import LocationFinding,SIR
from src.research.engine import generator,rollout,replay_likelihood,posterior_metrics,logmeanexp
from src.research.policies import make_policy,parameter_count,history_features,Posterior


class ResearchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.location=LocationFinding({})

    def test_location_published_mean(self):
        theta=torch.tensor([[0.,0.,1.,0.]],dtype=torch.float64)
        mean,_=self.location.forward(theta,self.location.initial((1,)),torch.tensor([[0.,1.]],dtype=torch.float64))
        self.assertAlmostEqual(float(mean),np.log(.1+1/1.0001+1/2.0001),places=12)

    def test_full_horizon_gradient(self):
        torch.manual_seed(6)
        policy=make_policy(self.location,3,'moe_sboed',8)
        policy.select_expert(0)
        param=policy.experts[0][-1].bias
        def score():return rollout(self.location,policy,3,3,77,8)['info'][:,-1].mean()
        value=score();value.backward();analytic=float(param.grad[0])
        with torch.no_grad():
            original=float(param[0]);param[0]=original+1e-5;plus=float(score())
            param[0]=original-1e-5;minus=float(score());param[0]=original
        self.assertAlmostEqual(analytic,(plus-minus)/2e-5,places=5)
        self.assertGreater(abs(analytic),1e-8)

    def test_replay_and_bound_denominators(self):
        policy=make_policy(self.location,3,'dad',8)
        result=rollout(self.location,policy,3,4,44,7)
        lp,_,_=replay_likelihood(self.location,result['particles'],result['actions'],result['observations'])
        torch.testing.assert_close(lp,result['log_likelihood'][:,1:])
        lower=result['info'][:,-1]
        upper=result['log_likelihood'][:,0]-logmeanexp(lp)
        self.assertTrue(torch.all(lower<=np.log(8)+1e-10))
        # Expected upper estimate need not exceed the lower estimate per sample.
        self.assertTrue(torch.isfinite(upper).all())

    def test_history_keeps_stage_order_and_matches_capacity(self):
        actions=[torch.tensor([[1.,2.]],dtype=torch.float64),torch.tensor([[3.,4.]],dtype=torch.float64)]
        ys=[torch.tensor([[5.]],dtype=torch.float64),torch.tensor([[6.]],dtype=torch.float64)]
        a=history_features(self.location,actions,ys,1,3)
        b=history_features(self.location,actions[::-1],ys[::-1],1,3)
        self.assertFalse(torch.equal(a,b))
        reference=parameter_count(make_policy(self.location,3,'moe_sboed',16))
        matched=parameter_count(make_policy(self.location,3,'matched_dense',16))
        self.assertLess(abs(reference-matched)/reference,.04)

    def test_sir_independent_integrator_and_action_gradient(self):
        benchmark=SIR({'sir_ode':{'integration_steps_per_stage':256}})
        theta=torch.tensor([[.5,.1]],dtype=torch.float64)
        action=torch.tensor([[12.]],dtype=torch.float64,requires_grad=True)
        mean,state=benchmark.forward(theta,benchmark.initial((1,)),action)
        def rhs(t,y):
            q=.5*y[0]*y[1]/500;r=.1*y[1]
            return [-q,q-r,r]
        reference=solve_ivp(rhs,[0,12],[498,2,0],rtol=1e-11,atol=1e-12).y[:,-1]
        np.testing.assert_allclose(state.detach().numpy()[0,:3],reference,rtol=2e-7,atol=2e-6)
        mean.sum().backward()
        expected=.5*reference[0]*reference[1]/500-.1*reference[1]
        self.assertAlmostEqual(float(action.grad),expected,places=5)
        means,second=benchmark.forward(theta,state.detach(),torch.tensor([[20.]],dtype=torch.float64))
        ref2=solve_ivp(rhs,[0,20],[498,2,0],rtol=1e-11,atol=1e-12).y[:,-1]
        np.testing.assert_allclose(second.detach().numpy()[0,:3],ref2,rtol=3e-7,atol=3e-6)
        self.assertAlmostEqual(float(second[0,:3].sum()),500,places=8)

    def test_sir_all_horizons_feasible(self):
        benchmark=SIR({})
        for horizon in [5,10]:
            actions=[]
            for stage in range(horizon):
                action=benchmark.feasible(torch.zeros(2,1,dtype=torch.float64),actions,horizon)
                if actions:self.assertTrue(torch.all(action-actions[-1]>=benchmark.gap-1e-10))
                self.assertTrue(torch.all(action<=benchmark.end-(horizon-stage-1)*benchmark.gap))
                actions.append(action)

    def test_mse_truth_never_used_for_inference(self):
        policy=make_policy(self.location,2,'dad',8)
        with torch.no_grad():
            result=rollout(self.location,policy,2,2,8,4)
            settings=dict(posterior_particles=16,posterior_chains=2,posterior_steps=4,posterior_proposal_scale=.1)
            first=posterior_metrics(self.location,result,settings,5)
            result['truth']=result['truth']+100
            second=posterior_metrics(self.location,result,settings,5)
        self.assertEqual(first[0]['estimate'],second[0]['estimate'])
        self.assertNotEqual(first[0]['mse'],second[0]['mse'])

    def test_posterior_and_uniform_routing(self):
        posterior=Posterior(5,2,8,4)
        result=posterior.log_prob(torch.zeros(3,5,dtype=torch.float64),torch.ones(3,2,dtype=torch.float64))
        (-result.mean()).backward()
        self.assertTrue(torch.isfinite(result).all())
        policy=make_policy(self.location,2,'uniform_moe',8)
        features=torch.zeros(2,9,dtype=torch.float64)
        weights,proposals=policy.components(features)
        torch.testing.assert_close(weights,torch.full((2,4),.25,dtype=torch.float64))
        torch.testing.assert_close(policy(features,0),proposals[:,0])

    def test_vsoed_never_differentiates_simulator(self):
        from src.research.training import train_vsoed
        from src.research.runner import DEFAULTS
        class SampleOnlyLocation(LocationFinding):
            def propagate(self,*args):
                if torch.is_grad_enabled():
                    raise AssertionError('vsOED tried to differentiate simulator')
                return super().propagate(*args)
        settings=DEFAULTS|dict(updates=2,batch_size=4,validation_systems=4,validation_contrasts=8,
                              validate_every=1,vsoed_posterior_updates=1,vsoed_critic_updates=1)
        with tempfile.TemporaryDirectory() as tmp:
            actor,record=train_vsoed(SampleOnlyLocation({}),2,settings,7,Path(tmp))
            checkpoint=torch.load(Path(tmp)/'vsoed.pt',weights_only=False)
            self.assertEqual(record['gamma'],1.)
            self.assertEqual(len(record['history']),3)
            self.assertEqual(checkpoint['kind'],'vsoed')


if __name__=='__main__':unittest.main()
