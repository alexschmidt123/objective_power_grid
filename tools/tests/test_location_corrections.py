import tempfile
import unittest
from pathlib import Path
import torch
from src.research.benchmarks import LocationFinding
from src.research.policies import make_policy, history_features, parameter_count
from src.research.engine import generator, rollout
from src.research.location_posterior import LocationSMC
from src.research.runner import DEFAULTS
from src.research.training import train_spce


def benchmark():
    return LocationFinding(dict(location_finding={'design_bounds':[-4,4]},
        research={'paper_protocol':{'corrected_methods':True}}))


class LocationCorrectionTests(unittest.TestCase):
    def test_shared_history_permutation_invariance_and_parameter_control(self):
        b=benchmark(); rng=generator(700)
        a=[b.feasible(b.random_latent(2,rng),[],30) for _ in range(4)]
        y=[torch.randn(2,1,generator=rng,dtype=torch.float64) for _ in range(4)]
        f=history_features(b,a,y,2,30)
        perm=[3,1,0,2]
        g=history_features(b,[a[i] for i in perm],[y[i] for i in perm],2,30)
        for kind in ['dad','moe_sboed','uniform_moe','matched_dense']:
            p=make_policy(b,30,kind,64)
            torch.testing.assert_close(p(f,4),p(g,4),rtol=1e-12,atol=1e-12)
        m=make_policy(b,30,'moe_sboed',64);d=make_policy(b,30,'matched_dense',64)
        self.assertLess(abs(parameter_count(m)-parameter_count(d))/parameter_count(m),.02)

    def test_full_trajectory_gradient_against_finite_difference(self):
        b=benchmark();torch.manual_seed(701)
        p=make_policy(b,5,'moe_sboed',8)
        v=p.experts[0][-1].bias
        value=rollout(b,p,5,4,702,31)['info'][:,-1].mean()
        value.backward();analytic=float(v.grad[0])
        self.assertGreater(float(p.router.weight.grad.norm()),1e-10)
        old=float(v[0]);eps=1e-5
        with torch.no_grad():
            v[0]=old+eps;plus=float(rollout(b,p,5,4,702,31)['info'][:,-1].mean())
            v[0]=old-eps;minus=float(rollout(b,p,5,4,702,31)['info'][:,-1].mean())
            v[0]=old
        self.assertAlmostEqual(analytic,(plus-minus)/(2*eps),delta=2e-5)

    def test_smc_matches_analytic_gaussian_posterior(self):
        b=benchmark();b.sigma=.2
        # A symmetric linear observation gives an exact posterior for theta0+theta2.
        b.propagate=lambda theta,state,action:((theta[...,0]+theta[...,2])[...,None],state)
        smc=LocationSMC(b,1,8192,generator(703),moves=8)
        smc.update(torch.zeros(1,2,dtype=torch.float64),torch.tensor([[.7]],dtype=torch.float64))
        total=smc.theta[0,:,0]+smc.theta[0,:,2]
        self.assertAlmostEqual(float(total.mean()),2/(2+.04)*.7,delta=.02)
        self.assertAlmostEqual(float(total.var()),2*.04/(2+.04),delta=.012)
        self.assertGreater(smc.diagnostics[-1]['mh_acceptance'][0],.01)
        smc.update(torch.zeros(1,2,dtype=torch.float64),torch.tensor([[-.3]],dtype=torch.float64))
        total=smc.theta[0,:,0]+smc.theta[0,:,2]
        self.assertAlmostEqual(float(total.mean()),(.7-.3)/.04/(.5+2/.04),delta=.02)
        self.assertAlmostEqual(float(total.var()),1/(.5+2/.04),delta=.01)

    def test_fixed_is_nonadaptive_and_scheduler_budget_is_recorded(self):
        cfg=DEFAULTS|dict(updates=10,batch_size=4,contrasts=8,validate_every=1,
            learning_rate=.0001,spce_lr_decay=.98,spce_lr_schedule_every=1)
        with tempfile.TemporaryDirectory() as d:
            p,r=train_spce(benchmark(),'fixed',3,cfg,704,Path(d),final_checkpoint=True,trajectory_budget=45)
        self.assertEqual(r['training_forward_trajectories'],45)
        self.assertAlmostEqual(r['history'][-1]['learning_rate'],.0001*.98**2)
        f=torch.randn(5,13,dtype=torch.float64)
        torch.testing.assert_close(p(f,1),p(torch.zeros_like(f),1))


if __name__=='__main__': unittest.main()
