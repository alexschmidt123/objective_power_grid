import unittest
import torch
from src.research.benchmarks import LocationFinding, Benchmark
from src.research.sequential_doptimal import SequentialDOptimal

class Linear(Benchmark):
    name='linear';param_dim=2;sigma=.5;calls=0
    def sample(self,shape,generator):return torch.randn(*shape,2,generator=generator,dtype=torch.float64)
    def feasible(self,x,actions,horizon):return x
    def propagate(self,theta,state,action):return (theta[...,0:1]*action+theta[...,1:2]),state

class Tests(unittest.TestCase):
    def test_analytic_gain_and_posterior(self):
        b=Linear();p=SequentialDOptimal(b,1,3,12,dict(myopic_map_restarts=2,myopic_map_max_nfev=40))
        a=[torch.tensor([[1.]],dtype=torch.float64)];y=[torch.tensor([[2.]],dtype=torch.float64)]
        z,cov,diag=p.fit(a,y,0)
        expected=torch.linalg.inv(torch.eye(2,dtype=torch.float64)+4*torch.ones(2,2,dtype=torch.float64))
        torch.testing.assert_close(cov,expected,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(z,torch.tensor([[8/9,8/9]],dtype=torch.float64),atol=1e-5,rtol=1e-5)
        candidates=torch.tensor([[0.],[1.],[2.]],dtype=torch.float64)
        score=p.scores(z,cov,candidates,a)
        jac=torch.cat((candidates,torch.ones_like(candidates)),1)
        exact=.5*torch.log1p((jac@cov*jac).sum(-1)/b.sigma**2)
        torch.testing.assert_close(score,exact)
    def test_episode_isolation_and_observation_adaptation(self):
        b=LocationFinding({'location_finding':{'design_bounds':[-4,4]}})
        settings=dict(myopic_map_restarts=2,myopic_map_max_nfev=12,myopic_prior_samples=4,myopic_design_candidates=12,myopic_design_rounds=2)
        p=SequentialDOptimal(b,2,3,99,settings);q=SequentialDOptimal(b,2,3,99,settings)
        actions=[torch.tensor([[.2,.4],[.2,.4]],dtype=torch.float64)]
        y=[torch.tensor([[1.],[1.]],dtype=torch.float64)];other=[torch.tensor([[1.],[4.]],dtype=torch.float64)]
        x=p.select(actions,y);xx=q.select(actions,other)
        torch.testing.assert_close(x[0],xx[0],rtol=0,atol=0)
        self.assertNotEqual(p.diagnostics[0]['episodes'][1]['posterior_mean_z'],q.diagnostics[0]['episodes'][1]['posterior_mean_z'])
        self.assertTrue(torch.isfinite(x).all())
        self.assertTrue(((b.feasible(x,actions,3)>=-4)&(b.feasible(x,actions,3)<=4)).all())
    def test_prior_stage_no_symmetric_zero_gradient(self):
        b=LocationFinding({'location_finding':{'design_bounds':[-4,4]}})
        p=SequentialDOptimal(b,1,30,9,dict(myopic_prior_samples=8,myopic_design_candidates=32))
        p.select([],[])
        self.assertGreater(p.diagnostics[0]['episodes'][0]['approximate_one_step_information'],0)

if __name__=='__main__':unittest.main()
