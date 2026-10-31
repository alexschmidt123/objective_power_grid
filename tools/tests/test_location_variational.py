import unittest
from unittest.mock import patch
import torch
import pyro
from src.research.benchmarks import LocationFinding
from src.research.location_variational import load_upstream,LocationVariationalMyopic
from src.research.runner import DEFAULTS
from src.research.engine import rollout


class VariationalMyopicTests(unittest.TestCase):
    def benchmark(self):
        return LocationFinding({'location_finding':{'design_bounds':[-4,4]},
            'research':{'paper_protocol':{'corrected_methods':True}}})

    def settings(self):
        return DEFAULTS|dict(myopic_vi_num_steps=2,myopic_oed_num_steps=2,
            myopic_oed_inner_samples=8,myopic_oed_outer_samples=4)

    def test_upstream_model_and_bounded_design(self):
        ns=load_upstream((-4.,4.));b=self.benchmark();net=ns['BatchDesignBaseline'](1,(1,2))
        model=ns['HiddenObjects'](net,p=2,K=2)
        theta=torch.tensor([[-1.,.2],[.4,1.]],dtype=torch.float64)
        a=torch.tensor([[.3,-.2]],dtype=torch.float64)
        expected,_=b.forward(theta.flatten()[None],None,a)
        torch.testing.assert_close(model.forward_map(a,theta),expected[0])
        with torch.no_grad():net.designs[0].fill_(100)
        self.assertTrue(bool((net()<=4).all()))

    def test_history_only_independent_episodes_and_rng_preserved(self):
        with patch('torch.cuda.is_available',return_value=False):
            b=self.benchmark();cfg=self.settings();results=[]
            for other in [0.,20.]:
                p=LocationVariationalMyopic(b,2,3,8154,cfg)
                a=[torch.zeros(2,2,dtype=torch.float64)]
                y=[torch.tensor([[1.],[other]],dtype=torch.float64)]
                before=torch.random.get_rng_state().clone();dtype=torch.get_default_dtype()
                with torch.no_grad():results.append(p.select(a,y))
                torch.testing.assert_close(before,torch.random.get_rng_state(),rtol=0,atol=0)
                self.assertEqual(dtype,torch.get_default_dtype())
                self.assertEqual(len(pyro.get_param_store()),0)
            torch.testing.assert_close(results[0][0],results[1][0],rtol=0,atol=0)

    def test_original_first_design_and_no_smc_planning(self):
        with patch('torch.cuda.is_available',return_value=False):
            b=self.benchmark();p=LocationVariationalMyopic(b,1,30,8155,self.settings())
            a=b.feasible(p.select([],[]),[],30)
            torch.testing.assert_close(a,torch.full((1,2),.01,dtype=torch.float64))
            with patch('src.research.location_posterior.LocationSMC',side_effect=AssertionError('SMC planner called')):
                with torch.no_grad():r=rollout(b,None,3,1,8156,8,method='myopic',settings=self.settings())
            self.assertEqual(r['planning_diagnostics'][-1]['implementation'],'DAD_Variational_Myopic')
            self.assertTrue(bool(torch.isfinite(r['info']).all()))


if __name__=='__main__':unittest.main()
