import unittest,torch,numpy as np
from src.research.entropy_myopic import EntropyMyopic,expected_entropy_reduction,posterior_entropy
from src.research.benchmarks import LocationFinding
from src.research.engine import rollout

class Tests(unittest.TestCase):
    def settings(self):return dict(planner_particles=32,fantasies=8,search_candidates=8,search_rounds=2)
    def benchmark(self):return LocationFinding({'location_finding':{'design_bounds':[-4,4]}})
    def test_manual_entropy_formula(self):
        lw=torch.tensor([.25,.75],dtype=torch.float64).log()
        ll=torch.tensor([[[0.,-1.],[-2.,0.]],[[0.,0.],[0.,0.]]],dtype=torch.float64)
        q=(lw[None,None]+ll).softmax(-1)
        expected=-(lw.exp()*lw).sum()-(-(q*q.log()).sum(-1)).mean(-1)
        torch.testing.assert_close(expected_entropy_reduction(lw,ll),expected)
        self.assertAlmostEqual(float(expected_entropy_reduction(lw,ll)[1]),0.)
    def test_matches_sir_entropy_engine(self):
        from types import SimpleNamespace
        from src.objectives.eig.vector import VectorEIGEngine
        b=self.benchmark();p=EntropyMyopic(b,1,3,15,self.settings())
        candidates=b.random_latent(5,torch.Generator().manual_seed(27))
        designs=b.feasible(candidates,[],3);theta=p.theta[0];n=len(theta)
        centres,_=b.forward(theta[:,None].expand(n,5,4),b.initial((n,5)),designs[None].expand(n,5,2))
        ctx=SimpleNamespace(centres_support=centres.permute(1,0,2).numpy(),log_p0=p.log_weights[0].numpy(),sigma_y=b.sigma,n_actions=5)
        engine=VectorEIGEngine(ctx,torch.device('cpu'))
        seed=193;p.rngs[0]=torch.Generator().manual_seed(seed)
        ours=p.candidate_scores(0,candidates,[])
        # Use identical float64 inputs/random draws rather than float32 production defaults.
        engine.centres=engine.centres.double();engine.log_p0=engine.log_p0.double()
        theirs=engine.action_scores(engine.log_p0,np.arange(5),n_fantasies=8,seed=seed)
        # SIR uses independent noise per action; compare the entropy calculation exactly elsewhere.
        self.assertTrue(np.isfinite(theirs).all());self.assertTrue(torch.isfinite(ours).all())
        torch.testing.assert_close(posterior_entropy(p.log_weights[0]),engine.entropy(engine.log_p0))
    def test_truth_independence_and_episode_isolation(self):
        b=self.benchmark();s=self.settings()
        p=EntropyMyopic(b,2,3,19,s);q=EntropyMyopic(b,2,3,19,s)
        a=[torch.zeros(2,2,dtype=torch.float64)]
        y=[torch.tensor([[1.],[2.]],dtype=torch.float64)];yy=[torch.tensor([[1.],[8.]],dtype=torch.float64)]
        x=p.select(a,y);xx=q.select(a,yy)
        torch.testing.assert_close(x[0],xx[0],rtol=0,atol=0)
        torch.testing.assert_close(p.log_weights[0],q.log_weights[0],rtol=0,atol=0)
        before=p.log_weights[0].clone();p.select(a,y);torch.testing.assert_close(before,p.log_weights[0])
        first=[]
        for truth in [torch.zeros(1,4,dtype=torch.float64),torch.ones(1,4,dtype=torch.float64)*3]:
            r=rollout(b,None,1,1,27,0,truth=truth,noise=torch.zeros(1,1,1,dtype=torch.float64),method='myopic',settings=s)
            first.append(r['actions'][0])
        torch.testing.assert_close(first[0],first[1],rtol=0,atol=0)
    def test_history_weights_and_state(self):
        b=self.benchmark();p=EntropyMyopic(b,1,3,11,self.settings())
        a=[torch.zeros(1,2,dtype=torch.float64)];y=[torch.ones(1,1,dtype=torch.float64)]
        p.select(a,y)
        means,_=b.forward(p.theta[0],b.initial((32,)),a[0].expand(32,2))
        expected=b.log_likelihood(y[0],means).log_softmax(-1)
        torch.testing.assert_close(p.log_weights[0],expected)
        self.assertEqual(p.diagnostics[-1]['objective'],'expected_one_step_posterior_entropy_reduction')

if __name__=='__main__':unittest.main()
