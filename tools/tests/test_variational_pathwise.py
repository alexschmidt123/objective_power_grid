import math
import tempfile
import unittest
from pathlib import Path
import torch
from src.research.variational_pathwise import FullCovariancePosterior,train_variational
from src.research.benchmarks import LocationFinding
from src.research.policies import make_policy
from src.research.engine import rollout
from src.research.runner import DEFAULTS

class Tests(unittest.TestCase):
    def b(self):return LocationFinding({'location_finding':{'design_bounds':[-4,4]},'research':{'paper_protocol':{'corrected_methods':True}}})
    def test_normalized_prior_and_label_symmetry(self):
        q=FullCovariancePosterior(13,4,16,1,True)
        x=torch.randn(7,13,dtype=torch.float64);z=torch.randn(7,4,dtype=torch.float64)
        torch.testing.assert_close(q.log_prob(x,z),-.5*z.square().sum(-1)-2*math.log(2*math.pi))
        with torch.no_grad():q.network[-1].weight.normal_(0,.05)
        torch.testing.assert_close(q.log_prob(x,z),q.log_prob(x,z.reshape(-1,2,2).flip(1).flatten(1)))
        w,_,s=q.parameters_for(x)
        torch.testing.assert_close(w.exp().sum(-1),torch.ones(7,dtype=torch.float64))
        self.assertTrue(torch.all(torch.diagonal(s,dim1=-2,dim2=-1)>0))
    def test_pathwise_gradient(self):
        torch.manual_seed(701);b=self.b();p=make_policy(b,3,'moe_sboed',16,{'moe_revision':2})
        q=FullCovariancePosterior(13,4,16,3,True)
        with torch.no_grad():q.network[-1].weight.normal_(0,.02)
        for v in q.parameters():v.requires_grad_(False)
        def score():
            r=rollout(b,p,3,4,702,0)
            return q.log_prob(r['final_features'],b.transform(r['truth']).detach()).mean()
        v=p.experts[0][-1].bias;score().backward();analytic=float(v.grad[0]);old=float(v[0]);eps=1e-5
        self.assertGreater(float(p.router[-1].weight.grad.norm()),1e-12)
        with torch.no_grad():
            v[0]=old+eps;plus=float(score());v[0]=old-eps;minus=float(score());v[0]=old
        self.assertAlmostEqual(analytic,(plus-minus)/(2*eps),delta=2e-5)
    def test_budget_and_policy_updates(self):
        cfg=DEFAULTS|dict(training_bound=2,updates=4,batch_size=3,hidden=8,moe_revision=2,
             variational_density_hidden=8,variational_mixtures=2,variational_density_updates=1,
             variational_warmup=1,validate_every=1)
        with tempfile.TemporaryDirectory() as folder:
            _,r=train_variational(self.b(),'moe_sboed',2,cfg,703,Path(folder),10)
        self.assertEqual(r['training_forward_trajectories'],10)
        self.assertEqual(r['contrast_training_trajectories'],0)
        self.assertEqual(r['policy_updates'],3)

if __name__=='__main__':unittest.main()
