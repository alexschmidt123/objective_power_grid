"""Direct MoE full-horizon endpoint-RoCoF gradient against finite differences."""
import unittest
from pathlib import Path
import numpy as np
import torch
from src.config import load_config
from src.policies.direct_moe import DirectMoEPolicy
from src.objectives.eig.continuous_eig import OnlineEIG
from src.objectives.eig.continuous_pathwise import pathwise_rollout

@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class DirectMoEPhysicsTests(unittest.TestCase):
    def test_endpoint_eig_gradient_t3_t4_t5(self):
        from src.domains.swing.continuous_rocof import EndpointRocofObserver
        cfg=load_config(Path(__file__).resolve().parents[2]/'configs/ieee9_eig.yaml')
        observer=EndpointRocofObserver(cfg,duration_bounds=(.2,3.),injection_bus=1,amplitude=.05,window=3.5)
        for T in (3,4,5):
            with self.subTest(T=T):
                torch.manual_seed(101)
                policy=DirectMoEPolicy(T,1)
                engine=OnlineEIG(cfg,observer,horizon=T,sigma=.005,contrasts=4)
                params=list(policy.parameters());bases=[p.detach().clone() for p in params]
                rng=np.random.default_rng(912)
                directions=[torch.tensor(rng.normal(size=p.shape),dtype=p.dtype) for p in params]
                norm=sum(float(z.square().sum()) for z in directions)**.5
                directions=[z/norm for z in directions]
                def score(delta):
                    with torch.no_grad():
                        for p,b,z in zip(params,bases,directions):p.copy_(b+delta*z)
                    return pathwise_rollout(engine,policy,np.random.default_rng(187),2)[0][:,-1].mean()
                initial=score(0.)
                grads=torch.autograd.grad(initial,params)
                gradient=sum(float((g*z).sum()) for g,z in zip(grads,directions))
                reference=float((score(.001)-score(-.001)).detach())/.002
                self.assertAlmostEqual(gradient,reference,delta=max(3e-5,.01*abs(reference)))
                score(0.)
                with torch.no_grad():
                    result=engine.rollout(policy,np.random.default_rng(187),2,stochastic=False)
                self.assertAlmostEqual(float(initial.detach()),float(result['info'][:,-1].mean()),places=9)
if __name__=='__main__':unittest.main()
