"""Direct policy replacement: representability, task gradients and constraints."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from src.policies.direct_moe import DirectMoEPolicy
from src.objectives.eig.continuous_eig import DurationPolicy
from src.objectives.eig.continuous_pathwise import pathwise_rollout

class DirectMoETests(unittest.TestCase):
    def test_can_represent_dad_exactly(self):
        for T in (3,4,5):
            torch.manual_seed(101)
            dad=DurationPolicy(T,1); moe=DirectMoEPolicy(T,1)
            for expert in moe.experts: expert.load_state_dict(dad.network.state_dict())
            with torch.no_grad():moe.stage_bias.copy_(dad.stage_bias.expand(4,-1))
            features=torch.randn(7,1+3*T)
            for stage in range(T):
                torch.testing.assert_close(moe(features,stage)[0].mean,dad(features,stage)[0].mean)
    def test_experts_independent_and_router_gets_task_gradient(self):
        torch.manual_seed(101); p=DirectMoEPolicy(3,1)
        x=torch.randn(9,10)
        p(x,1)[0].mean.square().mean().backward()
        self.assertGreater(float(p.router[-1].weight.grad.norm()),0)
        ids=[{id(x) for x in e.parameters()} for e in p.experts]
        for i,e in enumerate(p.experts):
            self.assertGreater(sum(float(x.grad.norm()) for x in e.parameters()),0)
            for j in range(i):self.assertFalse(ids[i]&ids[j])
    def test_full_horizon_gradient_and_order(self):
        class Observer:
            bounds=(.2,3.)
            def propagate(self,theta,state,duration):
                d=np.asarray(duration)[:,None]
                return SimpleNamespace(observations=theta[:,:1]*d+.3*state,
                                       terminal_state=state+.2*d)
        for T in (3,4,5):
            torch.manual_seed(19); p=DirectMoEPolicy(T,1)
            engine=SimpleNamespace(horizon=T,n_obs=1,sigma=.1,min_separation=.01,observer=Observer())
            engine.sample=lambda rng,batch:(rng.normal(size=(batch,5,1)),np.zeros((batch,5,1)))
            params=list(p.parameters()); base=[q.detach().clone() for q in params]
            zs=[torch.randn_like(q) for q in params]
            norm=sum(float(z.square().sum()) for z in zs)**.5
            zs=[z/norm for z in zs]
            def score(eps):
                with torch.no_grad():
                    for q,b,z in zip(params,base,zs):q.copy_(b+eps*z)
                info,actions,_=pathwise_rollout(engine,p,np.random.default_rng(91),2)
                d=torch.stack(actions,1)
                self.assertTrue(bool(torch.all(d[:,1:]-d[:,:-1]>=.01)))
                self.assertTrue(bool(torch.all((d>=.2)&(d<=3))))
                return info[:,-1].mean()
            val=score(0)
            gs=torch.autograd.grad(val,params)
            grad=sum(float((g*z).sum()) for g,z in zip(gs,zs))
            ref=float((score(.002)-score(-.002)).detach())/.004
            self.assertAlmostEqual(grad,ref,delta=max(2e-5,.02*abs(ref)))
    def test_equal_sequence_start_and_learnable_specialization(self):
        torch.manual_seed(101)
        dad=DurationPolicy(3,1);moe=DirectMoEPolicy(3,1)
        sequence=torch.tensor([-2.,-.7,.3])
        with torch.no_grad():
            dad.network[-1].weight.zero_();dad.network[-1].bias.zero_()
            dad.stage_bias.copy_(sequence)
        moe.initialize_sequence(sequence)
        features=torch.randn(32,10)
        for stage in range(3):
            torch.testing.assert_close(dad(features,stage)[0].mean,moe(features,stage)[0].mean)
        optimizer=torch.optim.Adam(moe.parameters(),lr=.001)
        target=features[:,1].tanh()
        for _ in range(3):
            optimizer.zero_grad()
            (moe(features,1)[0].mean-target).square().mean().backward()
            optimizer.step()
        weights,proposals=moe.components(features,1)
        self.assertGreater(float(proposals.std(-1).mean()),1e-5)
        self.assertGreater(float(moe.router[-1].weight.grad.norm()),1e-8)
        self.assertGreater(float(weights.std(0).sum()),1e-8)

    def test_checkpoint_roundtrip(self):
        p=DirectMoEPolicy(4,1); q=DirectMoEPolicy(4,1)
        q.load_state_dict(p.state_dict(),strict=True)
        x=torch.randn(3,13)
        torch.testing.assert_close(p(x,2)[0].mean,q(x,2)[0].mean)
if __name__=='__main__':unittest.main()
