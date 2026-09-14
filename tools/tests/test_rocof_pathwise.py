"""Full-objective max-RoCoF gradient regression, including fixed prefixes."""
import unittest
from pathlib import Path
import numpy as np
import torch
from src.config import load_config
from src.objectives.eig.continuous_eig import OnlineEIG,DurationPolicy,fixed_initializations
from src.objectives.eig.continuous_pathwise import pathwise_rollout,improve_pathwise


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class RocofPathwiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.domains.swing.continuous_rocof import MaxRocofObserver
        cls.cfg=load_config(Path(__file__).resolve().parents[2]/'configs/ieee9_eig.yaml')
        cls.observer=MaxRocofObserver(cls.cfg,duration_bounds=(.2,3.),injection_bus=1,amplitude=.05,window=3.)

    def check_direction(self,T,fixed,prefix=0):
        engine=OnlineEIG(self.cfg,self.observer,horizon=T,sigma=.005,contrasts=4)
        torch.manual_seed(101)
        policy=DurationPolicy(T,1,fixed=fixed)
        params=[p for n,p in policy.named_parameters() if n in {'sequence','stage_bias'} or n.startswith('network.')]
        bases=[p.detach().clone() for p in params]
        rng=np.random.default_rng(912)
        zs=[torch.tensor(rng.normal(size=p.shape),dtype=p.dtype) for p in params]
        norm=sum(float(z.square().sum()) for z in zs)**.5
        zs=[z/norm for z in zs]
        kwargs={}
        if prefix:
            theta,states=engine.sample(np.random.default_rng(199),2)
            actions=[];observations=[]
            for k in range(prefix):
                d=np.full(2,.3+.2*k)
                response=self.observer.propagate(theta.reshape(-1,6),states.reshape(-1,6),np.repeat(d,5))
                states=response.terminal_state.reshape(states.shape)
                actions.append(d);observations.append(response.observations.reshape(2,5,1)[:,0])
            kwargs=dict(initial=(theta,states),history=(actions,observations),stage_start=prefix)
        def value(delta):
            with torch.no_grad():
                for p,b,z in zip(params,bases,zs):p.copy_(b+delta*z)
            return pathwise_rollout(engine,policy,np.random.default_rng(187),2,**kwargs)[0][:,-1].mean()
        score=value(0.)
        gs=torch.autograd.grad(score,params)
        gradient=sum(float((g*z).sum()) for g,z in zip(gs,zs))
        eps=.001
        reference=float((value(eps)-value(-eps)).detach())/(2*eps)
        self.assertAlmostEqual(gradient,reference,delta=max(3e-5,.005*abs(reference)),
            msg=f'T={T}, fixed={fixed}, prefix={prefix}: gradient={gradient}, reference={reference}')
        for p,b in zip(params,bases):
            with torch.no_grad():p.copy_(b)
        forward=engine.rollout(policy,np.random.default_rng(187),2,stochastic=False,**kwargs)
        self.assertAlmostEqual(float(score.detach()),float(forward['info'][:,-1].mean()),places=10)

    def test_full_objective_gradients_t3_t4_t5(self):
        for T in (3,4,5):
            for fixed in (True,False):
                with self.subTest(T=T,fixed=fixed):self.check_direction(T,fixed)

    def test_conditional_refinement_gradients(self):
        for prefix in (1,3):
            with self.subTest(prefix=prefix):self.check_direction(5,False,prefix)

    def test_primal_active_sample_matches_maximum(self):
        engine=OnlineEIG(self.cfg,self.observer,horizon=5,sigma=.005,contrasts=4)
        theta,states=engine.sample(np.random.default_rng(20),2)
        theta=theta.reshape(-1,6);states=states.reshape(-1,6)
        for duration in (.2,.8,1.4,2.5,3.):
            forward=self.observer.propagate(theta,states,duration)
            active=self.observer.prepare_derivative(theta,states,duration)
            selected=self.observer.propagate_derivative(theta,states,duration,active)
            np.testing.assert_array_equal(forward.observations,selected.observations)
            np.testing.assert_array_equal(forward.terminal_state,selected.terminal_state)
            states=forward.terminal_state

    def test_fixed_interior_initializations_have_working_gradients(self):
        for T in (3,4,5):
            engine=OnlineEIG(self.cfg,self.observer,horizon=T,sigma=.005,contrasts=4)
            for label,logits in fixed_initializations(engine):
                with self.subTest(T=T,candidate=label):
                    policy=DurationPolicy(T,1,fixed=True)
                    with torch.no_grad():policy.sequence.copy_(torch.tensor(logits))
                    units=torch.sigmoid(policy.sequence)
                    self.assertGreater(float((units*(1-units)).min()),.01)
                    score,actions,_=pathwise_rollout(engine,policy,np.random.default_rng(187),2)
                    grad=torch.autograd.grad(score[:,-1].mean(),policy.sequence)[0]
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertGreater(float(grad.norm()),1e-7)
                    ds=np.array([a.detach().numpy()[0] for a in actions])
                    self.assertTrue(np.all(np.diff(ds)>.01))
                    self.assertGreater(ds[0],.2)
                    self.assertLess(ds[-1],3.)
                    before=policy.sequence.detach().clone()
                    optimizer=torch.optim.Adam(policy.parameters(),lr=.001)
                    improve_pathwise(engine,policy,optimizer,np.random.default_rng(187),2)
                    self.assertGreater(float((policy.sequence-before).abs().max()),1e-5)


if __name__=='__main__':unittest.main()
