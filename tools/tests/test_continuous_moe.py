import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from src.domains.swing.continuous import StageResponse
from src.objectives.eig.continuous_eig import OnlineEIG,train,evaluate
from src.objectives.eig.continuous_moe import ContinuousMoEPolicy,ARCHITECTURE,improve_moe


class Observer:
    n_obs=1
    bounds=(.2,3.)
    def initial_state(self,n):return np.zeros((n,6))
    def propagate(self,theta,state,duration):
        q=np.asarray(duration)[:,None]
        return StageResponse(theta[:,:1]*q+.03*state[:,:1],state+q)


def engine(horizon):
    cfg=SimpleNamespace(swing={'M_lower_nodes':[.1]*3,'M_upper_nodes':[.3]*3,
        'K_lower_nodes':[.1]*3,'K_upper_nodes':[.3]*3})
    return OnlineEIG(cfg,Observer(),horizon=horizon,sigma=.02,contrasts=8)


class ContinuousMoETests(unittest.TestCase):
    def test_sparse_dispatch_and_exact_mixture_density_gradients(self):
        torch.manual_seed(19)
        policy=ContinuousMoEPolicy(3,1)
        with torch.no_grad():
            policy.router.weight.zero_()
            policy.router.bias.copy_(torch.tensor([2.,1.,-2.,-3.]))
        counts=[0]*4
        hooks=[expert.register_forward_hook(lambda m,x,y,k=k:counts.__setitem__(k,counts[k]+len(x[0])))
               for k,expert in enumerate(policy.experts)]
        dist,_,selected,_=policy.components(torch.randn(6,10))
        self.assertEqual(counts,[6,6,0,0])
        z=torch.linspace(-1,1,6)
        expected=torch.logsumexp(dist.component_distribution.log_prob(z[:,None])+
                                  dist.mixture_distribution.logits,dim=1)
        torch.testing.assert_close(dist.log_prob(z),expected)
        (-dist.log_prob(z).mean()).backward()
        self.assertGreater(float(policy.router.bias.grad.abs().sum()),0.)
        for expert in policy.experts[:2]:
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in expert.parameters()),0.)
        for expert in policy.experts[2:]:self.assertTrue(all(p.grad is None for p in expert.parameters()))
        for hook in hooks:hook.remove()

    def test_stable_ppo_rolls_back_rejected_step(self):
        torch.manual_seed(29)
        p=ContinuousMoEPolicy(3,1);p.training_mode='stable';p.kl_limit=-1.
        before={k:v.clone() for k,v in p.state_dict().items()}
        optimizer=torch.optim.Adam(p.parameters(),lr=.001)
        improve_moe(engine(3),p,optimizer,np.random.default_rng(29),8)
        self.assertEqual(p.last_diagnostics['accepted_passes'],0)
        self.assertEqual(p.last_diagnostics['rejected_passes'],1)
        for key,value in p.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)
        self.assertEqual(len(optimizer.state),0)

    def test_stable_ppo_records_finite_gradients(self):
        p=ContinuousMoEPolicy(3,1);p.training_mode='stable'
        optimizer=torch.optim.Adam(p.parameters(),lr=.0003)
        improve_moe(engine(3),p,optimizer,np.random.default_rng(31),32)
        self.assertGreater(p.last_diagnostics['accepted_passes'],0)
        for row in p.last_diagnostics['update_diagnostics']:
            self.assertTrue(np.isfinite(row['approximate_kl']))
            self.assertTrue(np.isfinite(row['actor_gradient_norm']))
            self.assertEqual(len(row['expert_gradient_norms']),4)

    def test_latent_recovery_matches_logged_density_and_order(self):
        for horizon in (3,4,5):
            e=engine(horizon);p=ContinuousMoEPolicy(horizon,1)
            with torch.no_grad():r=e.rollout(p,np.random.default_rng(23),8,stochastic=True)
            for stage,q in enumerate(r['unit_actions']):
                z=torch.logit(torch.as_tensor(q)).float()
                x=e.features(r['actions'][:stage],r['observations'][:stage],8,stage)
                dist,_=p(x,stage)
                torch.testing.assert_close(dist.log_prob(z),r['logprobs'][stage])
            self.assertTrue((np.diff(np.stack(r['actions']),axis=0)>=.01).all())

    def test_independent_training_reload_and_evaluation(self):
        for horizon in (3,4,5):
            e=engine(horizon)
            args=SimpleNamespace(T=horizon,seed=101,learning_rate=.001,updates=2,batch_size=8,
                validate_every=1,validation_systems=8,methods='moe_sboed',eval_seed=1001,
                eval_systems=2,planner_particles=8,noise_sigma=.02,N_obs=0,objective='eig')
            with tempfile.TemporaryDirectory() as directory:
                with patch('src.objectives.eig.continuous_eig.improve_objective',side_effect=AssertionError('baseline trainer')):
                    p,_=train(e,'moe_sboed',args,Path(directory))
                saved=torch.load(Path(directory)/'moe_sboed.pth',weights_only=False)
                self.assertEqual(saved['architecture'],ARCHITECTURE)
                self.assertIsNone(saved['teacher'])
                loaded=ContinuousMoEPolicy(horizon,1)
                loaded.load_state_dict(saved['state_dict'],strict=True)
                rows=evaluate(e,{'moe_sboed':loaded},args)
                self.assertEqual(len(rows),2)
                self.assertTrue(all(np.isfinite(r['terminal_spce_nats']) for r in rows))
                self.assertEqual(len(rows[0]['routing_trace']),horizon)


if __name__=='__main__':unittest.main()
