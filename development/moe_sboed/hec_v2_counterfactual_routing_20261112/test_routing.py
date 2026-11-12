"""Numerical and leakage tests; synthetic fixtures do not establish scientific gains."""
import math
import unittest
import torch
from torch import nn
from routing import (gate_bank, fuse_options, routing_context, fit_router,
                     signal, terminal_scores, holm, CounterfactualRouter)


class FakeV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion = nn.Sequential(nn.Linear(384,128),nn.ReLU(),nn.Linear(128,2))
        self.response_fusion = nn.Linear(256,128,bias=False)
        self.base = nn.Sequential(nn.Linear(117,2),nn.Tanh())


class RoutingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        torch.set_num_threads(2)

    def test_options_and_manual_fusion(self):
        v2=FakeV2()
        x=torch.randn(3,117)
        info=dict(slots=torch.randn(3,4,128),history=torch.randn(3,128),
                  gates=torch.randn(3,4).softmax(-1))
        gates=gate_bank(info['gates'])
        self.assertEqual(tuple(gates.shape),(3,8,4))
        torch.testing.assert_close(gates.sum(-1),torch.ones(3,8))
        torch.testing.assert_close(gates[:,0],info['gates'])
        torch.testing.assert_close(gates[:,1],torch.full((3,4),.25))
        action=fuse_options(v2,x,info,gates)
        for k in range(8):
            g=gates[:,k,:,None]
            pre=v2.fusion[0](torch.cat([info['history'],(g*info['slots'][...,:64]).flatten(1)],-1))
            pre+=v2.response_fusion((g*info['slots'][...,64:]).flatten(1))
            delta=v2.fusion[2](v2.fusion[1](pre)).tanh()
            a0=v2.base(x)
            torch.testing.assert_close(action[:,k],a0+4*delta-a0*delta.abs())
        context=routing_context(v2,x,info,action)
        self.assertEqual(tuple(context.shape),(3,182))
        self.assertTrue(bool((action.abs()<=4).all()))

    def test_literal_full_trajectory_estimator(self):
        truth=torch.randn(2,4,dtype=torch.float64)
        actions=torch.randn(2,30,2)
        noise=torch.randn(2,30,dtype=torch.float64)
        y=signal(truth.reshape(2,1,2,2),actions.double())+.5*noise
        negatives=torch.randn(7,4,dtype=torch.float64)
        score=terminal_scores(actions,y,truth,negatives,chunk=3)
        literal=[]
        for i in range(2):
            theta=torch.cat([truth[i:i+1],negatives]).reshape(8,2,2)
            loglike=torch.zeros(8,dtype=torch.float64)
            for t in range(30):
                loglike+=-2*(y[i,t]-signal(theta,actions[i,t].double())).square()
            literal.append(loglike[0]-torch.logsumexp(loglike,0)+math.log(8))
        torch.testing.assert_close(score,torch.stack(literal),atol=1e-10,rtol=1e-10)
        torch.testing.assert_close(score,terminal_scores(actions,y,truth,negatives,chunk=7))
        self.assertTrue(bool((score<=math.log(8)+1e-10).all()))
        with self.assertRaises(ValueError):
            terminal_scores(actions[:,:29],y[:,:29],truth,negatives)

    def test_learns_history_dependent_choices_and_train_only_scaling(self):
        x=torch.randn(512,182)
        returns=torch.zeros(512,8)
        returns[:,0]=x[:,0]
        returns[:,1]=-x[:,0]
        returns[:,2:]=-2
        trained=fit_router(x[:384],returns[:384],ridge_penalty=10.)
        expected=torch.where(x[384:,0]>0,0,1)
        accuracy=float((trained.select(x[384:])==expected).float().mean())
        self.assertGreater(accuracy,.90)
        torch.testing.assert_close(trained.mean,x[:384].mean(0))
        # Stage control cannot see changes in non-stage history features.
        stage=CounterfactualRouter(stage_only=True)
        altered=x[:8].clone();altered[:,:-30]+=100
        torch.testing.assert_close(stage(x[:8]),stage(altered))

    def test_holm_preserves_family(self):
        tests={'a':{'p_two_sided':.01},'b':{'p_two_sided':.04},'c':{'p_two_sided':.5}}
        holm(tests)
        self.assertEqual([tests[k]['Holm_p'] for k in ['a','b','c']],[.03,.08,.5])


if __name__=='__main__':
    unittest.main()
