import math
import unittest
import torch
from torch import nn
from resolution import pathway_knockout,nested_scores,signal


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.coordinator=nn.Sequential(nn.Linear(384,64),nn.Tanh(),nn.Linear(64,4))
        self.response_gate=nn.Linear(256,4,bias=False)
        self.fusion=nn.Sequential(nn.Linear(384,128),nn.ReLU(),nn.Linear(128,2))
        self.response_fusion=nn.Linear(256,128,bias=False)


class Tests(unittest.TestCase):
    def setUp(self):torch.manual_seed(12);torch.set_num_threads(2)

    def test_removed_outputs_cannot_change_other_gates_or_action(self):
        model=Model();h=torch.randn(4,128);slots=torch.randn(4,4,128)
        legacy=torch.randn(4,4,64);responses=slots[...,64:];a0=torch.randn(4,2).tanh()*4
        for k in range(4):
            action,g=pathway_knockout(model,h,slots,legacy,responses,a0,k)
            changed=slots.clone();leg=legacy.clone();resp=responses.clone()
            changed[:,k]+=100;leg[:,k]-=100;resp[:,k]+=300
            aa,gg=pathway_knockout(model,h,changed,leg,resp,a0,k)
            torch.testing.assert_close(action,aa,rtol=0,atol=0)
            torch.testing.assert_close(g,gg,rtol=0,atol=0)
            self.assertTrue(bool((g[:,k]==0).all()))
            torch.testing.assert_close(g.sum(-1),torch.ones(4))
            self.assertTrue(bool((action.abs()<=4).all()))

    def test_nested_estimator_matches_literal_at_every_bank_size(self):
        theta=torch.randn(3,4,dtype=torch.float64);actions=torch.randn(3,30,2)
        y=signal(theta.reshape(3,1,2,2),actions.double())+.5*torch.randn(3,30,dtype=torch.float64)
        bank=torch.randn(13,4,dtype=torch.float64)
        scores=nested_scores(actions,y,theta,bank,[3,7,13],chunk=4)
        for count in [3,7,13]:
            exact=[]
            for row in range(3):
                all_theta=torch.cat([theta[row:row+1],bank[:count]]).reshape(count+1,1,2,2)
                prediction=signal(all_theta,actions[row].double()[None])
                ll=(-2*(y[row,None]-prediction).square()).sum(-1)
                exact.append(ll[0]-torch.logsumexp(ll,0)+math.log(count+1))
            torch.testing.assert_close(scores[count],torch.stack(exact),rtol=1e-10,atol=1e-10)
            self.assertTrue(bool((scores[count]<=math.log(count+1)+1e-10).all()))
            torch.testing.assert_close(scores[count],nested_scores(actions,y,theta,bank[:count],[count],chunk=6)[count])

    def test_rejects_invalid_interventions_and_levels(self):
        with self.assertRaises(ValueError):pathway_knockout(None,None,None,None,None,None,4)
        with self.assertRaises(ValueError):nested_scores(torch.zeros(2,30,2),torch.zeros(2,30),torch.zeros(2,4),torch.zeros(3,4),[3,2])


if __name__=='__main__':unittest.main()
