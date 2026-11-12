"""Small deterministic algebra regressions; actual checkpoint tests run in audit CLI."""
import pathlib
import sys
import unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
# Also runnable from the local authoring folder before installation under tools/tests.
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import torch
from torch import nn
from audit_hec_v2_code import head_parts, physical_head, normalized_gates, cka


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion=nn.Sequential(nn.Linear(384,128),nn.ReLU(),nn.Linear(128,2))
        self.response_fusion=nn.Linear(256,128,bias=False)


class AuditTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11012);torch.set_num_threads(2)
        self.model=Head();self.history=torch.randn(3,128)
        self.slots=torch.randn(3,4,128);self.a0=torch.randn(3,2).tanh()*4

    def test_complete_action_and_uniform_equivalence(self):
        g=torch.full((3,4),.25)
        shared,u=head_parts(self.model,self.history,self.slots)
        actual,_=physical_head(self.model,shared,u,g,self.a0)
        first=(g[:,:,None]*self.slots[...,:64]).flatten(1)
        response=(g[:,:,None]*self.slots[...,64:]).flatten(1)
        pre=self.model.fusion[0](torch.cat([self.history,first],-1))+self.model.response_fusion(response)
        delta=self.model.fusion[2](self.model.fusion[1](pre)).tanh()
        expected=self.a0+4*delta-self.a0*delta.abs()
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
        self.assertTrue(bool((actual.abs()<=4).all()))

    def test_single_expert_isolation_both_channels(self):
        for k in range(4):
            gates=torch.zeros(3,4);gates[:,k]=1
            s,u=head_parts(self.model,self.history,self.slots)
            before,_=physical_head(self.model,s,u,gates,self.a0)
            changed=self.slots.clone();changed[:,torch.arange(4)!=k]+=100
            s,u=head_parts(self.model,self.history,changed)
            after,_=physical_head(self.model,s,u,gates,self.a0)
            torch.testing.assert_close(before,after,rtol=0,atol=0)

    def test_router_derivative_matches_finite_difference(self):
        shared,u=head_parts(self.model,self.history,self.slots,torch.float64)
        shared,u=shared.detach(),u.detach()
        raw=torch.randn(3,4,dtype=torch.float64,requires_grad=True)
        action,_=physical_head(self.model,shared,u,normalized_gates(raw),self.a0)
        derivative=torch.autograd.grad(action.sum(),raw)[0]
        direction=torch.randn_like(raw);direction/=direction.norm()
        e=1e-6
        plus,_=physical_head(self.model,shared,u,normalized_gates(raw+e*direction),self.a0)
        minus,_=physical_head(self.model,shared,u,normalized_gates(raw-e*direction),self.a0)
        self.assertAlmostEqual(float(((plus-minus)/(2*e)).sum()),float((derivative*direction).sum()),places=7)
        self.assertGreater(float(derivative.norm()),0)

    def test_finite_expert_gradients_and_static_compensation(self):
        slots=self.slots.clone().requires_grad_()
        shared,u=head_parts(self.model,self.history,slots)
        fixed=torch.tensor([.1,.2,.3,.4]).expand(3,-1)
        action,_=physical_head(self.model,shared,u,fixed,self.a0)
        action.square().mean().backward()
        for k in range(4):
            self.assertTrue(bool(torch.isfinite(slots.grad[:,k]).all()))
            self.assertGreater(float(slots.grad[:,k].norm()),0)
        uniform=torch.full_like(fixed,.25)
        compensated,_=physical_head(self.model,shared,u*(4*fixed[:,:,None]),uniform,self.a0)
        torch.testing.assert_close(action,compensated,atol=1e-6,rtol=1e-5)

    def test_CKA_does_not_confuse_basis_rotation_with_difference(self):
        import numpy as np
        rng=np.random.default_rng(1);x=rng.normal(size=(64,8))
        q,_=np.linalg.qr(rng.normal(size=(8,8)))
        self.assertAlmostEqual(cka(x,x@q),1.,places=12)
        self.assertIsNone(cka(np.ones((8,3)),np.ones((8,4))))


if __name__=='__main__':unittest.main()
