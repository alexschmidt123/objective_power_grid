import torch
from torch import nn
from src.research.sparse_moe import stabilize_gradients

def test_dominant_group_does_not_suppress_small_group():
    p=nn.Linear(1,1).double()
    p.weight.grad=torch.full_like(p.weight,10000.)
    p.bias.grad=torch.full_like(p.bias,.1)
    stabilize_gradients(p,dict(large=[p.weight],small=[p.bias]),'component',5.)
    assert float(p.weight.grad.norm())<=5.000001
    torch.testing.assert_close(p.bias.grad,torch.full_like(p.bias,.1))

def test_global_mode_preserves_original_clipping():
    p=nn.Linear(1,1).double()
    p.weight.grad=torch.full_like(p.weight,10000.)
    p.bias.grad=torch.full_like(p.bias,.1)
    stabilize_gradients(p,dict(large=[p.weight],small=[p.bias]),'global',5.)
    assert float(p.bias.grad.norm())<.0001
