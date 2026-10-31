import unittest,torch,math
from tools.prototypes.sparse_moe import DSelectGate,SparseMoESBOED,MatchedDensePolicy
from src.research.benchmarks import LocationFinding
from src.research.engine import rollout
class Tests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1);torch.manual_seed(101)
 def test_gate_simplex_and_derivative(self):
  g=DSelectGate(3,2).double();x=torch.randn(4,3,dtype=torch.float64,requires_grad=True)
  w,_,_=g(x);torch.testing.assert_close(w.sum(-1),torch.ones(4,dtype=torch.float64))
  self.assertTrue(torch.autograd.gradcheck(lambda z:g(z)[0],(x,),eps=1e-6))
 def test_actual_sparse_dispatch_top1_top2(self):
  for k in [1,2]:
   p=SparseMoESBOED(2,1,3,8,k=k).eval();f=torch.zeros(1,13,dtype=torch.float64);calls=[0]*4
   with torch.no_grad():
    p.router.bits.weight.zero_();p.router.bits.bias.copy_(torch.tensor([-1.,-1.] if k==1 else [-1.,-1.,1.,1.]))
   hooks=[e.register_forward_hook(lambda m,a,o,i=i:calls.__setitem__(i,calls[i]+len(a[0]))) for i,e in enumerate(p.experts)]
   p(f,0)
   self.assertEqual(sum(calls),k);self.assertEqual(calls[1]+calls[2],0)
   for h in hooks:h.remove()
 def test_binary_relaxed_equals_hard_and_duplicate(self):
  g=DSelectGate(3,2).double();x=torch.randn(5,3,dtype=torch.float64)
  with torch.no_grad():g.bits.weight.zero_();g.bits.bias.fill_(1.)
  torch.testing.assert_close(g(x)[0],g(x,hard=True)[0],rtol=0,atol=0)
  self.assertTrue(bool(((g(x,hard=True)[0]>0).sum(-1)==1).all()))
 def test_spce_router_and_expert_gradient_T30(self):
  b=LocationFinding({});p=SparseMoESBOED(2,1,30,8,k=1).train()
  def objective():return rollout(b,p,30,3,101,8)['info'][:,-1].mean()
  value=objective();value.backward();g=float(p.router.bits.bias.grad[0]);self.assertGreater(abs(g),1e-9)
  self.assertTrue(any(x.grad is not None and x.grad.abs().sum()>0 for x in p.experts.parameters()))
  with torch.no_grad():p.router.bits.bias[0]+=1e-5
  hi=float(objective().detach())
  with torch.no_grad():p.router.bits.bias[0]-=2e-5
  lo=float(objective().detach())
  self.assertAlmostEqual(g,(hi-lo)/2e-5,delta=1e-5+abs(g)*1e-4)
 def test_no_auxiliary_and_parameter_match(self):
  p=SparseMoESBOED(2,1,30,64,dense=True);d=MatchedDensePolicy(2,1,30,64)
  n=lambda m:sum(x.numel() for x in m.parameters())
  self.assertLess(abs(n(p)-n(d))/n(p),.01)
  for name in ['critic_experts','posterior_experts','information_experts','feedback']:
   self.assertFalse(hasattr(p,name))
 def test_padding_and_empty_history(self):
  p=SparseMoESBOED(2,1,3,8);f=torch.zeros(2,13,dtype=torch.float64)
  first=p(f,0);f[:,1:]=100.;f[:,4::4]=0.
  torch.testing.assert_close(first,p(f,0),rtol=0,atol=0)
if __name__=='__main__':unittest.main()
