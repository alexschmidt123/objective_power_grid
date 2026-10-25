
import copy,math
import numpy as np,pytest,torch,yaml
from pathlib import Path
from unittest.mock import patch
from src.research.deepseek_rc import RoutedCalibrationPolicy
from src.research.deepseek_sboed import DeepSeekPolicy
from src.research.engine import rollout
from src.research.benchmarks import make_benchmark

def cfg():
 return yaml.safe_load(Path('experiments/10252026/10252026_deepseek_rc_seed101/configs/seed101.yaml').read_text())
def fixture():
 torch.manual_seed(101);return RoutedCalibrationPolicy(initial_gain=2.)
def test_scalar_bound():
 m=fixture();assert sum(p.numel() for p in m.parameters())==8435
 assert abs(float(m.gain)-2)<1e-14
 for x in [-1000.,1000.]:
  m.raw_gain.data.fill_(x);assert .25<=m.gain<=4
def test_v1_equivalence_gain_one():
 torch.manual_seed(101);a=DeepSeekPolicy()
 torch.manual_seed(101);b=RoutedCalibrationPolicy(initial_gain=1.)
 for t in [0,1,15,29]:
  f=torch.randn(8,121,dtype=torch.float64)
  assert torch.allclose(a(f,t),b(f,t),rtol=1e-13,atol=1e-14)
def test_spce_formula_and_full_horizon_gradients():
 m=fixture();b=make_benchmark(cfg()['benchmark']);r=rollout(b,m,30,4,8123001,11)
 ll=r['log_likelihood'];expected=ll[:,0]-torch.logsumexp(ll,-1)+math.log(12)
 assert torch.allclose(r['info'][:,-1],expected)
 r['actions'][0].retain_grad();loss=-r['info'][:,-1].mean();loss.backward()
 assert r['actions'][0].grad.norm()>0
 assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
 assert all(any(p.grad is not None and p.grad.norm()>0 for p in g) for g in m.groups().values())
def test_gain_gradient_finite_difference():
 m=fixture();b=make_benchmark(cfg()['benchmark'])
 def value():
  return -rollout(b,m,30,3,8123002,7)['info'][:,-1].mean()
 v=value();v.backward();analytic=float(m.raw_gain.grad);x=float(m.raw_gain);eps=1e-4
 m.raw_gain.data.fill_(x+eps);plus=float(value());m.raw_gain.data.fill_(x-eps);minus=float(value())
 assert abs(analytic-(plus-minus)/(2*eps))<1e-6
def test_gate_mass_and_selected_gradient():
 m=fixture();f=torch.randn(48,121,dtype=torch.float64);m(f,8).square().sum().backward()
 d=m.last_dispatch;assert torch.allclose(d['scores'].sum(-1),torch.ones(48,dtype=torch.float64))
 assert torch.all(d['selected_weights'].sum(-1)<=1)
 for i in d['indices'].unique():
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.routed[i].parameters())
 assert m.shared.down.weight.grad.norm()>0 and m.router.weight.grad.norm()>0
def test_action_feasible_and_reproducible():
 m=fixture();b=make_benchmark(cfg()['benchmark'])
 a=rollout(b,m,30,4,8123003,5);c=rollout(b,m,30,4,8123003,5)
 assert torch.equal(torch.stack(a['actions']),torch.stack(c['actions']))
 assert torch.stack(a['actions']).abs().max()<=4
def test_expert_swaps_influence_design():
 m=fixture();f=torch.randn(64,121,dtype=torch.float64);z=m.encode_context(f,5)
 w,i=m.router(z).softmax(-1).topk(3,-1);j=i.clone();j[:,-1]=m.router(z).softmax(-1).topk(4,-1).indices[:,-1]
 a=m.emitter(z+m.shared(z)+m.routed_feature_sum(z,w,i))
 b=m.emitter(z+m.shared(z)+m.routed_feature_sum(z,w,j))
 assert (a-b).norm()>1e-8
def test_checkpoint_strict_roundtrip(tmp_path):
 m=fixture();f=torch.randn(3,121,dtype=torch.float64);a=m(f,9)
 torch.save(m.state_dict(),tmp_path/'state.pt');n=fixture();n.load_state_dict(torch.load(tmp_path/'state.pt',weights_only=True))
 assert torch.equal(a,n(f,9))
def test_deployment_no_simulator_no_optimization():
 m=fixture().eval();f=torch.randn(3,121,dtype=torch.float64)
 with patch('src.research.engine.rollout',side_effect=AssertionError('online rollout')),patch('torch.optim.Adam.step',side_effect=AssertionError('online update')):
  with torch.no_grad():a=m(f,15)
 assert torch.isfinite(a).all()
 assert list(__import__('inspect').signature(m.forward).parameters)==['features','stage']
def test_stable_clipped_optimizer():
 m=fixture();o=torch.optim.Adam(m.parameters(),lr=5e-5,betas=(.8,.998));b=make_benchmark(cfg()['benchmark'])
 for k in range(2):
  o.zero_grad();r=rollout(b,m,30,4,8123010+k,11);(-r['info'][:,-1].mean()+m.balance_loss()).backward()
  torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True)
  assert sum(float(p.grad.square().sum()) for p in m.parameters() if p.grad is not None)**.5<=5.000001
  o.step()
 assert all(torch.isfinite(p).all() for p in m.parameters())

def test_seed_audit_complete_integer_tokens():
 import re
 pattern=r'(?<![0-9.])96251007(?![0-9.])'
 assert not re.search(pattern,'0.7896251007808661')
 assert re.search(pattern,'{"seed":96251007}')
 assert re.search(pattern,'confirmation_seed: 96251007')
