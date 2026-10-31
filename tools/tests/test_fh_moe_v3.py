import copy,hashlib,subprocess
from pathlib import Path
from unittest.mock import patch
import pytest,torch
from src.research.deepseek_sboed import DeepSeekPolicy
from src.research.engine import rollout
from src.research.fh_moe_v3_context import *
from tools.tests.test_deepseek_sboed import benchmark,history

torch.set_num_threads(2)
def test_foundation_files_are_exact():
 for name in ['deepseek_sboed','deepseek_sboed_experiment','engine','dad_location_policy','benchmarks','journal_validation']:
  p=Path('src/research')/(name+'.py')
  assert p.read_bytes()==subprocess.check_output(['git','show','13681d7091189a4e68056e6a2864dc58f9d5700b:'+str(p)])

def test_capture_preserves_exact_native_terminal_gradients():
 torch.manual_seed(91);m=DeepSeekPolicy();n=copy.deepcopy(m)
 with capture_contexts(m,[1,10,20,29]) as s:r=rollout(benchmark(),m,30,4,991,32)
 a=-r['info'][:,-1].mean()+m.balance_loss();a.backward()
 q=rollout(benchmark(),n,30,4,991,32);b=-q['info'][:,-1].mean()+n.balance_loss();b.backward()
 assert torch.equal(a,b)
 for p,t in zip(m.parameters(),n.parameters()):
  assert (p.grad is None)==(t.grad is None)
  if p.grad is not None:assert torch.equal(p.grad,t.grad)
 assert all(s[t]['u'].requires_grad for t in s)
 assert not m._forward_pre_hooks and not m.emitter._forward_pre_hooks

def test_terminal_context_adjoint_includes_complete_later_history_chain():
 torch.manual_seed(92);m=DeepSeekPolicy();m.ablate_routed=True
 with capture_contexts(m,[10]) as s:r=rollout(benchmark(),m,30,2,992,32)
 loss=-r['info'][:,-1].mean();analytic=torch.autograd.grad(loss,s[10]['u'])[0][0,0]
 values=[]
 for delta in [-1e-5,1e-5]:
  count=[-1]
  def pre(module,args):count[0]=args[1]
  def shift(module,args):
   if count[0]==10:
    d=torch.zeros_like(args[0]);d[0,0]=delta;return (args[0]+d,)
  h=m.register_forward_pre_hook(pre);j=m.emitter.register_forward_pre_hook(shift)
  with torch.no_grad():v=rollout(benchmark(),m,30,2,992,32);values.append(-v['info'][:,-1].mean())
  h.remove();j.remove()
 assert torch.allclose(analytic,(values[1]-values[0])/2e-5,rtol=2e-3,atol=2e-6)

def test_reference_history_gradients_are_live():
 m=DeepSeekPolicy();m.ablate_routed=True;f=history(stage=10,batch=3).requires_grad_()
 x=history_covariates(m,f,10);g=torch.autograd.grad(x.square().mean(),f)[0]
 assert torch.isfinite(g).all() and g[:,1:41].abs().sum()>0

def test_reference_independent_of_arbitrary_router_and_expert_changes():
 m=DeepSeekPolicy();m.ablate_routed=True;n=copy.deepcopy(m)
 with torch.no_grad():
  n.router.weight.normal_(100,100)
  for e in n.routed:
   for p in e.parameters():p.normal_(100,100)
 f=history(stage=10,batch=4)
 assert torch.equal(m(f,10),n(f,10))
 assert torch.equal(history_covariates(m,f,10),history_covariates(n,f,10))

def test_reference_never_executes_routed_modules():
 m=DeepSeekPolicy();m.ablate_routed=True
 with patch.object(m.router,'forward',side_effect=AssertionError('router called')):
  m(history(),12)

def data(seed,n=1200,signal=True):
 rng=torch.Generator().manual_seed(seed);x=torch.randn(n,4,generator=rng,dtype=torch.double)
 g=torch.zeros(n,16,dtype=torch.double);v=x[:,0] if signal else torch.randn(n,generator=rng,dtype=torch.double)
 g[:,0]=v.sign();g[:,1]=.1;g+=.04*torch.randn(n,16,generator=rng,dtype=torch.double)
 return x,normalized(g)

def test_reproducible_context_discovery_accepts_real_observable_signal():
 r,b=discovery_metrics(data(1),data(2),data(3));assert r['PASS'] and b is not None
 r2,b2=discovery_metrics(data(1),data(2),data(3));assert r==r2

def test_unpredictable_gradient_labels_fail():
 r,_=discovery_metrics(data(4,signal=False),data(5,signal=False),data(6,signal=False));assert not r['PASS']

def test_no_seven_group_requirement_or_expert_index_input():
 x,g=data(7);f=fit_conditional(x,g);c=two_directions(predict(f,x));assert c.shape==(2,16)
 r=responsibilities(f,c,x);assert r.shape==(1200,2) and torch.allclose(r.sum(-1),torch.ones(1200,dtype=torch.double))
 assert torch.isfinite(r).all() and (r>=0).all()

def test_train_statistics_do_not_depend_on_test_data():
 x,g=data(9);a=fit_conditional(x,g);b=fit_conditional(x,g)
 for k in a:assert torch.equal(a[k],b[k])
 y,_=data(10);assert torch.equal(a['mean'],x.mean(0));assert not torch.equal(a['mean'],y.mean(0))

def test_checkpoint_context_and_original_policy_roundtrip(tmp_path):
 m=DeepSeekPolicy();f=history();x,g=data(8);bank=dict(fit=fit_conditional(x,g));p=tmp_path/'context.pt';torch.save(bank,p);q=torch.load(p,weights_only=False)
 for k in bank['fit']:assert torch.equal(q['fit'][k],bank['fit'][k])
 path=tmp_path/'model.pt';torch.save(m.state_dict(),path);n=DeepSeekPolicy();n.load_state_dict(torch.load(path,weights_only=True));assert torch.equal(m(f,12),n(f,12))

def test_deployment_frozen_has_no_online_optimization_or_gradient_extraction():
 m=DeepSeekPolicy().eval();f=history()
 with patch('torch.autograd.grad',side_effect=AssertionError('online grad')),patch.object(torch.optim.Adam,'step',side_effect=AssertionError('online update')),torch.inference_mode():
  a=m(f,12);assert torch.isfinite(a).all()

@pytest.mark.parametrize('stage',[1,10,29])
def test_observable_covariates_only_and_constraints(stage):
 m=DeepSeekPolicy();f=history(stage,batch=3);x=history_covariates(m,f,stage)
 assert x.shape==(3,41) and torch.isfinite(x).all()
 assert benchmark().feasible(m(f,stage),[],30).abs().max()<=4
