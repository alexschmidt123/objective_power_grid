import copy,hashlib,subprocess
from pathlib import Path
import pytest
from tools.tests.test_deepseek_sboed import history
import torch
from src.research.deepseek_sboed import DeepSeekPolicy
from src.research.fh_moe_v2_responsibilities import *

torch.set_num_threads(2)
def synthetic_bank():
 b=GradientResponsibilities()
 d=torch.zeros(400,16,dtype=torch.double);w=torch.zeros(400,7,dtype=torch.double)
 for i in range(400):
  e=(i//2)%2;d[i,e]=1;w[i,e]=1
 for j in range(3):b.update(10,d,w)
 return b,d,w

def test_reproducible_distinct_demands_and_normalized_soft_targets():
 b,d,w=synthetic_bank();r,meta=b.targets(10,d)
 assert meta['active']==2 and meta['distinct'] and r is not None
 assert torch.allclose(r.sum(-1),torch.ones(400,dtype=torch.double))
 assert r[:,2:].sum()==0
 assert r[0,0]>r[0,1] and r[2,1]>r[2,0]
 assert torch.equal(r,b.targets(10,d)[0])

def test_no_forced_seven_clusters_for_identical_demands():
 b=GradientResponsibilities();d=torch.ones(400,16,dtype=torch.double);w=torch.ones(400,7,dtype=torch.double)
 b.update(10,d,w);r,m=b.targets(10,d)
 assert r is None and not m['distinct']

def test_independent_stream_disagreement_rejected():
 b=GradientResponsibilities();d=torch.zeros(400,16,dtype=torch.double);d[::2,0]=1;d[1::2,1]=1
 b.update(10,d,torch.ones(400,7,dtype=torch.double))
 assert b.targets(10,d)[0] is None

def test_zero_and_low_support_references_rejected():
 b=GradientResponsibilities();b.update(10,torch.zeros(8,16),torch.ones(8,7))
 assert b.targets(10,torch.ones(8,16))[0] is None

def test_reference_roundtrip():
 b,d,w=synthetic_bank();c=GradientResponsibilities();c.load_state_dict(b.state_dict())
 assert torch.equal(b.targets(10,d)[0],c.targets(10,d)[0])

def test_history_hook_is_exact_and_removed():
 m=DeepSeekPolicy().double();f=history(stage=3,batch=4).requires_grad_()
 expected=m(f,3)
 with capture_terminal_context(m,3) as s:got=m(f,3)
 assert torch.equal(expected,got)
 assert s['z'].requires_grad and s['u'].requires_grad and s['logits'].requires_grad
 assert not m._forward_pre_hooks and not m.router._forward_hooks and not m.emitter._forward_pre_hooks

def test_inactive_bank_preserves_exact_full_horizon_gradient_and_none():
 m=DeepSeekPolicy().double();n=copy.deepcopy(m);f=history(stage=3,batch=4)
 with capture_terminal_context(m,3) as s:y=m(f,3);loss=y.square().mean()
 (n(f,3).square().mean()+n.balance_loss()).backward()
 record,d,r=information_gradients(m,loss,m.balance_loss(),s,GradientResponsibilities(),3)
 assert not record['auxiliary_applied']
 for a,b in zip(m.parameters(),n.parameters()):
  assert (a.grad is None)==(b.grad is None)
  if a.grad is not None:assert torch.allclose(a.grad,b.grad,rtol=1e-12,atol=1e-12)

def test_auxiliary_scalar_gradient_finite_difference_with_fixed_references():
 m=DeepSeekPolicy().double();f=history(stage=3,batch=4)
 with capture_terminal_context(m,3) as s:y=m(f,3)
 d=torch.randn(4,16,dtype=torch.double);r=torch.rand(4,7,dtype=torch.double);r=r/r.sum(-1,keepdim=True)
 le,lr=auxiliary_losses(m,s,d,r);p=m.routed[0].down.weight
 analytic=torch.autograd.grad(le+lr,p)[0][0,0].item();old=p[0,0].item();eps=1e-5
 values=[]
 for sign in [1,-1]:
  with torch.no_grad():p[0,0]=old+sign*eps
  with capture_terminal_context(m,3) as ss:m(f,3)
  a,b=auxiliary_losses(m,ss,d,r);values.append(float((a+b).detach()))
 with torch.no_grad():p[0,0]=old
 assert analytic==pytest.approx((values[0]-values[1])/(2*eps),rel=1e-5,abs=1e-9)

def test_auxiliary_live_history_derivative():
 m=DeepSeekPolicy().double();f=history(stage=3,batch=4).requires_grad_()
 with capture_terminal_context(m,3) as s:m(f,3)
 d=torch.randn(4,16,dtype=torch.double);r=torch.rand(4,7,dtype=torch.double);r=r/r.sum(-1,keepdim=True)
 le,lr=auxiliary_losses(m,s,d,r)
 assert torch.autograd.grad(le+lr,f)[0].norm()>0

def test_norm_caps_and_exact_combination():
 base=[torch.tensor([1.,2.]),torch.tensor([.5])]
 x=[torch.tensor([20.,-10.]),torch.tensor([30.])]
 alpha=cap_coefficient(base,x,[[0],[1]])
 for i in range(2):assert (alpha*x[i]).norm()<=.05*base[i].norm()+1e-7

def test_frozen_architecture_exact_source_and_counts():
 p=Path('src/research/deepseek_sboed.py')
 assert p.read_bytes()==subprocess.check_output(['git','show','13681d7:'+str(p)])
 assert DeepSeekPolicy().counts()['total_parameters']==8434

def test_deployment_has_no_training_dependency(monkeypatch):
 def forbidden(*a,**kw):raise AssertionError('Forbidden online optimization')
 monkeypatch.setattr(torch.autograd,'grad',forbidden)
 monkeypatch.setattr(torch.optim.Adam,'step',forbidden)
 monkeypatch.setattr(GradientResponsibilities,'targets',forbidden)
 m=DeepSeekPolicy().eval()
 with torch.inference_mode():y=m(history(stage=0,batch=2),0)
 assert torch.isfinite(y).all()

def test_combined_estimator_matches_exact_frozen_reference_objective():
 m=DeepSeekPolicy().double();n=copy.deepcopy(m);f=history(stage=3,batch=4)
 bank,d,w=synthetic_bank();bank.stages[3]=copy.deepcopy(bank.stages[10])
 with capture_terminal_context(n,3) as s:y=n(f,3);loss=y.square().mean()
 demand=torch.autograd.grad(loss,s['u'],retain_graph=True)[0].detach()*4
 r,_=bank.targets(3,demand);assert r is not None
 params=list(n.parameters());base=gradient_list(loss+n.balance_loss(),params)
 le,lr=auxiliary_losses(n,s,demand,r);ge=gradient_list(le,params);gr=gradient_list(lr,params)
 groups=group_indices(n);ae=cap_coefficient(base,ge,groups);ar=cap_coefficient(base,gr,groups)
 exact=torch.autograd.grad(loss+n.balance_loss()+ae*le+ar*lr,params,allow_unused=True)
 with capture_terminal_context(m,3) as ss:yy=m(f,3)
 record,_,_=information_gradients(m,yy.square().mean(),m.balance_loss(),ss,bank,3)
 assert record['auxiliary_applied']
 for p,g in zip(m.parameters(),exact):
  # Mathematically zero auxiliaries do not wake an otherwise inactive Adam block.
  actual=torch.zeros_like(p) if p.grad is None else p.grad
  expected=torch.zeros_like(p) if g is None else g
  torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-12)

def test_full_30_stage_conditional_surrogate_history_gradient_finite_difference():
 from src.research.benchmarks import LocationFinding
 from src.research.engine import rollout
 torch.manual_seed(413);m=DeepSeekPolicy();b=LocationFinding({'location_finding':{'design_bounds':[-4,4]}})
 with capture_terminal_context(m,10) as s:r=rollout(b,m,30,3,98216001,32)
 base=-r['info'][:,-1].mean();d=torch.autograd.grad(base,s['u'],retain_graph=True)[0].detach()*3
 target=torch.rand(3,7,dtype=torch.double);target/=target.sum(-1,keepdim=True)
 le,lr=auxiliary_losses(m,s,d,target);loss=base+.001*le+.0001*lr
 param=m.time_vector;g=torch.autograd.grad(loss,param)[0][0].item();old=param[0].item();values=[]
 for sign in [1,-1]:
  with torch.no_grad():param[0]=old+sign*1e-5
  with capture_terminal_context(m,10) as ss:rr=rollout(b,m,30,3,98216001,32)
  a,c=auxiliary_losses(m,ss,d,target);v=-rr['info'][:,-1].mean()+.001*a+.0001*c
  values.append(float(v.detach()))
 with torch.no_grad():param[0]=old
 assert g==pytest.approx((values[0]-values[1])/2e-5,rel=.01,abs=1e-4)

def test_responsibility_checkpoint_with_optimizer_roundtrip(tmp_path):
 b,d,w=synthetic_bank();m=DeepSeekPolicy();o=torch.optim.Adam(m.parameters(),lr=5e-5)
 p=tmp_path/'checkpoint.pt';torch.save({'state':m.state_dict(),'optimizer':o.state_dict(),'references':b.state_dict()},p)
 x=torch.load(p,weights_only=False);n=DeepSeekPolicy();n.load_state_dict(x['state']);oo=torch.optim.Adam(n.parameters());oo.load_state_dict(x['optimizer'])
 c=GradientResponsibilities();c.load_state_dict(x['references'])
 assert torch.equal(b.targets(10,d)[0],c.targets(10,d)[0])
 assert oo.param_groups[0]['lr']==5e-5

def test_reference_targets_do_not_retain_hidden_or_reward_graphs():
 b,d,w=synthetic_bank();d.requires_grad_();r,_=b.targets(10,d)
 assert not r.requires_grad and r.grad_fn is None
 assert set(b.state_dict())=={'decay','temperature','stages'}

@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires native GPU')
def test_CPU_reference_state_under_CUDA_default_device():
 with torch.device('cuda'):
  b=GradientResponsibilities();b.update(10,torch.randn(400,16),torch.ones(400,7))
  assert b.stages[10]['moment'].device.type=='cpu'
