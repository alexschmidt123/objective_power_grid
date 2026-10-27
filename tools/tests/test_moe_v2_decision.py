import copy,inspect
from contextlib import ExitStack
from unittest.mock import patch
import torch,pytest
from src.research.moe_v2_decision import DecisionMoE,matched_models
from src.research.deepseek_sboed import DeepSeekPolicy
from tools.tests.test_deepseek_sboed import history,benchmark
from src.research.engine import rollout

def test_counts_initialization_and_independent_experts():
 m=matched_models(201);assert m['v1'].counts()['total_parameters']==8434
 assert m['v2'].counts()['total_parameters']==8658
 assert m['v2'].counts()['activated_parameters_per_context']==6994
 assert len({p.data_ptr() for p in m['v2'].parameters()})==len(list(m['v2'].parameters()))
 for name in ['v2','dense','shared']:
  for k,v in m['v1'].encoder.state_dict().items():torch.testing.assert_close(v,m[name].encoder.state_dict()[k],rtol=0,atol=0)

def test_sparse_gates_algebra_and_finite_difference():
 torch.manual_seed(201);m=DecisionMoE().eval();f=history(12,batch=3);a=m(f,12);z=m.encode_context(f,12);p=m.router(z).softmax(-1);w,i=p.topk(3,-1);expected=m.emitter(z+m.shared(z))
 for e in range(7):expected=expected+m.decoders[e](m.routed[e](z))*(w*(i==e)).sum(-1,keepdim=True)
 torch.testing.assert_close(a,expected,rtol=1e-12,atol=1e-12)
 loss=a.square().sum();loss.backward();v=m.router.weight;analytic=v.grad[0,0].item();old=v[0,0].item();values=[]
 with torch.no_grad():
  for d in [-1e-6,1e-6]:v[0,0]=old+d;values.append(m(f,12).square().sum().item())
  v[0,0]=old
 assert abs((values[1]-values[0])/2e-6-analytic)<1e-7
 calls=[0]*7;hooks=[e.register_forward_hook(lambda a,b,c,i=i:calls.__setitem__(i,calls[i]+1)) for i,e in enumerate(m.routed)]
 m(history(12,batch=1),12);assert sum(calls)==3
 for h in hooks:h.remove()

def test_full_horizon_terminal_and_history_gradients():
 torch.manual_seed(201);m=DecisionMoE();r=rollout(benchmark(),m,30,8,97502601,64);loss=-r['info'][:,-1].mean()+m.balance_loss();loss.backward()
 for layer in [m.encoder,m.shared,m.router,m.emitter]:assert sum(p.grad.abs().sum() for p in layer.parameters() if p.grad is not None)>0
 for e,d in zip(m.routed,m.decoders):
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in list(e.parameters())+list(d.parameters()))
 assert any(d.weight.grad is not None and d.weight.grad.abs().sum()>0 for d in m.decoders)
 f=history(20,batch=2).requires_grad_();m(f,20).square().sum().backward();assert f.grad[:,1:81].abs().sum()>0
 assert all(a.abs().max()<=4 for a in r['actions'])

def test_checkpoint_no_online_optimizer_and_intervention_mass(tmp_path):
 m=DecisionMoE().eval();p=tmp_path/'v2.pt';torch.save(m.state_dict(),p);q=DecisionMoE().eval();q.load_state_dict(torch.load(p,weights_only=True));before=copy.deepcopy(m.state_dict())
 with ExitStack() as stack:
  for target in ['torch.optim.Adam.step','torch.autograd.grad','torch.Tensor.backward','src.research.engine.rollout','scipy.optimize.minimize']:
   stack.enter_context(patch(target,side_effect=AssertionError(target)))
  with torch.inference_mode():
   for stage in [0,1,10,20,29]:torch.testing.assert_close(m(history(stage),stage),q(history(stage),stage),rtol=0,atol=0)
 for k,v in before.items():torch.testing.assert_close(v,m.state_dict()[k],rtol=0,atol=0)
 with torch.no_grad():
  f=history(20,batch=8);m(f,20);old=m.last_dispatch;m.substitution=(20,3);m(f,20);new=m.last_dispatch
  torch.testing.assert_close(old['selected_weights'].sum(-1),new['selected_weights'].sum(-1));assert all(len(set(row))==3 for row in new['indices'].tolist())


def test_terminal_pathwise_gradient_matches_full_rollout_finite_difference():
 torch.manual_seed(201);m=DecisionMoE();r=rollout(benchmark(),m,30,4,97502602,64);loss=-r['info'][:,-1].mean();loss.backward()
 options=[(float(d.weight.grad.abs().max()),d.weight) for d in m.decoders if d.weight.grad is not None]
 magnitude,p=max(options,key=lambda x:x[0]);idx=int(p.grad.abs().argmax());i,j=idx//p.shape[1],idx%p.shape[1];analytic=float(p.grad[i,j]);old=float(p[i,j].detach());vals=[]
 with torch.no_grad():
  for delta in [-1e-5,1e-5]:
   p[i,j]=old+delta;r=rollout(benchmark(),m,30,4,97502602,64);vals.append(float(-r['info'][:,-1].mean()))
  p[i,j]=old
 assert abs((vals[1]-vals[0])/2e-5-analytic)<1e-3*max(1,abs(analytic))

@pytest.mark.skipif(not torch.cuda.is_available(),reason='local GPU lock required')
def test_cuda_complete_rollout_expert_and_router_gradients():
 with torch.device('cuda:0'):
  torch.manual_seed(201);m=DecisionMoE();r=rollout(benchmark(),m,30,8,97502603,64);loss=-r['info'][:,-1].mean()+m.balance_loss();loss.backward()
  assert m.router.weight.grad.is_cuda and torch.isfinite(m.router.weight.grad).all()
  assert any(d.weight.grad is not None and d.weight.grad.abs().sum()>0 for d in m.decoders)
  assert all(torch.isfinite(a).all() and a.abs().max()<=4 for a in r['actions'])
