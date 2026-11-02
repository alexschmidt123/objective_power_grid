import copy
from unittest.mock import patch
from contextlib import ExitStack
import torch
from src.research.moe_v2_context import ContextDecisionMoE,context_matched_models
from tools.tests.test_deepseek_sboed import history,benchmark
from src.research.engine import rollout

def test_independent_modules_counts_and_common_initialization():
 m=context_matched_models(202)
 assert m['v2'].counts()['total_parameters']==8658
 assert m['v2'].counts()['activated_parameters_per_context']==6994
 assert len({p.data_ptr() for p in m['v2'].parameters()})==len(list(m['v2'].parameters()))
 for arm in ('v2','shared','dense'):
  for name,p in m['v1'].encoder.state_dict().items():torch.testing.assert_close(p,m[arm].encoder.state_dict()[name],rtol=0,atol=0)

def test_exact_readout_algebra_and_sparse_top3():
 torch.manual_seed(202);m=ContextDecisionMoE().eval();f=history(20,batch=4);actual=m(f,20);z=m.encode_context(f,20);weights,ids=m.router(z).softmax(-1).topk(3,-1)
 expected=m.emitter(z+m.shared(z))
 for e in range(7):expected=expected+m.decoders[e](z+m.routed[e](z))*(weights*(ids==e)).sum(-1,keepdim=True)
 torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
 calls=[0]*7;hooks=[e.register_forward_hook(lambda a,b,c,i=i:calls.__setitem__(i,calls[i]+1)) for i,e in enumerate(m.routed)]
 m(history(20,batch=1),20);assert sum(calls)==3
 for h in hooks:h.remove()

def test_context_path_provides_direct_design_gradient_without_expert_feature():
 torch.manual_seed(202);m=ContextDecisionMoE();f=history(10,batch=3)
 for e in m.routed:
  torch.nn.init.zeros_(e.down.weight)
 m(f,10).square().sum().backward()
 assert sum(float(d.weight.grad.abs().sum()) for d in m.decoders if d.weight.grad is not None)>0

def test_full_horizon_pathwise_derivative_and_history():
 torch.manual_seed(202);m=ContextDecisionMoE();r=rollout(benchmark(),m,30,4,97402762,64);loss=-r['info'][:,-1].mean();loss.backward()
 assert m.router.weight.grad.abs().sum()>0
 for layer in (m.encoder,m.shared,m.emitter):assert sum(p.grad.abs().sum() for p in layer.parameters() if p.grad is not None)>0
 d=max((d for d in m.decoders if d.weight.grad is not None),key=lambda d:float(d.weight.grad.abs().max()));p=d.weight;index=int(p.grad.abs().argmax());i,j=divmod(index,16);analytic=float(p.grad[i,j]);old=float(p[i,j].detach());values=[]
 with torch.no_grad():
  for delta in (-1e-5,1e-5):
   p[i,j]=old+delta;rr=rollout(benchmark(),m,30,4,97402762,64);values.append(float(-rr['info'][:,-1].mean()))
  p[i,j]=old
 assert abs((values[1]-values[0])/2e-5-analytic)<.002*max(1,abs(analytic))
 f=history(20,batch=2).requires_grad_();m(f,20).square().sum().backward();assert f.grad[:,1:81].abs().sum()>0
 assert all(torch.isfinite(a).all() and a.abs().max()<=4 for a in r['actions'])

def test_checkpoint_reproduction_no_online_optimizer(tmp_path):
 torch.manual_seed(202);m=ContextDecisionMoE().eval();p=tmp_path/'candidate.pt';torch.save(m.state_dict(),p);n=ContextDecisionMoE().eval();n.load_state_dict(torch.load(p,weights_only=True))
 with ExitStack() as stack:
  for target in ('torch.optim.Adam.step','torch.autograd.grad','torch.Tensor.backward','src.research.engine.rollout','scipy.optimize.minimize'):stack.enter_context(patch(target,side_effect=AssertionError(target)))
  with torch.inference_mode():
   for stage in (0,1,10,20,29):torch.testing.assert_close(m(history(stage),stage),n(history(stage),stage),rtol=0,atol=0)
