import copy
import torch,pytest
from src.research.deepseek_sboed import DeepSeekPolicy
from src.research.deepseek_sboed_v2 import DeepSeekV2Policy
from src.research.engine import rollout
from src.research.benchmarks import LocationFinding

def fixture(variant='full'):
 torch.manual_seed(101);p=DeepSeekPolicy(variant=variant);q=DeepSeekV2Policy(variant=variant);q.load_state_dict(p.state_dict());g=torch.Generator().manual_seed(10319001);f=torch.zeros(16,121,dtype=torch.float64);f[:,0]=12/30;slots=f[:,1:].reshape(16,30,4);slots[:,:12,:3]=torch.randn(16,12,3,generator=g,dtype=torch.float64);slots[:,:12,3]=1;return p,q,f

@pytest.mark.parametrize('variant',['full','routed_only','shared_only','dense'])
def test_equivalent_actions_losses_and_all_gradients(variant):
 p,q,f=fixture(variant);a=p(f,12);b=q(f,12);torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12);la=a.square().mean()+p.balance_loss();lb=b.square().mean()+q.balance_loss();la.backward();lb.backward();q.mask_unused_gradients()
 for (name,x),(n,y) in zip(p.named_parameters(),q.named_parameters()):
  assert name==n and (x.grad is None)==(y.grad is None)
  if x.grad is not None:torch.testing.assert_close(x.grad,y.grad,atol=1e-11,rtol=1e-9)
 assert p.counts()==q.counts()

def test_inactive_adam_momentum_is_not_updated():
 p,q,f=fixture();op=torch.optim.Adam(p.parameters(),lr=5e-5,betas=(.8,.998));oq=torch.optim.Adam(q.parameters(),lr=5e-5,betas=(.8,.998))
 for x in p.parameters():x.grad=torch.ones_like(x)*.01
 op.step();q.load_state_dict(p.state_dict());oq.load_state_dict(copy.deepcopy(op.state_dict()))
 with torch.no_grad():p.router.weight.zero_();q.router.weight.zero_()
 op.zero_grad(set_to_none=True);oq.zero_grad(set_to_none=True);a=p(f,12);b=q(f,12);(a.square().mean()+p.balance_loss()).backward();(b.square().mean()+q.balance_loss()).backward();q.mask_unused_gradients()
 selected=set(p.last_dispatch['indices'].flatten().tolist())
 for e in range(7):
  for x,y in zip(p.routed[e].parameters(),q.routed[e].parameters()):assert (x.grad is None)==(y.grad is None)==(e not in selected)
 op.step();oq.step()
 for x,y in zip(p.parameters(),q.parameters()):torch.testing.assert_close(x,y,atol=1e-12,rtol=1e-10)

def test_full_rollout_gradient_and_sparse_inference_equivalence():
 p,q,_=fixture();b=LocationFinding({'location_finding':{'design_bounds':[-4,4]}});r=rollout(b,p,30,4,10319002,64);s=rollout(b,q,30,4,10319002,64);torch.testing.assert_close(r['info'],s['info'],atol=1e-9,rtol=1e-9)
 (-r['info'][:,-1].mean()+p.balance_loss()).backward();(-s['info'][:,-1].mean()+q.balance_loss()).backward();q.mask_unused_gradients()
 for x,y in zip(p.parameters(),q.parameters()):
  if x.grad is not None:torch.testing.assert_close(x.grad,y.grad,atol=1e-8,rtol=1e-7)
 p.eval();q.eval();f=r['features'][15].detach()[:1];counts=[0]*7;hooks=[e.register_forward_hook(lambda a,b,c,i=i:counts.__setitem__(i,counts[i]+1)) for i,e in enumerate(q.routed)]
 torch.testing.assert_close(p(f,15),q(f,15),atol=0,rtol=0);assert sum(counts)==3
 for h in hooks:h.remove()


def test_saved_v2_name_and_existing_evaluator_roundtrip(tmp_path):
 from src.research.deepseek_sboed_experiment import save
 from src.research.deepseek_sboed_v2 import load_v2
 from src.research.journal_validation import CommonValidation
 p,q,f=fixture();path=tmp_path/'v2.pt';save(q,path,1,dict(test=True));loaded=load_v2(path)
 q.eval();torch.testing.assert_close(q(f,12),loaded(f,12),atol=0,rtol=0)
 raw={'system':{'name':'location_finding'},'location_finding':{'design_bounds':[-4,4]}}
 evaluator=CommonValidation(raw,30,10310001,4,64,batch=2,chunk=32,device='cpu');a,_=evaluator(p);b,_=evaluator(loaded)
 import numpy as np
 np.testing.assert_array_equal(a,b)
