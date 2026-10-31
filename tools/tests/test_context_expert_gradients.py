
import copy,inspect
from unittest.mock import patch
import torch
from src.research.deepseek_sboed import DeepSeekPolicy
from src.research.context_expert_gradients import *
from src.research.engine import rollout
from tools.tests.test_deepseek_sboed import benchmark,history

def test_projection_protects_anchor_preserves_norm_and_full_gradient_alignment():
 a=torch.tensor([1.,0.],dtype=torch.float64);b=torch.tensor([-.5,1.],dtype=torch.float64)
 out,r=protect_gradient(a,b,torch.tensor([.8,.2]))
 assert r['adjusted'] and r['protected_other_dot']>=-1e-14
 torch.testing.assert_close(out.norm(),(a+b).norm())
 assert torch.dot(out,a+b)>0 and not torch.equal(out,a+b)
def test_fallback_when_protection_opposes_total_information_gradient():
 a=torch.tensor([1.,0.]);b=torch.tensor([-2.,0.])
 out,r=protect_gradient(a,b,torch.tensor([.8,.2]))
 assert r['fallback'] and torch.equal(out,a+b)
def test_no_conflict_or_no_router_preference_means_exact_original_gradient():
 a=torch.tensor([1.,.2]);b=torch.tensor([.2,1.])
 out,r=protect_gradient(a,b,torch.tensor([.8,.2]));assert torch.equal(out,a+b) and not r['adjusted']
 out,r=protect_gradient(a,-.5*a,torch.tensor([.5,.5]));assert torch.equal(out,.5*a)
def test_context_is_observed_learned_history_only_and_detached():
 m=DeepSeekPolicy()
 with capture_context(m,12) as snap:m(history(batch=16),12)
 c=partition_context(snap)
 assert c['valid'] and not c['affinity'].requires_grad and c['group'].shape==(16,)
 assert not m._forward_hooks
 assert set(snap)=={'z','scores','indices','stage'}
def test_full_horizon_cohort_gradients_sum_and_match_finite_differences():
 torch.manual_seed(101);m=DeepSeekPolicy()
 with capture_context(m,10) as snap:r=rollout(benchmark(),m,30,8,97664001,31)
 c=partition_context(snap);assert c['valid']
 balance=m.balance_loss();g=collect_context_gradients(m,r['info'][:,-1],balance,c)
 loss=-r['info'][:,-1].mean()+balance;loss.backward()
 report=apply_context_gradients(m,g,c)
 assert max(x['cohort_sum_max_error'] for x in report['experts'])<1e-10
 # Independent finite difference of full terminal objective for a routed parameter.
 e=next(i for i,x in enumerate(report['experts']) if x['norm']>0)
 p=m.routed[e].down.weight;idx=sum(len(list(x.parameters())) for x in m.routed[:e])+2
 analytic=float((g[0][idx]+g[1][idx])[0,0]);original=float(p[0,0].detach());eps=1e-5
 values=[]
 with torch.no_grad():
  for sign in [-1,1]:
   p[0,0]=original+sign*eps;x=rollout(benchmark(),m,30,8,97664001,31)
   values.append(float(-x['info'][:,-1].mean()+m.balance_loss()))
  p[0,0]=original
 assert abs((values[1]-values[0])/(2*eps)-analytic)<1e-5
def test_nonrouted_gradients_and_architecture_unchanged():
 torch.manual_seed(101);a=DeepSeekPolicy();b=copy.deepcopy(a)
 with capture_context(a,10) as snap:r=rollout(benchmark(),a,30,8,97664002,31)
 c=partition_context(snap);bal=a.balance_loss();cg=collect_context_gradients(a,r['info'][:,-1],bal,c)
 (-r['info'][:,-1].mean()+bal).backward()
 before={n:p.grad.clone() for n,p in a.named_parameters() if not n.startswith('routed.') and p.grad is not None}
 apply_context_gradients(a,cg,c)
 for n,p in a.named_parameters():
  if n in before:assert torch.equal(p.grad,before[n])
 r_base=rollout(benchmark(),b,30,8,97664002,31)
 (-r_base['info'][:,-1].mean()+b.balance_loss()).backward()
 for n,p in b.named_parameters():
  if n in before:torch.testing.assert_close(p.grad,before[n],atol=1e-12,rtol=1e-12)
 assert a.counts()==b.counts() and a.counts()['total_parameters']==8434
 assert a.state_dict().keys()==b.state_dict().keys()
def test_zero_online_specialization_or_design_search():
 m=DeepSeekPolicy().eval()
 with patch('src.research.context_expert_gradients.collect_context_gradients',side_effect=AssertionError('online learning')),patch('torch.optim.Adam.step',side_effect=AssertionError('online update')),patch('src.research.engine.rollout',side_effect=AssertionError('online simulation')):
  with torch.no_grad():x=m(history(batch=2),12)
 assert torch.isfinite(x).all()
def test_constant_history_context_has_no_forced_specialist_labels():
 m=DeepSeekPolicy()
 with capture_context(m,0) as snap:m(torch.zeros(8,121,dtype=torch.float64),0)
 c=partition_context(snap);assert not c['valid']

def test_independent_evaluator_and_common_prefix_interventions():
 import numpy as np
 from src.research.journal_validation import CommonValidation
 from src.research.v1_specialization_evaluation import common_prefix_scores,FrozenIntervention
 raw={'system':{'name':'location_finding'},'observation':{'noise_sigma':.5},'location_finding':{'design_bounds':[-4,4],'random_design_distribution':'truncated_standard_normal'}}
 torch.manual_seed(101);m=DeepSeekPolicy().eval()
 v=CommonValidation(raw,30,97669901,8,64,batch=4,chunk=32,device='cpu')
 expected,_=v(m);actual,meta=common_prefix_scores(m,v,10,return_features=True)
 np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
 alternative,alt=common_prefix_scores(m,v,10,(0,1,2))
 assert alt['prefix_SHA256']==meta['prefix_SHA256']
 assert np.isfinite(alternative).all()
 p=FrozenIntervention(m,'uniform_subset',seed=97669902)
 a,_=v(p);b,_=v(p);np.testing.assert_array_equal(a,b)
def test_new_training_and_evaluation_import_no_retired_model_code():
 import ast
 from pathlib import Path
 forbidden=('ir_moe','deepseek_rc','deepseek_sboed_v2','deepseek_sboed_v3','sa_moe','pimoe')
 for name in ['context_expert_gradients.py','v1_specialization_train.py','v1_specialization_experiment.py','v1_specialization_evaluation.py']:
  tree=ast.parse((Path('src/research')/name).read_text())
  for node in ast.walk(tree):
   if isinstance(node,ast.ImportFrom):assert not any(x in (node.module or '') for x in forbidden)
