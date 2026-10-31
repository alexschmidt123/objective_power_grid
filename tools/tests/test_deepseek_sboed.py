import copy,math,inspect
from unittest.mock import patch
import torch,pytest
from src.research.deepseek_sboed import *
from src.research.benchmarks import LocationFinding
from src.research.engine import rollout

def benchmark():return LocationFinding({'location_finding':{'design_bounds':[-4,4]}})
def history(stage=12,batch=8):
 g=torch.Generator().manual_seed(101);f=torch.zeros(batch,121,dtype=torch.float64);f[:,0]=stage/30;slots=f[:,1:].reshape(batch,30,4);slots[:,:stage,:3]=torch.randn(batch,stage,3,generator=g,dtype=torch.float64);slots[:,:stage,3]=1;return f

def test_exact_paper_sparse_gates_and_inactive_experts_not_called():
 torch.manual_seed(101);m=DeepSeekPolicy().eval();m.router.weight.data.zero_();calls=[0]*7;shared=[]
 hooks=[e.register_forward_hook(lambda a,b,c,i=i:calls.__setitem__(i,calls[i]+1)) for i,e in enumerate(m.routed)];h=m.shared.register_forward_hook(lambda *a:shared.append(1))
 m(history(batch=1),12);d=m.last_dispatch;assert sum(calls)==3 and len(shared)==1 and d['indices'].shape==(1,3)
 torch.testing.assert_close(d['selected_weights'],torch.full((1,3),1/7,dtype=torch.float64));assert d['selected_weights'].sum()<1
 for hook in hooks+[h]:hook.remove()

def test_shared_selected_expert_router_gradients_and_full_objective():
 torch.manual_seed(101);m=DeepSeekPolicy();r=rollout(benchmark(),m,30,8,1017001,64);loss=-r['info'][:,-1].mean()+m.balance_loss();loss.backward()
 assert torch.isfinite(loss) and sum(p.grad.abs().sum() for p in m.shared.parameters())>0
 assert m.router.weight.grad is not None and torch.isfinite(m.router.weight.grad).all() and m.router.weight.grad.abs().sum()>0
 chosen=set(torch.cat([i.flatten() for _,i in m._gate_cache]).tolist())
 for e in chosen:assert sum(p.grad.abs().sum() for p in m.routed[e].parameters())>0
 assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
 assert all(a.abs().max()<=4 for a in r['actions'])

def test_dad_raw_encoder_empty_history_and_order_symmetry():
 torch.manual_seed(101);dad=LocationDAD(30,4.,3.,first_design_revision=1);m=DeepSeekPolicy();m.encoder.load_state_dict(dad.encoder.state_dict())
 for stage in [0,1,12,29]:
  f=history(stage);torch.testing.assert_close(m.encode(f,stage),dad.encode(f,stage));assert torch.isfinite(m.encode_context(f,stage)).all()
  if stage>0:assert float(m.input_norm(m.encode(f,stage)).square().mean().sqrt())<1.01
 f=history(12);p=f.clone();p[:,1:]=p[:,1:].reshape(8,30,4)[:,torch.cat([torch.arange(12).flip(0),torch.arange(12,30)])].reshape(8,-1);torch.testing.assert_close(m.encode(f,12),m.encode(p,12),atol=1e-12,rtol=1e-12)

def test_balance_formula_and_no_teacher_dependency():
 m=DeepSeekPolicy();scores=torch.tensor([[.4,.2,.1,.1,.08,.07,.05],[.1,.1,.3,.2,.15,.1,.05]],dtype=torch.float64,requires_grad=True);ids=scores.topk(3,-1).indices;m._gate_cache=[(scores,ids)]
 frequency=torch.nn.functional.one_hot(ids,7).sum(1).double().mean(0)*(7/3);expected=.001*(frequency*scores.mean(0)).sum();torch.testing.assert_close(m.balance_loss(),expected)
 source=inspect.getsource(__import__('src.research.deepseek_sboed',fromlist=['DeepSeekPolicy']));assert 'pimoe_ic' not in source and 'vsoed' not in source and 'NumericalOracle' not in source

def test_router_selected_score_gradient_matches_finite_difference():
 torch.manual_seed(101);m=DeepSeekPolicy().eval();f=history(batch=2);loss=m(f,12).square().sum();loss.backward();e=int(m.last_dispatch['indices'][0,0]);p=m.router.weight;analytic=float(p.grad[e,0]);original=float(p[e,0]);values=[]
 with torch.no_grad():
  for delta in [-1e-6,1e-6]:p[e,0]=original+delta;values.append(float(m(f,12).square().sum()))
  p[e,0]=original
 assert abs((values[1]-values[0])/2e-6-analytic)<1e-6

def test_small_controlled_learning_action_mapping_and_checkpoint(tmp_path):
 torch.manual_seed(101);m=DeepSeekPolicy();f=history(batch=16);target=f[:,1:3].tanh();opt=torch.optim.Adam(m.parameters(),lr=.005);first=float((m(f,12)-target).square().mean())
 for _ in range(80):opt.zero_grad();loss=(m(f,12)-target).square().mean()+m.balance_loss();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step()
 assert float(loss)<first*.35
 m.eval();path=tmp_path/'model.pt';torch.save(dict(method='DeepSeekMoE-sBOED',horizon=30,action_scale=4.,obs_scale=3.,variant='full',balance_factor=.001,state_dict=m.state_dict()),path);q=load_deepseek(path);torch.testing.assert_close(q(f,12),m(f,12));assert m.counts()['total_parameters']==8434
 a=benchmark().feasible(m(f,12),[],30);assert torch.isfinite(a).all() and (a.abs()<=4).all()

def test_dad_terminal_loss_gradient_matches_finite_difference():
 torch.manual_seed(101);p=LocationDAD(30,4.,3.,first_design_revision=1);b=benchmark();r=rollout(b,p,30,2,1017002,32);loss=-r['info'][:,-1].mean();loss.backward();parameter=p.emitter.bias;analytic=float(parameter.grad[0]);original=float(parameter[0]);values=[]
 with torch.no_grad():
  for delta in [-1e-5,1e-5]:parameter[0]=original+delta;v=rollout(b,p,30,2,1017002,32);values.append(float(-v['info'][:,-1].mean()))
 assert abs(analytic-(values[1]-values[0])/2e-5)<1e-3*max(1,abs(analytic))


def test_existing_common_evaluator_and_random_control_serialization():
 from src.research.journal_validation import CommonValidation
 from src.research.deepseek_sboed_experiment import RandomControl
 raw={'system':{'name':'location_finding'},'observation':{'noise_sigma':.5},'location_finding':{'design_bounds':[-4,4],'random_design_distribution':'truncated_standard_normal'}}
 v=CommonValidation(raw,30,1030,4,64,batch=2,chunk=32,device='cpu');m=DeepSeekPolicy();rollout(benchmark(),m,30,2,1017003,16)
 values,meta=v(m);assert len(values)==4 and all(math.isfinite(x) for x in values)
 values,meta=v(RandomControl(raw,701030));assert len(values)==4 and all(math.isfinite(x) for x in values)
