import copy,io,inspect,math,shutil
from pathlib import Path
import numpy as np,torch,pytest,yaml
from src.research.moe_sboed_v4 import RoleMoE,dense_control
from src.research.moe_v4_partition import PredictiveAdapter,fit_partition
from src.research.moe_v4_information import densities,targets,paired_training_draws,role_rollout
from src.research.engine import rollout
from src.research.deepseek_sboed import DeepSeekPolicy
ROOT=Path(__file__).resolve().parents[2];torch.set_num_threads(2)
def raw():return yaml.safe_load((ROOT/'configs/deepseek_sboed_location.yaml').read_text())['benchmark']
def history(stage=10,B=8,h=30,a=2,o=1):
 f=torch.zeros(B,1+h*(a+o+1),dtype=torch.float64);f[:,0]=stage/h;slots=f[:,1:].reshape(B,h,a+o+1);slots[:,:stage,:a+o]=torch.randn(B,stage,a+o,dtype=torch.float64);slots[:,:stage,-1]=1;return f
@pytest.fixture(scope='module')
def partition():
 a=PredictiveAdapter(raw());p,r=fit_partition(a,512,512);return a,p,r

def test_exact_four_term_density_identity_and_negative_realizations():
 g=torch.Generator().manual_seed(42);ell=torch.randn(16,65,5,generator=g,dtype=torch.float64)*5;labels=torch.randint(0,4,(16,65),generator=g);j=[densities(ell[:,:,t],labels) for t in range(5)];total=torch.cat((torch.zeros(16,1),torch.stack([x[0] for x in j],1)),1);coarse=torch.cat((torch.zeros(16,1),torch.stack([x[1] for x in j],1)),1);r=targets(total,coarse);torch.testing.assert_close(r.sum(-1),total[:,-1:]-total[:,:-1],atol=1e-12,rtol=1e-12);assert (r<0).any();assert torch.equal(r[:,-1,2:],torch.zeros_like(r[:,-1,2:]));assert not r.requires_grad

def test_log_domain_extremes_and_shift_invariance():
 ell=torch.tensor([[-1000000.,-1000100.,-999999.,-2000000.]],dtype=torch.float64);c=torch.tensor([[0,0,1,2]]);a=densities(ell,c);b=densities(ell+12345,c)
 for x,y in zip(a,b):assert torch.isfinite(x).all();torch.testing.assert_close(x,y,atol=1e-9,rtol=1e-10)

def test_chain_rule_against_exact_adaptive_discrete_model():
 p1=np.array([.1,.3,.7,.9]);p2={0:np.array([.2,.8,.3,.6]),1:np.array([.8,.4,.6,.1])};group=np.array([0,0,1,1]);expect=np.zeros(4);total1=coarse1=total2=coarse2=0.
 for theta in range(4):
  order=[theta]+[j for j in range(4) if j!=theta]
  for y1 in range(2):
   a=p1 if y1 else 1-p1
   for y2 in range(2):
    b=p2[y1] if y2 else 1-p2[y1];joint=a*b;weight=.25*joint[theta];ll1=torch.tensor(np.log(a[order])[None]);ll2=torch.tensor(np.log(joint[order])[None]);labels=torch.tensor(group[order][None]);t1,c1,_=densities(ll1,labels);t2,c2,_=densities(ll2,labels);rt=targets(torch.stack((torch.zeros_like(t1),t1,t2),1),torch.stack((torch.zeros_like(c1),c1,c2),1));expect+=weight*rt[0,0].numpy();total2+=weight*np.log(joint[theta]/joint.mean());coarse2+=weight*np.log(joint[group==group[theta]].mean()/joint.mean())
   total1+=.25*a[theta]*np.log(a[theta]/a.mean());coarse1+=.25*a[theta]*np.log(a[group==group[theta]].mean()/a.mean())
 exact=np.array([coarse1,total1-coarse1,coarse2-coarse1,(total2-coarse2)-(total1-coarse1)]);np.testing.assert_allclose(expect,exact,atol=1e-12);assert (exact>=-1e-12).all();assert abs(expect.sum()-total2)<1e-12

def test_predictive_partition_nonzero_variance_and_source_symmetry(partition):
 a,p,r=partition;assert r['valid'] and min(r['frequency'])>=.05 and r['between_fraction']>=.05;theta=a.b.sample((128,),torch.Generator().manual_seed(17));torch.testing.assert_close(a.signatures(theta),a.signatures(theta[:,[2,3,0,1]]),atol=1e-12,rtol=1e-12);assert torch.equal(p.classify(a,theta),p.classify(a,theta[:,[2,3,0,1]]))

def test_history_encoder_matches_v1_foundation():
 torch.manual_seed(17);v1=DeepSeekPolicy();torch.manual_seed(17);v4=RoleMoE();v4.history.encoder.load_state_dict(v1.encoder.state_dict())
 for stage in [0,1,10,29]:
  f=history(stage);torch.testing.assert_close(v4.history(f,stage),v1.encode_context(f,stage))

def test_full_rollout_and_gradients_equal_unmodified_dad_engine(partition):
 a,part,_=partition;torch.manual_seed(18);p=RoleMoE();q=copy.deepcopy(p);truth,bank,noise=paired_training_draws(a.b,30,4,64,991)
 r=role_rollout(a.b,p,30,truth,bank,noise,part,a);s=rollout(a.b,q,30,4,991,64,truth=truth,noise=noise,prior_particles=bank[None].expand(4,-1,-1));torch.testing.assert_close(r['info'],s['info'],atol=1e-12,rtol=1e-12)
 g=torch.autograd.grad(r['info'][:,-1].mean(),r['actions'][0],retain_graph=True)[0];assert g.abs().sum()>0
 (-r['info'][:,-1].mean()).backward();(-s['info'][:,-1].mean()).backward()
 for x,y in zip(p.parameters(),q.parameters()):
  if x.grad is None:assert y.grad is None
  else:torch.testing.assert_close(x.grad,y.grad,atol=1e-10,rtol=1e-10)
 torch.testing.assert_close(r['role_targets'].sum(-1),r['total_density'][:,-1:]-r['total_density'][:,:-1],atol=1e-12,rtol=1e-12)

def test_each_role_loss_updates_only_its_expert_and_head():
 torch.manual_seed(13);p=RoleMoE();f=[history(t,B=4) for t in range(5)];target=torch.randn(4,5,4,dtype=torch.float64,requires_grad=True)
 for e in range(4):
  p.zero_grad();losses,_=p.role_losses(f,target);losses[e].backward();assert target.grad is None
  for other,expert in enumerate(p.experts):
   size=sum(float(x.grad.abs().sum()) for x in expert.parameters() if x.grad is not None)
   assert size>0 if other==e else size==0
  assert all(x.grad is None for x in p.history.parameters());assert p.router.weight.grad is None;assert all(x.grad is None for x in p.design_head.parameters())

def test_sparse_two_experts_and_primary_router_gradients():
 torch.manual_seed(14);p=RoleMoE();calls=[0]*4;hooks=[e.register_forward_hook(lambda a,b,c,i=i:calls.__setitem__(i,calls[i]+len(c))) for i,e in enumerate(p.experts)];f=history(B=16);y=p(f,10);y.square().sum().backward();assert sum(calls)==32;assert p.router.weight.grad.abs().sum()>0;assert all(torch.isfinite(x.grad).all() for x in p.parameters() if x.grad is not None)
 for hook in hooks:hook.remove()
 p.routing='uniform_top2';p(f,10);assert p.last_dispatch['indices'].shape==(16,2);torch.testing.assert_close(p.last_dispatch['weights'].sum(-1),torch.ones(16,dtype=torch.float64));assert p.counts()['always_active_FFN_experts']==0

def test_selected_router_gradient_matches_finite_difference():
 torch.manual_seed(18);p=RoleMoE();f=history(B=2);loss=p(f,10).square().sum();loss.backward();par=p.router.weight;idx=tuple(torch.unravel_index(par.grad.abs().argmax(),par.shape));g=float(par.grad[idx]);old=float(par[idx]);v=[]
 with torch.no_grad():
  for delta in [-1e-5,1e-5]:par[idx]=old+delta;v.append(float(p(f,10).square().sum()))
  par[idx]=old
 assert abs(g-(v[1]-v[0])/2e-5)<1e-7

def test_future_heads_zero_and_deployment_reproducible():
 p=RoleMoE().eval();f=history(29);r=p.role_predictions(f,29);assert torch.equal(r[:,2:],torch.zeros_like(r[:,2:]));value=p(f,29);stream=io.BytesIO();torch.save(p,stream);stream.seek(0);q=torch.load(stream,weights_only=False);torch.testing.assert_close(value,q(f,29),atol=0,rtol=0);assert list(inspect.signature(p.forward).parameters)==['f','stage']
 source=inspect.getsource(__import__('src.research.moe_sboed_v4',fromlist=['RoleMoE']));assert 'make_benchmark' not in source and 'torch.optim' not in source and 'theta' not in inspect.getsource(RoleMoE.forward)

def test_dense_parameter_match():
 p=RoleMoE();d=dense_control(p.counts()['total_parameters']);assert abs(d.counts()['total_parameters']/p.counts()['total_parameters']-1)<.01;assert torch.isfinite(d(history(),10)).all()

def test_sir_interface_and_numerics():
 from src.research.benchmarks import SIR
 b=SIR({'sir_ode':{}});p=RoleMoE(5,1,1,1.,1.,ordered_history=True);r=rollout(b,p,5,2,701,2);assert torch.isfinite(r['info']).all();t=torch.stack(r['actions'],1)[:,:,0];assert (t[:,1:]-t[:,:-1]>=.1-1e-10).all();(-r['info'][:,-1].mean()).backward();assert p.history.encoder[0].weight.grad.abs().sum()>0

@pytest.mark.parametrize('name',['ieee9','ieee14'])
def test_grid_history_action_interfaces(name):
 cfg=yaml.safe_load((ROOT/f'configs/{name}_research_eig.yaml').read_text());h=cfg['experiment']['step_number'];p=RoleMoE(h,1,1,1.,1.,ordered_history=True);f=history(1,h=h,a=1,o=1);assert p(f,1).shape==(8,1);p(f,1).sum().backward()

@pytest.mark.parametrize('name',['ieee9','ieee14'])
def test_grid_unchanged_simulator(name):
 if shutil.which('nvcc') is None or not torch.cuda.is_available():pytest.skip('Grid numerical test requires GPU and compiler; no numerical substitute')
 cfg=yaml.safe_load((ROOT/f'configs/{name}_research_eig.yaml').read_text());a=PredictiveAdapter(cfg,config_path=ROOT/f'configs/{name}_research_eig.yaml');x=a.b.sample((2,),torch.Generator().manual_seed(21));assert torch.isfinite(a.signatures(x)).all()
 b=a.b;state=b.initial((2,));action=b.feasible(torch.zeros(2,b.action_dim,dtype=torch.float64),[],cfg['experiment']['step_number']).requires_grad_();mean,st=b.forward(x,state,action);grad=torch.autograd.grad(mean.sum(),action)[0];values=[]
 with torch.no_grad():
  for eps in [-1e-5,1e-5]:values.append(b.forward(x,state,action+eps)[0].sum())
 finite=(values[1]-values[0])/2e-5;torch.testing.assert_close(grad.sum(),finite,rtol=1e-3,atol=1e-5)


def test_dynamic_domain_order_is_retained():
 torch.manual_seed(42);p=RoleMoE(5,1,1,1.,1.,ordered_history=True);f=history(3,B=4,h=5,a=1,o=1);g=f.clone();slots=g[:,1:].reshape(4,5,3);slots[:,:3]=slots[:,:3].flip(1);assert not torch.allclose(p.history(f,3),p.history(g,3))

def test_expert_swaps_preserve_distinct_function_mapping():
 p=RoleMoE();p.expert_permutation=[2,1,0,3];f=history();p(f,10);ids=p.last_dispatch['indices'];mapped=torch.tensor(p.expert_permutation)[ids];assert (mapped[:,0]!=mapped[:,1]).all();p.expert_permutation=None;p.routing='uniform_top2';p.routing_seed=111;p._routing_rng=None;p(f,10);a=p.last_dispatch['indices'];p.routing_seed=222;p._routing_rng=None;p(f,10);assert not torch.equal(a,p.last_dispatch['indices'])
