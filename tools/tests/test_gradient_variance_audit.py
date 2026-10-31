import copy,inspect
import torch,pytest
from src.research.gradient_variance_audit import extract,joint_draws,bank_draws
from src.research.gradient_variance_analysis import estimator_sets
from src.research.deepseek_sboed import DeepSeekPolicy
from src.research.benchmarks import make_benchmark
from src.research.fh_moe_v3_discovery import RAW
from src.research.engine import rollout

torch.set_num_threads(2)
def test_exact_native_objective_and_full_parameter_gradient_when_capturing():
 torch.manual_seed(77);m=DeepSeekPolicy();m.ablate_routed=True;n=copy.deepcopy(m)
 truth,noise=joint_draws(3,707,'cpu');particles=bank_draws(3,12,708,'cpu')
 out=extract(m,truth,noise,particles)
 r=rollout(make_benchmark(RAW),n,30,3,0,12,truth=truth,noise=noise,prior_particles=particles)
 assert out['loss']==float(-r['info'][:,-1].mean().detach())
 assert out['raw'].shape==(3,4,16) and out['action'].shape==(3,4,2)
 assert out['x'].shape==(3,4,41)

def test_same_trajectory_independent_of_contrast_bank():
 m=DeepSeekPolicy();m.ablate_routed=True;t,e=joint_draws(3,1,'cpu');c=bank_draws(3,16,2,'cpu')
 a=extract(m,t,e,c);b=extract(m,t,e,-c)
 assert torch.equal(a['x'],b['x']) and not torch.equal(a['raw'],b['raw'])

def test_mean_raw_gradient_is_exact_gradient_of_mean_terminal_objective():
 torch.manual_seed(4);m=DeepSeekPolicy();m.ablate_routed=True;t,e=joint_draws(2,12,'cpu');c=bank_draws(2,16,13,'cpu');d=bank_draws(2,16,14,'cpu')
 # Compare ordinary parameter gradients; all30stage history graphs remain live.
 r=rollout(make_benchmark(RAW),m,30,2,0,16,truth=t,noise=e,prior_particles=c);s=rollout(make_benchmark(RAW),m,30,2,0,16,truth=t,noise=e,prior_particles=d)
 params=list(m.parameters());a=torch.autograd.grad(-r['info'][:,-1].mean(),params,retain_graph=True,allow_unused=True);b=torch.autograd.grad(-s['info'][:,-1].mean(),params,retain_graph=True,allow_unused=True)
 total=torch.autograd.grad(-(r['info'][:,-1].mean()+s['info'][:,-1].mean())/2,params,allow_unused=True)
 for x,y,z in zip(a,b,total):
  if x is None:assert y is None and z is None
  else:torch.testing.assert_close(z,(x+y)/2,atol=1e-12,rtol=1e-12)

def test_antithetic_transform_preserves_gaussian_marginals_not_iid_between_banks():
 c=bank_draws(1000,16,777,'cpu');assert torch.equal(-(-c),c)
 assert abs(float((-c).mean()))<.015 and abs(float((-c).var())-1)<.025
 assert torch.equal(c.square(),(-c).square())

def test_reproducible_draws_and_nested_contrasts():
 t,e=joint_draws(8,11,'cpu');u,f=joint_draws(8,11,'cpu');assert torch.equal(t,u) and torch.equal(e,f)
 c=bank_draws(8,32,12,'cpu');assert torch.equal(c[:,:16],c.clone()[:,:16])

def test_nonlinear_normalization_does_not_commute_with_averaging():
 from src.research.fh_moe_v3_context import normalized
 g=torch.tensor([[10.,0.],[0.,1.]])
 assert not torch.allclose(normalized(g.mean(0)),normalized(g).mean(0))

def test_estimators_use_raw_not_averaged_unit_vectors_and_matched_work():
 d={k:torch.randn(12,4,4,16,dtype=torch.double) for k in ['C','D','negative_C']};a=estimator_sets(d)
 torch.testing.assert_close(a['mean4_iid_L2000'],d['C'].mean(1))
 torch.testing.assert_close(a['mean4_antithetic_pairs_L2000'],((d['C']+d['negative_C'])/2).mean(1))
 assert a['mean4_iid_pairs_L2000'].shape==a['mean4_antithetic_pairs_L2000'].shape

def test_per_trajectory_adjoint_batch_scaling():
 torch.manual_seed(12);m=DeepSeekPolicy();m.ablate_routed=True;t,e=joint_draws(2,3,'cpu');p=bank_draws(2,16,4,'cpu')
 a=extract(m,t,e,p);b=extract(m,t[:1],e[:1],p[:1])
 torch.testing.assert_close(a['raw'][:1],b['raw'],rtol=1e-9,atol=1e-9)

def test_no_optimization_or_posterior_sampler_in_diagnostic():
 import src.research.gradient_variance_audit as module
 source=inspect.getsource(module)
 assert 'torch.optim.' not in source and 'loss.backward' not in source
 assert 'prior_particles=particles' in source


def test_saved_unit_vectors_below_normalization_floor_are_not_renormalized():
 from src.research.gradient_variance_analysis import distribution
 u=torch.zeros(4,16,dtype=torch.double);u[:,0]=torch.tensor([1.,1.,1e-20,1e-25])
 d=distribution(u,already_normalized=True)
 assert abs(d['unit_mean_norm']-float(u.mean(0).norm()))<1e-15
