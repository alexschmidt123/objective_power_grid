import math,copy
import numpy as np
import torch
from src.research.benchmarks import LocationFinding
from src.research.pimoe import *

def setup(n=256):
 torch.manual_seed(101);b=LocationFinding({'location_finding':{'design_bounds':[-4,4]},'observation':{'noise_sigma':.5}});p=AdaptiveBelief(b,n,1004,moves=8);a=torch.tensor([[-2.,0.],[0.,0.],[2.,1.]],dtype=DTYPE);return b,p,a

def test_simulator_symmetry_and_likelihood():
 b,p,a=setup();z=responses(b,p.theta,a);swap=p.theta[:,[2,3,0,1]];torch.testing.assert_close(z,responses(b,swap,a),rtol=0,atol=0)
 theta=p.theta.reshape(-1,2,2);ref=(.1+(1e-4+(theta[:,None]-a[None,:,None]).square().sum(-1)).reciprocal().sum(-1)).log();torch.testing.assert_close(z,ref)
 assert conditional_entropy(b)==.5*math.log(2*math.pi*math.e*.25)

def test_density_normalization_membership_symmetry_gradients():
 b,p,a=setup();m=PredictiveMoE();f=state_features(b,p,a);z=m(**f)
 torch.testing.assert_close(z['weights'].sum(-1),torch.ones(3,dtype=DTYPE));torch.testing.assert_close(z['membership'].sum(-1),torch.ones(3,256,dtype=DTYPE));assert (z['mass']>0).all()
 grid=torch.linspace(-20,30,12000,dtype=DTYPE)[None].expand(3,-1);prob=m.log_prob(z,grid).exp();torch.testing.assert_close(torch.trapezoid(prob,grid),torch.ones(3,dtype=DTYPE),atol=2e-4,rtol=0)
 q=p.fork(1005);q.theta=q.theta[:,[2,3,0,1]];other=m(**state_features(b,q,a));torch.testing.assert_close(other['means'],z['means'],rtol=0,atol=0)
 ref=direct_reference(f['means'],f['weights'],.5,1005,128);loss,info,_=predictive_loss(m,f,ref['y'],ref['EIG'],ref['SE']);loss.backward();assert torch.isfinite(loss);assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters());assert sum(v.grad.abs().sum() for v in m.router.parameters())>0
 old=copy.deepcopy(m.state_dict());torch.optim.Adam(m.parameters(),lr=.001).step();assert any(not torch.equal(v,old[k]) for k,v in m.state_dict().items())

def test_analytic_gaussian_entropy_and_information():
 d=dict(weights=torch.ones(1,1,dtype=DTYPE),means=torch.zeros(1,1,dtype=DTYPE),scales=torch.full((1,1),math.sqrt(1.25),dtype=DTYPE))
 h=PredictiveMoE.entropy(d);torch.testing.assert_close(h,torch.tensor([.5*math.log(2*math.pi*math.e*1.25)],dtype=DTYPE),rtol=1e-12,atol=1e-12)
 assert abs(float(h)-.5*math.log(2*math.pi*math.e*.25)-.5*math.log(5))<1e-12
 means=torch.randn(1,4096,generator=rng(1004),dtype=DTYPE);ref=direct_reference(means,torch.full((4096,),1/4096,dtype=DTYPE),.5,1005,1024);assert abs(float(ref['EIG'])-.5*math.log(5))<4*float(ref['SE'])+.03

def test_continuous_posterior_gaussian_update_and_fork():
 class Linear(LocationFinding):
  source_label_symmetry=False
  def propagate(self,theta,state,action):return theta[...,:1],state
 b=Linear({'observation':{'noise_sigma':.5}});p=AdaptiveBelief(b,4096,1004,moves=16);prior=p.theta.clone();p.update(torch.tensor([0.,0.],dtype=DTYPE),1.)
 assert abs(float(p.theta[:,0].mean())-.8)<.06;assert abs(float(p.theta[:,0].var())-.2)<.04
 assert p.records[-1]['continuous_rejuvenation'];assert not torch.isin(p.theta[:,0],prior[:,0]).all()
 q=p.fork(1005);q.update(torch.tensor([0.,0.],dtype=DTYPE),.9);assert len(p.actions)==1 and len(q.actions)==2
 # This analytic asymmetric test is not label-symmetric: disable permutation kernel.

def test_checkpoint_roundtrip_and_ties(tmp_path):
 b,p,a=setup();m=PredictiveMoE();z=m(**state_features(b,p,a));file=tmp_path/'m.pt';torch.save(m.state_dict(),file);q=PredictiveMoE();q.load_state_dict(torch.load(file,weights_only=True));torch.testing.assert_close(q(**state_features(b,p,a))['means'],z['means'],atol=0,rtol=0)
 f=state_features(b,p,a);y=direct_reference(f['means'],f['weights'],.5,1005,32)['y'];loss,info,_=predictive_loss(m,f,y,torch.zeros(3),torch.ones(3)*100);assert info['confident_pairs']==0 and info['ranking_loss']==0

def test_bounded_two_step_planner_no_truth():
 b,p,a=setup(128);m=PredictiveMoE();planner=PIMoEPlanner(m,b,1004,particles=128,candidates=4,shortlist=1,fantasies=1)
 x=planner.select();assert x.shape==(2,) and (x.abs()<=4).all();planner.observe(x,torch.tensor(1.));assert len(planner.belief.actions)==1


def test_counting_reference_and_capacity_control():
 b,p,a=setup();b.pimoe_work={};p.update(a[0],torch.tensor(1.));assert b.calls>0 and b.pimoe_work['scalar_conditional_likelihood_evaluations']>0
 m=PredictiveMoE();target=sum(v.numel() for v in m.parameters());width=min(range(16,257),key=lambda w:abs(sum(v.numel() for v in PredictiveMoE(1,w,12).parameters())-target));q=PredictiveMoE(1,width,12)
 assert abs(sum(v.numel() for v in q.parameters())-target)/target<.03

def test_quadrature_entropy_agrees_with_density_integration():
 b,p,a=setup();m=PredictiveMoE();z=m(**state_features(b,p,a));grid=torch.linspace(-25,35,12000,dtype=DTYPE)[None].expand(3,-1);logp=m.log_prob(z,grid);reference=-torch.trapezoid(logp.exp()*logp,grid);torch.testing.assert_close(m.entropy(z,nodes=64),reference,atol=1e-5,rtol=0)
