import copy,math
from pathlib import Path
import pytest,torch
from src.research.deepseek_sboed import DeepSeekPolicy
from src.research.benchmarks import LocationFinding
from src.research.engine import rollout
from src.research.moe_v5a_training import primary_backward,draws
from src.research.moe_v5a_sampler import sample_prefix,AuditedSMC

def bench():return LocationFinding({'location_finding':{'design_bounds':[-4,4]}})

def test_effective_batch_gradient_accumulation_preserves_native_loss():
 torch.manual_seed(611001);a=DeepSeekPolicy().train();b=copy.deepcopy(a);x=bench();truth,bank,noise=draws(x,4,16,611002,'cpu')
 r=rollout(x,a,30,4,611002,16,truth=truth,noise=noise,prior_particles=bank)
 loss=-r['info'][:,-1].mean()+a.balance_loss();loss.backward()
 stats=primary_backward(bench(),b,truth,bank,noise,611002,2)
 assert abs(stats['main_spce']-float(r['info'][:,-1].mean().detach()))<1e-12
 for p,q in zip(a.parameters(),b.parameters()):
  assert (p.grad is None)==(q.grad is None)
  if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=1e-8,atol=1e-10)

def test_prefix_only_sampler_and_source_swap_invariance():
 b=bench();action=[torch.tensor([[.1,-.3]],dtype=torch.float64)];past=[torch.tensor([[1.2]],dtype=torch.float64)]
 x,q,meta,pool=sample_prefix(b,action,past,16,611003,64,4)
 y,_,_,_=sample_prefix(bench(),action,past,16,611003,64,4)
 torch.testing.assert_close(x,y,rtol=0,atol=0);assert q.conditioned_observations==1 and not q.exact
 design=torch.tensor([[.2,.5]],dtype=torch.float64).expand(len(pool),-1)
 mu,_=b.forward(pool,None,design);swapped=pool.reshape(-1,2,2).flip(1).reshape(-1,4);other,_=b.forward(swapped,None,design)
 torch.testing.assert_close(mu,other,rtol=0,atol=0)
 assert meta['independent_draws_certified'] is False

def test_constant_and_low_ess_populations_rejected(monkeypatch):
 original=AuditedSMC.__init__
 def constant(self,*args,**kwargs):original(self,*args,**kwargs);self.theta.zero_()
 monkeypatch.setattr(AuditedSMC,'__init__',constant)
 with pytest.raises(ValueError):sample_prefix(bench(),[],[],8,611004,32,4)

def test_gaussian_exact_and_detached_sampling_gradient_bias_is_measured():
 g=torch.Generator().manual_seed(611005);B=128;L=256;sigma=.5
 truth=torch.randn(B,1,generator=g,dtype=torch.float64);prior=torch.randn(B,L,generator=g,dtype=torch.float64);eps=torch.randn(B,2,generator=g,dtype=torch.float64);z=torch.randn(B,L,generator=g,dtype=torch.float64)
 def estimate(detach):
  a=torch.tensor(.7,dtype=torch.float64,requires_grad=True);c=torch.tensor(1.3,dtype=torch.float64,requires_grad=True)
  y=a*truth[:,0]+sigma*eps[:,0];var=1/(1+a*a/sigma**2);mean=var*a*y/sigma**2
  post=mean[:,None]+var.sqrt()*z
  if detach:post=post.detach()
  def score(y,design,neg):
   particles=torch.cat((truth,neg),1);ll=-.5*((y[:,None]-design*particles)/sigma).square();return ll[:,0]-torch.logsumexp(ll,-1)+math.log(L+1)
  value=(score(y,a,prior)+score(c*truth[:,0]+sigma*eps[:,1],c,post)).mean()
  return torch.autograd.grad(value,(a,c))
 exact=estimate(False);stopped=estimate(True);analytic=.7/(sigma**2+.7**2+1.3**2)
 assert all(torch.isfinite(x) for x in exact+stopped)
 assert abs(float(exact[0])-analytic)<.25
 assert abs(float(stopped[0]-exact[0]))>.1 # recorded bias, not an invalid 'unbiased' claim
