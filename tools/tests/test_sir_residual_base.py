import json
import pytest
import torch
from src.research.benchmarks import SIR
from src.research.sir_dad import SIRSumMLP
from src.research.moe_family import BasePolicy,MoESBOEDPolicy,load_policy
from src.research.engine import rollout

RAW=dict(system=dict(name='sir_ode'),sir_ode=dict(population=500,i0=2,t_end=100,min_spacing=.1,beta_mean=.5,gamma_mean=.1,log_std=.5,integration_steps_per_stage=512),observation=dict(noise_sigma=1.))

def make_base(tmp_path):
 b=SIR(RAW);p=SIRSumMLP(5).eval();(tmp_path/'models').mkdir()
 (tmp_path/'run_config.json').write_text(json.dumps(dict(benchmark=RAW,horizon=5)))
 (tmp_path/'models/sum_single_training.json').write_text(json.dumps(dict(generated_training_trajectories=8,training_seconds=1.)))
 path=tmp_path/'models/sum_single.pt';torch.save(dict(kind=p.checkpoint_kind,horizon=5,hidden=64,state_dict=p.state_dict()),path)
 return b,p,path

@pytest.mark.parametrize('variant',['resmoe','switchresmoe'])
def test_sir_native_base_loading_disabled_corrections_and_roundtrip(tmp_path,variant):
 torch.set_num_threads(1)
 b,p,path=make_base(tmp_path);q=MoESBOEDPolicy(variant,b,5,base_checkpoint=path,residual_scale=0).eval()
 a=rollout(b,p,5,2,902,2);c=rollout(b,q,5,2,902,2)
 for key in ['actions','observations']:torch.testing.assert_close(torch.stack(a[key]),torch.stack(c[key]),atol=0,rtol=0)
 saved=tmp_path/'family.pt';torch.save(q.checkpoint(),saved);r=load_policy(saved,b)
 d=rollout(b,r,5,2,902,2)
 torch.testing.assert_close(torch.stack(c['actions']),torch.stack(d['actions']),atol=0,rtol=0)
 assert q.core.extra_encoder is None

def test_sir_base_rejects_different_prior(tmp_path):
 b,p,path=make_base(tmp_path);other=SIR(dict(RAW,sir_ode=dict(RAW['sir_ode'],beta_mean=.7)))
 with pytest.raises(ValueError,match='physics/prior'):BasePolicy.from_checkpoint(path,other,5)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cpu_initialized_sir_prior_and_rollout_on_cuda():
 b=SIR(RAW)
 with torch.device('cuda:0'):
  from src.research.engine import generator
  z=torch.randn(2,2,generator=generator(91),dtype=torch.float64);theta=b.inverse(z)
  torch.testing.assert_close(b.transform(theta),z)
  p=SIRSumMLP(5);r=rollout(b,p,5,2,101,2);loss=-r['info'][:,-1].mean();loss.backward()
  assert torch.isfinite(loss) and all(torch.isfinite(v.grad).all() for v in p.parameters() if v.grad is not None)
  times=torch.stack(r['actions'],1).squeeze(-1);assert (times[:,1:]-times[:,:-1]>=.1-1e-10).all() and (times<=100).all()

@pytest.mark.parametrize('residual',[False,True])
def test_sir_forced_rerouting_matches_nonswitching_variant(tmp_path,residual):
 b,_,path=make_base(tmp_path)
 kw=dict(base_checkpoint=path) if residual else {}
 torch.manual_seed(201);a=MoESBOEDPolicy('resmoe' if residual else 'moe',b,5,**kw).set_routing()
 torch.manual_seed(201);c=MoESBOEDPolicy('switchresmoe' if residual else 'switchmoe',b,5,routing_mode='always_reroute',**kw).set_routing('always_reroute')
 x=rollout(b,a,5,2,902,2);y=rollout(b,c,5,2,902,2)
 torch.testing.assert_close(torch.stack(x['actions']),torch.stack(y['actions']),atol=1e-10,rtol=1e-10)
 torch.testing.assert_close(x['info'],y['info'],atol=1e-9,rtol=1e-9)
