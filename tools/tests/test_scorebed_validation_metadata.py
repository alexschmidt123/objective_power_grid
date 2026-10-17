"""Regression: native validation must preserve deployed feasibility metadata."""
import copy
import numpy as np
import pytest
import torch
from src.research.journal_scorebed_bridge import export_candidate
from src.research.scorebed import OfficialScorePolicy
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout

def params():
 g=np.random.default_rng(819)
 return {'encoder_net':{'layers_0':{'kernel':g.normal(0,.05,(3,256)).astype('float32'),'bias':g.normal(0,.01,256).astype('float32')},'layers_2':{'kernel':g.normal(0,.05,(256,16)).astype('float32'),'bias':g.normal(0,.01,16).astype('float32')}},'emitter_net':{'layers_0':{'kernel':g.normal(0,.1,(16,2)).astype('float32'),'bias':np.array([2.,-2.],dtype='float32')}}}

def test_candidate_preserves_bounds_and_exact_weights():
 p=params();before=copy.deepcopy(p);a=export_candidate(p,{'raw':{'location_finding':{'design_bounds':[-4,4]}}})
 np.testing.assert_array_equal(a['common_design_bounds'],[-4,4])
 for s,ls in p.items():
  for l,vs in ls.items():
   for k,v in vs.items():
    np.testing.assert_array_equal(a[f'{s}/{l}/{k}'],v);np.testing.assert_array_equal(v,before[s][l][k])
@pytest.mark.parametrize('bounds',[[4,-4],[-4],[-4,4,5],[float('nan'),4]])
def test_invalid_metadata_rejected(bounds):
 with pytest.raises(ValueError):export_candidate(params(),{'raw':{'location_finding':{'design_bounds':bounds}}})
def test_unbounded_policy_does_not_gain_bounds():
 assert 'common_design_bounds'not in export_candidate(params(),{'raw':{}})
def test_validation_export_roundtrip_preserves_complete_bounded_rollout(tmp_path):
 torch.set_num_threads(1);a=export_candidate(params(),{'raw':{'location_finding':{'design_bounds':[-4,4]}}});np.savez(tmp_path/'candidate.npz',**a);b=make_benchmark({'system':{'name':'location_finding'},'observation':{'noise_sigma':.5},'location_finding':{'design_bounds':[-4,4]}})
 p=OfficialScorePolicy(a,30).eval();q=OfficialScorePolicy(np.load(tmp_path/'candidate.npz'),30).eval();generator=torch.Generator().manual_seed(192);truth=b.sample((4,),generator);noise=torch.randn(4,30,1,generator=generator,dtype=torch.float64)
 with torch.no_grad():
  r=rollout(b,p,30,4,1001,0,truth=truth,noise=noise);s=rollout(b,q,30,4,1001,0,truth=truth,noise=noise)
 for key in ['actions','observations']:
  for x,y in zip(r[key],s[key]):assert torch.equal(x,y)
 assert all(bool(((x>=-4)&(x<=4)).all())for x in r['actions'])
 # Removing the metadata reproduces the original validation defect.
 unbounded={k:v for k,v in a.items()if k!='common_design_bounds'};bad=OfficialScorePolicy(unbounded,30).eval();features=torch.zeros(4,121,dtype=torch.float64)
 assert not torch.equal(p(features,0),bad(features,0))
