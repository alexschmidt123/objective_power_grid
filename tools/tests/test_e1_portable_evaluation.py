import copy
import numpy as np
import pytest
import torch
from src.research.benchmarks import LocationFinding
from src.research.engine import generator
from src.research.dad_location_policy import LocationDAD
from src.research.runner import DEFAULTS,evaluate

def test_explicit_draws_reproduce_existing_evaluator_exactly(tmp_path):
 b=LocationFinding({'location_finding':{'design_bounds':[-4,4]},'observation':{'noise_sigma':.5}});h=5;n=3;L=16
 p=LocationDAD(h).eval();s=DEFAULTS|dict(eval_systems=n,evaluation_population_systems=n,eval_contrasts=L,evaluation_batch_size=3,evaluation_contrast_chunk=8)
 g=generator(1001);draws=dict(truth=b.sample((n,),g),noise=torch.randn(n,h,1,generator=g,dtype=torch.float64),contrasts=b.sample((L,),g))
 a=tmp_path/'old';z=tmp_path/'portable'
 for x in [a,z]:x.mkdir();(x/'eval').mkdir()
 with torch.no_grad():
  old,old_s=evaluate(b,p,'dad',h,s,1001,dict(training_seconds=0.),a,paper=True,infer_parameters=False)
  new,new_s=evaluate(b,p,'dad',h,s,1001,dict(training_seconds=0.),z,paper=True,infer_parameters=False,evaluation_draws=draws)
 assert [{k:v for k,v in row.items() if k!='mean_decision_seconds_per_trajectory'} for row in old]==[{k:v for k,v in row.items() if k!='mean_decision_seconds_per_trajectory'} for row in new]
 assert old_s['mean_spce']==new_s['mean_spce']
 np.testing.assert_array_equal(np.load(a/'eval/contrast_bank_seed1001.npy'),np.load(z/'eval/contrast_bank_seed1001.npy'))
 bad=dict(draws,contrasts=draws['contrasts'][:2])
 with pytest.raises(ValueError,match='shape mismatch'):evaluate(b,p,'dad',h,s,1001,{},z,paper=True,infer_parameters=False,evaluation_draws=bad)
