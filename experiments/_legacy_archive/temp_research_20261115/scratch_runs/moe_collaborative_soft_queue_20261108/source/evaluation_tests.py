"""CPU-only frozen scorer/history consistency; no policy training/evaluation bank."""
import os,sys,math
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
import torch
from native_common import *
from fresh_common import build_fresh
from evaluate_soft import native_x
from frozen_evaluator import contrast_scores
import types
beg=time.monotonic();torch.set_num_threads(2);torch.manual_seed(202611070713);a=build_fresh()['algorithm'];b,T,L=3,4,11;truth=torch.randn(b,4,dtype=torch.double);neg=torch.randn(L,4,dtype=torch.double);acts=[torch.randn(b,2,dtype=torch.double) for _ in range(T)];ys=[torch.randn(b,1,dtype=torch.double) for _ in range(T)];data=dict(actions=acts,observations=ys,truth=truth);actual,_=contrast_scores(types.SimpleNamespace(sigma=.5,calls=0),data,neg,3,'cpu');rows=[]
for i in range(b):
 theta=torch.cat([truth[i:i+1],neg],0).reshape(-1,2,2);lp=torch.zeros(L+1,dtype=torch.double);r=[]
 for t in range(T):
  mean=(.1+(1e-4+(theta-acts[t][i]).square().sum(-1)).reciprocal().sum(-1)).log();lp+=-2*(ys[t][i,0]-mean).square()-math.log(.5*math.sqrt(2*math.pi));r.append(lp[0]-torch.logsumexp(lp,0)+math.log(L+1))
 rows.append(torch.stack(r))
literal=torch.stack(rows);torch.testing.assert_close(actual,literal,atol=1e-12,rtol=1e-12)
for t in [0,5,14,29]:
 ds=torch.randn(b,t,2);obs=torch.randn(b,t,1);ours=native_x(list(ds.unbind(1)),list(obs.unbind(1)),b,'cpu');native=a.form_actor_input(t,ds,obs);torch.testing.assert_close(ours,native,atol=0,rtol=0)
policy_factor=torch.tensor([-2.,-4.,-7.],dtype=torch.double);lp=torch.randn(3,12,dtype=torch.double);before=lp[:,0]-torch.logsumexp(lp,1);after=lp[:,0]+policy_factor-torch.logsumexp(lp+policy_factor[:,None],1);torch.testing.assert_close(before,after,atol=1e-12,rtol=1e-12)
put(O/'evidence/EVALUATION_TEST_RESULTS.json',dict(status='PASS',frozen_scorer_manual_max_error=float((actual-literal).abs().max()),native_history_encoding_exact=True,stochastic_policy_likelihood_factors_cancel=True,scorer_sha256=sha(O/'source/frozen_evaluator.py'),GPU_used=False,training_updates=0,scientific_bank_accessed=False,seconds=time.monotonic()-beg));print('EVALUATION_FIXTURE_TESTS_PASS',flush=True)
