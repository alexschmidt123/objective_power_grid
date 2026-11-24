"""Frozen tiny synthetic scorer and changed-policy resimulation fixtures."""
import copy,math,types
from native_common import *
from evaluate_v2 import template,actor_load,simulate,lib
from frozen_evaluator import contrast_scores
from v2_policy import V2Actor
original.setup();leases=original.acquire();torch.manual_seed(202611099112);model=V2Actor(copy.deepcopy(template())).eval();n=4
with torch.no_grad(),torch.device('cuda'):
 theta=torch.randn(n,4,dtype=torch.float64);noise=torch.randn(n,30,dtype=torch.float64);u=torch.rand(n,30);d=simulate(model,theta,noise,u,'candidate',collect=True);repeat=simulate(model,theta,noise,u,'candidate');assert all(torch.equal(x,y) for x,y in zip(d['actions'],repeat['actions'])) and all(torch.equal(x,y) for x,y in zip(d['observations'],repeat['observations']));x=d['X'];commands=d['commands'];gates=d['gates']
 for kwargs in [dict(commands=torch.zeros(20)),dict(commands=commands.mean(0)),dict(commands=commands.flip(0)),dict(gates=gates.mean(0)),dict(gates=torch.full((30,4),.25))]+[dict(disable=k) for k in range(4)]+[dict(response_off=1),dict(response_off=3),dict(response_off=[0,1,2,3]),dict(replace_expert=1,donor_slots=d['slots'].flip(0))]:
  sim=simulate(model,theta,noise,u,'candidate',**kwargs);assert torch.isfinite(torch.stack(sim['actions'])).all()
 outside,_=actor_load('original')
 for method in ['sg','parent','reference','vsoed','ordinary','historical_control']:
  actor,_=actor_load(method);r=simulate(actor,theta,noise,u,method,outside=outside);assert torch.isfinite(torch.stack(r['actions'])).all()
 neg=torch.randn(11,4,dtype=torch.float64);actual,_=contrast_scores(types.SimpleNamespace(sigma=.5,calls=0),d,neg,3,'cuda');rows=[]
 for i in range(n):
  th=torch.cat([theta[i:i+1],neg]).reshape(-1,2,2);lp=torch.zeros(12,dtype=torch.float64);vals=[]
  for t in range(30):
   mean=(.1+(1e-4+(th-d['actions'][t][i]).square().sum(-1)).reciprocal().sum(-1)).log();lp+=-2*(d['observations'][t][i,0]-mean).square()-math.log(.5*math.sqrt(2*math.pi));vals.append(lp[0]-torch.logsumexp(lp,0)+math.log(12))
  rows.append(torch.stack(vals))
 torch.testing.assert_close(actual,torch.stack(rows).cpu(),rtol=1e-11,atol=1e-11)
put(O/'evidence/EVALUATION_CHECKS.json',dict(status='PASS',training_updates=0,scientific_bank_accessed=False,bitwise_repeated_full_rollout=True,all_command_coordination_removal_interfaces=True,all_MoE_native_baseline_loaders=True,literal_frozen_scorer_max_error=float((actual-torch.stack(rows).cpu()).abs().max()),positive_included_once=True));print('HEC_EVALUATION_CHECKS_PASS',flush=True)
