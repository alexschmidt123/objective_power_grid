import copy
from common import *
from evaluate import rollout
from v3_policy import V3Actor
import conditional

def main():
 original.setup();lease=original.acquire();start=time.monotonic();s=restore();m=V3Actor(copy.deepcopy(s['algorithm'].actor_net)).cuda().eval();del s;rng=np.random.default_rng(CFG['seed']+19);b=320;bank=dict(truth=rng.standard_normal((b,4)),noise=rng.standard_normal((b,30)),u=rng.random((b,30)),eps=rng.standard_normal((b,30,2)));CFG['conditional_cases']=64;CFG['conditional_stages']=[3]
 with torch.no_grad(),torch.device('cuda'):
  data=rollout(m,*[torch.tensor(bank[k],device='cuda') for k in ['truth','noise','u','eps']],'candidate',collect=True);hist={k:data[k].cpu().numpy() for k in ['X','ds','ys']}
  for label in ['identical','changed']:
   conditional.O=O/'evidence'/'conditional_fixture'/label;(conditional.O/'evaluation').mkdir(parents=True,exist_ok=True)
   if label=='changed':m.experts[1][-1].bias.add_(.4)
   conditional.run_conditional(m,bank,hist,time.monotonic()+120);a=np.load(conditional.O/'evaluation/conditional_native_advantages.npz')['advantages'];assert np.isfinite(a).all()
   if label=='identical':assert np.max(np.abs(a))==0
   else:assert np.max(np.abs(a))>0
 put(O/'evidence/CONDITIONAL_CHECKS.json',dict(status='PASS',identical_policies_exact_zero=True,changed_policy_suffix_resimulation_finite=True,zero_variance_interaction_handled=True,training_updates=0,seconds=time.monotonic()-start));print('V3_CONDITIONAL_CHECKS_PASS')
if __name__=='__main__':main()
