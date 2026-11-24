"""Frozen synthetic resimulation checks;no optimizer updates/evaluation bank."""
import time,copy
from native_common import *
from restore_parent import restore
from sg_policy import migrate
from evaluate_sg import simulate,lib
from conditional_diagnostics import suffix
from normalized_reward import normalize_native_post
original.setup();leases=original.acquire();state=restore('cuda');a=state['algorithm'];parent=copy.deepcopy(a.actor_net);migrate(a,True);normalize_native_post(a);actor=a.actor_net.eval();torch.manual_seed(2026110815601)
with torch.no_grad(),torch.device('cuda'):
 th=torch.randn(16,4);eps=torch.randn(16,30);u=torch.rand(16,30);data=simulate(actor,th,eps,u,'candidate',collect=True);ds=torch.stack(data['actions'],1);ys=torch.stack(data['observations'],1);slots=data['slots'].flip(0);routes=torch.cat([data['gates'],data['retention']],-1);stage=routes.mean(0)
 for routing in ['parent','uniform','stage','fixed','shuffle']:
  dr=routes.flip(0) if routing=='shuffle' else stage
  out=simulate(actor,th,eps,u,'candidate',routing=routing,donor_routes=dr,parent=parent)
  assert torch.isfinite(torch.stack(out['actions'])).all()
 for k in range(4):
  simulate(actor,th,eps,u,'candidate',replace_expert=k,donor_slots=slots);simulate(actor,th,eps,u,'candidate',disable=k)
 for mode in ['adaptive_off','retention_stage','anchors_live']:simulate(actor,th,eps,u,'candidate',mechanism=mode,donor_routes=stage)
 for t in [3,12,24]:
  one=suffix(actor,th,ds,ys,t,eps);two=suffix(actor,th,ds,ys,t,eps);assert all(torch.equal(x,y) for x,y in zip(one,two))
  changed=suffix(actor,th,ds,ys,t,eps,2,slots[:,t]);assert torch.equal(changed[0][:,:t],ds[:,:t]) and torch.equal(changed[1][:,:t],ys[:,:t]);q=a.post_approx.log_poi_post(29,*changed,th);assert torch.isfinite(q).all()
 assert torch.equal(actor.components(data['X'])[1],parent.parts(data['X'])[1])
put(O/'evidence/EVALUATION_PATH_CHECKS.json',dict(status='PASS',training_updates=0,all_routing_and_expert_intervention_interfaces=True,anchor_features_bitwise_preserved=True,conditional_resimulation_bitwise_repeat=True,conditional_native_scores_finite=True,scientific_bank_accessed=False));print('SG_EVALUATION_PATHS_PASS')
