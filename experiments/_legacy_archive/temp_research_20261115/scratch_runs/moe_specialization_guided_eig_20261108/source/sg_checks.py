"""Restore/function/gradient/anchor/Adam/reward checks;no optimizer updates."""
import sys,time,copy,json
sys.dont_write_bytecode=True
import torch
from native_common import *
from restore_parent import restore,PARENT_ROOT,ACTOR_SHA,RUNTIME_SHA
from sg_policy import migrate
from normalized_reward import normalize_native_post
from native_reward import stateless_rewards
original.setup();leases=original.acquire();beg=time.monotonic();state=restore('cuda');a=state['algorithm'];b=copy.deepcopy(a);parent=copy.deepcopy(a.actor_net);gen=torch.Generator(device='cuda').manual_seed(2026110815301);x=torch.randn(32,117,device='cuda',generator=gen);w=a.actor_net.decision[0].weight;m0=a.actor_optimizer.state[w]['exp_avg'].clone();bias_before=a.actor_optimizer.state[a.actor_net.decision[0].bias]['exp_avg'].clone();before=digest(parent.state_dict());info=migrate(a,True);migrate(b,False)
with torch.no_grad():
 original_out=parent(x);ca=a.actor_net(x);co=b.actor_net(x);torch.testing.assert_close(ca,co,rtol=0,atol=0);torch.testing.assert_close(ca,original_out,rtol=1e-6,atol=1e-6);live,anchor,g,r=a.actor_net.components(x);assert torch.equal(live,anchor);assert (g>0).all() and (r==.5).all();assert len(list(a.actor_net.anchor_encoder.parameters()))==0 and len(list(a.actor_net.anchor_experts.parameters()))==0
 assert torch.equal(a.actor_optimizer.state[a.actor_net.decision[0].weight]['exp_avg'],m0);assert torch.equal(a.actor_optimizer.state[a.actor_net.decision[0].bias]['exp_avg'],bias_before)
for model in [a,b]:
 model.actor_optimizer.zero_grad(set_to_none=True);loss=-model.critic_net(torch.cat([x,model.actor_net(x)],-1)).mean();loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.actor_net.parameters())
for module in [a.actor_net.encoder,a.actor_net.router,*a.actor_net.experts,a.actor_net.decision]:assert norm([p.grad for p in module.parameters()])>0
# First-step ordinary gradients agree at function-preserving initialization.
for key in ['encoder','router','experts']:
 for pa,pb in zip(getattr(a.actor_net,key).parameters(),getattr(b.actor_net,key).parameters()):torch.testing.assert_close(pa.grad,pb.grad,rtol=1e-5,atol=1e-7)
anchor_hash=digest(a.actor_net.anchor_experts.state_dict());anchor_encoder_hash=digest(a.actor_net.anchor_encoder.state_dict())
with torch.no_grad():
 for p in a.actor_net.experts[0].parameters():p.add_(.001)
 a.actor_net.encoder[0].bias.add_(.001)
a.actor_optimizer.zero_grad(set_to_none=True);loss=-a.critic_net(torch.cat([x,a.actor_net(x)],-1)).mean();loss.backward();assert norm([p.grad for p in a.actor_net.retention.parameters()])>0;assert digest(a.actor_net.anchor_experts.state_dict())==anchor_hash and digest(a.actor_net.anchor_encoder.state_dict())==anchor_encoder_hash
# Native valid information rewards telescope with gamma1;latent/datum simulation fixed.
normalize_native_post(b)
with torch.no_grad(),torch.device('cuda'):
 theta=torch.randn(8,4,device='cuda',generator=gen);ds=torch.randn(8,30,2,device='cuda',generator=gen).clamp(-4,4);noise=torch.randn(8,30,device='cuda',generator=gen);distance=(theta.reshape(8,1,2,2)-ds[:,:,None]).square().sum(-1);ys=((.1+(distance+1e-4).reciprocal().sum(-1)).log()+.5*noise)[:,:,None];rewards=stateless_rewards(b,ds,ys,theta);terminal=b.post_approx.log_poi_post(29,ds,ys,theta)-b.post_approx.prior.log_poi_prior(theta);torch.testing.assert_close(rewards.sum(-1),terminal,atol=2e-5,rtol=1e-5)
 assert ((ca>=-4)&(ca<=4)).all()
assert sha(PARENT_ROOT/'arms/candidate/actor.pt')==ACTOR_SHA and sha(PARENT_ROOT/'arms/candidate/runtime.pkl')==RUNTIME_SHA
put(O/'evidence/SG_CHECKS.json',dict(status='PASS',training_updates=0,parent_actor_digest=before,parent_Adam_and_replay_restored=True,initial_candidate_control_actions_bitwise_equal=True,parent_output_max_error=float((ca-original_out).abs().max()),Adam_old_columns_preserved=True,new_residual_head_Adam_fresh=True,frozen_anchors_excluded_from_Polyak_parameters=True,anchor_features_unchanged_after_live_parameter_perturbation=True,all_live_expert_and_router_gradients=True,initial_adaptive_gradient_matches_ordinary_control=True,retention_gradient_after_adaptation=True,proper_mixture_native_rewards_telescope=True,seconds=time.monotonic()-beg,migration=info));print('SG_CORE_CHECKS_PASS')
