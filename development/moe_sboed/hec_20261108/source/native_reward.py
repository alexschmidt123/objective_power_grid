import torch
def stateless_rewards(a,ds,ys,theta):
 """Exact location native incremental variational rewards,without live caches."""
 post=a.post_approx;assert post.poi_weight==1 and post.model_weight==0 and post.goal_weight==0 and post.stages_incre==list(range(30)) and not post.share_interm_net
 previous=post.prior.log_poi_prior(theta);reward=torch.zeros(len(theta),31,device=theta.device,dtype=theta.dtype)
 for t in range(30):
  now=post.log_poi_post(stage=t,ds_hist=ds[:,:t+1].contiguous(),ys_hist=ys[:,:t+1].contiguous(),params=theta);reward[:,t]=now-previous;previous=now
 assert torch.isfinite(reward).all();return reward

