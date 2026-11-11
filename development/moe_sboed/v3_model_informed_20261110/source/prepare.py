import copy,time,torch,numpy as np
from common import *
from v3_policy import V3Actor

def conditions(model,x):
 scores=[]
 with torch.no_grad():
  for xx in x.split(128):
   base,s,nodes,pw,var,bf=model.provider(xx);t=xx[:,:30].argmax(-1);F=xx.new_zeros(len(xx),4,4);ds=xx[:,30:88].reshape(-1,29,2)
   for j in range(int(t.max())):
    J=jacobian(nodes,ds[:,j]);F+=4*torch.einsum('bn,bni,bnj->bij',pw*(t>j)[:,None],J,J)
   scale=var.sqrt();ev=torch.linalg.eigvalsh((F*scale[:,:,None]*scale[:,None,:]).double()).clamp_min(0);scores.append(torch.stack([var.sum(-1),torch.expm1(bf[:,1]),-torch.log1p(ev[:,0]).float()],-1))
 return torch.cat(scores)

def dataset(a,model):
 rng=np.random.default_rng(CFG['prepare_seed']);n0=CFG['preparation_train_per_stage'];n1=CFG['preparation_validation_per_stage'];stages=torch.arange(30,device='cuda').repeat_interleave(n0+n1);idx=rng.choice(1000000,len(stages),replace=False);ids=torch.tensor(idx,device='cuda');x=prefix_inputs(a._runtime_buffer['ds_hist'][ids],a._runtime_buffer['ys_hist'][ids],stages);cal=torch.arange(len(x),device='cuda')%(n0+n1)<n0;scores=conditions(model,x);threshold=torch.stack([torch.quantile(scores[(stages==t)&cal],CFG['quantile'],dim=0) for t in range(30)]);special=CFG['specialist_weight_floor']+(scores>threshold[stages]).float();weights=torch.cat([torch.ones(len(x),1,device='cuda'),special],-1);weights=weights/weights[cal].mean(0);return x,weights,cal,scores,threshold,idx

def prepare():
 root=O/'preparation';assert not root.exists();root.mkdir();tick=time.monotonic();s=restore();a=s['algorithm'];model=V3Actor(copy.deepcopy(a.actor_net)).cuda();x,w,cal,score,threshold,idx=dataset(a,model);model.encoder.requires_grad_(False);model.router.requires_grad_(False);a.critic_net.requires_grad_(False);op=torch.optim.Adam(model.experts.parameters(),lr=CFG['preparation_lr']);rng=np.random.default_rng(CFG['prepare_seed']);gen=torch.Generator(device='cuda').manual_seed(CFG['prepare_seed']);train=cal.nonzero().flatten();val=(~cal).nonzero().flatten();COST.clear()
 def objective(xx,ww):
  _,mu,_=model.parts(xx);eps=torch.randn(len(xx),4,2,device='cuda',generator=gen);ac=4*(mu+CFG['latent_sigma']*eps).tanh();q=a.critic_net(torch.cat([xx[:,None].expand(-1,4,-1),ac],-1)).squeeze(-1);return -(ww*q).mean(),q
 with torch.no_grad():before=objective(x[val],w[val])[1].mean(0).tolist()
 for n in range(CFG['preparation_updates']):
  ids=train[torch.tensor(rng.integers(len(train),size=CFG['preparation_batch']),device='cuda')];op.zero_grad(set_to_none=True);l,q=objective(x[ids],w[ids]);assert torch.isfinite(l);l.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.experts.parameters());op.step()
  if n%50==0:print('PREPARATION',n,float(l.detach()),flush=True)
 with torch.no_grad():
  after=objective(x[val],w[val])[1].mean(0).tolist();ga,ma,_=model.parts(x[val]);torch.testing.assert_close(ga,torch.full_like(ga,.25),rtol=0,atol=0);model.stage_only=True;gb,mb,_=model.parts(x[val]);model.stage_only=False;assert torch.equal(ga,gb) and torch.equal(ma,mb)
 torch.save(dict(actor_state=model.state_dict(),preparation_optimizer=op.state_dict(),generator_state=gen.get_state()),root/'prepared.pt');np.savez_compressed(root/'prefixes.npz',X=x.cpu().numpy(),weights=w.cpu().numpy(),calibration=cal.cpu().numpy(),scores=score.cpu().numpy(),thresholds=threshold.cpu().numpy(),replay_indices=idx);put(root/'RESULT.json',dict(status='COMPLETE',steps=CFG['preparation_updates'],batch=CFG['preparation_batch'],seconds=time.monotonic()-tick,training_prefixes=int(cal.sum()),validation_prefixes=int((~cal).sum()),new_simulator_episodes=0,new_simulator_observations=0,condition_weighted_native_Q_preparation=True,critic_is_approximate_not_measured_EIG=True,validation_Q_before=before,validation_Q_after=after,counts=dict(COST),initial_gate_exact_uniform=True,initial_dynamic_and_stage_distributions_bitwise_identical=True));print('PREPARATION_COMPLETE',flush=True)
