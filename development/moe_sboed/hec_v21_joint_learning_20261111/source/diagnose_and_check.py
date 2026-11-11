import copy,time,warnings,torch,numpy as np
from common import *
from joint_policy import JointHEC,install,contributions,decision_features
from v2_tasks import native_logpost

def refusion(m,x,info,g):
 pre=m.fusion[0](torch.cat([info['history'],(g[...,None]*info['slots'][...,:64]).flatten(1)],-1))+m.response_fusion((g[...,None]*info['slots'][...,64:]).flatten(1));d=m.fusion[2](m.fusion[1](pre)).tanh();a0=info['probes'][:,0];return a0+4*d-a0*d.abs()

def main():
 original.setup();leases=original.acquire();start=time.monotonic();torch.manual_seed(CFG['validation_seed']);state=restore();a=state['algorithm'];parent=copy.deepcopy(a.actor_net);model=JointHEC(copy.deepcopy(a.actor_net));hist=np.load(OLD/'evaluation/diagnostic_candidate_histories.npz');xx=torch.tensor(hist['X'][:64][:,[3,12,24]],device='cuda').reshape(-1,117);t=xx[:,:30].argmax(-1)
 with torch.no_grad():
  pa,pi=parent(xx,collect=True);ca,ci=model(xx,collect=True);assert torch.equal(pa,ca);assert torch.equal(pi['gates'],ci['gates']);initial_error=float((pa-ca).abs().max());c,pre,d,v,vt=contributions(parent,pi['history'],pi['slots'],pi['gates'],pi['probes'][:,0]);shead=torch.nn.functional.linear(pi['history'],parent.fusion[0].weight[:,:128],parent.fusion[0].bias);weighted=(pi['gates'][...,None]*c).sum(1);raw=parent.coordinator(torch.cat([pi['history'],pi['legacy_features'].flatten(1)],-1))+parent.response_gate(pi['responses'].flatten(1));attenuation=1-torch.tanh(raw/10).square();diag={}
  for stage in [3,12,24]:
   ix=t==stage;g=pi['gates'][ix];diag[str(stage)]=dict(gate_mean=g.mean(0).tolist(),within_stage_gate_SD=g.std(0).tolist(),entropy=float(-(g*g.log()).sum(-1).mean()),raw_logit_tanh_derivative_mean=float(attenuation[ix].mean()),decision_tangent_RMS=float(vt[ix].square().mean().sqrt()),contribution_norms=c[ix].norm(dim=-1).mean(0).tolist(),weighted_cancellation_ratio=float((weighted[ix].norm(dim=-1)/(pi['gates'][ix]*c[ix].norm(dim=-1)).sum(-1).clamp_min(1e-12)).mean()),shared_history_to_expert_norm_ratio=float((shead[ix].norm(dim=-1)/weighted[ix].norm(dim=-1).clamp_min(1e-12)).mean()),physical_bound_fraction=float((pa[ix].abs()>3.96).float().mean()))
 # Native Q gradients at identical frozen histories:diagnostic,not new-policy EIG.
 selected=[];names=[]
 for name,p in parent.named_parameters():
  if p.requires_grad and name.startswith(('commander.','coordinator.','response_gate.','fusion.','response_fusion.','region.','discriminate.','sensitivity.','sequential.','adapter2.','adapter4.')):selected.append(p);names.append(name)
 loss=-a.critic_net(torch.cat([xx,parent(xx)],-1)).mean();grads=torch.autograd.grad(loss,selected,allow_unused=True);gradient={}
 for name,g in zip(names,grads):gradient.setdefault(name.split('.')[0],[]).append(g)
 gradient={k:norm(v) for k,v in gradient.items()};model.zero_grad();loss=-a.critic_net(torch.cat([xx,model(xx)],-1)).mean();headgrad=torch.autograd.grad(loss,list(model.relevance[-1].parameters()),retain_graph=True);assert norm(headgrad)>0
 # Initial native gradients on inherited parameters must match the parent.
 mp=dict(model.named_parameters());cg=torch.autograd.grad(loss,[mp[k] for k in names],allow_unused=True)
 for old,new in zip(grads,cg):
  assert (old is None)==(new is None)
  if old is not None:torch.testing.assert_close(old,new,rtol=3e-4,atol=3e-7)
 # Exact derivative of the actual collaborative action with respect to logits.
 logits=pi['gates'].log().detach().requires_grad_(True);aa=refusion(parent,xx,pi,logits.softmax(-1));auto=torch.stack([torch.autograd.grad(aa[:,q].sum(),logits,retain_graph=True)[0] for q in [0,1]],-1);torch.testing.assert_close(auto,vt,rtol=3e-4,atol=3e-6)
 th=torch.randn(7,4,device='cuda',dtype=torch.float64);ac=torch.randn(7,2,device='cuda',dtype=torch.float64,requires_grad=True);val=mean_signal(th[:,None],ac[:,None])[:,0,0];ja=torch.autograd.grad(val.sum(),ac)[0];torch.testing.assert_close(ja,-jacobian(th[:,None],ac.detach()).reshape(7,2,2).sum(1),rtol=1e-10,atol=1e-10)
 # Nonzero paths after activating only a disposable relevance head;no optimizer steps.
 with torch.no_grad():model.relevance[-1].weight.normal_(0,.001)
 loss=-(model(xx)*xx.new_tensor([1.,-.4])).mean();model.zero_grad();loss.backward();active={}
 for name in ['history','commander','coordinator','response_gate','region','discriminate','sensitivity','sequential','adapter2','adapter4','fusion','response_fusion','contribution_projection','relevance']:
  gg=[p.grad for p in getattr(model,name).parameters() if p.requires_grad];assert all(g is None or torch.isfinite(g).all() for g in gg);active[name]=norm(gg);assert active[name]>0,name
 # Commands/new relevance influence real bounded designs,not just logits.
 with torch.no_grad():
  ac,info=model(xx,collect=True);zero=model(xx,command_override=torch.zeros(20,device='cuda'));uniform=model(xx,routing_override=torch.full((4,),.25,device='cuda'));assert float((ac-zero).abs().max())>0 and float((ac-uniform).abs().max())>0;assert bool((ac.abs()<=4).all())
  for k in range(4):assert bool((model(xx,routing_override=torch.eye(4,device='cuda')[k]).abs()<=4).all())
 ds=a._runtime_buffer['ds_hist'][:30].clone();ys=a._runtime_buffer['ys_hist'][:30].clone();tt=torch.arange(30,device='cuda');before=prefix_inputs(ds,ys,tt)
 for k in range(30):ds[k,k:]=999;ys[k,k:]=-999
 assert torch.equal(before,prefix_inputs(ds,ys,tt));put(O/'evidence/ROUTING_DIAGNOSIS.json',dict(status='COMPLETE_FROZEN_AUDIT',cases=len(xx),stages=diag,native_gradient_norms=gradient,new_router_head_native_gradient_norm=norm(headgrad),fixed_history_critic_not_EIG=True,head_compensation_and_redundancy_not_causally_proved=True))
 del parent,model;torch.cuda.empty_cache();moment=a.actor_optimizer.state[a.actor_net.history[0].weight]['exp_avg'].clone();migration=install(a,True);assert torch.equal(moment,a.actor_optimizer.state[a.actor_net.history[0].weight]['exp_avg']);a.n_newtraj=4;a.n_critic_update=1;a.n_post_approx_update=1
 class Log:
  def __init__(self):self.logger=self;self.tabular=self
  def clear(self):pass
  def record(self,*a):pass
  def log(self,*a):pass
  def push_prefix(self,*a):pass
  def pop_prefix(self,*a):pass
  def dump_all(self):pass
 l=Log();a.dowel=l;a.post_approx.dowel=l;a.save_path=None;digest0=digest(a.actor_net.state_dict());calls=[0];episodes=[0];assess=a.asses
 for op in allopts(a):
  def noop(*args,**kwargs):calls[0]+=1
  op.step=noop
 def sim(*args,**kwargs):episodes[0]+=int(kwargs.get('n_traj',args[0] if args else 0));return assess(*args,**kwargs)
 a.asses=sim
 with torch.device('cuda'),warnings.catch_warnings():
  warnings.simplefilter('ignore');a.train(n_update=16002,n_newtraj=4,n_batch=10000,n_buffer_init=20000,n_buffer_max=1000000,buffer_device='cuda',discount=1.,n_critic_update=1,n_post_approx_update=1,actor_optimizer=a.actor_optimizer,critic_optimizer=a.critic_optimizer,actor_lr_scheduler=a.actor_lr_scheduler,critic_lr_scheduler=a.critic_lr_scheduler,lr_target=.1,design_noise_scale=a.design_noise_scale,design_noise_decay=.9999,on_policy=False,use_PCE=False,transition=0,frozen=-1,dowel=l,save_path=None)
 assert calls[0]==32 and episodes[0]==4 and a.update==16002 and digest(a.actor_net.state_dict())==digest0;assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.actor_net.parameters());assert len(a._runtime_buffer['params'])==1000000
 with torch.no_grad():
  n=len(a.params);terminal=native_logpost(a,a.ds_hist,a.ys_hist,a.params,torch.full((n,),30,device='cuda'));prior=native_logpost(a,a.ds_hist,a.ys_hist,a.params,torch.zeros(n,device='cuda',dtype=torch.long));torch.testing.assert_close(a.rewards_hist.sum(-1),terminal-prior,rtol=1e-4,atol=2e-4)
 put(O/'evidence/GRADIENT_VALIDATION.json',dict(status='PASS',seconds=time.monotonic()-start,scientific_optimizer_updates=0,initial_actions_bitwise_identical=True,initial_inherited_parameter_gradients_match=True,exact_collaborative_logit_action_tangent=True,exact_design_likelihood_derivative=True,new_native_router_head_gradient_positive=True,full_gradient_paths=active,command_and_weight_action_effects=True,prefix_future_mutation_invariant=True,compatible_Adam_objects_and_moments_preserved=True,full_native_batch=10000,actor_rows=300000,intercepted_steps=calls[0],fixture_episodes=episodes[0],native_gamma1_reward_telescope=True,checkpoints_unchanged=True,peak_GPU_bytes=torch.cuda.max_memory_allocated()));print('HECV21_NATIVE_GRADIENT_CHECKS_PASS',flush=True)
if __name__=='__main__':main()
