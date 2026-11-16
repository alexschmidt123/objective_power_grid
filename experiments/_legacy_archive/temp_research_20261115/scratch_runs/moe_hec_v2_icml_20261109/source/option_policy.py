"""H=3 complete-design options with an immutable original-policy fallback."""
import pathlib,sys,copy,inspect,textwrap,functools,math
import torch
from torch import nn
R=pathlib.Path.home()/'Documents/objective_power_grid_temp/scratch_runs';P=R/'moe_development_migration_20261105/imported_ubuntu_tree/scratch_runs/20261104T161647Z_direct_head_moe_vsoed_seed101';sys.path.insert(0,str(P/'source'))
from src.research.vendor.vsoed.pg_vsoed import PGvsOED
import src.research.vendor.vsoed.pg_vsoed as native_module
H=3
class ContextLayer(nn.Module):
 """Preserve the exact old Linear and Adam state;add a zero context adapter."""
 def __init__(self,base):
  super().__init__();self.base=base
  with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []):self.context=nn.Linear(5,base.out_features,bias=False,device=base.weight.device,dtype=base.weight.dtype)
  nn.init.zeros_(self.context.weight)
 def forward(self,x):return self.base(torch.cat([x[...,:117],x[...,-2:]],-1))+self.context(x[...,117:122])
def gate_from_z(actor,z):
 tail=actor.net[4];specialist=tail.gate(z).log_softmax(-1);f=tail.fallback_gate(z);return torch.cat([(torch.nn.functional.logsigmoid(-f)+specialist).exp(),torch.nn.functional.logsigmoid(f).exp()],-1)
def parts(actor,x):
 z=actor.net[:4](x);return gate_from_z(actor,z),torch.stack([-4+8*h(z).sigmoid() for h in actor.net[4].heads],-2)
def all_actions(a,x,target=False):
 actor=a.actor_target_net if target else a.actor_net;g,expert=parts(actor,x)
 with torch.no_grad():outside=a._frozen_fallback_actor(x)
 return g,torch.cat([expert,outside.unsqueeze(-2)],-2)
def selected_action(a,x,option,target=False):
 actor=a.actor_target_net if target else a.actor_net;_,expert=parts(actor,x);k=option.long().clamp(max=3);chosen=expert.gather(-2,k[...,None,None].expand(*k.shape,1,2)).squeeze(-2)
 with torch.no_grad():fallback=a._frozen_fallback_actor(x)
 return torch.where((option<4).unsqueeze(-1),chosen,fallback)
def critic_input(x,option,action):return torch.cat([x,torch.nn.functional.one_hot(option.long(),5).to(x.dtype),action],-1)
def sample_designs(a,stage,ds_hist,ys_hist,xps_hist=None,return_all_stages=False,use_target=False):
 assert not return_all_stages and not use_target and not a.use_encoder
 b=len(ds_hist);x=a.form_actor_input(stage,ds_hist,ys_hist)
 if stage==0:
  a._rollout_options=torch.empty(b,30,device=x.device,dtype=torch.int8);a._active_option=None
 if stage%H==0:
  g,_=parts(a.actor_net,x);behavior=.95*g+.05/5;a._active_option=torch.multinomial(behavior.detach(),1,generator=a._route_generator).squeeze(-1);a._boundary_counts+=torch.bincount(a._active_option,minlength=5)
 a._rollout_options[:,stage]=a._active_option.to(torch.int8)
 return selected_action(a,x,a._active_option).detach()
def replay_critic_input(a,stage,ds_hist,ys_hist,action,option):return critic_input(a.form_actor_input(stage,ds_hist,ys_hist),option,action)
def target_value(a,stage,ds_hist,ys_hist,active_option):
 x=a.form_actor_input(stage,ds_hist,ys_hist)
 if stage%H==0:
  g,actions=all_actions(a,x,True);xx=x.unsqueeze(-2).expand(*x.shape[:-1],5,117);options=torch.arange(5,device=x.device).expand(*x.shape[:-1],5);q=a.critic_target_net(critic_input(xx,options,actions))[...,0];return (g*q).sum(-1)
 action=selected_action(a,x,active_option,True);return a.critic_target_net(critic_input(x,active_option,action))[...,0]
def actor_loss(a,x,options):
 B,T=x.shape[:2];z=a.actor_net.net[:4](x);tail=a.actor_net.net[4];expert=torch.stack([-4+8*h(z).sigmoid() for h in tail.heads],-2);k=options.long().clamp(max=3);action=expert.gather(-2,k[...,None,None].expand(*k.shape,1,2)).squeeze(-2);valid=options<4
 with torch.no_grad():fallback=a._frozen_fallback_actor(x)
 action=torch.where(valid.unsqueeze(-1),action,fallback);q=a.critic_net(critic_input(x,options,action))[...,0];expert_loss=-(q*valid).sum()/(B*T)
 xb=x[:,::H];zb=z[:,::H];g=gate_from_z(a.actor_net,zb)
 with torch.no_grad():
  proposals=expert[:,::H].detach();outside=a._frozen_fallback_actor(xb);actions=torch.cat([proposals,outside.unsqueeze(-2)],-2);xx=xb.unsqueeze(-2).expand(*xb.shape[:-1],5,117);oo=torch.arange(5,device=x.device).expand(*xb.shape[:-1],5);Q=a.critic_net(critic_input(xx,oo,actions))[...,0];baseline=(g.detach()*Q).sum(-1,keepdim=True);adv=Q-baseline
 router_loss=-(g*adv).sum()/(B*T);loss=expert_loss+router_loss
 a._last_option=dict(expert_loss=float(expert_loss.detach()),router_loss=float(router_loss.detach()),gate_means=g.detach().mean((0,1)).cpu().tolist(),gate_entropy=float(-(g.detach()*g.detach().clamp_min(1e-30).log()).sum(-1).mean()),replay_option_counts=torch.bincount(options.long().flatten(),minlength=5).cpu().tolist(),option_Q_spread=float(Q.std(-1).mean()))
 return loss

def install(a,seed,fallback_state=None):
 assert not a.use_encoder and a.backend_net is None and not a.use_PCE
 a._frozen_fallback_actor=copy.deepcopy(a.actor_net).eval().requires_grad_(False)
 if fallback_state is not None:a._frozen_fallback_actor.load_state_dict(fallback_state,strict=True)
 device=next(a.actor_net.parameters()).device
 for actor in [a.actor_net,a.actor_target_net]:
  with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []):layer=nn.Linear(256,1,device=device)
  nn.init.zeros_(layer.weight);nn.init.constant_(layer.bias,math.log(.2/.8));actor.net[4].fallback_gate=layer
 a.actor_optimizer.param_groups[0]['params'].extend(list(a.actor_net.net[4].fallback_gate.parameters()))
 for critic in [a.critic_net,a.critic_target_net]:critic.net[0]=ContextLayer(critic.net[0])
 a.critic_optimizer.param_groups[0]['params'].append(a.critic_net.net[0].context.weight)
 a.critic_input_dimn+=5;a._route_generator=torch.Generator(device=device).manual_seed(seed);a._boundary_counts=torch.zeros(5,dtype=torch.long,device=device);a._last_option={}
 a.actor=functools.partial(sample_designs,a,return_all_stages=False,use_target=False);a.option_critic_input=functools.partial(replay_critic_input,a);a.option_target_value=functools.partial(target_value,a);a.option_actor_loss=functools.partial(actor_loss,a)
 # Historical soft-policy actions are valid off-policy transition data. Attach
 # fallback context as an artificial behavior-controller label,not a claim
 # that the old histories were generated by the frozen fallback policy.
 if hasattr(a,'_runtime_buffer') and 'option_hist' not in a._runtime_buffer:a._runtime_buffer['option_hist']=torch.full((len(a._runtime_buffer['params']),30),4,device=a._runtime_buffer['params'].device,dtype=torch.int8)
 src=inspect.getsource(PGvsOED.train)
 changes=[
 ("buffer['xps_hist'] = self.xps_hist.to(self.buffer_device)","buffer['xps_hist'] = self.xps_hist.to(self.buffer_device)\n            buffer['option_hist'] = self._rollout_options.to(self.buffer_device)"),
 ("buffer['xps_hist'] = torch.cat([buffer['xps_hist'][idx_left:], self.xps_hist.to(self.buffer_device)], 0)","buffer['xps_hist'] = torch.cat([buffer['xps_hist'][idx_left:], self.xps_hist.to(self.buffer_device)], 0)\n            buffer['option_hist'] = torch.cat([buffer['option_hist'][idx_left:], self._rollout_options.to(self.buffer_device)], 0)"),
 ("self.xps_hist = buffer['xps_hist'][idxs_pick].to(self.xps_hist.device)","self.xps_hist = buffer['xps_hist'][idxs_pick].to(self.xps_hist.device)\n            self._sampled_options = buffer['option_hist'][idxs_pick].to(self.ds_hist.device)"),
 ('self.n_stage, self.backend_input_dimn + self.n_design)','self.n_stage, self.backend_input_dimn + 5 + self.n_design)'),
 ('''                        X = self.form_critic_input(k,
                                                   self.ds_hist[:, :k],
                                                   self.ys_hist[:, :k],
                                                   self.ds_hist[:, k])''','''                        X = self.option_critic_input(k,
                                                   self.ds_hist[:, :k],
                                                   self.ys_hist[:, :k],
                                                   self.ds_hist[:, k],self._sampled_options[:, k])'''),
 ('''                            ds_next = self.actor_target(k + 1, 
                                self.ds_hist[:, :k + 1],
                                self.ys_hist[:, :k + 1],
                                None)
                            next_action_value = self.critic_target(k + 1, 
                                self.ds_hist[:, :k + 1],
                                self.ys_hist[:, :k + 1],
                                None,
                                ds_next)''','''                            next_action_value = self.option_target_value(k + 1,
                                self.ds_hist[:, :k + 1],self.ys_hist[:, :k + 1],
                                self._sampled_options[:, k])'''),
 ('''                X_actor = X_critic[..., :-self.n_design]
                designs = self.actor_net(X_actor)
                X = torch.cat([X_actor, designs], dim=-1)
                output = -self.critic_net(X).mean()''','''                X_actor = X_critic[..., :117]
                output = self.option_actor_loss(X_actor,self._sampled_options)''')]
 for old,new in changes:assert src.count(old)==1,old;src=src.replace(old,new)
 src=textwrap.dedent(src);namespace=dict(native_module.__dict__);exec(compile(src,'<native_H3_option_training>','exec'),namespace);a.train=functools.partial(namespace['train'],a);return src
