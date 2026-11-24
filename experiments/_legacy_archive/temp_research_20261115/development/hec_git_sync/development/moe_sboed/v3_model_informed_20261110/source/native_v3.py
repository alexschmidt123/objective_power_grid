import inspect,textwrap,functools,copy,torch
from common import *
from v3_policy import V3Actor
from src.research.vendor.vsoed.pg_vsoed import PGvsOED
import src.research.vendor.vsoed.pg_vsoed as native_module

def sample(a,stage,ds,ys,xps_hist=None,return_all_stages=False,use_target=False):
 assert not return_all_stages and not a.use_encoder
 model=a.actor_target_net if use_target else a.actor_net;x=a.form_actor_input(stage,ds,ys);out=[]
 for xx in x.split(CFG['actor_chunk']):
  u=torch.rand(len(xx),device=xx.device,generator=a._v3_generator);e=torch.randn(len(xx),2,device=xx.device,generator=a._v3_generator);ac,info=model.draw(xx,u,e);a._v3_selections+=torch.bincount(info['selected'],minlength=4);out.append(ac)
 return torch.cat(out).detach()
def expected(a,stage,ds,ys):return a.actor_target_net.target_value(a.form_actor_input(stage,ds,ys),a.critic_target_net,a._v3_generator)
def loss(a,x):return a.actor_net.actor_loss(x,a.critic_net,a._v3_generator)

def install(a,stage_only,prepared=None):
 old=a.actor_net;target=a.actor_target_net;enc_ids={id(p) for p in old.history.parameters()};old_state={p:v for p,v in a.actor_optimizer.state.items() if id(p) in enc_ids};actor=V3Actor(old,stage_only).cuda();tgt=V3Actor(target,stage_only).cuda();tgt.reference=actor.reference
 if prepared is not None:
  cp=torch.load(prepared,map_location='cuda',weights_only=False)['actor_state'];actor.load_state_dict(cp);new={k:v for k,v in cp.items() if not k.startswith(('encoder.','reference.'))};tgt.load_state_dict(new,strict=False)
 a.actor_net=actor;a.actor_target_net=tgt;a.actor_optimizer.param_groups[0]['params']=list(actor.parameters());a.actor_optimizer.state.clear();a.actor_optimizer.state.update(old_state);assert all(p in a.actor_optimizer.state for p in actor.encoder.parameters());assert len({id(p) for p in actor.parameters()}&enc_ids)==len(enc_ids)
 a.design_noise_scale=0.;a._v3_generator=torch.Generator(device='cuda').manual_seed(CFG['route_seed']);a._v3_selections=torch.zeros(4,device='cuda',dtype=torch.long);a.actor=functools.partial(sample,a);a.actor_target=functools.partial(sample,a,use_target=True);a.v3_target=functools.partial(expected,a);a.v3_loss=functools.partial(loss,a)
 src=inspect.getsource(PGvsOED.train);old_actor='''                designs = self.actor_net(X_actor)
                X = torch.cat([X_actor, designs], dim=-1)
                output = -self.critic_net(X).mean()''';assert src.count(old_actor)==1;src=src.replace(old_actor,'                output = self.v3_loss(X_actor)')
 old_target='''                            ds_next = self.actor_target(k + 1,
                                self.ds_hist[:, :k + 1],
                                self.ys_hist[:, :k + 1],
                                None)
                            next_action_value = self.critic_target(k + 1,
                                self.ds_hist[:, :k + 1],
                                self.ys_hist[:, :k + 1],
                                None,
                                ds_next)''';old_target=old_target.replace('k + 1,\n','k + 1, \n');assert src.count(old_target)==1;src=src.replace(old_target,'''                            next_action_value = self.v3_target(k + 1,
                                self.ds_hist[:, :k + 1], self.ys_hist[:, :k + 1])''');ns=dict(native_module.__dict__);exec(compile(textwrap.dedent(src),'<native_v3_two_policy_expectations>','exec'),ns);a.train=functools.partial(ns['train'],a);normalize_native_post(a)
 return dict(source_sha256=hashlib.sha256(src.encode()).hexdigest(),source=src,actor_Adam_encoder_preserved=True,new_expert_router_moments_fresh=True,native_critic_posterior_replay_preserved=True,additional_gaussian_exploration_disabled_for_consistent_stochastic_policy=True)

def instrument_simulator(a):
 """Count actual completed native assessor episodes and physical model calls."""
 fn,asses=a.m_f,a.asses;counts=dict(episodes=0,observations=0,batch_calls=0)
 def model(stage,theta,*args,**kwargs):
  result=fn(stage,theta,*args,**kwargs);counts['observations']+=len(theta);counts['batch_calls']+=1;return result
 def assess(*args,**kwargs):
  result=asses(*args,**kwargs);counts['episodes']+=int(kwargs.get('n_traj',args[0] if args else 10000));return result
 a.m_f=model;a.asses=assess;return (fn,asses),counts
