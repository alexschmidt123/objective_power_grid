"""One restored native-loop plumbing fixture;alloptimizerstepsintercepted."""
import sys,time,warnings
sys.dont_write_bytecode=True
from native_common import *
from restore_parent import restore
from sg_policy import migrate
from normalized_reward import normalize_native_post
class Log:
 def __init__(self):self.logger=self;self.tabular=self
 def clear(self):pass
 def record(self,*args):pass
 def log(self,*args):pass
 def push_prefix(self,*args):pass
 def pop_prefix(self,*args):pass
 def dump_all(self):pass
original.setup();leases=original.acquire();state=restore('cuda');a=state['algorithm'];migrate(a,True);normalize_native_post(a);a.n_newtraj=4;a.n_batch=10000;a.n_post_approx_update=1;a.discount=1.;a.dowel=l=Log();a.post_approx.dowel=l;a.save_path=None;before=digest(a.actor_net.state_dict());critic_before=digest(a.critic_net.state_dict());calls=[0];sim=[0];native_asses=a.asses

def asses(*args,**kwargs):
 sim[0]+=int(kwargs.get('n_traj',args[0] if args else 0));return native_asses(*args,**kwargs)
a.asses=asses
def no_step(*args,**kwargs):calls[0]+=1
for op in allopts(a):op.step=no_step
with torch.device('cuda'),warnings.catch_warnings():
 warnings.simplefilter('ignore');a.train(n_update=10002,n_newtraj=4,n_batch=10000,n_buffer_init=20000,n_buffer_max=1000000,buffer_device='cuda',discount=1.,n_critic_update=1,n_post_approx_update=1,actor_optimizer=a.actor_optimizer,critic_optimizer=a.critic_optimizer,actor_lr_scheduler=a.actor_lr_scheduler,critic_lr_scheduler=a.critic_lr_scheduler,lr_target=.1,design_noise_scale=a.design_noise_scale,design_noise_decay=.9999,on_policy=False,use_PCE=False,transition=0,frozen=-1,dowel=l,save_path=None)
assert calls[0]==32 and sim[0]==4 and a.update==10002 and len(a._runtime_buffer['params'])==1000000;assert digest(a.actor_net.state_dict())==before and digest(a.critic_net.state_dict())==critic_before
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in a.actor_net.parameters() if p.requires_grad)
put(O/'evidence/NATIVE_LOOP_CHECK.json',dict(status='PASS',scientific_optimizer_updates=0,intercepted_step_calls=32,new_fixture_episodes=4,extra_initial_replay_episodes=0,replay_size=1000000,actor_and_critic_weights_unchanged=True,full_native_batch=10000,finite_actor_gradients=True,GPU_peak_bytes=torch.cuda.max_memory_allocated()));print('SG_NATIVE_LOOP_ZERO_UPDATE_PASS')
