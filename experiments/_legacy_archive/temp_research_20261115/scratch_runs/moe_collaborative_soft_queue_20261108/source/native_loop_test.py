"""Exercise fresh native replay/TD plumbing;all optimizer steps intercepted."""
import os,sys,time,warnings
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
import torch
from native_common import *
from fresh_common import build_fresh,fallback_state
from soft_policy import install
class Log:
 def __init__(self):self.logger=self;self.tabular=self;self.data={}
 def clear(self):self.data.clear()
 def record(self,k,v):self.data[k]=v
 def log(self,*args):pass
 def push_prefix(self,*args):pass
 def pop_prefix(self,*args):pass
 def dump_all(self):pass
torch.set_num_threads(2);beg=time.monotonic();state=build_fresh();a=state['algorithm'];install(a,enabled=False);actor0=digest(a.actor_net.state_dict());critic0=digest(a.critic_net.state_dict());count=[0]
def no_step(*args,**kwargs):count[0]+=1
for op in allopts(a):op.step=no_step
a.dowel=l=Log();a.post_approx.dowel=l
with warnings.catch_warnings():
 warnings.simplefilter('ignore')
 a.train(n_update=1,n_newtraj=4,n_batch=4,n_buffer_init=8,n_buffer_max=16,buffer_device='cpu',discount=.9,n_critic_update=1,n_post_approx_update=1,actor_optimizer=a.actor_optimizer,critic_optimizer=a.critic_optimizer,actor_lr_scheduler=a.actor_lr_scheduler,critic_lr_scheduler=a.critic_lr_scheduler,lr_target=0.,design_noise_scale=.5,design_noise_decay=.9999,on_policy=False,use_PCE=False,transition=0,frozen=-1,dowel=l,save_path=None)
assert a.update==1 and a.ds_hist.shape==(4,30,2);assert a._runtime_buffer['params'].shape==(12,4);assert digest(a.actor_net.state_dict())==actor0 and digest(a.critic_net.state_dict())==critic0;assert all(len(op.state)==0 for op in allopts(a));assert count[0]==32
put(O/'evidence/NATIVE_LOOP_TEST.json',dict(status='PASS',optimizer_updates=0,intercepted_step_calls=count[0],toy_episodes=12,toy_simulator_observations=360,native_replay_and_TD_shapes_valid=True,online_actor_critic_parameters_unchanged=True,GPU_used=False,seconds=time.monotonic()-beg));print('NATIVE_LOOP_NO_UPDATE_TEST_PASS',flush=True)
