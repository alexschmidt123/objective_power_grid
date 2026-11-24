"""Shared supervised preparation; matched B=200,C=1800 native EIG updates."""
import os,time,gc,signal
from native_common import *
from restore_sg import restore
from migration import migrate
from hec_policy import COST
from normalized_reward import normalize_native_post
from prepare_specialists import prepare
CFG=json.loads((O/'CONFIG_LOCKED.json').read_text());STOP=False
class Halt(Exception):pass
class Table:
 def __init__(self):self.data={}
 def clear(self):self.data.clear()
 def record(self,k,v):self.data[k]=v
class Logger:
 def __init__(self,a,arm,start):self.algorithm=a;self.arm=arm;self.start=time.monotonic();self.global_start=start;self.logger=self;self.tabular=Table();self.counts=dict(actor=0,critic=0,posterior=0);self.gradient={}
 def push_prefix(self,*args):pass
 def pop_prefix(self,*args):pass
 def log(self,*args):pass
 def dump_all(self):
  a=self.algorithm;n=int(a.update)-12001;assert self.counts==dict(actor=n,critic=5*n,posterior=150*n)
  if n==200:a.actor_net.set_phase('C')
  if n==1 or n%50==0 or n==2000:
   row=dict(arm=self.arm,continuation_updates=n,target_updates=2000,phase='B' if n<=200 else 'C',native_iteration=int(a.update),native_optimizer_steps=dict(self.counts),new_simulator_episodes=1000*n,new_simulator_observations=30000*n,seconds=time.monotonic()-self.start,whole_pair_seconds=time.monotonic()-self.global_start,actor_lr=a.actor_optimizer.param_groups[0]['lr'],exploration=float(a.design_noise_scale),discount=1.,auxiliary_loss=0,gradient=self.gradient,gpu_peak_bytes=torch.cuda.max_memory_allocated());put(O/f'TRAINING_LOGS/{self.arm}_status.json',row)
   with (O/f'TRAINING_LOGS/{self.arm}_metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   print(json.dumps(row),flush=True)
  if n%500==0:torch.save(dict(actor_state=a.actor_net.state_dict(),dynamic=self.arm=='candidate',native_iteration=int(a.update)),O/f'arms/{self.arm}/actor_update{n}.pt')
  if STOP or (O/'STOP_REQUESTED').exists() or time.monotonic()-self.global_start>=21600:raise Halt('Fixed stopping rule')
def run_arm(arm,start):
 state=restore();a=state['algorithm'];migration=migrate(a,arm=='candidate');normalize_native_post(a);a.dowel=l=Logger(a,arm,start);a.post_approx.dowel=l;a.save_path=None;root=O/'arms'/arm;root.mkdir(parents=True,exist_ok=True);assert not (root/'STARTED.json').exists();initial=digest(a.actor_net.state_dict());frozen={k:sha0 for k,sha0 in []};buffer_before=digest(dict(a.actor_net.named_buffers()));COST.clear()
 if arm=='candidate':assert initial==json.loads((O/'arms/control/STARTED.json').read_text())['initial_actor_digest']
 for op,key in [(a.actor_optimizer,'actor'),(a.critic_optimizer,'critic')]+[(op,'posterior') for oo in a.post_approx.poi_post_optimizers.values() for op in oo]:
  def count(*args,key=key,**kwargs):l.counts[key]+=1
  op.register_step_post_hook(count)
 def monitor(opt,args,kwargs):
  n=int(a.update)-12000
  if n==1 or n%50==0 or n==2000:
   groups={}
   for name,p in a.actor_net.named_parameters():
    if p.grad is not None:assert torch.isfinite(p.grad).all(),name
    groups.setdefault(name.split('.')[0],[]).append(p.grad)
   l.gradient={k:norm(v) for k,v in groups.items()}
 a.actor_optimizer.register_step_pre_hook(monitor);original.restore(state['rng']);torch.cuda.reset_peak_memory_stats();put(root/'STARTED.json',dict(time=time.time(),pid=os.getpid(),initial_actor_digest=initial,migration=migration,parent_update=12001,initial_actor_lr=a.actor_optimizer.param_groups[0]['lr'],initial_exploration=float(a.design_noise_scale),initial_replay_size=len(a._runtime_buffer['params']),history_Adam_preserved=True));status='COMPLETE'
 try:
  profile=json.loads((P/'frozen_protocol.json').read_text())['profile'];profile['discount']=1.;original.train_native(a,l,profile,target=14001)
 except Halt:status='INCOMPLETE_STOP_OR_RUNTIME'
 except Exception as exc:
  import traceback
  traceback.print_exc();put(root/'ERROR.json',dict(type=type(exc).__name__,message=str(exc),native_iteration=int(a.update)));status='FAILED'
 for op in allopts(a):op._optimizer_step_pre_hooks.clear();op._optimizer_step_post_hooks.clear()
 assert digest(dict(a.actor_net.named_buffers()))==buffer_before,'Frozen inference/SG anchors changed';torch.save(dict(actor_state=a.actor_net.state_dict(),dynamic=arm=='candidate',native_iteration=int(a.update)),root/'actor.pt')
 with (root/'runtime.pkl').open('wb') as f:cloudpickle.dump(dict(algorithm=a,rng=original.rng(),method='HEC',arm=arm),f,protocol=5)
 n=int(a.update)-12001;result=dict(status=status,arm=arm,continuation_updates=n,native_iteration=int(a.update),native_optimizer_steps=l.counts,new_simulator_episodes=n*1000,new_simulator_observations=n*30000,seconds=time.monotonic()-l.start,gpu_peak_bytes=torch.cuda.max_memory_allocated(),actor_sha256=sha(root/'actor.pt'),runtime_sha256=sha(root/'runtime.pkl'),model_computation_calls=dict(COST),parameters=sum(p.numel() for p in a.actor_net.parameters()),deployed_buffer_weights=sum(v.numel() for v in a.actor_net.buffers()),phase_B_updates=min(n,200),phase_C_updates=max(n-200,0),auxiliary_native_updates=0,frozen_model_information_unchanged=True,history_Adam_and_native_critic_posterior_replay_schedules_preserved=True);put(root/'RESULT.json',result);print('ARM_END',json.dumps(result),flush=True)
 if status=='COMPLETE':assert n==2000 and l.counts==dict(actor=2000,critic=10000,posterior=300000)
 return status

def run_pair(start):
 assert not (O/'evidence/TRAINING_STARTED.json').exists();manifest=json.loads((O/'PRETRAIN_MANIFEST.json').read_text())
 for n,h in manifest['files'].items():assert sha(O/n)==h,n
 assert json.loads((O/'evidence/CHECKS.json').read_text())['status']=='PASS';signal.signal(signal.SIGTERM,lambda *args:globals().__setitem__('STOP',True));put(O/'evidence/TRAINING_STARTED.json',dict(time=time.time(),pid=os.getpid(),GPU=torch.cuda.get_device_name(),one_pair=True));prepare(start);gc.collect();torch.cuda.empty_cache();statuses={}
 for arm in ['control','candidate']:
  statuses[arm]=run_arm(arm,start);gc.collect();torch.cuda.empty_cache()
  if statuses[arm]!='COMPLETE':break
 ok=len(statuses)==2 and all(v=='COMPLETE' for v in statuses.values());put(O/'evidence/PAIR_COMPLETION.json',dict(status='COMPLETE' if ok else 'INCOMPLETE',arms=statuses));return ok
