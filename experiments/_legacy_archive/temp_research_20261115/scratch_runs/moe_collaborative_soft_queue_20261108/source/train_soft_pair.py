"""Full verified10001-iteration native schedule from fresh seed101 in each arm."""
import os,sys,time,pathlib,json,signal,gc,copy,subprocess
sys.dont_write_bytecode=True
import torch
from native_common import *
from fresh_common import build_fresh,fallback_state
from soft_policy import install
STOP=False
class Halt(Exception):pass
class Table:
 def __init__(self):self.data={}
 def clear(self):self.data.clear()
 def record(self,k,v):self.data[k]=v
class Logger:
 def __init__(self,a,arm,global_start):self.algorithm=a;self.arm=arm;self.logger=self;self.tabular=Table();self.start=time.monotonic();self.global_start=global_start;self.counts=dict(actor=0,critic=0,posterior=0);self.gradient={}
 def push_prefix(self,*args):pass
 def pop_prefix(self,*args):pass
 def log(self,*args):pass
 def dump_all(self):
  a=self.algorithm;n=int(a.update);elapsed=time.monotonic()-self.start;assert self.counts==dict(actor=n,critic=5*n,posterior=150*n)
  if n==1 or n%50==0 or n==10001:
   record=dict(arm=self.arm,native_iterations=n,target_iterations=10001,native_optimizer_steps=self.counts,new_training_episodes=20000+n*1000,simulator_observations=(20000+n*1000)*30,training_prior_contrasts=0,seconds=elapsed,whole_pair_seconds=time.monotonic()-self.global_start,actor_lr=a.actor_optimizer.param_groups[0]['lr'],exploration=float(a.design_noise_scale),variational_reward=self.tabular.data.get('Reward/MeanReward'),gradient=self.gradient,gpu_peak_bytes=torch.cuda.max_memory_allocated())
   record.update(tasks=dict(steps=a._soft_tasks.steps,hypothesis_density_calls=a._soft_tasks.calls,host_seconds=a._soft_tasks.seconds,last=a._soft_tasks.last))
   put(O/'TRAINING_LOGS'/f'{self.arm}_status.json',record)
   with (O/'TRAINING_LOGS'/f'{self.arm}_metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
   print(json.dumps(record),flush=True)
  if n%1000==0:
   torch.save(dict(actor_state=a.actor_net.state_dict(),native_iterations=n,checkpoint_selection='diagnostic_only_not_EIG_selected'),O/f'arms/{self.arm}/actor_iteration{n}.pt')
  if STOP or (O/'STOP_REQUESTED').exists() or time.monotonic()-self.global_start>=41400:raise Halt('Original graceful whole-pair runtime cap or stop at iteration boundary')
def posterior_digest(a):return {str(key):[digest(net.state_dict()) for net in nets] for key,nets in a.post_approx.poi_post_nets.items()}
def run_arm(arm,global_start):
 state=build_fresh('cuda');a=state['algorithm'];root=O/'arms'/arm;root.mkdir(parents=True,exist_ok=True);assert not (root/'STARTED.json').exists(),'No arm restart'
 initial=dict(actor=digest(a.actor_net.state_dict()),critic=digest(a.critic_net.state_dict()),posterior=posterior_digest(a));assert all(len(op.state)==0 for op in allopts(a));assert not hasattr(a,'_runtime_buffer') and a.update==0
 src=install(a,enabled=arm=='candidate');assert src==(O/'source/native_train_snapshot.py.txt').read_text()
 assert digest(a.actor_net.state_dict())==torch.load(O/'inputs/fresh_soft_initial.pt',map_location='cpu',weights_only=False)['actor_digest']
 if arm=='candidate':
  assert digest(a.actor_net.state_dict())==json.loads((O/'arms/control/STARTED.json').read_text())['effective_actor_digest']
  assert initial['actor']==json.loads((O/'arms/control/STARTED.json').read_text())['fresh_initial_digests']['actor'];assert initial['critic']==json.loads((O/'arms/control/STARTED.json').read_text())['fresh_initial_digests']['critic'];assert initial['posterior']==json.loads((O/'arms/control/STARTED.json').read_text())['fresh_initial_digests']['posterior']
 a.dowel=l=Logger(a,arm,global_start);a.post_approx.dowel=l;a.save_path=None
 for op,key in [(a.actor_optimizer,'actor'),(a.critic_optimizer,'critic')]+[(op,'posterior') for oo in a.post_approx.poi_post_optimizers.values() for op in oo]:
  def count(*args,key=key,**kwargs):l.counts[key]+=1
  op.register_step_post_hook(count)
 def monitor(opt,args,kwargs):
  step=int(a.update)+1;groups={'encoder':[],'router':[],'decision':[],'task_heads':[],**{f'expert{k+1}':[] for k in range(4)}}
  for name,p in a.actor_net.named_parameters():
   if p.grad is None:
    assert name.startswith('task_heads.'),name
    continue
   assert torch.isfinite(p.grad).all(),name
   group='task_heads' if name.startswith('task_heads.') else 'router' if name.startswith('router.') else 'decision' if name.startswith('decision.') else next((f'expert{k+1}' for k in range(4) if f'experts.{k}.' in name),'encoder');groups[group].append(p.grad)
  if step==1 or step%50==0 or step==10001:l.gradient=dict(sample_iteration=step,group_norms={k:norm(v) for k,v in groups.items()})
 a.actor_optimizer.register_step_pre_hook(monitor);original.restore(state['rng']);torch.cuda.reset_peak_memory_stats();put(root/'STARTED.json',dict(pid=os.getpid(),time=time.time(),initialization='original_fresh_seed101',fresh_initial_digests=initial,initial_native_counter=0,initial_replay_imported=False,initial_optimizer_state_entries=[len(op.state) for op in allopts(a)],initial_actor_lr=a.actor_optimizer.param_groups[0]['lr'],initial_actor_Adam_steps=[],independent_expert_digests=[digest(h.state_dict()) for h in a.actor_net.experts],effective_actor_digest=digest(a.actor_net.state_dict()),source_manifest_sha256=sha(O/'PRETRAIN_MANIFEST.json')));status='COMPLETE'
 try:original.train_native(a,l,state['profile'],target=10001)
 except Halt:status='INCOMPLETE_RUNTIME_OR_STOP'
 except Exception as exc:
  import traceback
  traceback.print_exc();put(root/'ERROR.json',dict(type=type(exc).__name__,message=str(exc),native_iterations=int(a.update)));status='FAILED'
 for op in allopts(a):op._optimizer_step_pre_hooks.clear();op._optimizer_step_post_hooks.clear()
 torch.save(dict(actor_state=a.actor_net.state_dict(),method='collaborative_soft_distinct_tasks_native_AC' if arm=='candidate' else 'collaborative_soft_no_task_gradients_native_AC',native_iterations=int(a.update),fresh_seed=101),root/'actor.pt')
 with (root/'runtime.pkl').open('wb') as f:cloudpickle.dump(dict(algorithm=a,rng=original.rng(),logger=l,method=arm,initialization='fresh_seed101'),f,protocol=5)

 result=dict(status=status,arm=arm,native_iterations=int(a.update),full_native_schedule_completed=int(a.update)==10001,native_optimizer_steps=l.counts,new_training_episodes=20000+int(a.update)*1000,simulator_observations=(20000+int(a.update)*1000)*30,training_prior_contrasts=0,seconds=time.monotonic()-l.start,gpu_peak_bytes=torch.cuda.max_memory_allocated(),actor_sha256=sha(root/'actor.pt'),runtime_sha256=sha(root/'runtime.pkl'),actor_trainable_parameters=sum(p.numel() for p in a.actor_net.parameters()),critic_trainable_parameters=sum(p.numel() for p in a.critic_net.parameters()),deployment_parameters=sum(p.numel() for name,p in a.actor_net.named_parameters() if not name.startswith('task_heads.')),task_head_parameters=sum(p.numel() for p in a.actor_net.task_heads.parameters()),differentiated_tasks=arm=='candidate',auxiliary_updates_computed=a._soft_tasks.steps,auxiliary_hypothesis_density_calls=a._soft_tasks.calls,auxiliary_host_seconds=a._soft_tasks.seconds,use_PCE=False,discount=.9,initialization='fresh_seed101',trained_parent_Adam_or_replay_imported=False)
 put(root/'RESULT.json',result);print('TRAINING_END',json.dumps(result),flush=True)
 if status=='COMPLETE':assert a.update==10001 and l.counts==dict(actor=10001,critic=50005,posterior=1500150)
 return status

def run_pair(global_start):
 assert not (O/'evidence/TRAINING_STARTED.json').exists(),'One full pair only';m=json.loads((O/'PRETRAIN_MANIFEST.json').read_text())
 for p,h in m['files'].items():assert sha(O/p)==h,p
 assert json.loads((O/'evidence/SOFT_TEST_RESULTS.json').read_text())['status']=='PASS';assert json.loads((O/'evidence/NATIVE_LOOP_TEST.json').read_text())['status']=='PASS';put(O/'evidence/TRAINING_STARTED.json',dict(pid=os.getpid(),time=time.time(),initialization='fresh_seed101',GPU=torch.cuda.get_device_name(),CPU_threads=torch.get_num_threads(),manifest_sha256=sha(O/'PRETRAIN_MANIFEST.json')));signal.signal(signal.SIGTERM,lambda *args:globals().__setitem__('STOP',True));statuses={}
 for arm in ['control','candidate']:
  statuses[arm]=run_arm(arm,global_start);gc.collect();torch.cuda.empty_cache()
  if statuses[arm]!='COMPLETE':break
 put(O/'evidence/PAIR_COMPLETION.json',dict(status='COMPLETE' if len(statuses)==2 and all(v=='COMPLETE' for v in statuses.values()) else 'INCOMPLETE',arms=statuses,fresh_initialization=True));return all(v=='COMPLETE' for v in statuses.values()) and len(statuses)==2
