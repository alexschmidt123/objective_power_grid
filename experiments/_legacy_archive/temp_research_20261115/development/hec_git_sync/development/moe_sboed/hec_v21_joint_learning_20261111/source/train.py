import gc,signal,resource
from common import *
from joint_policy import install
STOP=False
class Halt(Exception):pass
class Logger:
 def __init__(self,a,arm,start):self.algorithm=a;self.arm=arm;self.start=time.monotonic();self.cpu_start=time.process_time();self.global_start=start;self.logger=self;self.tabular=self;self.counts=dict(actor=0,critic=0,posterior=0)
 def clear(self):pass
 def record(self,*args):pass
 def log(self,*args):pass
 def push_prefix(self,*args):pass
 def pop_prefix(self,*args):pass
 def dump_all(self):
  a=self.algorithm;n=int(a.update)-16001;assert self.counts==dict(actor=n,critic=5*n,posterior=150*n);assert self.simulator==dict(episodes=n*1000,observations=n*30000,batch_calls=n*30)
  if n==1 or n%50==0 or n==2000:
   with torch.no_grad():
    x=a.form_actor_input(12,a.ds_hist[:64,:12],a.ys_hist[:64,:12]);_,info=a.actor_net(x,collect=True);g=info['gates'];entropy=float(-(g*g.clamp_min(1e-30).log()).sum(-1).mean())
   groups={}
   for name,p in a.actor_net.named_parameters():
    if p.grad is not None:assert torch.isfinite(p.grad).all(),name
    groups.setdefault(name.split('.')[0],[]).append(p.grad)
   row=dict(arm=self.arm,updates=n,target=2000,native_iteration=int(a.update),seconds=time.monotonic()-self.start,driver_seconds=time.monotonic()-self.global_start,optimizer_steps=self.counts.copy(),actual_simulator_counts=self.simulator.copy(),gradient_norms={k:norm(v) for k,v in groups.items()},gate_entropy=entropy,gate_means=g.mean(0).tolist(),actor_lr=a.actor_optimizer.param_groups[0]['lr'],exploration=float(a.design_noise_scale),discount=1.,main_auxiliary_loss=0,GPU_peak_bytes=torch.cuda.max_memory_allocated());put(O/f'TRAINING_LOGS/{self.arm}_status.json',row)
   with (O/f'TRAINING_LOGS/{self.arm}_metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   print('TRAINING',json.dumps(row),flush=True)
  if n%500==0:torch.save(dict(actor_state=a.actor_net.state_dict(),method=self.arm,native_iteration=int(a.update)),O/f'arms/{self.arm}/actor_update{n}.pt')
  if STOP or (O/'STOP_REQUESTED').exists() or time.monotonic()-self.global_start>=CFG['training_grace_seconds']:raise Halt('Predeclared stopping rule')

def instrument(a):
 m,asses=a.m_f,a.asses;counts=dict(episodes=0,observations=0,batch_calls=0)
 def model(stage,theta,*args,**kwargs):
  result=m(stage,theta,*args,**kwargs);counts['observations']+=len(theta);counts['batch_calls']+=1;return result
 def assessor(*args,**kwargs):
  result=asses(*args,**kwargs);counts['episodes']+=int(kwargs.get('n_traj',args[0] if args else 10000));return result
 a.m_f=model;a.asses=assessor;return (m,asses),counts

def arm_run(arm,start):
 state=restore();a=state['algorithm'];migration=install(a,arm=='candidate');normalize_native_post(a);a.dowel=l=Logger(a,arm,start);a.post_approx.dowel=l;a.save_path=None;original_sim,l.simulator=instrument(a);root=O/'arms'/arm;root.mkdir(parents=True);initial=digest({k:v for k,v in a.actor_net.state_dict().items() if not k.startswith(('contribution_projection.','relevance.'))});frozen=digest(dict(a.actor_net.named_buffers()))
 if arm=='candidate':assert initial==json.loads((O/'arms/continuation/STARTED.json').read_text())['initial_inherited_digest']
 for op,key in [(a.actor_optimizer,'actor'),(a.critic_optimizer,'critic')]+[(op,'posterior') for oo in a.post_approx.poi_post_optimizers.values() for op in oo]:
  def count(*args,key=key,**kwargs):l.counts[key]+=1
  op.register_step_post_hook(count)
 def finite_guard(*args):
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.actor_net.parameters()),'Nonfinite actor gradient before Adam step'
 a.actor_optimizer.register_step_pre_hook(finite_guard);original.restore(state['rng']);torch.cuda.reset_peak_memory_stats();COST.clear();put(root/'STARTED.json',dict(time=time.time(),initial_inherited_digest=initial,parent_native_iteration=16001,initial_replay_size=len(a._runtime_buffer['params']),migration=migration));init=O/'initial';init.mkdir(exist_ok=True);torch.save(dict(actor_state=a.actor_net.state_dict(),method=arm,native_iteration=16001),init/f'{arm}.pt');status='COMPLETE'
 try:
  profile=json.loads((P/'frozen_protocol.json').read_text())['profile'];profile['discount']=1.;original.train_native(a,l,profile,target=18001)
 except Halt:status='INCOMPLETE_STOP_OR_RUNTIME'
 except Exception as exc:
  import traceback
  traceback.print_exc();status='FAILED';put(root/'ERROR.json',dict(type=type(exc).__name__,message=str(exc),native_iteration=int(a.update)))
 a.m_f,a.asses=original_sim
 for op in allopts(a):op._optimizer_step_pre_hooks.clear();op._optimizer_step_post_hooks.clear()
 assert digest(dict(a.actor_net.named_buffers()))==frozen,'Frozen competency/inference buffers changed';torch.save(dict(actor_state=a.actor_net.state_dict(),method=arm,native_iteration=int(a.update)),root/'actor.pt')
 with (root/'runtime.pkl').open('wb') as f:cloudpickle.dump(dict(algorithm=a,rng=original.rng(),method=CFG['method'],arm=arm),f,protocol=5)
 n=int(a.update)-16001;result=dict(status=status,arm=arm,updates=n,seconds=time.monotonic()-l.start,seconds_per_update=(time.monotonic()-l.start)/max(1,n),process_CPU_seconds=time.process_time()-l.cpu_start,process_max_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,optimizer_steps=l.counts,actual_simulator_counts=l.simulator,model_queries=dict(COST),parameters=sum(p.numel() for p in a.actor_net.parameters()),frozen_buffer_elements=sum(v.numel() for v in a.actor_net.buffers()),GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),actor_sha256=sha(root/'actor.pt'),runtime_sha256=sha(root/'runtime.pkl'),all_frozen_buffers_unchanged=True);put(root/'RESULT.json',result);print('ARM_END',json.dumps(result),flush=True)
 if status=='COMPLETE':assert n==2000 and l.counts==dict(actor=2000,critic=10000,posterior=300000)
 return status

def run_pair(start):
 assert not (O/'evidence/TRAINING_STARTED.json').exists();signal.signal(signal.SIGTERM,lambda *args:globals().__setitem__('STOP',True));put(O/'evidence/TRAINING_STARTED.json',dict(pid=os.getpid(),time=time.time(),candidate_and_matched_continuation=True));results={}
 for arm in ['continuation','candidate']:
  results[arm]=arm_run(arm,start);gc.collect();torch.cuda.empty_cache()
  if results[arm]!='COMPLETE':break
 complete=len(results)==2 and all(v=='COMPLETE' for v in results.values());put(O/'evidence/PAIR_COMPLETION.json',dict(status='COMPLETE' if complete else 'INCOMPLETE',arms=results));return complete
