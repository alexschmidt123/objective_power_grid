import gc,signal,copy
from common import *
from native_v3 import install,instrument_simulator
from prepare import prepare
STOP=False
class Halt(Exception):pass
class Logger:
 def __init__(self,a,arm,global_start):self.algorithm=a;self.arm=arm;self.start=time.monotonic();self.global_start=global_start;self.logger=self;self.tabular=self;self.counts=dict(actor=0,critic=0,posterior=0)
 def clear(self):pass
 def record(self,*args):pass
 def log(self,*args):pass
 def push_prefix(self,*args):pass
 def pop_prefix(self,*args):pass
 def dump_all(self):
  a=self.algorithm;n=int(a.update)-16001;assert self.counts==dict(actor=n,critic=5*n,posterior=150*n)
  assert self.simulator==dict(episodes=1000*n,observations=30000*n,batch_calls=30*n)
  if n==1 or n%50==0 or n==2000:
   with torch.no_grad():
    xx=a.form_actor_input(12,a.ds_hist[:64,:12],a.ys_hist[:64,:12]);g,mu,info=a.actor_net.parts(xx);entropy=float(-(g*g.log()).sum(-1).mean());sim=float(torch.pdist(info['proposals'].transpose(0,1).flatten(1)).mean()/math.sqrt(64))
   groups={}
   for name,p in a.actor_net.named_parameters():
    if p.grad is not None:assert torch.isfinite(p.grad).all(),name
    groups.setdefault(name.split('.')[0],[]).append(p.grad)
   row=dict(arm=self.arm,updates=n,target=2000,native_iteration=int(a.update),seconds=time.monotonic()-self.start,driver_seconds=time.monotonic()-self.global_start,optimizer_steps=self.counts.copy(),new_simulator_episodes=self.simulator['episodes'],new_simulator_observations=self.simulator['observations'],simulator_batch_calls=self.simulator['batch_calls'],gradient_norms={k:norm(v) for k,v in groups.items()},gate_entropy=entropy,gate_means=g.mean(0).tolist(),cumulative_behavior_selections=a._v3_selections.tolist(),proposal_disagreement_physical=sim,actor_lr=a.actor_optimizer.param_groups[0]['lr'],latent_sigma=CFG['latent_sigma'],external_clipped_exploration=0,main_auxiliary_loss=0,gpu_peak_bytes=torch.cuda.max_memory_allocated());put(O/f'TRAINING_LOGS/{self.arm}_status.json',row)
   with (O/f'TRAINING_LOGS/{self.arm}_metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   print('TRAINING',json.dumps(row),flush=True)
  if n%500==0:torch.save(dict(actor_state=a.actor_net.state_dict(),stage_only=self.arm=='stage',native_iteration=int(a.update)),O/f'arms/{self.arm}/actor_update{n}.pt')
  if STOP or (O/'STOP_REQUESTED').exists() or time.monotonic()-self.global_start>=CFG['training_grace_seconds']:raise Halt('Predeclared stopping rule')

def arm_run(arm,start):
 s=restore();a=s['algorithm'];migration=install(a,arm=='stage',O/'preparation/prepared.pt');a.dowel=l=Logger(a,arm,start);a.post_approx.dowel=l;a.save_path=None;original_sim,l.simulator=instrument_simulator(a);root=O/'arms'/arm;root.mkdir(parents=True);frozen=digest(dict(a.actor_net.named_buffers()));initial=digest(a.actor_net.state_dict())
 if arm=='candidate':assert initial==json.loads((O/'arms/stage/STARTED.json').read_text())['initial_digest']
 for op,key in [(a.actor_optimizer,'actor'),(a.critic_optimizer,'critic')]+[(op,'posterior') for oo in a.post_approx.poi_post_optimizers.values() for op in oo]:
  def count(*args,key=key,**kwargs):l.counts[key]+=1
  op.register_step_post_hook(count)
 original.restore(s['rng']);torch.cuda.reset_peak_memory_stats();COST.clear();put(root/'STARTED.json',dict(time=time.time(),initial_digest=initial,parent_native_iteration=16001,initial_replay_size=len(a._runtime_buffer['params']),migration={k:v for k,v in migration.items() if k!='source'}));initial_dir=O/'initial';initial_dir.mkdir(exist_ok=True);torch.save(dict(actor_state=a.actor_net.state_dict(),stage_only=arm=='stage'),initial_dir/f'{arm}.pt');status='COMPLETE'
 try:
  profile=json.loads((P/'frozen_protocol.json').read_text())['profile'];profile['discount']=1.;original.train_native(a,l,profile,target=18001)
 except Halt:status='INCOMPLETE_STOP_OR_RUNTIME'
 except Exception as exc:
  import traceback
  traceback.print_exc();status='FAILED';put(root/'ERROR.json',dict(type=type(exc).__name__,message=str(exc),native_iteration=int(a.update)))
 a.m_f,a.asses=original_sim
 for op in allopts(a):op._optimizer_step_pre_hooks.clear();op._optimizer_step_post_hooks.clear()
 assert digest(dict(a.actor_net.named_buffers()))==frozen,'Frozen provider changed';torch.save(dict(actor_state=a.actor_net.state_dict(),stage_only=arm=='stage',native_iteration=int(a.update)),root/'actor.pt')
 with (root/'runtime.pkl').open('wb') as f:cloudpickle.dump(dict(algorithm=a,rng=original.rng(),method=CFG['method'],arm=arm),f,protocol=5)
 n=int(a.update)-16001;result=dict(status=status,arm=arm,updates=n,seconds=time.monotonic()-l.start,seconds_per_update=(time.monotonic()-l.start)/max(1,n),optimizer_steps=l.counts,new_simulator_episodes=l.simulator['episodes'],new_simulator_observations=l.simulator['observations'],simulator_batch_calls=l.simulator['batch_calls'],model_queries=dict(COST),parameters=sum(p.numel() for p in a.actor_net.parameters()),frozen_buffer_elements=sum(v.numel() for v in a.actor_net.buffers()),gpu_peak_bytes=torch.cuda.max_memory_allocated(),actor_sha256=sha(root/'actor.pt'),runtime_sha256=sha(root/'runtime.pkl'));put(root/'RESULT.json',result);print('ARM_END',json.dumps(result),flush=True)
 if status=='COMPLETE':assert n==2000 and l.counts==dict(actor=2000,critic=10000,posterior=300000)
 return status

def train_pair(start):
 assert not (O/'evidence/TRAINING_STARTED.json').exists();signal.signal(signal.SIGTERM,lambda *args:globals().__setitem__('STOP',True));put(O/'evidence/TRAINING_STARTED.json',dict(time=time.time(),pid=os.getpid(),two_fixed_arms=True));prepare();gc.collect();torch.cuda.empty_cache();results={}
 for arm in ['stage','candidate']:
  results[arm]=arm_run(arm,start);gc.collect();torch.cuda.empty_cache()
  if results[arm]!='COMPLETE':break
 status='COMPLETE' if len(results)==2 and all(v=='COMPLETE' for v in results.values()) else 'INCOMPLETE';put(O/'evidence/PAIR_COMPLETION.json',dict(status=status,arms=results));return status=='COMPLETE'
