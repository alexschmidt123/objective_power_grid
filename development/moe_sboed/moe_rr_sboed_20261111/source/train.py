import gc,signal,resource
from common import *
from integration import install,bind,load_prepared
STOP=False
class Halt(Exception):pass
class Table:
    def __init__(self):self.data={}
    def clear(self):self.data.clear()
    def record(self,k,v):self.data[k]=v
class Logger:
    def __init__(self,a,arm,start,offset,fixture=False):
        self.algorithm=a;self.arm=arm;self.offset=offset;self.global_start=start;self.start=time.monotonic();self.cpu=time.process_time()
        self.logger=self;self.tabular=Table();self.counts=dict(actor=0,critic=0,posterior=0);self.sim=dict(episodes=0,observations=0,batch_calls=0);self.fixture=fixture
    def push_prefix(self,*args):pass
    def pop_prefix(self,*args):pass
    def log(self,*args):pass
    def dump_all(self):
        a=self.algorithm;n=int(a.update)-self.offset
        assert self.counts==dict(actor=n,critic=5*n,posterior=150*n),(n,self.counts)
        assert a._rr_last['actor_rows']==300000
        assert self.sim==dict(episodes=n*1000,observations=n*30000,batch_calls=n*30),self.sim
        if self.fixture:return
        target=CFG['preparation_updates'] if self.arm=='preparation' else CFG['updates_per_arm']
        if n==1 or n%25==0 or n==target:
            with torch.no_grad():
                f=a._rr_cache[:256,12];p=a.actor_net.routing(f,a._rr_uniform,CFG['joint_pair_exploration'])
                actions,delta=a.actor_net.all_actions(f)
                mask=torch.nn.functional.one_hot(a.actor_net.pairs,4).sum(1).float();margins=p@mask
            row=dict(arm=self.arm,updates=n,target=target,native_iteration=int(a.update),seconds=time.monotonic()-self.start,
                     seconds_per_update=(time.monotonic()-self.start)/n,optimizer_steps=self.counts.copy(),actual_simulator_counts=self.sim.copy(),
                     pair_selection_counts=a._rr_selected.tolist(),pair_probabilities=p.mean(0).tolist(),expert_inclusion_probabilities=margins.mean(0).tolist(),
                     pair_entropy=float(-(p*p.clamp_min(1e-30).log()).sum(-1).mean()),mean_correction_norm=float(delta.norm(dim=-1).mean()),
                     actor=a._rr_last.copy(),actor_lr=a.actor_optimizer.param_groups[0]['lr'],native_exploration=float(a.design_noise_scale),
                     native_reward=self.tabular.data.get('Reward/MeanReward'),GPU_peak_bytes=torch.cuda.max_memory_allocated())
            put(O/f'TRAINING_LOGS/{self.arm}_status.json',row)
            with (O/f'TRAINING_LOGS/{self.arm}_metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print('TRAINING',json.dumps(row),flush=True)
        if self.arm!='preparation' and n%500==0:torch.save(dict(actor_state=a.actor_net.state_dict(),native_iteration=int(a.update)),O/f'arms/{self.arm}/actor_update{n}.pt')
        if STOP or (O/'STOP_REQUESTED').exists() or time.monotonic()-self.global_start>CFG['training_grace_seconds']:raise Halt('Locked stop rule')

def instrument(a,l):
    original_model,original_asses=a.m_f,a.asses
    def model(stage,theta,*args,**kwargs):
        value=original_model(stage,theta,*args,**kwargs);l.sim['observations']+=len(theta);l.sim['batch_calls']+=1;return value
    def asses(*args,**kwargs):
        value=original_asses(*args,**kwargs);l.sim['episodes']+=int(kwargs.get('n_traj',args[0] if args else 10000));return value
    a.m_f=model;a.asses=asses
    for opt,key in [(a.actor_optimizer,'actor'),(a.critic_optimizer,'critic')]+[(p,'posterior') for oo in a.post_approx.poi_post_optimizers.values() for p in oo]:
        def count(*args,key=key,**kwargs):l.counts[key]+=1
        opt.register_step_post_hook(count)
    return original_model,original_asses

def run_stage(arm,start):
    if arm=='preparation':
        state=restore();a=state['algorithm'];src=install(a);offset=16001;target=CFG['joint_start_iteration'];root=O/'preparation'
    else:
        state=load_prepared();a=state['algorithm'];src=bind(a,arm=='control');offset=CFG['joint_start_iteration'];target=offset+CFG['updates_per_arm'];root=O/'arms'/arm
    root.mkdir(parents=True);frozen=digest(dict(a.actor_net.base.named_buffers()));initial=digest(a.actor_net.state_dict())
    if arm=='control':assert initial==json.loads((O/'arms/full/STARTED.json').read_text())['initial_actor_digest']
    a.dowel=l=Logger(a,arm,start,offset);a.post_approx.dowel=l;a.save_path=None;originals=instrument(a,l)
    original.restore(state['rng']);torch.cuda.reset_peak_memory_stats();COST.clear()
    put(root/'STARTED.json',dict(time=time.time(),pid=os.getpid(),initial_actor_digest=initial,source_sha256=hashlib.sha256(src.encode()).hexdigest(),native_start=offset,uniform_top2=a._rr_uniform))
    status='COMPLETE'
    try:
        profile=json.loads((P/'frozen_protocol.json').read_text())['profile'];profile['discount']=1.
        original.train_native(a,l,profile,target=target)
    except Halt:status='INCOMPLETE_STOP_OR_CAP'
    except Exception as exc:
        import traceback
        traceback.print_exc();status='FAILED';put(root/'ERROR.json',dict(type=type(exc).__name__,message=str(exc)))
    a.m_f,a.asses=originals;a._rr_cache=None
    for opt in allopts(a):opt._optimizer_step_pre_hooks.clear();opt._optimizer_step_post_hooks.clear()
    assert digest(dict(a.actor_net.base.named_buffers()))==frozen,'Frozen uniform HEC base changed'
    assert digest(dict(a.actor_target_net.base.named_buffers()))==frozen,'Target frozen base changed'
    torch.save(dict(actor_state=a.actor_net.state_dict(),critic_state=a.critic_net.state_dict(),native_iteration=int(a.update),arm=arm),root/'actor.pt')
    with (root/'runtime.pkl').open('wb') as f:cloudpickle.dump(dict(algorithm=a,rng=original.rng()),f,protocol=5)
    result=dict(status=status,arm=arm,updates=int(a.update)-offset,seconds=time.monotonic()-l.start,
                process_CPU_seconds=time.process_time()-l.cpu,process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                optimizer_steps=l.counts,simulator_counts=l.sim,pair_selection_counts=a._rr_selected.tolist(),model_queries=dict(COST),
                GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),actor_sha256=sha(root/'actor.pt'),runtime_sha256=sha(root/'runtime.pkl'),
                frozen_base_unchanged=True,trainable_parameters=sum(p.numel() for p in a.actor_net.parameters()))
    put(root/'RESULT.json',result);print('STAGE_END',json.dumps(result),flush=True)
    if status=='COMPLETE':assert int(a.update)==target
    return status

def run_pair(start):
    signal.signal(signal.SIGTERM,lambda *args:globals().__setitem__('STOP',True))
    assert not (O/'evidence/TRAINING_STARTED.json').exists()
    put(O/'evidence/TRAINING_STARTED.json',dict(time=time.time(),pid=os.getpid(),arms=CFG['arms'],shared_preparation_updates=CFG['preparation_updates']))
    outcomes={}
    for stage in ['preparation',*CFG['arms']]:
        outcomes[stage]=run_stage(stage,start);gc.collect();torch.cuda.empty_cache()
        if outcomes[stage]!='COMPLETE':break
    complete=len(outcomes)==3 and all(v=='COMPLETE' for v in outcomes.values())
    put(O/'evidence/PAIR_COMPLETION.json',dict(status='COMPLETE' if complete else 'INCOMPLETE',stages=outcomes))
    return complete
