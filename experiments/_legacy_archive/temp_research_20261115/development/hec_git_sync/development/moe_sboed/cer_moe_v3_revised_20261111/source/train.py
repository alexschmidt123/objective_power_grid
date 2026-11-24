import gc,signal,resource
from common import *
from integration import make_policy,install,exposure
from preparation import prepare
STOP=False

class Halt(Exception):pass
class Table:
    def __init__(self):self.data={}
    def clear(self):self.data.clear()
    def record(self,k,v):self.data[k]=v

class Logger:
    def __init__(self,a,arm,start,fixture=False):
        self.algorithm=a;self.arm=arm;self.global_start=start;self.start=time.monotonic();self.cpu=time.process_time()
        self.logger=self;self.tabular=Table();self.counts=dict(actor=0,critic=0,posterior=0);self.sim=dict(episodes=0,observations=0,batch_calls=0);self.fixture=fixture
    def push_prefix(self,*args):pass
    def pop_prefix(self,*args):pass
    def log(self,*args):pass
    def dump_all(self):
        a=self.algorithm;n=int(a.update)-CFG['initial_native_iteration']
        assert self.counts==dict(actor=n,critic=5*n,posterior=150*n),(n,self.counts)
        assert a._cer_last['actor_rows']==10000*30
        assert self.sim==dict(episodes=n*1000,observations=n*30000,batch_calls=n*30),self.sim
        if self.fixture:return
        if n==1 or n%25==0 or n==CFG['updates_per_arm']:
            with torch.no_grad():
                x=a.form_actor_input(12,a.ds_hist[:512,:12],a.ys_hist[:512,:12]);_,q,p=a.actor_net.distribution(x,a.critic_net,a._cer_uniform)
            row=dict(arm=self.arm,updates=n,native_iteration=int(a.update),seconds=time.monotonic()-self.start,
                     seconds_per_update=(time.monotonic()-self.start)/n,epsilon=exposure(a),optimizer_steps=self.counts.copy(),
                     actual_simulator_counts=self.sim.copy(),selected_counts=a._cer_selected.tolist(),
                     gate_means=p.mean(0).tolist(),gate_entropy=float(-(p*p.clamp_min(1e-30).log()).sum(-1).mean()),
                     temperature=float(a.actor_net.log_temperature.exp().clamp(.1,10.)),
                     critic_candidate_value_SD=float(q.std(-1).mean()),actor=a._cer_last.copy(),
                     actor_lr=a.actor_optimizer.param_groups[0]['lr'],native_Gaussian_exploration=float(a.design_noise_scale),
                     native_reward=self.tabular.data.get('Reward/MeanReward'),GPU_peak_bytes=torch.cuda.max_memory_allocated())
            put(O/f'TRAINING_LOGS/{self.arm}_status.json',row)
            with (O/f'TRAINING_LOGS/{self.arm}_metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print('TRAINING',json.dumps(row),flush=True)
        if n%500==0:torch.save(dict(actor_state=a.actor_net.state_dict(),native_iteration=int(a.update)),O/f'arms/{self.arm}/actor_update{n}.pt')
        if STOP or (O/'STOP_REQUESTED').exists() or time.monotonic()-self.global_start>CFG['training_grace_seconds']:raise Halt('Fixed stopping rule')

def instrument(a,logger):
    original_model,original_asses=a.m_f,a.asses
    def model(stage,theta,*args,**kwargs):
        value=original_model(stage,theta,*args,**kwargs)
        logger.sim['observations']+=len(theta);logger.sim['batch_calls']+=1;return value
    def asses(*args,**kwargs):
        value=original_asses(*args,**kwargs)
        logger.sim['episodes']+=int(kwargs.get('n_traj',args[0] if args else 10000));return value
    a.m_f=model;a.asses=asses
    for op,key in [(a.actor_optimizer,'actor'),(a.critic_optimizer,'critic')]+[(p,'posterior') for oo in a.post_approx.poi_post_optimizers.values() for p in oo]:
        def count(*args,key=key,**kwargs):logger.counts[key]+=1
        op.register_step_post_hook(count)
    return original_model,original_asses

def arm_run(arm,start):
    state=restore();a=state['algorithm'];policy=make_policy(a)
    policy.load_state_dict(torch.load(O/'preparation/actor.pt',map_location='cuda',weights_only=False)['actor_state'])
    source=install(a,policy,arm=='control');root=O/'arms'/arm;root.mkdir(parents=True)
    initial=digest(a.actor_net.state_dict());buffers=digest(dict(a.actor_net.named_buffers()))
    if arm=='control':assert initial==json.loads((O/'arms/full/STARTED.json').read_text())['initial_actor_digest']
    a.dowel=l=Logger(a,arm,start);a.post_approx.dowel=l;a.save_path=None
    originals=instrument(a,l);original.restore(state['rng']);torch.cuda.reset_peak_memory_stats();COST.clear()
    put(root/'STARTED.json',dict(time=time.time(),initial_actor_digest=initial,source_sha256=hashlib.sha256(source.encode()).hexdigest(),parent_iteration=16001))
    status='COMPLETE'
    try:
        profile=json.loads((P/'frozen_protocol.json').read_text())['profile'];profile['discount']=1.
        original.train_native(a,l,profile,target=16001+CFG['updates_per_arm'])
    except Halt:status='INCOMPLETE_STOP_OR_CAP'
    except Exception as exc:
        import traceback
        traceback.print_exc();status='FAILED';put(root/'ERROR.json',dict(type=type(exc).__name__,message=str(exc)))
    a.m_f,a.asses=originals
    for opt in allopts(a):opt._optimizer_step_pre_hooks.clear();opt._optimizer_step_post_hooks.clear()
    assert digest(dict(a.actor_net.named_buffers()))==buffers,'Frozen Bayesian model changed'
    torch.save(dict(actor_state=a.actor_net.state_dict(),critic_state=a.critic_net.state_dict(),native_iteration=int(a.update),arm=arm),root/'actor.pt')
    with (root/'runtime.pkl').open('wb') as f:cloudpickle.dump(dict(algorithm=a,rng=original.rng()),f,protocol=5)
    result=dict(status=status,arm=arm,updates=int(a.update)-16001,seconds=time.monotonic()-l.start,
                process_CPU_seconds=time.process_time()-l.cpu,process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                optimizer_steps=l.counts,simulator_counts=l.sim,selected_counts=a._cer_selected.tolist(),model_queries=dict(COST),
                GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),actor_sha256=sha(root/'actor.pt'),runtime_sha256=sha(root/'runtime.pkl'),
                final_epsilon=exposure(a),frozen_belief_unchanged=True)
    put(root/'RESULT.json',result);print('ARM_END',json.dumps(result),flush=True)
    if status=='COMPLETE':assert result['updates']==CFG['updates_per_arm']
    return status

def run_pair(start):
    signal.signal(signal.SIGTERM,lambda *args:globals().__setitem__('STOP',True))
    assert not (O/'evidence/TRAINING_STARTED.json').exists()
    put(O/'evidence/TRAINING_STARTED.json',dict(time=time.time(),pid=os.getpid(),arms=CFG['arms']))
    state=restore();prepare(state['algorithm']);del state;gc.collect();torch.cuda.empty_cache()
    outcomes={}
    for arm in CFG['arms']:
        outcomes[arm]=arm_run(arm,start);gc.collect();torch.cuda.empty_cache()
        if outcomes[arm]!='COMPLETE':break
    success=len(outcomes)==2 and all(v=='COMPLETE' for v in outcomes.values())
    put(O/'evidence/PAIR_COMPLETION.json',dict(status='COMPLETE' if success else 'INCOMPLETE',arms=outcomes))
    return success
