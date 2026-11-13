"""Native training/logging only; no retired expert architecture."""
import json,time,random
import numpy as np
import torch
from common import n
class Stop(Exception):pass
class Table:
    def __init__(self):self.data={}
    def record(self,k,v):self.data[k]=v
    def clear(self):self.data.clear()

def all_optimizers(a):
    return [(a.actor_optimizer,'actor'),(a.critic_optimizer,'critic')]+[(o,'posterior') for oo in a.post_approx.poi_post_optimizers.values() for o in oo]

class Logger:
    def __init__(self,a,phase,out,cfg,start):
        self.algorithm=a;self.phase=phase;self.out=out;self.cfg=cfg;self.start=start
        self.base=int(a.update);self.logger=self;self.tabular=Table();self.counts=dict(actor=0,critic=0,posterior=0)
    def push_prefix(self,*unused):pass
    def pop_prefix(self,*unused):pass
    def log(self,*unused):pass
    def dump_all(self):
        u=int(self.algorithm.update)-self.base
        assert self.counts==dict(actor=u,critic=5*u,posterior=150*u)
        if u==1 or u%25==0 or u==self.cfg['updates']:
            row=dict(status='TRAINING',phase=self.phase,updates=u,target=self.cfg['updates'],
                     optimizer_steps=self.counts,new_episodes=1000*u,new_observations=30000*u,seconds=time.monotonic()-self.start)
            n.put(self.out/'STATUS.json',row);print(json.dumps(row),flush=True)
        if (self.out/'STOP_REQUESTED').exists() or time.monotonic()-self.start>self.cfg['wall_seconds']:raise Stop('Bounded stop')

def phase(a,name,updates,out,start,wall_seconds,seed=None):
    root=out/name;root.mkdir();cfg=dict(updates=updates,wall_seconds=wall_seconds)
    logger=Logger(a,name,out,cfg,start);a.dowel=logger;a.post_approx.dowel=logger;a.save_path=None
    for opt,key in all_optimizers(a):
        opt._optimizer_step_pre_hooks.clear();opt._optimizer_step_post_hooks.clear()
        def count(*unused,key=key):logger.counts[key]+=1
        opt.register_step_post_hook(count)
    def monitor(*unused):
        for key,p in a.actor_net.named_parameters():
            if p.grad is not None:assert torch.isfinite(p.grad).all(),key
            if not p.requires_grad:assert p.grad is None,key
    a.actor_optimizer.register_step_pre_hook(monitor)
    if seed is not None:
        torch.manual_seed(seed);np.random.seed(seed%(2**32));random.seed(seed)
    before_buffers=n.digest(dict(a.actor_net.named_buffers()))
    initial=n.digest(a.actor_net.state_dict());base=int(a.update);status='COMPLETE'
    n.put(root/'STARTED.json',dict(phase=name,parent_update=base,initial_actor_digest=initial,seed=seed,
                                  trainable_parameters=sum(p.numel() for p in a.actor_net.parameters() if p.requires_grad)))
    profile=json.loads((n.P/'frozen_protocol.json').read_text())['profile'];profile['discount']=1.
    try:n.original.train_native(a,logger,profile,target=base+updates)
    except Stop:status='INCOMPLETE'
    assert n.digest(dict(a.actor_net.named_buffers()))==before_buffers
    for opt,_ in all_optimizers(a):opt._optimizer_step_pre_hooks.clear();opt._optimizer_step_post_hooks.clear()
    torch.save(dict(actor_state=a.actor_net.state_dict(),actor_target_state=a.actor_target_net.state_dict(),native_iteration=int(a.update)),root/'actor.pt')
    with (root/'runtime.pkl').open('wb') as f:n.cloudpickle.dump(dict(algorithm=a,rng=n.original.rng(),phase=name),f,protocol=5)
    result=dict(status=status,phase=name,updates=int(a.update)-base,optimizer_steps=logger.counts,
       actor_sha256=n.sha(root/'actor.pt'),runtime_sha256=n.sha(root/'runtime.pkl'),frozen_buffers_preserved=True)
    n.put(root/'RESULT.json',result)
    return result
