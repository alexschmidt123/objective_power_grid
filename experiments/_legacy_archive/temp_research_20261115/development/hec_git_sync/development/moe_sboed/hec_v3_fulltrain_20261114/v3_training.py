"""Unmodified native sBOED updates with scoped parameter selection and receipts."""
import copy,json,random,time
import numpy as np
import torch
from v3_native import n
class Stop(Exception):pass
class Table:
    def __init__(self):self.data={}
    def record(self,k,v):self.data[k]=v
    def clear(self):self.data.clear()
def optimizers(a):return [(a.actor_optimizer,'actor'),(a.critic_optimizer,'critic')]+[(o,'posterior') for group in a.post_approx.poi_post_optimizers.values() for o in group]
def configure(a,phase,replace=False):
    a.actor_net.configure(phase)
    if replace:
        a.actor_target_net=copy.deepcopy(a.actor_net);a.actor_target_net.configure('frozen')
    if not getattr(a,'_fulltrain_optimizer_initialized',False):
        lr=a.actor_optimizer.param_groups[0]['lr']
        # Include history now, so unfreezing preserves all Adam/scheduler state.
        a.actor_optimizer=torch.optim.Adam(list(a.actor_net.parameters()),lr=lr)
        a.actor_lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(a.actor_optimizer,gamma=.9999)
        a._fulltrain_optimizer_initialized=True
def frozen(a):
    values={k:p for k,p in a.actor_net.named_parameters() if not p.requires_grad}
    values.update({'buffer:'+k:v for k,v in a.actor_net.named_buffers()});return n.digest(values)
class Logger:
    def __init__(self,a,name,out,steps,start,cap):
        self.a=a;self.name=name;self.out=out;self.steps=steps;self.start=start;self.cap=cap;self.base=int(a.update)
        self.logger=self;self.tabular=Table();self.counts=dict(actor=0,critic=0,posterior=0)
    def push_prefix(self,*unused):pass
    def pop_prefix(self,*unused):pass
    def log(self,*unused):pass
    def dump_all(self):
        done=int(self.a.update)-self.base;assert self.counts==dict(actor=done,critic=5*done,posterior=150*done)
        if done==1 or done%25==0 or done==self.steps:
            row=dict(phase=self.name,updates=done,native_iteration=int(self.a.update),seconds=time.monotonic()-self.start,counts=self.counts.copy(),actor_lr=self.a.actor_optimizer.param_groups[0]['lr'])
            row['native_metrics']={str(k):float(v) for k,v in self.tabular.data.items() if isinstance(v,(int,float,np.number))}
            with (self.out/'TRAINING_METRICS.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if done==1 or done%25==0 or done==self.steps:
            n.put(self.out/'STATUS.json',dict(status='TRAINING',phase=self.name,updates=done,target=self.steps,counts=self.counts.copy(),
                new_episodes=1000*done,new_observations=30000*done,seconds=time.monotonic()-self.start))
        if time.monotonic()-self.start>self.cap or (self.out/'STOP_REQUESTED').exists():raise Stop('Bounded stop; partial artifacts retained')
def phase(a,name,steps,out,start,cap):
    root=out/name;root.mkdir();logger=Logger(a,name,out,steps,start,cap);a.dowel=logger;a.post_approx.dowel=logger;a.save_path=None
    for op,key in optimizers(a):
        op._optimizer_step_pre_hooks.clear();op._optimizer_step_post_hooks.clear()
        def count(*unused,key=key):logger.counts[key]+=1
        op.register_step_post_hook(count)
    def check(*unused):
        for name,p in a.actor_net.named_parameters():
            if p.requires_grad:
                if hasattr(a.actor_net,'configure'):assert p.grad is not None,name
                if p.grad is not None:assert torch.isfinite(p.grad).all(),name
            else:assert p.grad is None,name
    a.actor_optimizer.register_step_pre_hook(check)
    before=frozen(a);base=int(a.update);status='COMPLETE'
    n.put(root/'STARTED.json',dict(parent_update=base,updates=steps,trainable_parameters=sum(p.numel() for p in a.actor_net.parameters() if p.requires_grad),
        actor_loss='Unmodified native -mean Q(h,actor(h))',native_EIG_objective_unchanged=True,KL_penalty=False,new_value_estimator=False))
    profile=json.loads((n.P/'frozen_protocol.json').read_text())['profile'];profile['discount']=1.
    try:n.original.train_native(a,logger,profile,target=base+steps)
    except Stop:status='INCOMPLETE'
    assert frozen(a)==before
    for op,_ in optimizers(a):op._optimizer_step_pre_hooks.clear();op._optimizer_step_post_hooks.clear()
    torch.save(dict(actor_state=a.actor_net.state_dict(),actor_target_state=a.actor_target_net.state_dict(),native_iteration=int(a.update)),root/'actor.pt')
    with (root/'runtime.pkl').open('wb') as f:n.cloudpickle.dump(dict(algorithm=a,rng=n.original.rng(),phase=name),f,protocol=5)
    receipt=dict(status=status,updates=int(a.update)-base,counts=logger.counts,actor_sha256=n.sha(root/'actor.pt'),runtime_sha256=n.sha(root/'runtime.pkl'),frozen_actor_state_preserved=True)
    n.put(root/'RESULT.json',receipt);return receipt
