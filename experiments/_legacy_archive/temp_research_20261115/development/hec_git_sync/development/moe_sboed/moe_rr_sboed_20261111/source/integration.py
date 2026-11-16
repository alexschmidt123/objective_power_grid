"""Native actor-critic with exact finite Top2 expectations and frozen cache."""
import functools,inspect,textwrap
from common import *
from rr_policy import RRPolicy
from src.research.vendor.vsoed.pg_vsoed import PGvsOED
import src.research.vendor.vsoed.pg_vsoed as native_module

def cache_batch(a):
    n=len(a.ds_hist);device=a.ds_hist.device
    cache=torch.empty(n,30,642,device=device)
    with torch.no_grad():
        for stage in range(30):
            x=a.form_actor_input(stage,a.ds_hist[:,:stage],a.ys_hist[:,:stage])
            cache[:,stage]=a.actor_net.frozen(x)
    a._rr_cache=cache

def draw_design(a,stage,ds_hist,ys_hist,xps_hist=None,return_all_stages=False,use_target=False):
    assert not return_all_stages and not a.use_encoder
    x=a.form_actor_input(stage,ds_hist,ys_hist);actor=a.actor_target_net if use_target else a.actor_net
    with torch.no_grad():
        action,info=actor.select(x,u=torch.rand(len(x),device=x.device,generator=a._rr_rng),uniform=a._rr_uniform,
                                epsilon=CFG['joint_pair_exploration'],collect=True)
    if not use_target:a._rr_selected+=torch.bincount(info['chosen_pair'],minlength=6)
    return action.detach()

def target_value(a,stage,ds_hist,ys_hist):
    x=a.form_actor_input(stage,ds_hist,ys_hist);features=a._rr_cache[:,stage]
    out=[]
    with torch.no_grad():
        for lo in range(0,len(x),CFG['actor_chunk']):
            xx=x[lo:lo+CFG['actor_chunk']];f=features[lo:lo+len(xx)]
            actions,p,_=a.actor_target_net.distribution(f,a._rr_uniform,CFG['joint_pair_exploration'])
            q=a.critic_target_net(torch.cat([xx[:,None].expand(-1,6,-1),actions],-1))[...,0]
            out.append((p*q).sum(-1))
    return torch.cat(out)

def loss_for(a,x,features):
    actions,p,_=a.actor_net.distribution(features,a._rr_uniform,CFG['joint_pair_exploration'])
    q=a.critic_net(torch.cat([x[:,None].expand(-1,6,-1),actions],-1))[...,0]
    return -(p*q).sum(-1).mean()

def actor_update(a,x):
    a.actor_optimizer.zero_grad();a.critic_net.requires_grad_(False)
    flat=x.reshape(-1,117);features=a._rr_cache.reshape(-1,642);total=0.
    for lo in range(0,len(flat),CFG['actor_chunk']):
        xx=flat[lo:lo+CFG['actor_chunk']];f=features[lo:lo+len(xx)]
        loss=loss_for(a,xx,f)*len(xx)/len(flat);total+=float(loss.detach());loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.actor_net.parameters())
    a._rr_last=dict(actor_rows=len(flat),loss=total,adaptation_gradients=[norm([p.grad for p in m.parameters()]) for m in a.actor_net.adaptations],
        router_gradient=norm([p.grad for p in a.actor_net.router.parameters()]),correction_gradient=norm([p.grad for p in a.actor_net.correction.parameters()]))
    a.actor_optimizer.step();a.critic_net.requires_grad_(True)

def bind(a,uniform):
    a._rr_uniform=uniform;a._rr_cache=None;a._rr_last={}
    a._rr_rng=torch.Generator(device=next(a.actor_net.parameters()).device).manual_seed(CFG['routing_seed'])
    a._rr_selected=torch.zeros(6,dtype=torch.long,device=next(a.actor_net.parameters()).device)
    a.actor=functools.partial(draw_design,a,use_target=False);a.actor_target=functools.partial(draw_design,a,use_target=True)
    a.rr_cache=functools.partial(cache_batch,a);a.rr_target=functools.partial(target_value,a);a.rr_actor_update=functools.partial(actor_update,a)
    src='\n'.join(line.rstrip() for line in inspect.getsource(PGvsOED.train).splitlines())+'\n'
    assert '_runtime_buffer' in src
    old='''                designs = self.actor_net(X_actor)
                X = torch.cat([X_actor, designs], dim=-1)
                output = -self.critic_net(X).mean()'''
    assert src.count(old)==1;src=src.replace(old,'                output = None')
    old='''            self.actor_optimizer.zero_grad()
            output.backward()
            self.actor_optimizer.step()'''
    assert src.count(old)==1;src=src.replace(old,'            self.rr_actor_update(X_actor)')
    old='''                            ds_next = self.actor_target(k + 1,
                                self.ds_hist[:, :k + 1],
                                self.ys_hist[:, :k + 1],
                                None)
                            next_action_value = self.critic_target(k + 1,
                                self.ds_hist[:, :k + 1],
                                self.ys_hist[:, :k + 1],
                                None,
                                ds_next)'''
    assert src.count(old)==1;src=src.replace(old,'''                            next_action_value = self.rr_target(k + 1,
                                self.ds_hist[:, :k + 1], self.ys_hist[:, :k + 1])''')
    old='''            # Form the inputs and target values of critic network, and form the'''
    assert src.count(old)==1;src=src.replace(old,'''            self.rr_cache()
'''+old)
    namespace=dict(native_module.__dict__);exec(compile(textwrap.dedent(src),'<rr_native_top2_expectations>','exec'),namespace)
    a.train=functools.partial(namespace['train'],a);return src

def install(a):
    device=next(a.actor_net.parameters()).device;lr=a.actor_optimizer.param_groups[0]['lr']
    with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
        torch.manual_seed(CFG['seed']);policy=RRPolicy(a.actor_net,CFG)
    a.actor_net=policy;a.actor_target_net=copy.deepcopy(policy)
    a.actor_optimizer=torch.optim.Adam(policy.parameters(),lr=lr)
    a.actor_lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(a.actor_optimizer,CFG['lr_decay'])
    return bind(a,True)

def load_prepared():
    import io
    path=O/'preparation/runtime.pkl';receipt=json.loads((O/'preparation/RESULT.json').read_text());assert sha(path)==receipt['runtime_sha256']
    old=torch.storage._load_from_bytes
    # CPU ByteTensor RNG states must stay on CPU when a CUDA Generator is
    # deserialized; remap only storages that were originally CUDA storages.
    torch.storage._load_from_bytes=lambda b:torch.load(io.BytesIO(b),map_location=lambda storage,loc:storage.cuda(0) if loc.startswith('cuda') else storage,weights_only=False)
    try:
        with path.open('rb') as f:state=cloudpickle.load(f)
    finally:torch.storage._load_from_bytes=old
    for opt in allopts(state['algorithm']):opt._optimizer_step_pre_hooks.clear();opt._optimizer_step_post_hooks.clear()
    assert state['algorithm'].update==CFG['joint_start_iteration']
    return state
