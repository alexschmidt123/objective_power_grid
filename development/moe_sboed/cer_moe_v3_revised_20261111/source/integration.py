"""Change only the native policy expectations; preserve native replay/returns."""
import functools,inspect,textwrap
from common import *
from policy import CERPolicy,chunked_loss
from src.research.vendor.vsoed.pg_vsoed import PGvsOED
import src.research.vendor.vsoed.pg_vsoed as native_module

def make_policy(a):
    from src.research.vendor.vsoed.utils import Net
    device=next(a.actor_net.parameters()).device
    native=Net([117,256,256,256,2],torch.nn.ReLU,torch.tensor([[-4.,4.]]*2,device=device),'actor').to(device)
    native.load_state_dict(torch.load(OLD/'inputs/vsoed.pt',map_location=device,weights_only=False)['actor_state'])
    return CERPolicy(native,a.actor_net.belief,CFG['final_epsilon'],CFG['initial_temperature']).to(device)

def exposure(a):
    n=int(a.update)-CFG['initial_native_iteration']
    if a._cer_uniform:return 1.
    fraction=min(1.,max(0.,(n-CFG['uniform_exposure_updates'])/CFG['exposure_transition_updates']))
    return 1.-fraction*(1.-CFG['final_epsilon'])

def draw_design(a,stage,ds_hist,ys_hist,xps_hist=None,return_all_stages=False,use_target=False):
    assert not return_all_stages and not a.use_encoder
    x=a.form_actor_input(stage,ds_hist,ys_hist)
    actor=a.actor_target_net if use_target else a.actor_net
    critic=a.critic_target_net if use_target else a.critic_net
    actor.epsilon=exposure(a)
    with torch.no_grad():
        act,info=actor.select(x,critic,a._cer_uniform,u=torch.rand(len(x),device=x.device,generator=a._cer_generator))
    if not use_target:a._cer_selected+=torch.bincount(info['selected'],minlength=4)
    return act.detach()

def target_value(a,stage,ds_hist,ys_hist):
    x=a.form_actor_input(stage,ds_hist,ys_hist);a.actor_target_net.epsilon=exposure(a)
    outs=[]
    with torch.no_grad():
        for xx in x.split(CFG['actor_chunk']):
            _,q,p=a.actor_target_net.distribution(xx,a.critic_target_net,a._cer_uniform)
            outs.append((p*q).sum(-1))
    return torch.cat(outs)

def actor_update(a,x):
    a.actor_net.epsilon=exposure(a)
    a.actor_optimizer.zero_grad()
    for p in a.critic_net.parameters():p.requires_grad_(False)
    total=0.;flat=x.reshape(-1,117)
    # Backward each chunk immediately: exact full native batch accumulation,
    # bounded activations; no dropped episodes/stages or batch-size substitution.
    for xx in flat.split(CFG['actor_chunk']):
        from policy import expected_loss
        loss=expected_loss(a.actor_net,a.critic_net,xx,a._cer_uniform)*(len(xx)/len(flat))
        total+=float(loss.detach());loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.actor_net.parameters())
    a._cer_last=dict(actor_rows=len(flat),actor_loss=total,epsilon=exposure(a),
                     expert_gradients=[norm([p.grad for p in e.parameters()]) for e in a.actor_net.experts])
    a.actor_optimizer.step()
    for p in a.critic_net.parameters():p.requires_grad_(True)

def install(a,policy,uniform=False):
    assert not a.use_encoder and a.backend_net is None and not a.use_PCE and a.discount==1
    lr=a.actor_optimizer.param_groups[0]['lr']
    a.actor_net=policy;a.actor_target_net=copy.deepcopy(policy)
    a.actor_optimizer=torch.optim.Adam(policy.parameters(),lr=lr)
    a.actor_lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(a.actor_optimizer,CFG['lr_decay'])
    a._cer_uniform=uniform;a._cer_generator=torch.Generator(device=next(policy.parameters()).device).manual_seed(CFG['training_seed'])
    a._cer_selected=torch.zeros(4,dtype=torch.long,device=next(policy.parameters()).device)
    a._cer_last={}
    a.actor=functools.partial(draw_design,a,use_target=False)
    a.actor_target=functools.partial(draw_design,a,use_target=True)
    a.cer_target=functools.partial(target_value,a)
    a.cer_actor_update=functools.partial(actor_update,a)
    src='\n'.join(line.rstrip() for line in inspect.getsource(PGvsOED.train).splitlines())+'\n'
    assert '_runtime_buffer' in src,'Requires the verified resumable native source'
    old='''                designs = self.actor_net(X_actor)
                X = torch.cat([X_actor, designs], dim=-1)
                output = -self.critic_net(X).mean()'''
    assert src.count(old)==1
    src=src.replace(old,'                output = None')
    old='''            self.actor_optimizer.zero_grad()
            output.backward()
            self.actor_optimizer.step()'''
    assert src.count(old)==1
    src=src.replace(old,'            self.cer_actor_update(X_actor)')
    old='''                            ds_next = self.actor_target(k + 1,
                                self.ds_hist[:, :k + 1],
                                self.ys_hist[:, :k + 1],
                                None)
                            next_action_value = self.critic_target(k + 1,
                                self.ds_hist[:, :k + 1],
                                self.ys_hist[:, :k + 1],
                                None,
                                ds_next)'''
    assert src.count(old)==1
    src=src.replace(old,'''                            next_action_value = self.cer_target(k + 1,
                                self.ds_hist[:, :k + 1], self.ys_hist[:, :k + 1])''')
    namespace=dict(native_module.__dict__)
    exec(compile(textwrap.dedent(src),'<cer_native_policy_expectations>','exec'),namespace)
    a.train=functools.partial(namespace['train'],a)
    return src
