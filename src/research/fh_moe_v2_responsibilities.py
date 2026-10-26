"""Offline terminal-information-gradient responsibilities for unchanged DeepSeekMoE-v1.
No deployed model, routing architecture or simulator is defined in this module.
"""
import copy
from contextlib import contextmanager
import torch

@contextmanager
def capture_terminal_context(model,stage):
    state={};current=[-1]
    def pre(module,args):current[0]=int(args[1])
    def router_pre(module,args):
        if current[0]==stage:state['z']=args[0]
    def router_post(module,args,out):
        if current[0]==stage:state['logits']=out
    def emitter_pre(module,args):
        if current[0]==stage:state['u']=args[0]
    hooks=[model.register_forward_pre_hook(pre),model.router.register_forward_pre_hook(router_pre),
           model.router.register_forward_hook(router_post),model.emitter.register_forward_pre_hook(emitter_pre)]
    try:yield state
    finally:
        for h in hooks:h.remove()

class GradientResponsibilities:
    """Two independent EMA reference streams; seven expert-indexed soft components,
    NOT a requirement to discover seven distinct clusters. Unstable/collapsed banks
    produce no auxiliary update. Targets use ONLY previously observed gradients.
    """
    def __init__(self,decay=.95,temperature=.25):
        self.decay=decay;self.temperature=temperature;self.stages={}
    def state_dict(self):return dict(decay=self.decay,temperature=self.temperature,stages=copy.deepcopy(self.stages))
    def load_state_dict(self,state):
        self.decay=state['decay'];self.temperature=state['temperature'];self.stages=copy.deepcopy(state['stages'])
    @staticmethod
    def exposure(logits):
        p=logits.detach().softmax(-1);ids=p.topk(3,-1).indices
        return p*torch.nn.functional.one_hot(ids,7).sum(1)
    def update(self,stage,demand,exposure):
        d=demand.detach().double().cpu();w=exposure.detach().double().cpu()
        norm=d.norm(dim=-1);valid=norm>1e-12;d=d/norm.clamp_min(1e-12)[:,None];w=w*valid[:,None]
        if stage not in self.stages:self.stages[stage]=dict(moment=torch.zeros(2,7,d.shape[-1],dtype=torch.double,device='cpu'),mass=torch.zeros(2,7,dtype=torch.double,device='cpu'),squared_mass=torch.zeros(2,7,dtype=torch.double,device='cpu'),updates=0)
        s=self.stages[stage]
        for half in [0,1]:
            x=d[half::2];a=w[half::2]
            s['moment'][half]=self.decay*s['moment'][half]+a.T@x
            s['mass'][half]=self.decay*s['mass'][half]+a.sum(0)
            s['squared_mass'][half]=self.decay**2*s['squared_mass'][half]+a.square().sum(0)
        s['updates']+=1
    def references(self,stage,device,dtype):
        if stage not in self.stages:return None,dict(active=0,reason='NO_PAST_REFERENCE')
        s=self.stages[stage];mean=s['moment']/s['mass'].clamp_min(1e-30)[:,:,None]
        strength=mean.norm(dim=-1);unit=mean/strength.clamp_min(1e-30)[:,:,None]
        agreement=(unit[0]*unit[1]).sum(-1)
        ess=s['mass'].square()/s['squared_mass'].clamp_min(1e-30)
        reliable=(agreement>=.9)&(strength.min(0).values>=.1)&(ess.min(0).values>=100)
        c=mean.mean(0);c=c/c.norm(dim=-1).clamp_min(1e-30)[:,None]
        ids=reliable.nonzero().flatten();distinct=False
        if len(ids)>=2:
            cos=c[ids]@c[ids].T
            distinct=bool(cos.min()<.8)
        record=dict(active=int(reliable.sum()),agreement=agreement.tolist(),strength=strength.tolist(),ESS=ess.tolist(),distinct=distinct,reference_updates=s['updates'])
        if not distinct:return None,{**record,'reason':'NO_REPRODUCIBLE_SEPARATED_DEMANDS'}
        return (c.to(device=device,dtype=dtype),reliable.to(device)),record
    def targets(self,stage,demand):
        ref,record=self.references(stage,demand.device,demand.dtype)
        if ref is None:return None,record
        c,valid=ref;g=demand.detach();g=g/g.norm(dim=-1).clamp_min(1e-30)[:,None]
        logits=(g@c.T)/self.temperature;logits=logits.masked_fill(~valid[None],-torch.inf)
        r=logits.softmax(-1).detach()
        return r,record

def auxiliary_losses(model,snapshot,demand,responsibility):
    """Conditional scalar surrogates with fixed terminal adjoint/targets.
    History z and router logits remain LIVE: their valid auxiliary history
    derivatives and the full original pathwise history gradient are retained.
    """
    z=snapshot['z'];features=torch.stack([e(z) for e in model.routed],1)
    expert=(responsibility.detach()*(features*demand.detach()[:,None,:]).sum(-1)).sum(-1).mean()
    router=-(responsibility.detach()*snapshot['logits'].log_softmax(-1)).sum(-1).mean()
    return expert,router

def gradient_list(loss,parameters,retain_graph=True):
    raw=torch.autograd.grad(loss,parameters,retain_graph=retain_graph,allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for p,g in zip(parameters,raw)]
def group_indices(model):
    names=[n for n,p in model.named_parameters()]
    groups=[]
    for prefix in ['encoder.','shared.','routed.','router.','emitter.']:
        ids=[i for i,n in enumerate(names) if n.startswith(prefix)]
        if ids:groups.append(ids)
    covered=set(i for g in groups for i in g);groups.append([i for i in range(len(names)) if i not in covered])
    return [g for g in groups if g]
def cap_coefficient(base,extra,groups,fraction=.05):
    """Fixed coefficient for this conditional auxiliary step; norm contribution
    <=5% of original gradient in EVERY parameter group. No unbiased-gradient
    claim is made for the resulting reference-dependent training objective.
    """
    coefficient=1.
    for ids in groups:
        a=sum(base[i].detach().square().sum() for i in ids).sqrt()
        b=sum(extra[i].detach().square().sum() for i in ids).sqrt()
        if b>0:coefficient=min(coefficient,float(fraction*a/b))
    return coefficient
def information_gradients(model,information_loss,balance,snapshot,bank,stage):
    params=list(model.parameters());B=len(snapshot['u'])
    demand=torch.autograd.grad(information_loss,snapshot['u'],retain_graph=True)[0].detach()*B
    targets,record=bank.targets(stage,demand)
    if targets is not None:
        le,lr=auxiliary_losses(model,snapshot,demand,targets)
        ge=gradient_list(le,params);gr=gradient_list(lr,params)
    # The original full-horizon pass is LAST and frees the complete trajectory graph.
    # Retaining it after assembling detached optimizer gradients wastes GPU memory.
    raw_base=torch.autograd.grad(information_loss+balance,params,retain_graph=False,allow_unused=True)
    base=[torch.zeros_like(p) if g is None else g for p,g in zip(params,raw_base)]
    if targets is None:
        final=base;record.update(auxiliary_applied=False,expert_coefficient=0.,router_coefficient=0.)
    else:
        groups=group_indices(model);ae=cap_coefficient(base,ge,groups);ar=cap_coefficient(base,gr,groups)
        final=[a+ae*b+ar*c for a,b,c in zip(base,ge,gr)]
        record.update(auxiliary_applied=bool(ae>0 and ar>0),expert_coefficient=ae,router_coefficient=ar,
                      expert_loss=float(le.detach()),router_loss=float(lr.detach()),
                      responsibility_mean=targets.mean(0).tolist(),
                      gradient_target_entropy=float(-(targets*targets.clamp_min(1e-30).log()).sum(-1).mean()))
    # Fresh sample enters future references only AFTER target construction.
    bank.update(stage,demand,bank.exposure(snapshot['logits']))
    for p,g,original in zip(params,final,raw_base):
        if not torch.isfinite(g).all():raise FloatingPointError('Nonfinite full-horizon responsibility gradient')
        p.grad=None if original is None and not bool((g!=0).any()) else g.detach()
    return record,demand,targets
