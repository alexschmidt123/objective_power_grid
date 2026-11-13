"""V3: independently executable design policies with categorical selection."""
import copy
import torch
from torch import nn

def observed_prefix(x):
    if x.shape[-1]!=117 or x.dtype!=torch.float32:
        raise ValueError('History must be float32 [...,117]')
    if not torch.isfinite(x).all():raise ValueError('History must be finite')
    flat=x.reshape(-1,117);clock=flat[:,:30]
    if not bool(((clock==0)|(clock==1)).all()) or not bool((clock.sum(-1)==1).all()):
        raise ValueError('Exactly one of30 stage indicators must be active')
    stage=clock.argmax(-1);past=torch.arange(29,device=x.device)[None]<stage[:,None]
    actions=torch.where(past[...,None],flat[:,30:88].reshape(-1,29,2),0)
    observations=torch.where(past,flat[:,88:],0)
    return torch.cat([clock,actions.flatten(1),observations],-1).reshape_as(x)

class DecisionExpert(nn.Module):
    """A COMPLETE V2 actor, including its own action head, no peer dependency."""
    def __init__(self,parent):
        super().__init__();self.core=copy.deepcopy(parent)
    def forward(self,history):return self.core(observed_prefix(history))

class DecisionMoE(nn.Module):
    def __init__(self,parent,experts=3,seed=202611133001):
        super().__init__()
        if experts<2:raise ValueError('At least two independent expert policies')
        self.experts=nn.ModuleList([DecisionExpert(parent) for _ in range(experts)])
        self.n_experts=experts
        device=next(parent.parameters()).device
        with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
            torch.manual_seed(seed)
            self.router=nn.Sequential(nn.Linear(118+2*experts,128),nn.Tanh(),nn.Linear(128,64),nn.Tanh(),nn.Linear(64,experts)).to(device)
        # Identical initial proposals make a uniform initial selector appropriate.
        nn.init.zeros_(self.router[-1].weight);nn.init.zeros_(self.router[-1].bias)
        self.phase='router';self.set_phase('router')

    def set_phase(self,phase):
        if phase not in ['router','joint']:raise ValueError(phase)
        self.phase=phase
        for expert in self.experts:
            # Keep diagnostic heads frozen exactly as in the V2 native phase.
            for name,p in expert.core.named_parameters():
                p.requires_grad_(phase=='joint' and not name.startswith(('readouts.','response_readouts.')))
                p.grad=None
        self.router.requires_grad_(True)

    def proposals_and_weights(self,history):
        clean=observed_prefix(history)
        proposals=torch.stack([expert.core(clean) for expert in self.experts],-2)
        if not torch.isfinite(proposals).all() or not bool((proposals.abs()<=4.000001).all()):
            raise ValueError('Expert produced invalid design')
        stage=clean[...,:30].argmax(-1)
        remaining=(30-stage).to(clean.dtype)[...,None]/30
        # Router can inspect proposals, but cannot reshape them through this path.
        context=torch.cat([clean,remaining,proposals.detach().flatten(-2)],-1)
        logits=self.router(context);weights=logits.softmax(-1)
        return clean,proposals,logits,weights

    def forward(self,history,mode='learned',expert=None,uniforms=None,collect=False):
        if mode=='expert':
            if expert not in range(self.n_experts):raise ValueError('Expert index outside range')
            action=self.experts[expert](history)
            if collect:return action,dict(selected_expert=torch.full(history.shape[:-1],expert,device=history.device,dtype=torch.long))
            return action
        if mode not in ['learned','uniform','argmax']:raise ValueError(mode)
        _,proposals,logits,weights=self.proposals_and_weights(history)
        if mode=='uniform':weights=torch.full_like(weights,1/self.n_experts)
        if mode=='argmax':chosen=weights.argmax(-1)
        else:
            if uniforms is None:uniforms=torch.rand(weights.shape[:-1],device=history.device,dtype=history.dtype)
            if uniforms.shape!=weights.shape[:-1] or not bool(((uniforms>=0)&(uniforms<1)).all()):
                raise ValueError('Uniforms must have history batch shape and lie in[0,1)')
            chosen=(uniforms[...,None]>=weights.cumsum(-1)).sum(-1).clamp_max(self.n_experts-1)
        action=proposals.gather(-2,chosen[...,None,None].expand(*chosen.shape,1,2)).squeeze(-2)
        if collect:return action,dict(proposals=proposals,logits=logits,weights=weights,selected_expert=chosen)
        return action

    def native_actor_loss(self,critic,history):
        """Exact expectation over categorical routing, with native Q targets.

        No straight-through gradient, soft-action surrogate, auxiliary reward,
        or online optimization. Critic is still trained by native full-horizon
        posterior-derived returns; this computes E_k[-Q(h,a_k)].
        """
        clean,proposals,_,weights=self.proposals_and_weights(history)
        values=torch.stack([critic(torch.cat([clean,proposals[...,k,:]],-1)).squeeze(-1)
                            for k in range(self.n_experts)],-1)
        return -(weights*values).sum(-1).mean()
