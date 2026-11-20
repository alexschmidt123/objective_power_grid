"""Training-only frozen-belief residual critic. Not used by the deployed actor."""
import copy,torch
from torch import nn

class BeliefCritic(nn.Module):
    def __init__(self,base,belief):
        super().__init__();self.base=base;self.belief=copy.deepcopy(belief).eval().requires_grad_(False)
        assert sum(v.numel() for v in self.belief.parameters())==0
        self.residual=nn.Sequential(nn.Linear(133,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
        self.to(next(base.parameters()));self._cached_prefix=None;self._cached_summary=None;self.context_evaluations=0;self.cache_hits=0

    @torch.no_grad()
    def summary(self,prefix):
        # Value-based comparison prevents stale reuse when tensor storage is recycled/mutated.
        if self._cached_prefix is not None and self._cached_prefix.shape==prefix.shape and torch.equal(prefix,self._cached_prefix):
            self.cache_hits+=1;return self._cached_summary
        w,mu,sd=self.belief.params(prefix)
        mean=(w[...,None]*mu).sum(1);var=(w[...,None]*(mu.square()+sd.square())).sum(1)-mean.square();scale=var.clamp_min(1e-5).sqrt()
        cov=torch.einsum('bm,bmi,bmj->bij',w,mu-mean[:,None],mu-mean[:,None]);ij=torch.triu_indices(4,4,1,device=prefix.device)
        correlation=cov[:,ij[0],ij[1]]/(scale[:,ij[0]]*scale[:,ij[1]])
        result=torch.cat([mean/3,scale.log(),correlation],-1);assert result.shape==(len(prefix),14) and torch.isfinite(result).all()
        self._cached_prefix=prefix.detach().clone();self._cached_summary=result.detach();self.context_evaluations+=1
        return result

    def forward(self,x):
        shape=x.shape[:-1];flat=x.reshape(-1,119)
        # Native replay prefixes are already observed-history masked. Match the actor guard too.
        from frozen_feature_policy import observed_prefix
        prefix=observed_prefix(flat[:,:117]);summary=self.summary(prefix)
        residual=self.residual(torch.cat([flat,summary],-1))
        return self.base(x)+residual.reshape(*shape,1)

    def __getstate__(self):
        state=self.__dict__.copy();state['_cached_prefix']=None;state['_cached_summary']=None
        return state


def install(a,belief,seed):
    torch.manual_seed(seed%(2**32));torch.cuda.manual_seed_all(seed%(2**32))
    old_optimizer=a.critic_optimizer;old_parameters=list(a.critic_net.parameters())
    online=BeliefCritic(a.critic_net,belief);target=BeliefCritic(a.critic_target_net,belief)
    target.residual.load_state_dict(online.residual.state_dict());target.requires_grad_(False)
    a.critic_net=online;a.critic_target_net=target
    # Preserve every inherited base-critic Adam state. Add only fresh residual parameters.
    before=[id(v) for g in old_optimizer.param_groups for v in g['params']]
    lr=old_optimizer.param_groups[0]['lr'];old_optimizer.add_param_group(dict(params=list(online.residual.parameters()),lr=lr))
    assert [id(v) for v in old_parameters]==before
    a.critic_lr_scheduler.base_lrs.append(a.critic_lr_scheduler.base_lrs[0])
    if hasattr(a.critic_lr_scheduler,'_last_lr'):a.critic_lr_scheduler._last_lr.append(lr)
    return online,target
