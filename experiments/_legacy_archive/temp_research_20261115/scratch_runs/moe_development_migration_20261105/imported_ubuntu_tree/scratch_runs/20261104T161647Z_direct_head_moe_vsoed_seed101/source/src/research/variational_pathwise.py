"""Pathwise optimization of the normalized Barber-Agakov EIG lower bound.

E[log q(theta|history)-log p(theta)] <= I(theta;history). The gap is the
expected posterior KL to q. No action-value critic, replay, MSE utility, or
online fitting is used. Parameter labels are offline density-training targets.
"""
import copy
import math
import time
import torch
from torch import nn
from src.research.policies import mlp,make_policy,parameter_count
from src.research.engine import rollout


class FullCovariancePosterior(nn.Module):
    def __init__(self,inputs,parameters,hidden=128,mixtures=16,source_symmetry=False):
        super().__init__()
        self.dim,self.mixtures,self.source_symmetry=parameters,mixtures,source_symmetry
        if source_symmetry and parameters!=4:raise ValueError('Source symmetry requires four coordinates')
        self.per_component=1+parameters+parameters*(parameters+1)//2
        self.network=mlp(inputs,hidden,mixtures*self.per_component).double()
        row,col=torch.tril_indices(parameters,parameters,device=self.network[-1].weight.device)
        self.register_buffer('row',row);self.register_buffer('col',col)
        with torch.no_grad():
            self.network[-1].weight.zero_();self.network[-1].bias.zero_()
            bias=self.network[-1].bias.reshape(mixtures,self.per_component)
            bias[:,1+parameters+torch.nonzero(row==col).flatten()]=math.log(math.expm1(1-.001))
            if mixtures>1:bias[:,1:1+parameters].normal_(0,.05)

    def parameters_for(self,features):
        raw=self.network(features).reshape(len(features),self.mixtures,self.per_component)
        logits=raw[...,0].log_softmax(-1);means=raw[...,1:1+self.dim]
        scale=raw.new_zeros(len(features),self.mixtures,self.dim,self.dim)
        scale[...,self.row,self.col]=raw[...,1+self.dim:]
        diag=torch.nn.functional.softplus(torch.diagonal(scale,dim1=-2,dim2=-1))+.001
        scale=torch.tril(scale,diagonal=-1)+torch.diag_embed(diag)
        return logits,means,scale

    def log_prob(self,features,z):
        logits,means,scale=self.parameters_for(features)
        def density(value):
            delta=torch.linalg.solve_triangular(scale,(value[:,None]-means)[...,None],upper=False).squeeze(-1)
            component=-.5*delta.square().sum(-1)-torch.diagonal(scale,dim1=-2,dim2=-1).log().sum(-1)-.5*self.dim*math.log(2*math.pi)
            return torch.logsumexp(logits+component,-1)
        direct=density(z)
        if not self.source_symmetry:return direct
        swapped=density(z.reshape(-1,2,2).flip(1).flatten(1))
        return torch.logaddexp(direct,swapped)-math.log(2.)


def train_variational(benchmark,kind,horizon,settings,seed,directory,trajectory_budget):
    torch.manual_seed(seed)
    policy=make_policy(benchmark,horizon,kind,settings['hidden'],settings)
    inputs=1+horizon*(benchmark.action_dim+benchmark.obs_dim+1)
    density=FullCovariancePosterior(inputs,benchmark.param_dim,settings['variational_density_hidden'],
                                  settings['variational_mixtures'],benchmark.name=='location_finding')
    policy_opt=torch.optim.Adam(policy.parameters(),lr=settings['variational_policy_lr'],
                   betas=(settings.get('spce_adam_beta1',.9),settings.get('spce_adam_beta2',.999)))
    density_opt=torch.optim.Adam(density.parameters(),lr=settings['variational_density_lr'])
    scheduler=torch.optim.lr_scheduler.ExponentialLR(policy_opt,settings.get('spce_lr_decay',1.))
    updates=settings['updates'];batch_size=settings['batch_size']
    if trajectory_budget is not None:updates=min(updates,math.ceil(trajectory_budget/batch_size))
    warmup=min(settings['variational_warmup'],updates-1)
    started=time.perf_counter();calls=benchmark.calls;primary=0;records=[];actual=0
    for update in range(1,updates+1):
        batch=batch_size if trajectory_budget is None else min(batch_size,trajectory_budget-primary)
        if batch<1:break
        if update<=warmup:
            with torch.no_grad():result=rollout(benchmark,policy,horizon,batch,seed+update*104729,0)
        else:result=rollout(benchmark,policy,horizon,batch,seed+update*104729,0)
        primary+=batch;actual=update
        features=result['final_features'];z=benchmark.transform(result['truth']).detach()
        for _ in range(settings['variational_density_updates']):
            post_loss=-density.log_prob(features.detach(),z).mean()
            if not torch.isfinite(post_loss):raise FloatingPointError('Nonfinite posterior NLL')
            density_opt.zero_grad();post_loss.backward()
            torch.nn.utils.clip_grad_norm_(density.parameters(),5.);density_opt.step()
        for p in density.parameters():p.requires_grad_(False)
        bound=(density.log_prob(features,z)-benchmark.log_prior_z(z)).mean()
        if not torch.isfinite(bound):raise FloatingPointError('Nonfinite variational information bound')
        grad=None
        if update>warmup:
            policy_opt.zero_grad();(-bound).backward()
            grad=torch.nn.utils.clip_grad_norm_(policy.parameters(),5.)
            if not torch.isfinite(grad):raise FloatingPointError('Nonfinite variational policy gradient')
            policy_opt.step()
            if (update-warmup-1)%settings.get('spce_lr_schedule_every',1000)==0:scheduler.step()
        for p in density.parameters():p.requires_grad_(True)
        if update%settings['validate_every']==0 or update==updates:
            row=dict(update=update,variational_bound=float(bound.detach()),posterior_nll=float(post_loss.detach()),
                     gradient_norm=None if grad is None else float(grad),warmup=update<=warmup,
                     learning_rate=policy_opt.param_groups[0]['lr'])
            records.append(row);print(f'{kind} update {update}: variational EIG bound={row["variational_bound"]:.6f}',flush=True)
    state=copy.deepcopy(policy.state_dict())
    torch.save(dict(kind=kind,horizon=horizon,hidden=settings['hidden'],state_dict=state,selected_update=actual,
          policy_settings={k:settings[k] for k in ['moe_revision','moe_router_temperature','training_bound']},
          density_state=density.state_dict(),density_parameters=dict(inputs=inputs,parameters=benchmark.param_dim,
          hidden=settings['variational_density_hidden'],mixtures=settings['variational_mixtures'],source_symmetry=benchmark.name=='location_finding')),
          directory/f'{kind}.pt')
    return policy,dict(training='pathwise_variational_Barber_Agakov',policy_parameters=parameter_count(policy),
          posterior_parameters=parameter_count(density),history=records,selected_update=actual,
          checkpoint_selection='final_predeclared',validation_spce=None,training_seconds=time.perf_counter()-started,
          actual_updates=actual,policy_updates=max(0,actual-warmup),posterior_updates=actual*settings['variational_density_updates'],
          generated_training_trajectories=primary,contrast_training_trajectories=0,training_forward_trajectories=primary,
          training_trajectory_budget=trajectory_budget,forward_particle_stages=benchmark.calls-calls,
          batch_size=batch_size,policy_learning_rate=settings['variational_policy_lr'],posterior_learning_rate=settings['variational_density_lr'],
          warmup=warmup,moe_revision=settings['moe_revision'],posterior_source_symmetry=benchmark.name=='location_finding',
          objective='total EIG via normalized variational posterior lower bound',
          deployment='policy only; no posterior fitting or online refinement',
          limitation='Training posterior bound is not the independent EIG evaluation metric')
