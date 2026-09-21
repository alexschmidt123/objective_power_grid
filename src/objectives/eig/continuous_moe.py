"""Independent sparse Gaussian-mixture design policy, trained by on-policy PPO.

Only ordered observed history enters the actor. No baseline actor, teacher,
simulator state, or true parameter is used as a policy input. The common IEEE9
engine supplies sPCE rewards and the chronological continuous feasibility map.
"""
from __future__ import annotations
import copy
import json
import os
import time
import numpy as np
import torch

ARCHITECTURE = 'independent_sparse_continuous_moe_v1'


class ContinuousMoEPolicy(torch.nn.Module):
    def __init__(self, horizon, n_obs, n_experts=4, top_k=2, hidden=64):
        super().__init__()
        self.horizon, self.n_obs = horizon, n_obs
        self.n_experts, self.top_k = n_experts, top_k
        if not 1 <= top_k <= n_experts:
            raise ValueError('Invalid sparse routing budget')
        width = 1+horizon*(n_obs+2)
        self.encoder = torch.nn.Sequential(torch.nn.Linear(width,hidden),torch.nn.Tanh(),
                                           torch.nn.Linear(hidden,hidden),torch.nn.Tanh())
        self.router = torch.nn.Linear(hidden,n_experts)
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(hidden,hidden),torch.nn.Tanh(),
                                torch.nn.Linear(hidden,2)) for _ in range(n_experts)])
        self.critic = torch.nn.Sequential(torch.nn.Linear(width,hidden),torch.nn.Tanh(),
                                          torch.nn.Linear(hidden,hidden),torch.nn.Tanh(),
                                          torch.nn.Linear(hidden,1))

    def components(self, features):
        encoded = self.encoder(features)
        gate_logits = self.router(encoded)
        selected_logits, selected = gate_logits.topk(self.top_k,dim=-1)
        flat = torch.zeros(len(features)*self.top_k,2,device=features.device)
        for expert_id,expert in enumerate(self.experts):
            rows,slots = torch.where(selected == expert_id)
            if len(rows):
                flat = flat.index_copy(0,rows*self.top_k+slots,expert(encoded[rows]))
        parameters = flat.reshape(len(features),self.top_k,2)
        means = 4.*torch.tanh(parameters[...,0]/4.)
        stds = (parameters[...,1]-.5).clamp(-3.,1.).exp()
        distribution = torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(logits=selected_logits),
            torch.distributions.Normal(means,stds))
        return distribution, gate_logits, selected, means

    def forward(self, features, stage):
        distribution,_,_,_ = self.components(features)
        return distribution,self.critic(features).squeeze(-1)

    @torch.no_grad()
    def routing_record(self,features):
        distribution,logits,selected,means = self.components(features)
        return {'selected_experts':selected.tolist(),
                'selected_weights':distribution.mixture_distribution.probs.tolist(),
                'router_probabilities':logits.softmax(-1).tolist(),
                'selected_expert_latent_means':means.tolist()}


def improve_moe(engine,policy,optimizer,rng,batch):
    # Rollouts have no simulator Jacobian: PPO uses the mixture log density.
    with torch.no_grad():
        result = engine.rollout(policy,rng,batch,stochastic=True)
        features = [engine.features(result['actions'][:t],result['observations'][:t],batch,t)
                    for t in range(engine.horizon)]
        # OnlineEIG stores the exact double-precision sigmoid of the latent draw.
        latent = torch.stack([torch.logit(torch.as_tensor(q)).float()
                              for q in result['unit_actions']],dim=1)
        if not torch.isfinite(latent).all():
            raise FloatingPointError('Non-finite recovered latent action')
        old_logp = torch.stack(result['logprobs'],dim=1)
        old_values = torch.stack(result['values'],dim=1)
        info = result['info']
        returns = torch.tensor(info[:,-1,None]-np.c_[np.zeros(batch),info[:,:-1]],dtype=torch.float32)
        advantage = returns-old_values
        advantage = (advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-6)
    losses=[]; measures=[]
    stable=getattr(policy,'training_mode','legacy')=='stable'
    critic_parameters=list(policy.critic.parameters())
    critic_ids={id(p) for p in critic_parameters}
    actor_parameters=[p for p in policy.parameters() if id(p) not in critic_ids]
    accepted=0;rejected=0
    with torch.no_grad():
        original_selected=torch.cat([policy.components(x)[2] for x in features])
    for _ in range(4):
        logprobs,values,gates=[],[],[]
        for stage,x in enumerate(features):
            distribution,logits,_,_ = policy.components(x)
            logprobs.append(distribution.log_prob(latent[:,stage]))
            values.append(policy.critic(x).squeeze(-1))
            gates.append(logits.softmax(-1))
        logp=torch.stack(logprobs,1)
        ratio=(logp-old_logp).exp()
        actor=-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()
        value_loss=.5*(torch.stack(values,1)-returns).square().mean()
        mean_usage=torch.cat(gates).mean(0)
        balance=policy.n_experts*mean_usage.square().sum()-1.
        loss=actor+value_loss+.01*balance
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite MoE PPO loss')
        optimizer.zero_grad()
        loss.backward()
        actor_norm=float(torch.sqrt(sum(p.grad.square().sum() for p in actor_parameters if p.grad is not None)))
        critic_norm=float(torch.sqrt(sum(p.grad.square().sum() for p in critic_parameters if p.grad is not None)))
        expert_norms=[float(torch.sqrt(sum(p.grad.square().sum() for p in expert.parameters() if p.grad is not None)))
                      if any(p.grad is not None for p in expert.parameters()) else 0. for expert in policy.experts]
        if stable:
            torch.nn.utils.clip_grad_norm_(actor_parameters,1.)
            torch.nn.utils.clip_grad_norm_(critic_parameters,5.)
            before=copy.deepcopy(policy.state_dict());before_optimizer=copy.deepcopy(optimizer.state_dict())
        else:
            torch.nn.utils.clip_grad_norm_(policy.parameters(),5.)
        optimizer.step()
        with torch.no_grad():
            after_logp=torch.stack([policy.components(x)[0].log_prob(latent[:,stage]) for stage,x in enumerate(features)],1)
            logratio=after_logp-old_logp
            approximate_kl=float((logratio.exp()-1.-logratio).mean())
            clip_fraction=float(((logratio.exp()-1.).abs()>.2).float().mean())
            selected=torch.cat([policy.components(x)[2] for x in features])
            switches=float((selected.sort(1).values!=original_selected.sort(1).values).any(1).float().mean())
        measures.append({'approximate_kl':approximate_kl,'clip_fraction':clip_fraction,
            'expert_pair_switch_fraction':switches,'actor_gradient_norm':actor_norm,
            'critic_gradient_norm':critic_norm,'expert_gradient_norms':expert_norms})
        if stable and (not np.isfinite(approximate_kl) or approximate_kl>getattr(policy,'kl_limit',.02)):
            policy.load_state_dict(before);optimizer.load_state_dict(before_optimizer)
            rejected+=1
            break
        accepted+=1
        losses.append(float(loss.detach()))
    policy.last_diagnostics={'ppo_loss':float(np.mean(losses)) if losses else None,
        'dense_router_mean_probabilities':mean_usage.detach().tolist(),
        'ppo_epochs':4,'balance_coefficient':.01,'training_trajectories':batch,
        'training_mode':getattr(policy,'training_mode','legacy'),'accepted_passes':accepted,'rejected_passes':rejected,
        'update_diagnostics':measures}
    return float(info[:,-1].mean())


def train_moe(engine,args,directory):
    torch.manual_seed(args.seed)
    policy=ContinuousMoEPolicy(args.T,engine.n_obs)
    policy.training_mode=getattr(args,'moe_training_mode','legacy')
    optimizer=torch.optim.Adam(policy.parameters(),lr=args.learning_rate)
    rng=np.random.default_rng(args.seed)
    records=[];best=-float('inf');best_state=None;best_update=None
    started=time.monotonic()
    update_history=[]
    for update in range(args.updates+1):
        utility=improve_moe(engine,policy,optimizer,rng,args.batch_size) if update else None
        if update:update_history.append({'update':update,**policy.last_diagnostics})
        if update%args.validate_every==0 or update==args.updates:
            with torch.no_grad():
                validation=engine.rollout(policy,np.random.default_rng(args.seed+900000),
                                          args.validation_systems,stochastic=False)
            score=float(validation['info'][:,-1].mean())
            if not np.isfinite(score):raise FloatingPointError('Non-finite validation score')
            routing=[]
            for stage in range(args.T):
                x=engine.features(validation['actions'][:stage],validation['observations'][:stage],
                                  args.validation_systems,stage)
                record=policy.routing_record(x)
                counts=np.bincount(np.asarray(record['selected_experts']).ravel(),minlength=4)
                routing.append({'stage':stage,'selection_counts':counts.tolist(),
                    'mean_selected_expert_latent_disagreement':float(np.std(record['selected_expert_latent_means'],axis=1).mean())})
            records.append({'update':update,'training_utility':utility,'validation_utility':score,
                'routing':routing,'ppo':getattr(policy,'last_diagnostics',{}),'elapsed_seconds':time.monotonic()-started})
            if score>best:
                best,best_state,best_update=score,copy.deepcopy(policy.state_dict()),update
                temporary=directory/'moe_sboed.pth.tmp'
                torch.save({'state_dict':best_state,'architecture':ARCHITECTURE,'method':'moe_sboed',
                    'horizon':args.T,'n_obs':engine.n_obs,'settings':vars(args),
                    'n_experts':4,'top_k':2,'teacher':None,'best_update':best_update},temporary)
                os.replace(temporary,directory/'moe_sboed.pth')
            (directory/'moe_sboed_optimizer_diagnostics.json').write_text(json.dumps(update_history,indent=2)+'\n')
            (directory/'moe_sboed_training.json').write_text(json.dumps(records,indent=2)+'\n')
            print(f'[independent-continuous-moe] update={update}/{args.updates} validation_spce={score:.6f} elapsed={time.monotonic()-started:.1f}s',flush=True)
    policy.load_state_dict(best_state)
    (directory/'moe_sboed_training_diagnostics.json').write_text(json.dumps({
        'architecture':ARCHITECTURE,'best_update':best_update,'best_validation_utility':best,
        'completed_updates':args.updates,'n_experts':4,'top_k':2,'teacher':None,
        'parameter_count':sum(p.numel() for p in policy.parameters()),
        'actor_inputs':'ordered observed history only; no true MK or simulator state',
        'training':'PPO on sparse Gaussian-mixture latent-action log density; four passes per batch',
        'evaluation':'sigmoid of mixture-weighted latent mean, followed by ordered feasible map',
        'training_mode':policy.training_mode,'total_training_trajectories':args.updates*args.batch_size,
        'kl_limit':.02 if policy.training_mode=='stable' else None,
        'convergence_established':False},indent=2)+'\n')
    return policy,time.monotonic()-started
