"""RL-sBOED-I: documented location adapter of Blau et al. (2022) SAC/REDQ.

Two independent summary/Q networks, tanh-Gaussian policy, replay, target
networks and incremental PCE rewards. No simulator derivatives. Common
undiscounted EIG and forward cap replace the original gamma=.9/L=100000.
"""
import copy,math,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from src.research.engine import rollout

class Summary(nn.Module):
    def __init__(self,action_dim,obs_dim,horizon,unbounded=False):
        super().__init__();self.horizon=horizon;self.width=action_dim+obs_dim+1;self.unbounded=unbounded
        self.encoder=nn.Sequential(nn.Linear(action_dim+obs_dim,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,64))
    def forward(self,f):
        slots=f[:,1:].reshape(len(f),self.horizon,self.width)
        inputs=slots[...,:-1].float()
        if self.unbounded:inputs=torch.asinh(inputs)
        tokens=self.encoder(inputs)*slots[...,-1:].float()
        pooled=tokens.sum(1)
        if self.unbounded:
            pooled=pooled/slots[...,-1:].float().sum(1).clamp_min(1.)
            pooled=F.layer_norm(pooled,(pooled.shape[-1],))
        return torch.cat((pooled,f[:,:1].float()),-1)

class RLPolicy(nn.Module):
    architecture='rl_sboed_incremental_pce_unbounded_stable_v2'
    kind='rl_sboed';outputs_design=True
    def __init__(self,benchmark,horizon):
        super().__init__();self.sequential=benchmark.name=='sir_ode';self.outputs_design=not self.sequential;self.raw_latent=self.sequential
        self.summary=Summary(benchmark.action_dim,benchmark.obs_dim,horizon,not self.sequential and benchmark.design_bounds is None)
        self.head=nn.Sequential(nn.Linear(65,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,2*benchmark.action_dim))
        self.action_dim=benchmark.action_dim;self.bound=1. if self.sequential else benchmark.action_scale;self.stochastic=False;self.unbounded=not self.sequential and benchmark.design_bounds is None
        with torch.no_grad():self.head[-1].bias[self.action_dim:].fill_(math.log(math.sqrt(1/3)))
    def sample(self,f,deterministic=False,rng=None):
        mean,logstd=self.head(self.summary(f)).chunk(2,-1)
        dist=torch.distributions.Normal(mean,logstd.clamp(-20,0).exp())
        z=mean if deterministic else (dist.rsample() if rng is None else mean+dist.scale*torch.randn(mean.shape,dtype=mean.dtype,device=mean.device,generator=rng))
        action=z if self.unbounded or self.raw_latent else z.tanh()
        if self.raw_latent:
            logp=(dist.log_prob(z)-math.log(2)+F.softplus(-z)+F.softplus(z)).sum(-1)
        else:logp=dist.log_prob(z).sum(-1) if self.unbounded else (dist.log_prob(z)-2*(math.log(2)-z-F.softplus(-2*z))).sum(-1)
        return action,logp
    def distribution(self,f):
        mean,logstd=self.head(self.summary(f)).chunk(2,-1)
        return torch.distributions.Normal(mean,logstd.clamp(-20,0).exp())
    def forward(self,f,stage):
        return (self.sample(f,deterministic=not self.stochastic)[0]*self.bound).double()

class QNetwork(nn.Module):
    def __init__(self,b,h):
        super().__init__();self.unbounded=b.name!='sir_ode' and b.design_bounds is None;self.sequential=b.name=='sir_ode';self.summary=Summary(b.action_dim,b.obs_dim,h,self.unbounded)
        self.head=nn.Sequential(nn.Linear(65+b.action_dim,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1))
    def forward(self,f,a):return self.head(torch.cat((self.summary(f),torch.asinh(a.float()) if self.unbounded or self.sequential else a.float()),-1)).squeeze(-1)

def checked_step(loss,optimizer,parameters,label):
    parameters=list(parameters)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite '+label+' loss before update')
    optimizer.zero_grad();loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in parameters):raise FloatingPointError('Nonfinite '+label+' parameters')
    return float(norm)

class Replay:
    def __init__(self,capacity,inputs,actions,device):
        self.capacity=capacity;self.position=self.count=0
        self.data=torch.empty(capacity,2*inputs+actions+2,device=device,dtype=torch.float32)
    def add(self,x):
        ids=(torch.arange(len(x),device=x.device)+self.position)%self.capacity
        self.data[ids]=x;self.position=(self.position+len(x))%self.capacity;self.count=min(self.capacity,self.count+len(x))
    def sample(self,size):return self.data[torch.randint(self.count,(size,),device=self.data.device)]

def incremental_rewards(info):
    return info-torch.cat((torch.zeros_like(info[:,:1]),info[:,:-1]),1)

def train_rl_sboed(b,h,s,seed,directory,trajectory_budget=None):
    if b.name not in {'location_finding','sir_ode'}:raise ValueError('RL-sBOED adapter supports location/SIR only')
    if not getattr(b,'likelihood_available',True):raise ValueError('PCE RL-sBOED needs explicit likelihood')
    torch.manual_seed(seed);started=time.perf_counter()
    p=RLPolicy(b,h);device=next(p.parameters()).device
    q=nn.ModuleList([QNetwork(b,h) for _ in range(2)]);target=copy.deepcopy(q)
    for parameter in target.parameters():parameter.requires_grad_(False)
    actoropt=torch.optim.Adam(p.parameters(),lr=.0001);qopt=torch.optim.Adam(q.parameters(),lr=.0003)
    logalpha=nn.Parameter(torch.tensor(math.log(.01),device=device));alphaopt=torch.optim.Adam([logalpha],lr=.0001)
    inputs=1+h*(b.action_dim+b.obs_dim+1)
    capacity=s.get('rl_replay_capacity',1000000);replay=Replay(capacity,inputs,b.action_dim,device)
    batch=s.get('rl_batch_size',100);replaybatch=s.get('rl_replay_batch',4096)
    minbuffer=s.get('rl_min_buffer',100000);gradient_steps=s.get('rl_gradient_steps',64)
    contrast=s['contrasts'];generated=spent=update=gradupdates=0;records=[]
    maxupdates=s['updates'] if trajectory_budget is None else math.ceil(trajectory_budget/(batch*(1 if s.get('primary_episode_budget',0) else contrast+1)))
    while update<maxupdates:
        n=batch if trajectory_budget is None else min(batch,(trajectory_budget-spent)//(1 if s.get('primary_episode_budget',0) else contrast+1))
        if n<1:break
        update+=1;p.stochastic=True
        with torch.no_grad():r=rollout(b,p,h,n,seed+update*104729,contrast)
        p.stochastic=False;generated+=n;spent+=n*(1 if s.get('primary_episode_budget',0) else contrast+1)
        f=torch.stack(r['features'],1).reshape(-1,inputs).float()
        nf=torch.stack(r['features'][1:]+[r['final_features']],1).reshape_as(f).float()
        a=torch.stack(r['latents'] if p.sequential else r['actions'],1).reshape(-1,b.action_dim).float()/p.bound
        reward=incremental_rewards(r['info']).reshape(-1,1).float()
        done=torch.zeros(n,h,device=device);done[:,-1]=1
        replay.add(torch.cat((f,a,reward,done.reshape(-1,1),nf),-1))
        reference=copy.deepcopy(p) if p.unbounded else None
        rejected=0;lastkl=0.
        if replay.count>=minbuffer:
            for _ in range(gradient_steps):
                data=replay.sample(replaybatch)
                state=data[:,:inputs];action=data[:,inputs:inputs+b.action_dim]
                rewards=data[:,inputs+b.action_dim];terminal=data[:,inputs+b.action_dim+1];nextstate=data[:,-inputs:]
                alpha=logalpha.exp().detach()
                with torch.no_grad():
                    na,nlp=p.sample(nextstate)
                    tq=torch.stack([net(nextstate,na) for net in target]).min(0).values
                    expected=rewards+(1-terminal)*(tq-alpha*nlp)
                values=torch.stack([net(state,action) for net in q]);qloss=(values-expected).square().mean()
                checked_step(qloss,qopt,q.parameters(),'critic')
                for parameter in q.parameters():parameter.requires_grad_(False)
                proposed,lp=p.sample(state)
                aloss=(alpha*lp-torch.stack([net(state,proposed) for net in q]).min(0).values).mean()
                if reference is not None:
                    with torch.no_grad():old_dist=reference.distribution(state)
                    kl=torch.distributions.kl_divergence(old_dist,p.distribution(state)).sum(-1).mean()
                    aloss=aloss+kl
                    saved=copy.deepcopy(p.state_dict());saved_opt=copy.deepcopy(actoropt.state_dict())
                checked_step(aloss,actoropt,p.parameters(),'actor')
                if reference is not None:
                    with torch.no_grad():lastkl=float(torch.distributions.kl_divergence(old_dist,p.distribution(state)).sum(-1).mean())
                    if not math.isfinite(lastkl) or lastkl>.02:
                        p.load_state_dict(saved);actoropt.load_state_dict(saved_opt);rejected+=1
                for parameter in q.parameters():parameter.requires_grad_(True)
                alphloss=-(logalpha*(lp.detach()-b.action_dim)).mean()
                checked_step(alphloss,alphaopt,[logalpha],'temperature')
                with torch.no_grad():
                    logalpha.clamp_(-10,2)
                    for original,destination in zip(q.parameters(),target.parameters()):destination.lerp_(original,.001)
                gradupdates+=1
                if not all(torch.isfinite(x) for x in (qloss,aloss,alphloss)):raise FloatingPointError('Nonfinite RL-sBOED update')
        if update%max(1,s.get('rl_log_every',10))==0 or update==maxupdates:
            row=dict(update=update,training_EIG=float(r['info'][:,-1].mean()),training_forward_trajectories=generated*(contrast+1),
                replay_transitions=replay.count,gradient_updates=gradupdates,alpha=float(logalpha.exp().detach()),
                policy_kl=lastkl,rejected_actor_steps=rejected,maximum_rollout_action=float(a.abs().max()))
            records.append(row);print(f'rl_sboed update {update}: training EIG={row["training_EIG"]:.6f}; forward={spent}; gradient_updates={gradupdates}',flush=True)
    p.stochastic=False
    torch.save(dict(kind='rl_sboed',architecture=p.architecture,horizon=h,state_dict=p.state_dict(),selected_update=update),directory/'rl_sboed.pt')
    return p,dict(training='RL-sBOED-I SAC_REDQ_two_critic_adaptation',policy_architecture=p.architecture,
        reward='incremental PCE; rewards telescope to terminal PCE',gamma=1.,checkpoint_selection='final',
        generated_training_trajectories=generated,contrast_training_trajectories=generated*contrast,training_forward_trajectories=generated*(contrast+1),
        training_trajectory_budget=trajectory_budget,training_budget_unit='primary_episodes' if s.get('primary_episode_budget',0) else 'forward_including_contrasts',training_seconds=time.perf_counter()-started,history=records,
        actual_updates=update,gradient_updates=gradupdates,critic_count=2,target_subset=2,
        target_rate=.001,actor_lr=.0001,critic_lr=.0003,replay_capacity=capacity,
        replay_batch=replaybatch,minimum_replay_transitions=minbuffer,gradient_steps_per_update=gradient_steps,
        target_entropy=-b.action_dim,entropy_coordinates='normalized_feasible_fraction_minus1_plus1' if p.sequential else 'unbounded_Gaussian' if p.unbounded else 'normalized_tanh_minus1_plus1',
        unbounded_stabilization=dict(history='asinh,masked_mean,layer_norm',critic_actions='asinh',critic_actor_value='minimum_two_Q',gradient_clip=1.,policy_kl_penalty=1.,maximum_rollout_reference_mean_kl=.02) if p.unbounded else None,
        policy_parameters=sum(x.numel() for x in p.parameters()),simulator_gradients_required=False,
        upstream='https://github.com/csiro-mlai/RL-BOED',upstream_commit='54fdc7b6d23849ab1df8fd85d42af7c8961f930e',
        adaptations=['gamma1 undiscounted EIG','common L and forward cap','shared fixed history normalization with stage index','stochastic evaluation with independent decision RNG','standalone Torch implementation without Garage'],
        publication_algorithm_equivalence=False)
