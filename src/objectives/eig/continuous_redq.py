"""REDQ-style continuous RL-sBOED (Blau et al., 2022, Appendix C.1).

Ordered histories replace the exchangeable summary for the non-reset model.
Actions are feasible-interval coordinates in [0,1]; rewards telescope to the
terminal utility: sPCE, negative MSC or negative posterior MOCU.
Uses replay, an ensemble of target critics, random-subset minimum targets and
an entropy-regularized tanh-Gaussian actor. Numerical budgets are study settings.
"""
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def mlp(inputs,outputs):
    return nn.Sequential(nn.Linear(inputs,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,outputs))


class REDQPolicy(nn.Module):
    unit_scale=2.
    def __init__(self,horizon,n_obs):
        super().__init__()
        self.network=mlp(1+horizon*(n_obs+2),2)
    def forward(self,features,stage=None):
        mean,log_std=self.network(features).unbind(-1)
        return torch.distributions.Normal(mean,log_std.clamp(-5,1).exp()),torch.zeros_like(mean)
    def sample_action(self,features):
        dist,_=self(features)
        z=dist.rsample()
        unit=(torch.tanh(z)+1)/2
        log_jac=math.log(2)-2*z-2*F.softplus(-2*z)
        return unit,dist.log_prob(z)-log_jac


class REDQTrainer:
    def __init__(self,policy,args):
        self.policy=policy
        self.dimension=1+args.T*(args.N_obs+2)
        self.critics=nn.ModuleList([mlp(self.dimension+1,1) for _ in range(getattr(args,'redq_critics',10))])
        self.targets=copy.deepcopy(self.critics)
        for p in self.targets.parameters():p.requires_grad_(False)
        self.actor_optimizer=torch.optim.Adam(policy.parameters(),lr=3e-4)
        self.critic_optimizer=torch.optim.Adam(self.critics.parameters(),lr=3e-4)
        self.log_alpha=nn.Parameter(torch.tensor(math.log(.01)))
        self.alpha_optimizer=torch.optim.Adam([self.log_alpha],lr=3e-4)
        self.capacity=100000
        self.replay=np.empty((self.capacity,2*self.dimension+3),dtype=np.float32)
        self.position=self.count=0
        self.updates=getattr(args,'redq_updates_per_batch',5)
        self.rng=np.random.default_rng(args.seed+330000)

    def observe(self,engine,result,batch):
        info=result['info']
        for stage in range(engine.horizon):
            s=engine.features(result['actions'][:stage],result['observations'][:stage],batch,stage).numpy()
            ns=engine.features(result['actions'][:stage+1],result['observations'][:stage+1],batch,stage+1).numpy()
            reward=info[:,stage]-(info[:,stage-1] if stage else 0)
            data=np.c_[s,result['unit_actions'][stage],reward,
                       np.full(batch,float(stage==engine.horizon-1)),ns]
            indices=(np.arange(batch)+self.position)%self.capacity
            self.replay[indices]=data
            self.position=(self.position+batch)%self.capacity
            self.count=min(self.count+batch,self.capacity)

    def step(self,engine,rng,batch):
        with torch.no_grad():result=engine.rollout(self.policy,rng,batch,stochastic=True)
        self.observe(engine,result,batch)
        for _ in range(self.updates):
            data=torch.from_numpy(self.replay[self.rng.integers(self.count,size=256)])
            n=self.dimension
            s,a,reward,done,ns=data[:,:n],data[:,n],data[:,n+1],data[:,n+2],data[:,n+3:]
            alpha=self.log_alpha.exp().detach()
            with torch.no_grad():
                na,nlp=self.policy.sample_action(ns)
                target_input=torch.cat([ns,na[:,None]],dim=1)
                subset=self.rng.choice(len(self.targets),size=2,replace=False)
                target_q=torch.stack([self.targets[j](target_input).squeeze(-1) for j in subset]).min(0).values
                # gamma=1 preserves the undiscounted sequential information objective.
                target=reward+(1-done)*(target_q-alpha*nlp)
            inputs=torch.cat([s,a[:,None]],dim=1)
            q=torch.stack([critic(inputs).squeeze(-1) for critic in self.critics])
            qloss=(q-target).square().mean()
            self.critic_optimizer.zero_grad();qloss.backward();self.critic_optimizer.step()
            for p in self.critics.parameters():p.requires_grad_(False)
            action,logp=self.policy.sample_action(s)
            actor_input=torch.cat([s,action[:,None]],dim=1)
            qa=torch.stack([critic(actor_input).squeeze(-1) for critic in self.critics]).mean(0)
            aloss=(alpha*logp-qa).mean()
            self.actor_optimizer.zero_grad();aloss.backward();self.actor_optimizer.step()
            for p in self.critics.parameters():p.requires_grad_(True)
            self.alpha_optimizer.zero_grad()
            alpha_loss=-(self.log_alpha*(logp.detach()+1)).mean()
            alpha_loss.backward();self.alpha_optimizer.step()
            with torch.no_grad():
                self.log_alpha.clamp_(-10,2)
                for target_net,net in zip(self.targets,self.critics):
                    for target_p,p in zip(target_net.parameters(),net.parameters()):
                        target_p.lerp_(p,.005)
        return float(result['info'][:,-1].mean())
