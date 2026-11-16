"""MoE-sBOED: history attention and four policy experts, trained with sPCE."""
import math
import torch
from torch import nn

class MoEPolicy(nn.Module):
    architecture = 'attention_policy_moe_v1'
    def __init__(self, action_dim, obs_dim, horizon, hidden=64, kind='moe_sboed', temperature=1., first_action=None):
        super().__init__()
        if temperature<=0:raise ValueError('Router temperature must be positive')
        self.kind,self.horizon,self.action_dim,self.obs_dim=kind,horizon,action_dim,obs_dim
        self.temperature=temperature;self.hidden=hidden
        self.tokens=nn.Sequential(nn.Linear(action_dim+obs_dim+1,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh())
        self.queries=nn.Parameter(torch.randn(4,hidden)/math.sqrt(hidden))
        self.empty=nn.Parameter(torch.zeros(hidden))
        context=3*hidden+1
        def head(width):return nn.Sequential(nn.Linear(context,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,action_dim))
        self.router=nn.Sequential(nn.Linear(context,hidden),nn.Tanh(),nn.Linear(hidden,4))
        nn.init.zeros_(self.router[-1].weight);nn.init.zeros_(self.router[-1].bias)
        if kind=='matched_dense':
            target=4*((context+1)*hidden+(hidden+1)*hidden+(hidden+1)*action_dim)
            def count(w):return (4*context+1)*w+(w+1)*w+(w+1)*action_dim
            width=min(range(1,8*hidden+1),key=lambda w:abs(count(w)-target))
            self.network=nn.Sequential(nn.Linear(4*context,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,action_dim))
        else:self.experts=nn.ModuleList([head(hidden) for _ in range(4)])
        if kind!='moe_sboed':
            for p in self.router.parameters():p.requires_grad_(False)
        if first_action is not None:self.first_design=nn.Parameter(torch.tensor(first_action,dtype=torch.float64))
        self.double()
    def context(self,features):
        b=len(features);slots=features[:,1:].reshape(b,self.horizon,self.action_dim+self.obs_dim+1)
        valid=slots[...,-1]>0
        pos=torch.arange(self.horizon,device=features.device,dtype=features.dtype)/max(self.horizon-1,1)
        tokens=self.tokens(torch.cat((slots[...,:-1],pos[None,:,None].expand(b,-1,1)),-1))
        logits=torch.einsum('bth,eh->bet',tokens,self.queries)/math.sqrt(self.hidden)
        attention=logits.masked_fill(~valid[:,None,:],-1e9).softmax(-1)*valid[:,None,:]
        attention=attention/attention.sum(-1,keepdim=True).clamp_min(1.)
        pools=torch.einsum('bet,bth->beh',attention,tokens)
        empty=~valid.any(-1)
        pools=torch.where(empty[:,None,None],self.empty[None,None,:],pools)
        mean=(tokens*valid[...,None]).sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
        last_index=(valid.sum(-1)-1).clamp_min(0)
        last=tokens[torch.arange(b,device=features.device),last_index]
        mean=torch.where(empty[:,None],self.empty[None,:],mean)
        last=torch.where(empty[:,None],self.empty[None,:],last)
        context=torch.cat((pools,mean[:,None,:].expand(-1,4,-1),last[:,None,:].expand(-1,4,-1),features[:,:1,None].transpose(1,2).expand(-1,4,-1)),-1)
        return context
    def components(self,features):
        context=self.context(features)
        proposals=torch.stack([expert(context[:,i]) for i,expert in enumerate(self.experts)],1)
        weights=(self.router(context.mean(1))/self.temperature).softmax(-1) if self.kind=='moe_sboed' else torch.full_like(proposals[...,0],.25)
        if hasattr(self,'first_design'):
            proposals=torch.where((features[:,0]==0)[:,None,None],self.first_design[None,None,:],proposals)
        return weights,proposals
    def forward(self,features,stage):
        if hasattr(self,'first_design') and stage==0:return self.first_design[None,:].expand(len(features),-1)
        if self.kind=='matched_dense':return self.network(self.context(features).flatten(1))
        weights,proposals=self.components(features)
        return (weights[...,None]*proposals).sum(1)
