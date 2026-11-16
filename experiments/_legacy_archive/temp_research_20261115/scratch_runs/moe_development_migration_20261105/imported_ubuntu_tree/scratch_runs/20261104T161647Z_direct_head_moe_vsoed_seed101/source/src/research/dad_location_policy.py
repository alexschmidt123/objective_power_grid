"""Location DAD set encoder architecture from ae-foster/dad, with box mapping.

Copyright (c) 2021 Adam Foster, Desi R. Ivanova. MIT license retained in
vendor/dad/LICENSE. Encoder/emitter dimensions follow the location README:
256 hidden units and a 16-dimensional sum-pooled history representation.
The common sPCE training loop and bounded design mapping are adaptations.
"""
import torch
from torch import nn


class LocationDAD(nn.Module):
    kind = 'dad'
    def __init__(self, horizon, action_scale=4., obs_scale=3., kind='dad', hidden=64, first_design_revision=1):
        super().__init__()
        self.horizon, self.action_scale, self.obs_scale = horizon, action_scale, obs_scale
        self.kind = kind
        self.first_design_revision=first_design_revision
        if first_design_revision==2:
            physical=torch.full((2,),.01,dtype=torch.float64)
            self.first_design=nn.Parameter(torch.logit((physical+action_scale)/(2*action_scale)))
        self.encoder = nn.Sequential(nn.Linear(3,256),nn.ReLU(),nn.Linear(256,16))
        self.emitter = nn.Linear(16,2)
        if kind in {'moe_sboed','uniform_moe'}:
            from src.research.policies import mlp
            self.experts = nn.ModuleList([mlp(16,hidden,2) for _ in range(4)])
            self.router = nn.Linear(16,4)
            nn.init.zeros_(self.router.weight); nn.init.zeros_(self.router.bias)
            del self.emitter
        elif kind=='matched_dense':
            from src.research.policies import mlp
            target=4*((16+1)*hidden+(hidden+1)*hidden+(hidden+1)*2)+17*4
            width=min(range(1,hidden*8),key=lambda w:abs((16+1)*w+(w+1)*w+(w+1)*2-target))
            self.emitter=mlp(16,width,2)
        self.double()
    def encode(self, features, stage):
        slots = features[:,1:].reshape(len(features),self.horizon,4)
        if stage:
            raw = torch.cat((slots[:,:stage,:2]*self.action_scale,
                             slots[:,:stage,2:3]*self.obs_scale),-1)
            encoding = self.encoder(raw).sum(1)
        else:
            encoding = torch.zeros(len(features),16,dtype=features.dtype,device=features.device)
        return encoding
    def components(self, features):
        stage=int(round(float(features[0,0])*self.horizon))
        encoding=self.encode(features,stage)
        proposals=torch.stack([expert(encoding) for expert in self.experts],1)
        if stage==0 and self.first_design_revision==2:
            proposals=self.first_design[None,None,:].expand(len(features),4,2)
        weights=self.router(encoding).softmax(-1) if self.kind=='moe_sboed' else torch.full_like(proposals[...,0],.25)
        return weights,proposals
    def forward(self, features, stage):
        if stage==0 and self.first_design_revision==2:
            return self.first_design[None,:].expand(len(features),2)
        if self.kind in {'moe_sboed','uniform_moe'}:
            weights,proposals=self.components(features)
            return (weights[...,None]*proposals).sum(1)
        return self.emitter(self.encode(features,stage))
