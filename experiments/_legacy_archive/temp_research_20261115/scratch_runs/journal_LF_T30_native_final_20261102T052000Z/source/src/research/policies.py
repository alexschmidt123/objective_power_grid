"""Ordered-history policies and the variational posterior used by vsOED.

The history encoder is identical across architecture controls. All experts are
evaluated under dense routing. No policy sees true parameters or physical states.
"""
import math
import torch
from torch import nn


def history_features(benchmark, actions, observations, batch, horizon):
    stage = len(actions)
    slots = []
    for t in range(horizon):
        if t < stage:
            slots.append(torch.cat((benchmark.encode_action(actions[t]),
                                    observations[t]/benchmark.obs_scale,
                                    torch.ones(batch,1,dtype=torch.float64)), -1))
        else:
            slots.append(torch.zeros(batch,benchmark.action_dim+benchmark.obs_dim+1,dtype=torch.float64))
    return torch.cat((torch.full((batch,1),stage/horizon,dtype=torch.float64), *slots), -1)


def mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs,hidden),nn.Tanh(),
                         nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,outputs))


class Policy(nn.Module):
    def __init__(self, inputs, action_dim, horizon, kind='dad', hidden=64):
        super().__init__()
        self.kind, self.horizon = kind, horizon
        if kind == 'fixed':
            self.sequence = nn.Parameter(torch.zeros(horizon,action_dim,dtype=torch.float64))
        else:
            self.network = mlp(inputs,hidden,action_dim)
        self.double()

    def forward(self, features, stage):
        if self.kind == 'fixed':
            return self.sequence[stage].expand(len(features),-1)
        return self.network(features)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def make_policy(benchmark, horizon, kind, hidden, settings=None):
    settings=settings or {}
    if kind not in {'dad','fixed'}:raise ValueError('Retired MoE; sparse replacement is validation-only')
    if kind=='dad' and getattr(benchmark,'paper_dad',False):
        from src.research.dad_location_policy import LocationDAD
        policy=LocationDAD(horizon,benchmark.action_scale,benchmark.obs_scale,kind,hidden,settings.get('first_design_revision',1))
        if benchmark.design_bounds is None and hasattr(policy,'first_design'):
            with torch.no_grad():policy.first_design.fill_(.01)
        return policy
    inputs = 1+horizon*(benchmark.action_dim+benchmark.obs_dim+1)
    width = hidden
    return Policy(inputs,benchmark.action_dim,horizon,kind,width)


class Posterior(nn.Module):
    """History-conditioned joint mixture of diagonal normals in prior coordinates.

    Unlike independent scalar mixtures, the joint mixture can capture dependence
    between unknown parameters. This is a documented vsOED adaptation.
    """
    def __init__(self, inputs, parameters, hidden=128, mixtures=8):
        super().__init__()
        self.parameters_dim, self.mixtures = parameters, mixtures
        self.network = mlp(inputs,hidden,mixtures*(1+2*parameters)).double()

    def log_prob(self, features, z):
        raw = self.network(features).reshape(len(features),self.mixtures,-1)
        weights = raw[...,0].log_softmax(-1)
        mean = raw[...,1:1+self.parameters_dim]
        log_scale = raw[...,1+self.parameters_dim:].clamp(-5,3)
        lp = (-.5*((z[:,None]-mean)*(-log_scale).exp()).square()
              - log_scale-.5*math.log(2*math.pi)).sum(-1)
        return torch.logsumexp(weights+lp,-1)
