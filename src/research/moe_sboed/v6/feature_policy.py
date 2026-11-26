"""Independent V6 actor with ordered history and a soft mixture of normalized features."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from src.research.policies import mlp
from src.research.moe_sboed.v6.policy import METHODS


class FeaturePolicy(nn.Module):
    kind = "independent_feature_v6"

    def __init__(self, method, horizon, action_dim, observation_dim):
        super().__init__()
        if method not in METHODS or min(horizon, action_dim, observation_dim) < 1:
            raise ValueError("Invalid independent feature policy configuration")
        self.horizon = horizon
        self.action_dim = action_dim
        self.width = action_dim + observation_dim
        self.history = nn.Linear(horizon * self.width, 128)
        token_width = 2 * self.width + action_dim * observation_dim
        self.token = nn.Sequential(nn.Linear(token_width, 16), nn.SiLU())
        self.context = nn.Sequential(nn.Linear(128 + 4 * 16, 64), nn.SiLU())
        # All four configurations draw the same fresh seed-specific slots.
        slots = [nn.Sequential(mlp(65, 64, 128), nn.LayerNorm(128, elementwise_affine=False))
                 for _ in range(4)]
        self.experts = nn.ModuleList(slots if method in METHODS[:2] else [slots[METHODS.index(method)-2]])
        self.head = nn.Sequential(nn.Linear(128, 32), nn.SiLU(), nn.Linear(32, action_dim))
        self.router = None
        if method == METHODS[0]:
            self.router = nn.Sequential(nn.Linear(65 + 4 * 128, 64), nn.SiLU(), nn.Linear(64, 4))
            nn.init.zeros_(self.router[-1].weight)
            nn.init.zeros_(self.router[-1].bias)
        index = torch.arange(horizon, dtype=torch.float64)
        position = (index + .5) / max(1, horizon - 1)
        self.register_buffer("basis", torch.stack((torch.ones_like(index), position,
                             torch.cos(math.pi * position), torch.ones_like(index)), -1), persistent=False)
        self.register_buffer("decay", torch.tensor([1., 1., 1., math.exp(-.2)], dtype=torch.float64),
                             persistent=False)
        self.double()

    def start_history(self, sample):
        batch = len(sample)
        return (self.history.bias.expand(batch, -1), sample.new_zeros(batch, 4, 16),
                sample.new_zeros(batch, self.width))

    def update_history(self, state, token, stage):
        history, moments, previous = state
        action = token[:, :self.action_dim]
        observation = token[:, self.action_dim:].tanh()
        pair = torch.cat((action, observation), -1)
        interaction = (action.unsqueeze(-1) * observation.unsqueeze(-2)).flatten(1)
        embedding = self.token(torch.cat((pair, interaction, pair - previous), -1))
        weight = self.history.weight[:, stage*self.width:(stage+1)*self.width]
        history = history + F.linear(pair, weight)
        # The most recent observation is one stage old when the next action is taken.
        moments = (moments + embedding.unsqueeze(1) * self.basis[stage, :, None]) * self.decay[:, None]
        return history, moments, pair

    def mixture(self, state, stage):
        history, moments, _ = state
        ctx = self.context(torch.cat((F.silu(history), moments.flatten(1) / max(1, stage)), -1))
        ctx = torch.cat((ctx, ctx.new_full((len(ctx), 1), (self.horizon-stage)/self.horizon)), -1)
        features = torch.stack([expert(ctx) for expert in self.experts], 1)
        if self.router is None:
            weights = ctx.new_full((len(ctx), len(self.experts)), 1 / len(self.experts))
        else:
            weights = self.router(torch.cat((ctx, features.flatten(1)), -1)).softmax(-1)
        return features, weights

    def design_logits(self, state, stage):
        features, weights = self.mixture(state, stage)
        # The location feasibility map converts 2*x to 4*tanh(x) on [-4,4].
        return 2 * self.head((features * weights.unsqueeze(-1)).sum(1))

    def forward(self, features, stage):
        tokens = features[:, 1:].reshape(len(features), self.horizon, self.width + 1)
        state = self.start_history(features)
        for i in range(stage):
            state = self.update_history(state, tokens[:, i, :self.width], i)
        return self.design_logits(state, stage)
