"""Dense feature MoE for the static, two-source location experiment."""
import math

import torch
from torch import nn

from src.research.moe_sboed.v6.policy import METHODS
from src.research.policies import mlp


class HistoryExpert(nn.Module):
    def __init__(self, memory_width):
        super().__init__()
        context_width = memory_width + 1
        self.preference = nn.Linear(context_width, memory_width)
        self.features = mlp(context_width + memory_width, 64, context_width)

    def forward(self, context, tokens):
        # A token represents an already observed (design, signal) pair.
        if tokens.shape[1]:
            preference = self.preference(context)
            logits = torch.bmm(tokens, preference.unsqueeze(-1)) / math.sqrt(tokens.shape[-1])
            reading = (logits.softmax(dim=1) * tokens).sum(dim=1)
        else:
            reading = context.new_zeros(len(context), tokens.shape[-1])
        return self.features(torch.cat((context, reading), dim=-1))


class ObservedHistoryMoE(nn.Module):
    kind = "observed_history_moe"

    def __init__(self, method, horizon):
        super().__init__()
        if method not in METHODS or horizon < 1:
            raise ValueError("Expected a registered location policy and positive horizon")
        self.horizon = horizon
        width = 16
        self.encoder = nn.Sequential(nn.Linear(3, 256), nn.ReLU(), nn.Linear(256, width))
        self.head = nn.Linear(width + 1, 2)
        slots = [HistoryExpert(width) for _ in range(4)]
        self.experts = nn.ModuleList(slots if method in METHODS[:2] else [slots[METHODS.index(method)-2]])
        self.router = None
        if method == METHODS[0]:
            self.router = nn.Sequential(nn.Linear(width + 1, 32), nn.Tanh(), nn.Linear(32, 4))
            nn.init.zeros_(self.router[-1].weight)
            nn.init.zeros_(self.router[-1].bias)
        self.double()

    def start_history(self, sample):
        # Only batch size enters. The simulator owns all hidden parameters.
        empty = self.head.weight.new_zeros(len(sample), self.head.in_features - 1)
        return empty, empty[:, None, :][:, :0]

    def update_history(self, state, token, stage):
        total, tokens = state
        encoded = self.encoder(token)
        return total + encoded, torch.cat((tokens, encoded[:, None]), dim=1)

    def mixture(self, state, stage):
        total, tokens = state
        return self.mix_readings(total, [tokens] * len(self.experts), stage)

    def mix_readings(self, total, readings, stage):
        context = torch.cat((total, total.new_full((len(total), 1), stage/self.horizon)), dim=1)
        features = torch.stack([expert(context, tokens) for expert, tokens in zip(self.experts, readings)], dim=1)
        if self.router is None:
            weights = context.new_full(features.shape[:2], 1/len(self.experts))
        else:
            weights = self.router(context).softmax(dim=-1)
        return context, features, weights

    def design_logits(self, state, stage):
        context, features, weights = self.mixture(state, stage)
        return self.head(context + (weights.unsqueeze(-1) * features).sum(dim=1))

    def forward(self, features, stage):
        history = features[:, 1:].reshape(len(features), self.horizon, 4)
        tokens = self.encoder(history[:, :stage, :3])
        return self.design_logits((tokens.sum(dim=1), tokens), stage)
