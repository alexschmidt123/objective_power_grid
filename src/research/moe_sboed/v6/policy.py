"""Homogeneous residual feature experts for the T=30 location benchmark."""
import torch
from torch import nn

from src.research.policies import mlp


METHODS = ("moe-sboed", "moe-sboed-uniform", *(f"moe-sboed-e{i}" for i in range(1, 5)))


class LocationMoE(nn.Module):
    kind = "location_moe_v6"

    def __init__(self, method, horizon):
        super().__init__()
        if method not in METHODS:
            raise ValueError(method)
        self.method = method
        self.horizon = horizon
        self.encoder = nn.Sequential(nn.Linear(3, 256), nn.ReLU(), nn.Linear(256, 16))
        self.head = nn.Linear(17, 2)
        # Draw the identical seed-specific four slots for every arm. A standalone
        # arm keeps only its own fresh slot; no other slot is registered or trained.
        slots = [mlp(17, 64, 17) for _ in range(4)]
        self.experts = nn.ModuleList(slots if method in METHODS[:2] else [slots[METHODS.index(method) - 2]])
        self.router = nn.Linear(17, 4) if method == METHODS[0] else None
        self.double()

    def weights(self, context):
        if self.router is not None:
            return self.router(context).softmax(-1)
        return context.new_full((len(context), len(self.experts)), 1 / len(self.experts))

    def design_logits(self, encoded, stage):
        context = torch.cat((encoded, encoded.new_full((len(encoded), 1), stage / self.horizon)), -1)
        expert_features = torch.stack([expert(context) for expert in self.experts], 1)
        mixed = (self.weights(context).unsqueeze(-1) * expert_features).sum(1)
        return self.head(context + mixed)

    def forward(self, features, stage):
        # Interface used by the existing rollout. Location history stores x/4 and y/3.
        slots = features[:, 1:].reshape(len(features), self.horizon, 4)
        encoded = self.encoder(slots[:, :stage, :3]).sum(1)
        return self.design_logits(encoded, stage)
