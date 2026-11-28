"""Predict one action as a learned offset from the recorded sensor positions."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from src.research.moe_sboed.v6.observed_history import ObservedHistoryMoE


class RelativeHistoryMoE(ObservedHistoryMoE):
    kind = "relative_history_moe"

    def __init__(self, method, horizon):
        super().__init__(method, horizon)
        width = self.head.in_features - 1
        # The existing head predicts the offset. This shared projection supplies its
        # reference point and step size; all four experts contribute to both.
        self.reference = nn.Linear(width + 1, width + 3, dtype=self.head.weight.dtype)
        nn.init.zeros_(self.reference.weight)
        nn.init.zeros_(self.reference.bias)

    def start_history(self, sample):
        total, tokens = super().start_history(sample)
        return total, tokens, total.new_empty(len(sample), 0, 2)

    def update_history(self, state, token, stage):
        total, encoded = super().update_history(state[:2], token, stage)
        return total, encoded, torch.cat((state[2], token[:, None, :2]), dim=1)

    def mixture(self, state, stage):
        return super().mixture(state[:2], stage)

    def design_logits(self, state, stage):
        context, features, weights = self.mixture(state, stage)
        fused = context + (weights.unsqueeze(-1) * features).sum(dim=1)
        offset = self.head(fused)
        if stage == 0:
            return offset
        _, tokens, positions = state
        query, decay, origin_score, scale = self.reference(fused).split((tokens.shape[-1], 1, 1, 1), dim=1)
        scores = (tokens @ query.unsqueeze(-1)).squeeze(-1) / math.sqrt(tokens.shape[-1])
        age = torch.arange(stage - 1, -1, -1, dtype=tokens.dtype, device=tokens.device)
        scores = scores - F.softplus(decay) * age
        # One extra option centers the design at the origin so it can explore afresh.
        attention = torch.cat((scores, origin_score), dim=1).softmax(-1)[:, :-1]
        center = (attention.unsqueeze(-1) * positions).sum(dim=1)
        # The benchmark maps logits to 4*tanh(logits/2). Positions are already x/4.
        return 2 * torch.atanh(center.clamp(-1 + 1e-9, 1 - 1e-9)) + 2 * scale.sigmoid() * offset

    def forward(self, features, stage):
        history = features[:, 1:].reshape(len(features), self.horizon, 4)[:, :stage, :3]
        tokens = self.encoder(history)
        return self.design_logits((tokens.sum(dim=1), tokens, history[:, :, :2]), stage)
