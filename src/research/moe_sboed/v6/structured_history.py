"""Four location-specific views computed from actual, normalized measurements."""
import torch

from src.research.moe_sboed.v6.observed_history import ObservedHistoryMoE
from src.research.moe_sboed.v6.policy import METHODS


class StructuredHistoryMoE(ObservedHistoryMoE):
    kind = "structured_history_moe"

    def __init__(self, method, horizon):
        if method not in METHODS[:2]:
            raise ValueError("The structured pilot defines four-expert learned and uniform arms")
        super().__init__(method, horizon)

    def start_history(self, sample):
        total, empty = super().start_history(sample)
        views = empty.unsqueeze(2).expand(-1, -1, 4, -1)
        raw_zero = total.new_zeros(len(total), 3)
        return total, views, raw_zero, raw_zero

    def role_inputs(self, token, previous, raw_sum, stage):
        position, signal = token[:, :2], token[:, 2:]
        geometry = torch.cat((position, position.square().sum(-1, keepdim=True)), dim=-1)
        # The three relational views have no observation pair at the first step.
        if stage == 0:
            zero = torch.zeros_like(token)
            return torch.stack((geometry, zero, zero, zero), dim=1)
        contrast = token - raw_sum / stage
        displacement = position - previous[:, :2]
        delta_signal = signal - previous[:, 2:]
        distance_sq = displacement.square().sum(-1, keepdim=True)
        # Ridge-stabilized secant along the actual sensor displacement.
        # All coordinates are normalized. Ridge length is 0.1 (0.4 physical units).
        slope = displacement * delta_signal / (distance_sq + 0.01)
        sensitivity = torch.cat((slope, distance_sq), dim=-1)
        sequential = torch.cat((signal, delta_signal, torch.full_like(signal, stage / self.horizon)), dim=-1)
        return torch.stack((geometry, contrast, sensitivity, sequential), dim=1)

    def update_history(self, state, token, stage):
        total, past_views, previous, raw_sum = state
        role_inputs = self.role_inputs(token, previous, raw_sum, stage)
        embedded = self.encoder(torch.cat((token.unsqueeze(1), role_inputs), dim=1))
        views = torch.cat((past_views, embedded[:, None, 1:]), dim=1)
        return total + embedded[:, 0], views, token, raw_sum + token

    def mixture(self, state, stage):
        total, views, _, _ = state
        readings = [views[:, :, 0], *(views[:, 1:, index] for index in (1, 2, 3))]
        return self.mix_readings(total, readings, stage)

    def forward(self, features, stage):
        history = features[:, 1:].reshape(len(features), self.horizon, 4)
        state = self.start_history(features)
        for index in range(stage):
            state = self.update_history(state, history[:, index, :3], index)
        return self.design_logits(state, stage)
