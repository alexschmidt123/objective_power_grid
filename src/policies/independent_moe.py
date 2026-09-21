"""Independent sparse MoE policy: no teacher, base policy, or privileged expert."""
from __future__ import annotations

import torch
import math
from torch import nn


ARCHITECTURE = "independent_sparse_moe_v1"


class IndependentMoEPolicy(nn.Module):
    """History/posterior router and independent action-logit experts.

    All stages, including the first, use the same learned routing rule.
    Only top-k experts execute for each input. The common encoder is trained
    jointly; no parameters or action labels come from another design method.
    """

    def __init__(self, n_actions, config, *, n_experts=4, top_k=2,
                 expert_hidden=None, balance_coefficient=0.01, routing="learned"):
        super().__init__()
        if not 1 <= top_k <= n_experts or (n_experts > 1 and top_k < 2):
            raise ValueError("Use 2 <= top_k <= n_experts (or one dense expert)")
        self.n_actions, self.config = int(n_actions), config
        self.n_experts, self.top_k = int(n_experts), int(top_k)
        self.expert_hidden = int(expert_hidden or config.hidden)
        self.balance_coefficient = float(balance_coefficient)
        if routing not in {"learned", "uniform"}:
            raise ValueError(routing)
        self.routing = routing
        # Weighted nonlinear particle features retain more than posterior moments.
        self.particle_encoder = nn.Sequential(
            nn.Linear(config.particle_dim, 32), nn.SiLU(), nn.Linear(32, 32))
        width = config.max_steps * (config.obs_dim + 2) + 32 + 3
        self.encoder = nn.Sequential(nn.Linear(width, config.hidden), nn.LayerNorm(config.hidden),
                                     nn.SiLU(), nn.Linear(config.hidden, config.hidden), nn.SiLU())
        self.router = nn.Linear(config.hidden, self.n_experts) if routing == "learned" else None
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(config.hidden, self.expert_hidden), nn.SiLU(),
                          nn.Linear(self.expert_hidden, self.n_actions))
            for _ in range(self.n_experts)])

    def features(self, actions, observations, mask, belief, steps, particles, weights):
        del belief  # No control summaries or method-specific guidance.
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-20)
        encoded_particles = self.particle_encoder(particles)
        posterior = (weights[..., None] * encoded_particles).sum(1)
        obs = observations if observations.ndim == 3 else observations[..., None]
        history = torch.cat((actions.float()[..., None] / max(self.n_actions - 1, 1),
                             obs.clamp(-8, 8), mask[..., None]), dim=-1)
        history = history * mask[..., None]
        entropy = -(weights * weights.clamp_min(1e-30).log()).sum(-1)
        entropy = entropy / max(math.log(weights.shape[-1]), 1.0)
        ess = 1.0 / (weights.square().sum(-1) * weights.shape[-1]).clamp_min(1e-20)
        extra = torch.stack((steps.float().reshape(-1) / self.config.max_steps, entropy, ess), -1)
        return self.encoder(torch.cat((history.flatten(1), posterior, extra), -1))

    def routed(self, *inputs):
        features = self.features(*inputs)
        if self.routing == "uniform":
            dense = features.new_full((len(features), self.n_experts), 1.0 / self.n_experts)
            indices = torch.arange(self.n_experts, device=features.device)[None].expand(len(features), -1)
            logits = torch.stack([expert(features) for expert in self.experts], 1).mean(1)
            return logits, dense, indices, dense
        dense = self.router(features).softmax(-1)
        values, indices = dense.topk(self.top_k, dim=-1)
        values = values / values.sum(-1, keepdim=True)
        logits = features.new_zeros((features.shape[0], self.n_actions))
        # True sparse execution: an expert sees only rows routed to it.
        for expert_id, expert in enumerate(self.experts):
            rows, slots = torch.where(indices == expert_id)
            if rows.numel():
                contribution = expert(features[rows]) * values[rows, slots, None]
                logits = logits.index_add(0, rows, contribution)
        return logits, dense, indices, values

    def forward(self, *inputs):
        logits, _, _, _ = self.routed(*inputs[:7])
        if len(inputs) == 8:
            if not bool(inputs[7].any(-1).all()):
                raise ValueError("No feasible design")
            logits = logits.masked_fill(~inputs[7], -1e9)
        return logits

    def distribution(self, *inputs):
        return torch.distributions.Categorical(logits=self(*inputs))

    def routing_loss(self, *inputs):
        if self.routing == "uniform":
            return inputs[1].new_zeros(()), {"router_entropy": math.log(self.n_experts),
                "expert_load": [1.0 / self.n_experts] * self.n_experts, "balance_loss": 0.0}
        dense = self.router(self.features(*inputs[:7])).softmax(-1)
        selected = dense.topk(self.top_k, dim=-1).indices
        load = torch.nn.functional.one_hot(selected, self.n_experts).float().mean((0, 1))
        balance = self.n_experts * (load.detach() * dense.mean(0)).sum()
        entropy = -(dense * dense.clamp_min(1e-20).log()).sum(-1).mean()
        return self.balance_coefficient * balance, {
            "router_entropy": float(entropy.detach()),
            "expert_load": load.detach().cpu().tolist(),
            "balance_loss": float(balance.detach()),
        }

    @torch.no_grad()
    def routing_record(self, *inputs):
        _, dense, indices, values = self.routed(*inputs[:7])
        return {"dense_weights": dense.cpu().tolist(),
                "selected_experts": indices.cpu().tolist(),
                "selected_weights": values.cpu().tolist()}
