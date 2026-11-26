"""Location policy with a causal discrete approximation to the two-source posterior."""
import torch
from torch import nn

from src.research.moe_sboed.v6.feature_policy import FeaturePolicy


class BeliefHistoryPolicy(FeaturePolicy):
    kind = "belief_expert_history_v6"

    def __init__(self, method, cfg, benchmark):
        super().__init__(method, cfg["horizon"], benchmark.action_dim, benchmark.obs_dim,
                         expert_history=True)
        self.benchmark = benchmark
        count = cfg["belief"]["particles"]
        if benchmark.name != "location_finding" or count < 2 or count & (count - 1):
            raise ValueError("Location belief requires a power-of-two particle count >= 2")
        engine = torch.quasirandom.SobolEngine(4, scramble=True, seed=cfg["belief"]["scramble_seed"])
        unit = engine.draw_base2(count.bit_length() - 1, dtype=torch.float64)
        eps = torch.finfo(unit.dtype).eps
        particles = torch.special.ndtri(unit.clamp(eps, 1 - eps))
        self.register_buffer("belief_particles", particles)
        a, b = particles.split(2, dim=-1)
        center = (a + b) / 2
        half_difference = (a - b) / 2

        def products(x):
            return torch.stack((x[:, 0].square(), x[:, 0] * x[:, 1], x[:, 1].square()), -1)

        # All eight moments are invariant to exchanging the two source labels.
        self.register_buffer("belief_moments", torch.cat(
            (center, products(center), products(half_difference)), dim=-1))
        self.belief_projection = nn.Linear(9, 64, bias=False, dtype=torch.float64)
        nn.init.zeros_(self.belief_projection.weight)

    def start_history(self, sample):
        # Only batch shape, dtype and device are used. No episode truth enters the policy.
        prior = sample.new_zeros(len(sample), len(self.belief_particles))
        return (*super().start_history(sample), prior)

    def update_history(self, state, token, stage):
        action = token[:, :2] * self.benchmark.action_scale
        observation = token[:, 2:] * self.benchmark.obs_scale
        means, _ = self.benchmark.propagate(self.belief_particles, None, action[:, None])
        log_likelihood = self.benchmark.log_likelihood(observation[:, None], means)
        return (*super().update_history(state[:-1], token, stage), state[-1] + log_likelihood)

    def shared_context(self, state, stage):
        ctx = super().shared_context(state, stage)
        weights = state[-1].softmax(-1)
        moments = weights @ self.belief_moments
        relative_ess = (len(self.belief_particles) * weights.square().sum(-1, keepdim=True)).reciprocal()
        return ctx + self.belief_projection(torch.cat((moments, relative_ess), dim=-1))
