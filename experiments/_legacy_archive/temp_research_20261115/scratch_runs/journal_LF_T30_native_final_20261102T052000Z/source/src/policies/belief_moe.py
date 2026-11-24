"""Belief-conditioned, hard-routed mixture of adaptive design experts.

Domain adapters supply normalized particle tokens, weights and ordered history.
All experts propose actions. A shared action-value model routes by estimated
remaining utility; there is no action averaging or baseline teacher.
"""
import torch


ARCHITECTURE = 'belief_value_routed_moe_v1'


class BeliefMoEPolicy(torch.nn.Module):
    def __init__(self, particle_width, context_width, n_experts=2, hidden=64):
        super().__init__()
        self.n_experts = n_experts
        self.particle_width, self.context_width = particle_width, context_width
        self.particle_encoder = torch.nn.Sequential(
            torch.nn.Linear(particle_width, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden), torch.nn.Tanh())
        width = 2*hidden + context_width
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(width, hidden), torch.nn.Tanh(),
                                torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
                                torch.nn.Linear(hidden, 2)) for _ in range(n_experts)])
        # Separate representation for value learning: critic updates cannot move
        # the actor representation without an actor learning signal.
        self.value_particle_encoder = torch.nn.Sequential(
            torch.nn.Linear(particle_width, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden), torch.nn.Tanh())
        self.value_model = torch.nn.Sequential(
            torch.nn.Linear(width+1, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 1))

    @staticmethod
    def aggregate(encoder, inputs):
        tokens, weights, context = inputs
        embedded = encoder(tokens)
        mean = (weights[..., None]*embedded).sum(1)
        variance = (weights[..., None]*(embedded-mean[:, None]).square()).sum(1)
        return torch.cat([mean, variance, context], -1)

    def distributions(self, inputs):
        features = self.aggregate(self.particle_encoder, inputs)
        raw = torch.stack([expert(features) for expert in self.experts], 1)
        means = 4.*torch.tanh(raw[..., 0]/4.)
        scales = (raw[..., 1]-.7).clamp(-2.5, .5).exp()
        return torch.distributions.Normal(means, scales)

    def q_values(self, inputs, latent):
        features = self.aggregate(self.value_particle_encoder, inputs)
        if latent.ndim == 1:
            return self.value_model(torch.cat([features, torch.sigmoid(latent[:, None])], -1)).squeeze(-1)
        features = features[:, None].expand(-1, latent.shape[1], -1)
        return self.value_model(torch.cat([features, torch.sigmoid(latent[..., None])], -1)).squeeze(-1)

    def choose(self, inputs, *, stochastic=False, exploration=.15):
        distribution = self.distributions(inputs)
        with torch.no_grad():
            values = self.q_values(inputs, distribution.mean)
            selected = values.argmax(-1)
            if stochastic:
                explore = torch.rand(len(selected)) < exploration
                selected = torch.where(explore, torch.randint(self.n_experts, selected.shape), selected)
        latent_all = distribution.sample() if stochastic else distribution.mean
        latent = latent_all.gather(1, selected[:, None]).squeeze(1)
        logp = distribution.log_prob(latent[:, None]).gather(1, selected[:, None]).squeeze(1)
        return latent, logp, selected, values, distribution.mean
