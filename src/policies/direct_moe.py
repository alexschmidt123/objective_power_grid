"""History-only soft MoE: a drop-in policy for full-horizon pathwise EIG."""
import torch
from torch import nn

class DirectMoEPolicy(nn.Module):
    architecture = 'history_soft_moe_pathwise_v2'
    def __init__(self, horizon, n_obs, n_experts=4, hidden=64):
        super().__init__()
        self.horizon, self.n_obs = horizon, n_obs
        self.n_experts = n_experts
        width = 1 + horizon * (n_obs + 2)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(width, hidden), nn.Tanh(),
                          nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            for _ in range(n_experts)])
        self.stage_bias = nn.Parameter(torch.zeros(n_experts, horizon))
        self.router = nn.Sequential(nn.Linear(width, hidden), nn.Tanh(),
                                    nn.Linear(hidden, n_experts))
        # Uniform initial routing; experts themselves are independently initialized.
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def initialize_sequence(self, sequence):
        """Same initial deterministic action as DAD, independent expert hidden layers."""
        sequence=torch.as_tensor(sequence,dtype=self.stage_bias.dtype,device=self.stage_bias.device)
        if sequence.shape != (self.horizon,):
            raise ValueError('Sequence must contain one latent action per stage')
        with torch.no_grad():
            for expert in self.experts:
                expert[-1].weight.zero_()
                expert[-1].bias.zero_()
            self.stage_bias.copy_(sequence.clamp(-4.,4.).expand(self.n_experts,-1))
            self.router[-1].weight.zero_()
            self.router[-1].bias.zero_()

    def validation_diagnostics(self, engine, rollout):
        records=[]
        batch=len(rollout['actions'][0])
        with torch.no_grad():
            for stage in range(self.horizon):
                features=engine.features(rollout['actions'][:stage],
                    rollout['observations'][:stage],batch,stage)
                weights,proposals=self.components(features,stage)
                records.append({
                    'stage':stage,
                    'mean_router_weights':weights.mean(0).tolist(),
                    'router_weight_std_across_histories':weights.std(0,unbiased=False).tolist(),
                    'mean_router_entropy':float(-(weights*weights.clamp_min(1e-12).log()).sum(-1).mean()),
                    'mean_expert_latent_range':float((proposals.max(-1).values-proposals.min(-1).values).mean())})
        return records

    def components(self, features, stage):
        proposals = torch.cat([expert(features) for expert in self.experts], -1)
        proposals = proposals + self.stage_bias[:, stage]
        weights = self.router(features).softmax(-1)
        return weights, proposals

    def forward(self, features, stage):
        weights, proposals = self.components(features, stage)
        mean = (weights * proposals).sum(-1)
        # The shared pathwise trainer/evaluator uses the deterministic mean only.
        return torch.distributions.Normal(mean, torch.ones_like(mean)), torch.zeros_like(mean)

    def routing_record(self, features):
        stage = int(round(float(features[0, 0]) * self.horizon))
        weights, proposals = self.components(features, stage)
        return {'routing_type':'soft_latent_action_mixture',
                'weights':weights.detach().tolist(),
                'expert_latent_means':proposals.detach().tolist()}
