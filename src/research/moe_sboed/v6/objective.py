"""Full-horizon pathwise location sPCE using the repository likelihood."""
import torch
from src.research.engine import logmeanexp


def trajectory(benchmark, policy, truth, noise):
    batch, horizon = noise.shape
    encoded = truth.new_zeros((batch, 16))
    actions, observations = [], []
    for stage in range(horizon):
        raw = policy.design_logits(encoded, stage)
        action = benchmark.feasible(raw, actions, horizon)
        mean, _ = benchmark.propagate(truth, None, action)
        observation = mean + benchmark.sigma * noise[:, stage:stage + 1]
        actions.append(action)
        observations.append(observation)
        if stage + 1 < horizon:
            token = torch.cat((benchmark.encode_action(action), observation / benchmark.obs_scale), -1)
            encoded = encoded + policy.encoder(token)
    return torch.stack(actions, 1), torch.stack(observations, 1)


def cumulative_likelihood(benchmark, particles, actions, observations):
    # Each particle replays the same observed history. Broadcast over particles and stages.
    means, _ = benchmark.propagate(particles[:, :, None, :], None, actions[:, None])
    return benchmark.log_likelihood(observations[:, None], means).sum(-1)


def scores(benchmark, policy, truth, negatives, noise, likelihood=cumulative_likelihood):
    actions, observations = trajectory(benchmark, policy, truth, noise)
    particles = torch.cat((truth[:, None], negatives), 1)
    values = likelihood(benchmark, particles, actions, observations)
    return values[:, 0] - logmeanexp(values)
