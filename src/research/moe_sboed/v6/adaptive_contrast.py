"""Sequential ACE for the two-source location model (Foster et al., 2021, appendix).

Use a normalized proposal conditioned only on observed designs and outcomes.
No fitted model or additional optimizer is needed.
"""
import math
import torch
from src.research.engine import logmeanexp
from src.research.moe_sboed.v6.objective import trajectory, cumulative_likelihood


def source_proposal(actions, observations, min_scale):
    # Strong signals can indicate a nearby source. This heuristic remains a
    # valid normalized proposal even when it approximates the posterior poorly.
    weights = observations.softmax(dim=1)
    center = (weights * actions).sum(dim=1, keepdim=True)
    variance = (weights * (actions - center).square()).sum(dim=1, keepdim=True)
    return center.unsqueeze(2), (variance + min_scale**2).sqrt().unsqueeze(2)


def prior_over_proposal(sources, center, scale, prior_fraction):
    log_prior = -.5 * sources.square().sum(-1)
    log_local = -.5 * ((sources - center) / scale).square().sum(-1) - scale.log().sum(-1)
    # The omitted bivariate Gaussian normalization constant cancels.
    log_q = torch.logaddexp(math.log(prior_fraction) + log_prior,
                            math.log1p(-prior_fraction) + log_local)
    return (log_prior - log_q).sum(dim=-1)


def adaptive_scores(benchmark, policy, truth, base_negatives, noise, use_prior,
                    prior_fraction, min_scale, likelihood=cumulative_likelihood):
    if not (0 < prior_fraction < 1 and min_scale > 0):
        raise ValueError("sACE requires a defensive prior fraction in (0, 1) and a positive scale")
    actions, observations = trajectory(benchmark, policy, truth, noise)
    center, scale = source_proposal(actions, observations, min_scale)
    standard = base_negatives.unflatten(-1, (2, 2))
    proposals = torch.where(use_prior.unsqueeze(-1), standard, center + scale * standard)
    particles = torch.cat((truth[:, None], proposals.flatten(-2)), dim=1)
    values = likelihood(benchmark, particles, actions, observations)
    # Importance-correct every denominator term, including the positive truth.
    ratios = prior_over_proposal(particles.unflatten(-1, (2, 2)), center, scale, prior_fraction)
    return values[:, 0] - logmeanexp(values + ratios)
