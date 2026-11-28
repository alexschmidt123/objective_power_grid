"""Two-source location training diagnostics and a full-horizon D-optimality auxiliary."""
import torch

from src.research.engine import logmeanexp
from src.research.moe_sboed.v6.objective import trajectory


def signal_jacobian(truth, actions):
    """Derivative of the known log-signal with respect to the four source coordinates at fixed design."""
    displacement = truth.reshape(-1, 1, 2, 2) - actions.unsqueeze(2)
    inverse = (1e-4 + displacement.square().sum(-1)).reciprocal()
    power = 0.1 + inverse.sum(-1, keepdim=True)
    return ((-2 * inverse.square() / power).unsqueeze(-1) * displacement).flatten(2)


def local_precision(truth, actions, sigma):
    gradients = signal_jacobian(truth, actions) / sigma
    identity = torch.eye(4, dtype=actions.dtype, device=actions.device)
    precision = identity + gradients.transpose(1, 2) @ gradients
    return torch.linalg.cholesky(precision).diagonal(dim1=-2, dim2=-1).log().sum(-1)


def training_terms(bench, model, truth, negatives, noise, likelihood):
    # Hidden coordinates enter the simulator and the offline objective only.
    actions, obs = trajectory(bench, model, truth, noise)
    particles = torch.cat((truth[:, None], negatives), dim=1)
    ll = likelihood(bench, particles, actions, obs)
    return ll[:, 0] - logmeanexp(ll), local_precision(truth, actions, bench.sigma)
