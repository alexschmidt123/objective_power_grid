"""Counterfactual coordinator for frozen HEC-v2; no critic or model refitting."""
import math
import torch
from torch import nn


def gate_bank(original):
    """Original, uniform, six conservative unordered-pair interventions."""
    if original.ndim != 2 or original.shape[1] != 4:
        raise ValueError('Expected gates [B,4]')
    uniform = torch.full_like(original, 0.25)
    options = [original, uniform]
    for i in range(4):
        for j in range(i + 1, 4):
            pair = torch.zeros_like(original)
            pair[:, i] = pair[:, j] = 0.5
            options.append(0.5 * uniform + 0.5 * pair)
    return torch.stack(options, dim=1)


def fuse_options(v2, x, info, gates):
    """Reuse actual first/response fusion matrices and the physical action map."""
    b, k, e = gates.shape
    if e != 4 or info['slots'].shape != (b, 4, 128):
        raise ValueError('Incompatible V2 packet/gate shapes')
    first, response = info['slots'].split(64, -1)
    first = (gates[..., None] * first[:, None]).flatten(2)
    response = (gates[..., None] * response[:, None]).flatten(2)
    history = info['history'][:, None].expand(-1, k, -1)
    pre = v2.fusion[0](torch.cat([history, first], -1)) + v2.response_fusion(response)
    delta = v2.fusion[2](v2.fusion[1](pre)).tanh()
    a0 = v2.base(x)[:, None]
    return a0 + 4 * delta - a0 * delta.abs()


def routing_context(v2, x, info, actions):
    """182 observed-history features, no latent theta, future noise or scores."""
    first, response = info['slots'].split(64, -1)
    # Actual consumed per-expert hidden contributions, excluding history/bias.
    weight = v2.fusion[0].weight[:, 128:].reshape(128, 4, 64)
    rweight = v2.response_fusion.weight.reshape(128, 4, 64)
    contribution = (torch.einsum('bei,hei->beh', first, weight)
                    + torch.einsum('bei,hei->beh', response, rweight))
    norms = contribution.square().mean(-1).sqrt().log1p()
    return torch.cat([info['history'], info['gates'],
                      (actions - actions[:, 1:2]).flatten(1), norms, x[:, :30]], -1)


class CounterfactualRouter(nn.Module):
    """Small value-regression selector; one gate pattern is executed, not averaged."""
    def __init__(self, input_dim=182, stage_only=False):
        super().__init__()
        self.stage_only = stage_only
        self.register_buffer('mean', torch.zeros(input_dim))
        self.register_buffer('scale', torch.ones(input_dim))
        self.value = nn.Linear(input_dim, 8)
        nn.init.zeros_(self.value.weight)
        nn.init.zeros_(self.value.bias)

    def forward(self, context):
        x = (context - self.mean) / self.scale
        if self.stage_only:
            x = torch.cat([torch.zeros_like(x[:, :-30]), x[:, -30:]], -1)
        return self.value(x)

    @torch.no_grad()
    def select(self, context):
        return self(context).argmax(-1)


def fit_router(context, returns, *, stage_only=False, ridge_penalty=10.):
    """Exact ridge value regression using training data only; no early stopping."""
    if returns.shape != (len(context), 8) or len(context) < 2:
        raise ValueError('Need at least two complete labeled histories and eight returns')
    if ridge_penalty <= 0 or not torch.isfinite(context).all() or not torch.isfinite(returns).all():
        raise ValueError('Need finite data and a positive ridge penalty')
    router = CounterfactualRouter(context.shape[1], stage_only).to(context.device)
    with torch.no_grad():
        router.mean.copy_(context.mean(0))
        router.scale.copy_(context.std(0).clamp_min(0.1))
        x = (context - router.mean) / router.scale
        if stage_only:
            x = torch.cat([torch.zeros_like(x[:, :-30]), x[:, -30:]], -1)
        design = torch.cat([torch.ones_like(x[:, :1]), x], -1).double()
        target = (returns - returns[:, 1:2]).double()
        regularizer = torch.eye(design.shape[1], device=context.device, dtype=torch.float64) * ridge_penalty
        regularizer[0, 0] = 0.  # intercept unpenalized
        coefficients = torch.linalg.solve(design.T @ design + regularizer, design.T @ target)
        router.value.bias.copy_(coefficients[0].float())
        router.value.weight.copy_(coefficients[1:].T.float())
    return router.eval()


def signal(theta, action):
    """Broadcast theta [...,2,2] against physical action [...,2]."""
    distance = (theta - action[..., None, :]).square().sum(-1)
    return (0.1 + (distance + 1e-4).reciprocal().sum(-1)).log()


def terminal_scores(actions, observations, truth, contrasts, chunk=16384):
    """Literal positive-included-once prior-sPCE, full prefix AND suffix."""
    b, t, _ = actions.shape
    if t != 30 or observations.shape != (b, 30) or truth.shape != (b, 4):
        raise ValueError('Expected complete T30 trajectories')
    if len(contrasts) < 1 or chunk < 1:
        raise ValueError('Need a nonempty independent prior bank and positive chunk size')
    actions = actions.double()
    observations = observations.double()
    positive = -2 * (observations - signal(truth.double().reshape(b, 1, 2, 2), actions)).square()
    positive = positive.sum(-1)
    negative_sum = torch.full_like(positive, -torch.inf)
    for start in range(0, len(contrasts), chunk):
        theta = contrasts[start:start + chunk].double().reshape(1, -1, 2, 2)
        cumulative = torch.zeros(b, theta.shape[1], dtype=torch.float64, device=actions.device)
        for stage in range(30):
            mean = signal(theta, actions[:, stage, None])
            cumulative += -2 * (observations[:, stage, None] - mean).square()
        negative_sum = torch.logaddexp(negative_sum, torch.logsumexp(cumulative, -1))
    # Gaussian constants cancel because all terms have exactly the same T and sigma.
    return positive - torch.logaddexp(positive, negative_sum) + math.log(len(contrasts) + 1)


def paired_stats(values):
    from scipy.stats import t
    v = values.detach().double().cpu()
    if v.numel() < 2 or not torch.isfinite(v).all():
        raise ValueError('Need at least two finite independent episode scores')
    mean = float(v.mean())
    se = float(v.std() / math.sqrt(len(v)))
    width = float(t.ppf(0.975, len(v)-1)) * se
    p = float(2*t.sf(abs(mean/se), len(v)-1)) if se else (1. if mean == 0 else 0.)
    return dict(n=len(v), mean=mean, MCSE=se, CI95=[mean-width, mean+width], p_two_sided=p)


def holm(tests):
    previous = 0.
    for rank, name in enumerate(sorted(tests, key=lambda name: tests[name]['p_two_sided'])):
        previous = max(previous, min(1., (len(tests)-rank)*tests[name]['p_two_sided']))
        tests[name]['Holm_p'] = previous
