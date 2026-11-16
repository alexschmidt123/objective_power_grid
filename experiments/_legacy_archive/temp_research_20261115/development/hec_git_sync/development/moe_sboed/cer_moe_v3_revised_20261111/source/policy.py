"""Complete physical proposals; categorical selection, never coordinate fusion."""
import copy, torch
from torch import nn
from torch.utils.checkpoint import checkpoint

class CERPolicy(nn.Module):
    def __init__(self, native, belief, epsilon=.1, temperature=1.):
        super().__init__()
        # Native vsOED's validated first two history layers, not HEC's action head.
        self.encoder = copy.deepcopy(native.net[:4])
        self.belief = copy.deepcopy(belief).requires_grad_(False)
        self.experts = nn.ModuleList()
        for _ in range(4):
            head = nn.Sequential(nn.Linear(272, 256), nn.ReLU(), nn.Linear(256, 2))
            head = head.to(next(native.parameters()).device)
            with torch.no_grad():
                head[0].weight[:, :256].copy_(native.net[4].weight)
                head[0].weight[:, 256:].zero_()
                head[0].bias.copy_(native.net[4].bias)
                head[2].weight.copy_(native.net[6].weight)
                head[2].bias.copy_(native.net[6].bias)
            self.experts.append(head)
        self.log_temperature = nn.Parameter(torch.tensor(float(temperature)).log().to(next(native.parameters()).device))
        self.epsilon = float(epsilon)

    def context(self, x):
        with torch.no_grad():
            w, mu, sd = self.belief.params(x)
            mean = (w[..., None] * mu).sum(1)
            variance = ((w[..., None] * (mu.square() + sd.square())).sum(1) - mean.square()).clamp_min(1e-6)
            info = torch.cat([mean / 3, torch.log1p(variance), w], -1)
        return torch.cat([self.encoder(x), info], -1)

    def proposals(self, x):
        shape = x.shape[:-1]
        z = self.context(x.reshape(-1, 117))
        # Each slot contains its own ENTIRE action head and is valid at all30steps.
        return torch.stack([-4 + 8 * head(z).sigmoid() for head in self.experts], 1).reshape(*shape, 4, 2)

    def distribution(self, x, critic, uniform=False):
        proposals = self.proposals(x)
        xx = x.unsqueeze(-2).expand(*x.shape[:-1], 4, 117)
        values = critic(torch.cat([xx, proposals], -1))[..., 0]
        if uniform:
            probabilities = torch.full_like(values, .25)
        else:
            temperature = self.log_temperature.exp().clamp(.1, 10.)
            probabilities = (1 - self.epsilon) * (values / temperature).softmax(-1) + self.epsilon / 4
        return proposals, values, probabilities

    def select(self, x, critic, uniform=False, u=None, expert=None):
        proposals, values, probabilities = self.distribution(x, critic, uniform)
        if expert is None:
            u = torch.rand_like(probabilities[..., 0]) if u is None else u
            selected = (u[..., None] >= probabilities.cumsum(-1)).sum(-1).clamp_max(3)
        else:
            selected = torch.full_like(probabilities[..., 0], expert, dtype=torch.long)
        action = proposals.gather(-2, selected[..., None, None].expand(*selected.shape, 1, 2)).squeeze(-2)
        return action, dict(proposals=proposals, values=values, probabilities=probabilities, selected=selected)

def expected_loss(policy, critic, x, uniform=False):
    _, values, probabilities = policy.distribution(x, critic, uniform)
    # Exact categorical enumeration: both d(pi)*Q and pi*d(Q) are present.
    # Critic parameters are held fixed by the caller. No straight-through samples.
    return -(probabilities * values).sum(-1).mean()

def chunked_loss(policy, critic, x, uniform=False, chunk=2048):
    flat = x.reshape(-1, 117)
    loss = flat.new_zeros(())
    for rows in flat.split(chunk):
        loss = loss + checkpoint(lambda xx: expected_loss(policy, critic, xx, uniform), rows,
                                 use_reentrant=False) * (len(rows) / len(flat))
    return loss
