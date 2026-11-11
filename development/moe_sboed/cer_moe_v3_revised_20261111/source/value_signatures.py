"""Independent posterior-conditional full suffixes; no saved future observations.

Value = E[log q30(theta|new full history)-log qt(theta|prefix)].
The terminal variational score is native; SNIS corrects the conditional sampling
law using the exact likelihood. Its finite-sample bias is explicitly audited.
"""
from common import *

def gmm_log(theta, w, mu, sd):
    lp = -.5 * ((theta[:, :, None] - mu[:, None]) / sd[:, None]).square().sum(-1)
    lp = lp - sd.log().sum(-1)[:, None] - 2 * math.log(2 * math.pi)
    return torch.logsumexp(lp + w.clamp_min(1e-38).log()[:, None], -1)

def prefix_log_likelihood(x, theta):
    stages = x[:, :30].argmax(-1)
    ds = x[:, 30:88].reshape(-1, 29, 2)
    ys = x[:, 88:117]
    lp = theta.new_zeros(theta.shape[:2])
    for t in range(int(stages.max())):
        prediction = mean_signal(theta, ds[:, t:t+1])[:, 0]
        lp += -2 * (ys[:, t, None] - prediction).square() * (t < stages[:, None])
    return lp

@torch.no_grad()
def conditional_particles(belief, x, count, generator):
    w, mu, sd = belief.params(x)
    # Full-support .1prior+.9amortized-GMM importance proposal.
    k = torch.multinomial(w, count, replacement=True, generator=generator)
    mm = mu.gather(1, k[..., None].expand(-1, -1, 4))
    ss = sd.gather(1, k[..., None].expand(-1, -1, 4))
    th = mm + ss * torch.randn(mm.shape, generator=generator, device=x.device)
    use_prior = torch.rand((len(x), count), generator=generator, device=x.device) < .1
    prior_draw = torch.randn(th.shape, generator=generator, device=x.device)
    th = torch.where(use_prior[..., None], prior_draw, th)
    prior_lp = -.5 * th.square().sum(-1) - 2 * math.log(2 * math.pi)
    proposal_lp = torch.logaddexp(gmm_log(th, w, mu, sd) + math.log(.9), prior_lp + math.log(.1))
    log_weights = prior_lp + prefix_log_likelihood(x, th) - proposal_lp
    weights = log_weights.softmax(-1)
    ess = weights.square().sum(-1).reciprocal()
    return th, weights, ess

@torch.no_grad()
def terminal_log_q(belief, ds, ys, theta):
    net = belief.nets[29]
    inp = torch.cat([ds.reshape(len(ds), 60), ys.reshape(len(ds), 30)], -1)
    features = net.feature_net(inp)
    w = net.weight_net(features).exp()
    mu = net.mu_net(features).sigmoid().reshape(-1, 8, 4) * (net.mu_bounds[:, 1] - net.mu_bounds[:, 0]) + net.mu_bounds[:, 0]
    sd = net.sigma_net(features).sigmoid().reshape(-1, 8, 4) * net.max_sigmas + 1e-5
    core = gmm_log(theta[:, None], w, mu, sd)[:, 0]
    prior = -.5 * theta.square().sum(-1) - 2 * math.log(2 * math.pi)
    # Preserve the normalized native full-support density, including its tiny
    # prior component. This matters in posterior tails after counterfactuals.
    return torch.logaddexp(core, prior + math.log(1e-27)) - math.log1p(1e-27)

@torch.no_grad()
def suffix_values(model, x, actions, particles, weights, repeats, generator):
    # x is same-stage; random draws are shared across candidate actions only.
    stage = int(x[0, :30].argmax())
    assert bool((x[:, :30].argmax(-1) == stage).all())
    b, choices = actions.shape[:2]
    ids = torch.multinomial(weights, repeats, replacement=True, generator=generator)
    theta = particles.gather(1, ids[..., None].expand(-1, -1, 4))
    noise = torch.randn((b, repeats, 30-stage), generator=generator, device=x.device)
    theta = theta[:, None].expand(-1, choices, -1, -1).reshape(-1, 4)
    prefix = x[:, None, None].expand(-1, choices, repeats, -1).reshape(-1, 117).clone()
    ds = x.new_zeros(len(theta), 30, 2)
    ys = x.new_zeros(len(theta), 30, 1)
    ds[:, :stage] = prefix[:, 30:30+2*stage].reshape(-1, stage, 2)
    ys[:, :stage, 0] = prefix[:, 88:88+stage]
    initial_log_q = model.belief.log_prob(prefix, theta)
    for t in range(stage, 30):
        action = actions[:, :, None].expand(-1, -1, repeats, -1).reshape(-1, 2) if t == stage else model(prefix)
        y = mean_signal(theta[:, None], action[:, None])[:, 0, 0]
        y += .5 * noise[:, None, :, t-stage].expand(-1, choices, -1).reshape(-1)
        ds[:, t] = action; ys[:, t, 0] = y
        if t < 29: prefix = advance(prefix, action, y)
    result = (terminal_log_q(model.belief, ds, ys, theta) - initial_log_q).reshape(b, choices, repeats)
    return result
