"""Four independent specialist decision heads; categorical physical execution."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from frozen_feature_policy import observed_prefix
from hec_policy import mean_signal, jacobian, squash, upper, COST
from query_response import paired_discrimination, prospective_opportunity

ROLES = ['Posterior geometry', 'Hypothesis discrimination', 'Parameter sensitivity', 'Sequential opportunity']
COMMANDS = [4, 6, 6, 4]


class SharedInference(nn.Module):
    """Only frozen posterior inference and deterministic likelihood summaries.

    No inherited base policy, history encoder, commander, fusion or action head.
    """
    def __init__(self, parent):
        super().__init__()
        self.belief = copy.deepcopy(parent.context.belief)
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        w, mu, sd = self.belief.params(x)
        nodes, pw = self.belief.nodes(w, mu, sd)
        mean = (w[..., None]*mu).sum(1)
        var = (w[..., None]*(mu.square()+sd.square())).sum(1)-mean.square()
        scale = var.clamp_min(1e-5).sqrt()
        cov = torch.einsum('bm,bmi,bmj->bij', w, mu-mean[:, None], mu-mean[:, None])
        pair = torch.triu_indices(4, 4, 1, device=x.device)
        correlation = cov[:, pair[0], pair[1]]/(scale[:, pair[0]]*scale[:, pair[1]])
        summary = torch.cat([mean/3, scale.log(), correlation], -1)
        stage = x[:, :30].argmax(-1)
        remaining = (30-stage).to(x.dtype)[:, None]/30
        # Posterior centroid is a deterministic model-query location, not an expert action mixture.
        anchor = mean.reshape(-1, 2, 2).mean(1).clamp(-4, 4)
        probes = (anchor[:, None]+x.new_tensor([[0., 0.], [1., 0.], [-1., 0.]])[None]).clamp(-4, 4)
        pred = mean_signal(nodes, probes)
        pm = (pw[:, None]*pred).sum(-1)
        pv = (pw[:, None]*(pred-pm[:, :, None]).square()).sum(-1)
        likelihood = torch.cat([pm/5, torch.log1p(pv)], -1)
        # Known N(0,I4) prior, sigma=.5 and physical [-4,4]^2 bounds.
        prior = x.new_tensor([0.]*4+[1.]*4+[.5, -4., -4., 4., 4.]).expand(len(x), -1)
        return dict(x=x, w=w, mu=mu, sd=sd, nodes=nodes, pw=pw, mean=mean,
                    scale=scale, summary=summary, stage=stage, remaining=remaining,
                    a0=anchor, prior=prior, likelihood=likelihood)


class SpecialistFeatures(nn.Module):
    """V2 semantic computations, without any inherited decoder projections."""
    def __init__(self, parent_expert, index):
        super().__init__()
        self.index = index
        for name, module in parent_expert.named_children():
            setattr(self, name, copy.deepcopy(module))

    def configure(self):
        self.requires_grad_(True)
        for name in ['anchor1', 'anchor3', 'opportunity']:
            if hasattr(self, name):
                getattr(self, name).requires_grad_(False)

    def forward(self, c, cmd):
        x = c['x']
        w, mu, sd, nodes, pw = [c[k] for k in ['w', 'mu', 'sd', 'nodes', 'pw']]
        mean, scale, a0 = c['mean'], c['scale'], c['a0']
        if self.index == 0:
            center = (mu-mean[:, None])/scale[:, None]
            att = (w.clamp_min(1e-38).log()+(center*cmd[:, None]).sum(-1)/2).softmax(-1)
            tokens = torch.cat([mu/3, sd.log(), w[..., None]], -1)
            legacy = (att[..., None]*self.region(tokens)).sum(1)
            first = (att[..., None]*self.anchor1(tokens)).sum(1)
            response = legacy-first
            private = torch.cat([tokens, cmd[:, None].expand(-1, mu.shape[1], -1)], -1)
            innovation = (att[..., None]*self.innovation(private)).sum(1)
        elif self.index == 1:
            direction = cmd[:, :2]+x.new_tensor([1., 0.])
            direction = direction/(direction.square().sum(-1, keepdim=True)+1e-6).sqrt()
            probes = (a0[:, None]+x.new_tensor([0., 1., -1.])[None, :, None]*direction[:, None]).clamp(-4, 4)
            pred = mean_signal(nodes, probes)
            att = (pw.clamp_min(1e-38).log()+((nodes-mean[:, None])/scale[:, None]*cmd[:, None, 2:]).sum(-1)/2).softmax(-1)
            first = (att[..., None]*self.discriminate(torch.cat([pred.transpose(1, 2)/5, att[..., None]], -1))).sum(1)
            semantic, _, _, hp, hm = paired_discrimination(nodes, pw, pred, mean, scale, cmd)
            response = torch.cat([semantic, self.adapter2(torch.cat([semantic, hp/3, hm/3, a0/4, cmd, mean/3, scale.log()], -1))], -1)
            innovation = self.innovation(torch.cat([semantic, cmd, a0/4, mean/3, scale.log()], -1))
        elif self.index == 2:
            design = (a0+cmd[:, 4:]).clamp(-4, 4)
            J = jacobian(nodes, design)
            fisher = 4*torch.einsum('bn,bni,bnj->bij', pw, J, J)
            direction = cmd[:, :4]+x.new_tensor([1., -1., 1., -1.])
            direction = direction/(direction.square().sum(-1, keepdim=True)+1e-6).sqrt()
            inp = torch.cat([squash(upper(fisher)), direction], -1)
            legacy = self.sensitivity(inp)
            first = self.anchor3(inp)
            response = legacy-first
            innovation = self.innovation(torch.cat([inp, cmd], -1))
        else:
            direction = cmd[:, :2]+x.new_tensor([0., 1.])
            direction = direction/(direction.square().sum(-1, keepdim=True)+1e-6).sqrt()
            radius = 1+.5*cmd[:, 2:3]
            probes = (a0[:, None]+x.new_tensor([0., 1., -1.])[None, :, None]*radius[:, None]*direction[:, None]).clamp(-4, 4)
            COST['private_opportunity_rows'] += len(x)*3
            q = self.opportunity(torch.cat([x[:, None].expand(-1, 3, -1), probes], -1)).squeeze(-1)
            qdiff = (q-q[:, :1])/10
            pred = mean_signal(nodes, probes)
            pm = (pw[:, None]*pred).sum(-1)
            pv = (pw[:, None]*(pred-pm[:, :, None]).square()).sum(-1)
            first = self.sequential(torch.cat([qdiff, pm/5, torch.log1p(pv), c['remaining'], cmd], -1))
            semantic = prospective_opportunity(nodes, pw, probes, cmd, c['stage'])
            response = torch.cat([semantic, self.adapter4(torch.cat([semantic, qdiff, c['remaining'], cmd], -1))], -1)
            innovation = self.innovation(torch.cat([semantic, probes.flatten(1)/4, qdiff, c['remaining'], cmd], -1))
        return first, response, innovation


class PrivateExpert(nn.Module):
    def __init__(self, parent, index):
        super().__init__()
        self.index, self.role = index, ROLES[index]
        self.history = copy.deepcopy(parent.context.history)
        old = parent.context.commander
        start, width = sum(COMMANDS[:index]), COMMANDS[index]
        self.commander = nn.Sequential(copy.deepcopy(old[0]), nn.Tanh(), nn.Linear(64, width).to(old[0].weight))
        with torch.no_grad():
            self.commander[-1].weight.copy_(old[-1].weight[start:start+width])
            self.commander[-1].bias.copy_(old[-1].bias[start:start+width])
        self.offset = nn.Parameter(parent.context.offset[:, start:start+width].detach().clone())
        self.features = SpecialistFeatures(parent.experts[index], index)
        # 128 history +64 first +64 response +128 innovation +14 belief +1 horizon +13 prior +6 likelihood.
        self.head = nn.Sequential(nn.Linear(418, 128), nn.SiLU(), nn.Linear(128, 2)).to(old[0].weight)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.configure()

    def configure(self):
        self.requires_grad_(True)
        self.features.configure()

    def forward(self, x, c):
        h = self.history(x)
        # Preserve original command normalization; all weights belong to this expert.
        command_context = torch.cat([h, c['a0']/4, c['mean'], torch.log1p(c['scale'].square()), c['summary'][:, 8:]], -1)
        assert command_context.shape[-1] == 144
        cmd = (self.commander(command_context)+self.offset[c['stage']]).tanh()
        first, response, innovation = self.features(c, cmd)
        z = torch.cat([h, first, response, innovation, c['summary'], c['remaining'], c['prior'], c['likelihood']], -1)
        assert z.shape[-1] == 418
        return 4*self.head(z).tanh()


class StandalonePolicy(nn.Module):
    def __init__(self, inference, expert):
        super().__init__()
        self.inference, self.expert = copy.deepcopy(inference), copy.deepcopy(expert)
        self.chunk = 4096

    def configure(self):
        self.requires_grad_(False)
        self.expert.configure()

    def forward(self, x):
        shape = x.shape[:-1]
        flat = observed_prefix(x).reshape(-1, 117)
        outputs = []
        for xx in flat.split(self.chunk):
            def fn(v):
                return self.expert(v, self.inference(v))
            outputs.append(checkpoint(fn, xx, use_reentrant=False) if torch.is_grad_enabled() else fn(xx))
        return torch.cat(outputs).reshape(*shape, 2)


class DecisionMoE(nn.Module):
    def __init__(self, inference, experts, arm):
        super().__init__()
        assert arm in ['learned', 'uniform']
        self.inference = copy.deepcopy(inference)
        self.experts = nn.ModuleList([copy.deepcopy(e) for e in experts])
        self.arm, self.chunk = arm, 4096
        # 117 observed history +8 proposals +6 pair distances +1 horizon +14 belief summary.
        self.router = nn.Sequential(nn.Linear(146, 64), nn.SiLU(), nn.Linear(64, 4)).to(next(experts[0].head.parameters()))
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self.configure()

    def configure(self):
        self.requires_grad_(False)
        for e in self.experts:
            e.configure()
        self.router.requires_grad_(self.arm == 'learned')
        for p in self.parameters():
            p.grad = None

    def proposals_and_probabilities(self, x):
        c = self.inference(x)
        proposals = torch.stack([e(x, c) for e in self.experts], 1)
        if self.arm == 'uniform':
            p = x.new_full((len(x), 4), .25)
            assert not p.requires_grad and torch.equal(p, torch.full_like(p, .25))
        else:
            distances = torch.stack([(proposals[:, i]-proposals[:, j]).norm(dim=-1)/8
                                    for i in range(4) for j in range(i+1, 4)], -1)
            inp = torch.cat([x, proposals.flatten(1)/4, distances, c['remaining'], c['summary']], -1)
            assert inp.shape[-1] == 146
            # Do not detach: exact derivative of sum_k p_theta(k|h,a)Q(h,a_k)
            # includes routing's dependence on expert proposals.
            p = self.router(inp).softmax(-1)
        return proposals, p

    def forward(self, x, uniform=None, collect=False):
        shape = x.shape[:-1]
        flat = observed_prefix(x).reshape(-1, 117)
        if uniform is None:
            uniform = torch.rand(len(flat), device=x.device, dtype=x.dtype)
        else:
            uniform = torch.as_tensor(uniform, device=x.device, dtype=x.dtype).reshape(-1)
            assert len(uniform) == len(flat) and (uniform >= 0).all() and (uniform < 1).all()
        acts, indices, weights, proposed = [], [], [], []
        for xx, uu in zip(flat.split(self.chunk), uniform.split(self.chunk)):
            proposals, p = self.proposals_and_probabilities(xx)
            k = (uu[:, None] >= p.cumsum(-1)).sum(-1).clamp_max(3)
            action = proposals.gather(1, k[:, None, None].expand(-1, 1, 2)).squeeze(1)
            acts.append(action)
            if collect:
                indices.append(k)
                weights.append(p)
                proposed.append(proposals)
        result = torch.cat(acts).reshape(*shape, 2)
        if not collect:
            return result
        return result, dict(selected=torch.cat(indices), gates=torch.cat(weights), proposals=torch.cat(proposed))

    def expected_actor_loss(self, x, critic):
        flat = observed_prefix(x).reshape(-1, 117)
        sums = []
        for xx in flat.split(self.chunk):
            def fn(v):
                proposals, p = self.proposals_and_probabilities(v)
                inputs = torch.cat([v[:, None].expand(-1, 4, -1), proposals], -1)
                q = critic(inputs).squeeze(-1)
                return -(p*q).sum(-1).sum()
            sums.append(checkpoint(fn, xx, use_reentrant=False) if torch.is_grad_enabled() else fn(xx))
        return torch.stack(sums).sum()/len(flat)

    def standalone(self, index):
        return StandalonePolicy(self.inference, self.experts[index])
