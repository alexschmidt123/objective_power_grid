"""Frozen posterior and four specialist query computations;no action policy/head.
Copied byte-for-byte class bodies from the isolated categorical pilot;policy classes excluded.
"""
import copy
import torch
from torch import nn
from hec_policy import mean_signal,jacobian,squash,upper,COST
from query_response import paired_discrimination,prospective_opportunity

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
