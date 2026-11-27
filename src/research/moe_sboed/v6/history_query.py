"""V5-style ordered history, bounded query conditioning and soft feature fusion."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class HistoryPosterior(nn.Module):
    """Amortized mixture over the two locations, trained from simulated prefixes."""
    def __init__(self, horizon):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4*horizon, 128), nn.SiLU(), nn.Linear(128, 128),
                                 nn.SiLU(), nn.Linear(128, 8*9))

    def forward(self, history):
        raw = self.net(history.flatten(1)).reshape(len(history), 8, 9)
        logw = raw[:, :, 0].log_softmax(1)
        mu = 5 * (raw[:, :, 1:5] / 5).tanh()
        sd = .025 + F.softplus(raw[:, :, 5:])
        return logw, mu, sd


class QueryExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.command = nn.Linear(79, 6)
        self.condition = nn.Sequential(nn.Linear(85, 32), nn.SiLU(), nn.Linear(32, 262))
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)
        self.features = nn.Sequential(nn.Linear(91, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                                      nn.Linear(64, 256))
        self.projection = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.LayerNorm(128, elementwise_affine=False))

    def query(self, ctx):
        cmd = self.command(ctx).tanh()
        delta, gain = self.condition(torch.cat((ctx, cmd), -1)).split((6,256), -1)
        return cmd + .25 * delta.tanh() * (1-cmd.abs()), 1 + .5*gain.tanh()

    def encode(self, ctx, cmd, gain, response):
        return self.projection(gain*self.features(torch.cat((ctx, cmd, response), -1)))


class HistoryQueryPolicy(nn.Module):
    kind = "fresh_history_query"
    def __init__(self, method, cfg, bench):
        super().__init__()
        if method not in ("moe-sboed", "moe-sboed-uniform") or bench.name != "location_finding":
            raise ValueError("History-query pilot requires a full or uniform location policy")
        self.bench = bench
        self.horizon = t = cfg["horizon"]
        self.posterior = HistoryPosterior(t).requires_grad_(False)
        self.history = nn.Sequential(nn.Linear(4*t, 128), nn.SiLU())
        self.token = nn.Sequential(nn.Linear(8, 16), nn.SiLU())
        self.combine = nn.Sequential(nn.Linear(192, 64), nn.SiLU())
        self.experts = nn.ModuleList(QueryExpert() for _ in range(4))
        self.head = nn.Sequential(nn.Linear(128, 32), nn.SiLU(), nn.Linear(32,2))
        self.router = None
        if method == "moe-sboed":
            self.router = nn.Sequential(nn.Linear(4*t+512+15, 64), nn.SiLU(), nn.Linear(64,4))
            nn.init.zeros_(self.router[-1].weight)
            nn.init.zeros_(self.router[-1].bias)
        pos = (torch.arange(t)+.5)/max(1,t-1)
        self.register_buffer("basis",torch.stack((torch.ones_like(pos),pos,(math.pi*pos).cos(),torch.ones_like(pos)),1))
        self.register_buffer("decay",torch.tensor([1.,1.,1.,math.exp(-.2)]))
        self.register_buffer("pairs",torch.triu_indices(4,4,1))
        self.register_buffer("points",torch.cat((2*torch.eye(4),-2*torch.eye(4))))
        self.float()

    def start_history(self, sample):
        return (self.basis.new_zeros(len(sample),self.horizon,4),
                self.basis.new_zeros(len(sample),4,16),self.basis.new_zeros(len(sample),3))

    def update_history(self,state,token,stage):
        hist,mom,last = state
        # Location rollout encodes physical action / 4 and raw observation / 3.
        x,y = token.float().split((2,1),-1)
        y = (.6*y).tanh()
        pair = torch.cat((x,y),-1)
        enc = self.token(torch.cat((pair,x*y,y-last[:,2:],x-last[:,:2]),-1))
        hist = hist.clone()
        hist[:,stage] = torch.cat((pair,torch.ones_like(y)),-1)
        return hist,(mom+enc[:,None]*self.basis[stage,:,None])*self.decay[:,None],pair

    def belief(self,hist,stage):
        if stage == 0:
            w = hist.new_full((len(hist),8),1/8)
            mu = hist.new_zeros(len(hist),8,4)
            sd = torch.ones_like(mu)
        else:
            logw,mu,sd = self.posterior(hist)
            w = logw.exp()
        mean = (w[:,:,None]*mu).sum(1)
        centered = mu-mean[:,None]
        scale = (w[:,:,None]*(centered.square()+sd.square())).sum(1).clamp_min(1e-6).sqrt()
        norm = centered/scale[:,None]
        i,j = self.pairs
        corr = (w[:,:,None]*norm[:,:,i]*norm[:,:,j]).sum(1)
        summary = torch.cat((mean/3,scale.log(),corr),-1)
        nodes = (mu[:,:,None]+sd[:,:,None]*self.points).flatten(1,2)
        return w,mean,summary,nodes

    def mixture(self,state,stage):
        hist,mom,_ = state
        flat = hist.flatten(1)
        h = self.combine(torch.cat((self.history(flat),mom.flatten(1)/max(1,stage)),-1))
        w,mean,summary,nodes = self.belief(hist,stage)
        rem = h.new_full((len(h),1),(self.horizon-stage)/self.horizon)
        ctx = torch.cat((h,summary,rem),-1)
        queries = [expert.query(ctx) for expert in self.experts]
        cmd = torch.stack([q[0] for q in queries],1)
        centroid = mean.reshape(-1,2,2).mean(1)
        designs = (centroid[:,None,None]+2*cmd.reshape(-1,4,3,2)).clamp(-4,4)
        signals,_ = self.bench.propagate(nodes[:,None,None],None,designs[:,:,:,None])
        signals = signals.squeeze(-1).reshape(len(h),4,3,8,8)
        pm = (signals.mean(-1)*w[:,None,None]).sum(-1)
        pv = ((signals-pm[:,:,:,None,None]).square().mean(-1)*w[:,None,None]).sum(-1)
        response = torch.cat((pm/5,pv.log1p()),-1)
        z = torch.stack([expert.encode(ctx, c, g, response[:,i])
                         for i,(expert,(c,g)) in enumerate(zip(self.experts,queries))],1)
        rates = (self.router(torch.cat((flat,z.flatten(1),rem,summary),-1)).softmax(-1)
                 if self.router is not None else h.new_full((len(h),4),.25))
        return z,rates

    def design_logits(self,state,stage):
        z,w = self.mixture(state,stage)
        return (2*self.head((z*w[:,:,None]).sum(1))).double()


@torch.no_grad()
def training_prefix(actions, observations, rng):
    """One uniform, nonempty decision prefix per fresh training episode."""
    batch, horizon = actions.shape[:2]
    stage = torch.randint(1,horizon,(batch,),device=actions.device,generator=rng)
    valid = torch.arange(horizon,device=actions.device) < stage[:,None]
    x,y = actions.float()/4,(observations.float()/5).tanh()
    return torch.cat((x,y,torch.ones_like(y)),-1)*valid[:,:,None]


def update_posterior(provider, optimizer, history, truth):
    # Labels are simulator parameters for the training episodes only.
    provider.requires_grad_(True)
    try:
        optimizer.zero_grad(set_to_none=True)
        logw,mean,sd = provider(history)
        def density(theta):
            logp = -.5*((theta[:,None]-mean)/sd).square()-sd.log()-.5*math.log(2*math.pi)
            return (logw+logp.sum(-1)).logsumexp(1)
        loss = -(torch.logaddexp(density(truth),density(truth.roll(2,-1)))-math.log(2)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(provider.parameters(),10,error_if_nonfinite=True)
        optimizer.step()
        return loss.detach().item()
    finally:
        optimizer.zero_grad(set_to_none=True)
        provider.requires_grad_(False)
