"""Frozen uniform HEC-v2 plus strictly routed feature corrections."""
import copy,itertools,math
import torch
from torch import nn
from sg_policy import frozen_buffers
PAIRS=list(itertools.combinations(range(4),2))

def pair_probabilities(logits,epsilon=0.):
    """Unordered Plackett-Luce Top2: sum of both ordered selections."""
    log_first=logits.log_softmax(-1);ordered=[]
    for i,j in PAIRS:
        mask_i=torch.arange(4,device=logits.device)==i
        mask_j=torch.arange(4,device=logits.device)==j
        ij=log_first[...,i]+logits[...,j]-torch.logsumexp(logits.masked_fill(mask_i,-torch.inf),-1)
        ji=log_first[...,j]+logits[...,i]-torch.logsumexp(logits.masked_fill(mask_j,-torch.inf),-1)
        ordered.append(torch.logaddexp(ij,ji))
    # Normalization removes accumulated roundoff, not missing probability mass.
    p=torch.stack(ordered,-1).softmax(-1)
    return (1-epsilon)*p+epsilon/6

class RRPolicy(nn.Module):
    def __init__(self,parent,cfg):
        super().__init__();self.base=frozen_buffers(copy.deepcopy(parent));self.radius=cfg['correction_radius']
        device=next(self.base.buffers()).device
        self.adaptations=nn.ModuleList([nn.Sequential(nn.Linear(256,32),nn.SiLU(),nn.Linear(32,16)) for _ in range(4)]).to(device)
        self.history_gate=nn.Linear(128,32).to(device)
        self.specialist_projection=nn.Linear(16,32,bias=False).to(device)
        self.correction=nn.Linear(32,2,bias=False).to(device)
        self.router=nn.Sequential(nn.Linear(128,32),nn.SiLU(),nn.Linear(32,4)).to(device)
        self.register_buffer('pairs',torch.tensor(PAIRS,device=device))
        with torch.no_grad():
            self.correction.weight.zero_();self.router[-1].weight.zero_();self.router[-1].bias.zero_()
        assert not list(self.base.parameters()),'Frozen base must be buffers, outside Adam and Polyak'

    @torch.no_grad()
    def frozen(self,x):
        shape=x.shape[:-1];flat=x.reshape(-1,117);out=[]
        for xx in flat.split(2048):
            action,info=self.base(xx,routing_override=xx.new_full((4,),.25),collect=True)
            out.append(torch.cat([action,info['history'],info['slots'].flatten(1)],-1))
        return torch.cat(out).reshape(*shape,642)

    def routing(self,features,uniform=False,epsilon=.1):
        h=features[...,2:130];raw=self.router(h);logits=10*torch.tanh(raw/10)
        p=pair_probabilities(logits,epsilon)
        return torch.full_like(p,1/6) if uniform else p

    def tokens(self,features,disable=None):
        h=features[...,2:130];expert=features[...,130:].reshape(*features.shape[:-1],4,128)
        z=torch.stack([adapt(torch.cat([h,expert[...,k,:]],-1)) for k,adapt in enumerate(self.adaptations)],-2)
        tokens=torch.nn.functional.silu(self.specialist_projection(z))*self.history_gate(h).sigmoid().unsqueeze(-2)
        if disable is not None:
            mask=tokens.new_ones(4);mask[disable]=0;tokens=tokens*mask[...,None]
        return tokens

    def all_actions(self,features,disable=None):
        tokens=self.tokens(features,disable)
        pooled=(tokens[...,self.pairs[:,0],:]+tokens[...,self.pairs[:,1],:])/math.sqrt(2)
        delta=self.radius*torch.tanh(self.correction(pooled))
        return (features[...,:2].unsqueeze(-2)+delta).clamp(-4,4),delta

    def distribution(self,features,uniform=False,epsilon=.1,disable=None):
        actions,delta=self.all_actions(features,disable)
        return actions,self.routing(features,uniform,epsilon),delta

    def select(self,x,u=None,uniform=False,epsilon=.1,disable=None,collect=False):
        features=self.frozen(x);p=self.routing(features,uniform,epsilon)
        u=torch.rand_like(p[...,0]) if u is None else u
        chosen=(u[...,None]>=p.cumsum(-1)).sum(-1).clamp_max(5)
        selected=self.pairs[chosen];shape=chosen.shape;flat=features.reshape(-1,642);selected_flat=selected.reshape(-1,2)
        h=flat[:,2:130];ee=flat[:,130:].reshape(-1,4,128)
        gate=self.history_gate(h).sigmoid();pooled=h.new_zeros(len(h),32)
        # Deployment executes exactly the two selected small adaptations.
        for k,adapt in enumerate(self.adaptations):
            idx=(selected_flat==k).any(-1).nonzero().flatten()
            if len(idx) and k!=disable:
                z=adapt(torch.cat([h[idx],ee[idx,k]],-1))
                pooled[idx]+=torch.nn.functional.silu(self.specialist_projection(z))*gate[idx]
        delta=(self.radius*torch.tanh(self.correction(pooled/math.sqrt(2)))).reshape(*shape,2)
        action=(features[...,:2]+delta).clamp(-4,4)
        if not collect:return action
        return action,dict(probabilities=p,chosen_pair=chosen,selected=selected,delta=delta,base_action=features[...,:2])

    def forward(self,x):return self.select(x)
