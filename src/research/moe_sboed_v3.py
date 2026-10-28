"""Development process MoE. No simulator, parameter truth, or online optimizer here."""
import math
import torch
from torch import nn
from torch.nn import functional as F

METHOD = 'moe_sboed_v3'
STATUS = 'DEVELOPMENT_UNVALIDATED'

class HistoryAdapter(nn.Module):
    """Dimension-only adapter; scales/feasibility belong to the benchmark."""
    def __init__(self, action_dim, obs_dim, horizon, width=16):
        super().__init__()
        self.action_dim, self.obs_dim, self.horizon = action_dim, obs_dim, horizon
        self.token = nn.Sequential(nn.Linear(action_dim+obs_dim+1,256),nn.ReLU(),nn.Linear(256,width))
        self.norm = nn.LayerNorm(width)
        self.time = nn.Linear(1,width,bias=False)
    def forward(self, features, stage):
        slots=features[:,1:].reshape(len(features),self.horizon,self.action_dim+self.obs_dim+1)
        # Explicit normalized position retains order for dynamical systems.
        if stage:
            t=torch.arange(1,stage+1,device=features.device,dtype=features.dtype)/self.horizon
            tokens=torch.cat((slots[:,:stage,:-1],t[None,:,None].expand(len(features),-1,1)),dim=-1)
            z=self.token(tokens).sum(1)/math.sqrt(stage)
        else:z=features.new_zeros(len(features),16)
        return self.norm(z)+self.time(features[:,:1])

class ConditionalMixture(nn.Module):
    """Proper conditional density surrogate in benchmark-transformed coordinates.

    It is an approximation, not a posterior sample oracle. One joint theta/history
    pair supplies an unbiased stochastic conditional cross-entropy training term.
    """
    def __init__(self, dimension, width=16, components=4):
        super().__init__();self.d=dimension;self.k=components
        self.head=nn.Linear(width,components*(1+dimension+dimension*(dimension+1)//2))
        nn.init.normal_(self.head.weight,std=.02);nn.init.zeros_(self.head.bias)
        self.register_buffer('tri',torch.tril_indices(dimension,dimension))
    def forward(self,z):
        x=self.head(z).reshape(len(z),self.k,-1);w=x[:,:,0].softmax(-1)
        mu=3*torch.tanh(x[:,:,1:1+self.d]/3)
        raw=x[:,:,1+self.d:];L=z.new_zeros(len(z),self.k,self.d,self.d)
        L[:,:,self.tri[0],self.tri[1]]=.25*torch.tanh(raw)
        diag=torch.arange(self.d,device=z.device)
        rawdiag=raw[:,:,self.tri[0]==self.tri[1]]
        L[:,:,diag,diag]=.05+F.softplus(rawdiag)
        return w,mu,L
    @staticmethod
    def log_prob(distribution,target):
        w,mu,L=distribution;delta=target[:,None]-mu
        whitened=torch.linalg.solve_triangular(L,delta.unsqueeze(-1),upper=False).squeeze(-1)
        lp=-.5*whitened.square().sum(-1)-L.diagonal(dim1=-2,dim2=-1).log().sum(-1)-.5*mu.shape[-1]*math.log(2*math.pi)
        return torch.logsumexp(w.clamp_min(1e-300).log()+lp,-1)

def process_views(distribution):
    """Four non-additive, differentiable operators; coordinates are anonymous.

    Total C=W+B; E1 uses C, E2 between-component contrasts, E3 standardized C,
    E4 within-component information (inverse covariance). None is an EIG term.
    """
    w,mu,L=distribution;d=mu.shape[-1];Ck=L@L.transpose(-1,-2)
    mean=(w[...,None]*mu).sum(1);delta=mu-mean[:,None]
    W=(w[...,None,None]*Ck).sum(1)
    B=(w[...,None,None]*delta.unsqueeze(-1)*delta.unsqueeze(-2)).sum(1);C=W+B
    eye=torch.eye(d,dtype=C.dtype,device=C.device)
    C=C+1e-5*eye;W=W+1e-5*eye
    sd=C.diagonal(dim1=-2,dim2=-1).sqrt()
    corr=C/(sd[:,:,None]*sd[:,None,:])
    # B retains orientation of competing explanations; precision is smoothly
    # bounded after a stable Cholesky solve (no raw unstable inverse).
    precision=torch.cholesky_inverse(torch.linalg.cholesky(W))
    def pack(matrix):return torch.cat((mean,torch.tanh(matrix.flatten(1))),-1)
    return [pack(torch.log1p(C.diagonal(dim1=-2,dim2=-1)).diag_embed()+C/(1+C.abs())),
            pack(B/(1+C.diagonal(dim1=-2,dim2=-1).sum(-1))[:,None,None]),
            pack(corr),pack(precision/(1+precision.diagonal(dim1=-2,dim2=-1).sum(-1))[:,None,None])]

class ProcessMoE(nn.Module):
    kind=METHOD;outputs_design=False
    def __init__(self,action_dim=2,obs_dim=1,param_dim=4,horizon=30,variant='full',dense_width=None):
        super().__init__();self.horizon=horizon;self.variant=variant;self.param_dim=param_dim
        self.adapter=HistoryAdapter(action_dim,obs_dim,horizon)
        self.uncertainty=ConditionalMixture(param_dim)
        n=param_dim+param_dim*param_dim
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(16+n,16),nn.SiLU(),nn.Linear(16,16)) for _ in range(4)]) if variant in ('full','noaux') else nn.ModuleList()
        self.router=nn.Linear(16+4*n,4) if len(self.experts) else None
        if self.router is not None:nn.init.normal_(self.router.weight,std=.02);nn.init.zeros_(self.router.bias)
        # Ordered zero-filled slots allow cross-expert nonlinear interactions.
        inputs=80 if len(self.experts) else 16+4*n if variant=='dense' else 16
        self.fusion=nn.Sequential(nn.LayerNorm(inputs),nn.Linear(inputs,dense_width or 32),nn.SiLU(),nn.Linear(dense_width or 32,action_dim))
        nn.init.normal_(self.fusion[-1].weight,std=.01);nn.init.zeros_(self.fusion[-1].bias)
        self.routing_mode='learned';self.mask_routed=False;self.substitution=None
        self._routing_rng=None
        self.last_dispatch=None;self._aux=[];self._last_stage=None;self.double()
    def __getstate__(self):
        s=super().__getstate__();s['_aux']=[];s['last_dispatch']=None;s['_last_stage']=None;s['_routing_rng']=None;return s
    def forward(self,features,stage):
        if stage==0 or (self._last_stage is not None and stage<=self._last_stage):self._aux=[]
        self._last_stage=stage;z=self.adapter(features,stage);dist=self.uncertainty(z)
        if self.training:self._aux.append(dist)
        views=process_views(dist)
        if not len(self.experts):return self.fusion(torch.cat((z,*views),-1) if self.variant=='dense' else z)
        logits=self.router(torch.cat((z,*views),-1));scores=logits.softmax(-1)
        if self.routing_mode=='random':
            # Fixed independent routing realization, separate from simulator RNG.
            if self._routing_rng is None:self._routing_rng=torch.Generator(device=z.device).manual_seed(97402890)
            key=torch.rand(len(z),4,device=z.device,dtype=z.dtype,generator=self._routing_rng)
            ids=key.topk(2,-1).indices;weights=torch.full_like(scores[:,:2],.5)
        else:
            ids=scores.topk(2,-1).indices;weights=scores.gather(1,ids);weights=weights/weights.sum(-1,keepdim=True)
            if self.routing_mode=='uniform':weights=torch.full_like(weights,.5)
        slots=z.new_zeros(len(z),4,16)
        if not self.mask_routed:
            for e in ids.unique().tolist():
                rows,col=(ids==e).nonzero(as_tuple=True)
                selected=e if self.substitution is None else self.substitution
                value=self.experts[selected](torch.cat((z[rows],views[selected][rows]),-1))
                slots[rows,e]=value*weights[rows,col,None]
        self.last_dispatch=dict(indices=ids.detach(),weights=weights.detach(),scores=scores.detach())
        return self.fusion(torch.cat((z,slots.flatten(1)),-1))
    def auxiliary_loss(self,target):
        # Detaching BOTH generating theta and history path avoids auxiliary
        # incentives to choose histories that merely make density fitting easy.
        # Re-evaluation on detached histories is supplied by auxiliary_on below.
        return -torch.stack([ConditionalMixture.log_prob(x,target.detach()).mean() for x in self._aux]).mean()
    def auxiliary_on(self,features,target):
        values=[]
        for stage,f in enumerate(features):
            z=self.adapter(f.detach(),stage)
            values.append(-ConditionalMixture.log_prob(self.uncertainty(z),target.detach()).mean())
        return torch.stack(values).mean()/self.param_dim
    def counts(self):
        total=sum(p.numel() for p in self.parameters());inactive=2*sum(p.numel() for p in self.experts[0].parameters()) if len(self.experts) else 0
        return dict(total_parameters=total,active_parameters=total-inactive,experts=len(self.experts),top_k=2 if len(self.experts) else 0)

def dense_match(target,**kwargs):
    probe=ProcessMoE(variant='dense',dense_width=16,**kwargs)
    base=probe.counts()['total_parameters'];step=probe.fusion[1].in_features+1+probe.fusion[-1].out_features
    width=max(1,round(16+(target-base)/step))
    return ProcessMoE(variant='dense',dense_width=width,**kwargs)
