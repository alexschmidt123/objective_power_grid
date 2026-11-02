"""Teacher-free competence-specialized full-horizon policy MoE.

Only this module is needed by deployment. No predictive model, planner,
posterior-density experts, offline evaluator or optimization routine is imported.
"""
import math,time
import torch
from torch import nn

class HistoryEncoder(nn.Module):
    def __init__(self,horizon=30):
        super().__init__();self.horizon=horizon
        self.token=nn.Sequential(nn.Linear(4,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())
        self.context=nn.Sequential(nn.Linear(65,64),nn.Tanh())
        self.double()
    def forward(self,features):
        slots=features[:,1:].reshape(-1,self.horizon,4);mask=slots[:,:,-1:]
        index=torch.arange(self.horizon,device=features.device,dtype=features.dtype)[None,:,None].expand(len(features),-1,-1)/self.horizon
        embedding=self.token(torch.cat([slots[:,:,:3],index],-1))*mask
        pooled=embedding.sum(1)/mask.sum(1).clamp_min(1).sqrt()
        return self.context(torch.cat([pooled,features[:,:1]],-1))

def expert(width):return nn.Sequential(nn.Linear(64,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,2))

class SAMoE(nn.Module):
    kind='sa_moe';outputs_design=True
    def __init__(self,single=False):
        super().__init__();self.single=single;self.encoder=HistoryEncoder()
        self.experts=nn.ModuleList([expert(159)] if single else [expert(64) for _ in range(4)])
        self.router=None if single else nn.Sequential(nn.Linear(64,32),nn.Tanh(),nn.Linear(32,4))
        self.routing_mode='learned';self.double();self.inference_times=[];self.last_expert=None
    def context(self,features):return self.encoder(features)
    def probabilities(self,z):
        return z.new_ones((len(z),1)) if self.single else self.router(z).softmax(-1)
    def all_proposals(self,z):return 4*torch.stack([net(z).tanh() for net in self.experts],1)
    def forward(self,features,stage):
        start=time.perf_counter();z=self.context(features);encoded=time.perf_counter();p=self.probabilities(z)
        if self.single:index=torch.zeros(len(z),dtype=torch.long,device=z.device)
        elif self.routing_mode=='uniform':index=torch.randint(4,(len(z),),device=z.device)
        else:index=p.argmax(-1)
        routed=time.perf_counter();self.last_expert=index.detach()
        if self.training:
            hard=torch.nn.functional.one_hot(index,len(self.experts)).to(p)
            weight=hard+p-p.detach();action=(weight[:,:,None]*self.all_proposals(z)).sum(1)
        else:
            # Sparse grouped dispatch: one expert per history, no coordinate average.
            action=z.new_empty((len(z),2))
            for e in index.unique().tolist():
                selected=index==e;action[selected]=4*self.experts[e](z[selected]).tanh()
        end=time.perf_counter()
        if not self.training:self.inference_times.append(dict(encoder=encoded-start,router=routed-encoded,expert=end-routed,total=end-start))
        return action
    def forced_first(self,features,expert_index):
        return 4*self.experts[expert_index](self.context(features)).tanh()

def competence_responsibilities(q,se,saturated=None):
    """Decision competence, NEVER posterior hypothesis mass.

    Unresolved contrasts produce soft targets. Log-softmax is scale safe.
    """
    temp=torch.maximum(q.new_tensor(.1),2*se.max(-1,keepdim=True).values)
    log_r=((q-q.max(-1,keepdim=True).values)/temp).log_softmax(-1)
    if saturated is not None:log_r=torch.where(saturated[:,None],torch.full_like(log_r,-math.log(q.shape[-1])),log_r)
    return log_r,log_r.exp()

def conditional_risk(log_r,loss):
    # Unit-sum per-expert weights avoid the historical small-mass ESS bug.
    normalized=log_r.log_softmax(0).exp()
    return (normalized*loss).sum(0).mean(),normalized.square().sum(0).reciprocal()

def load_sa_checkpoint(path,device='cpu'):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('method')!='SA-MoE-sBOED':raise ValueError('Wrong checkpoint')
    model=SAMoE(state['single']);model.load_state_dict(state['state_dict'],strict=True);return model.to(device).eval()
