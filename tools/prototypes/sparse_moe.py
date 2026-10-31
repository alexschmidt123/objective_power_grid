"""Isolated sPCE-only MoE routing prototype; not wired into active campaigns.

DSelect-k adaptation for four independently parameterized policy experts.
Training relaxes two selector bits; inference executes <=k unique experts.
No posterior, Q-value, information prediction, or branch-specific objectives.
"""
import torch
from torch import nn
from src.research.shared_history import MoEPolicy

class DSelectGate(nn.Module):
    def __init__(self,context_dim,k=1,gamma=1.):
        super().__init__()
        if k not in (1,2) or gamma<=0:raise ValueError('Require k=1/2 and positive gamma')
        self.k,self.gamma=k,float(gamma)
        self.bits=nn.Linear(context_dim,2*k)
        self.mixing=nn.Linear(context_dim,k)
        nn.init.normal_(self.bits.weight,std=.001);nn.init.uniform_(self.bits.bias,-.05,.05)
        nn.init.zeros_(self.mixing.weight);nn.init.zeros_(self.mixing.bias)
        codes=torch.tensor([[0,0],[0,1],[1,0],[1,1]],dtype=torch.float64)
        self.register_buffer('codes',codes)
    def forward(self,context,hard=False):
        logits=self.bits(context).reshape(-1,self.k,2)
        if hard:bits=(logits>=0).to(logits.dtype)
        else:
            t=(logits/self.gamma).clamp(-.5,.5)
            bits=-2*t.pow(3)+1.5*t+.5
        selectors=torch.where(self.codes[None,None].bool(),bits[:,:,None],1-bits[:,:,None]).prod(-1)
        mix=self.mixing(context).softmax(-1)
        weights=(selectors*mix[:,:,None]).sum(1)
        return weights,selectors,bits

class SparseMoESBOED(MoEPolicy):
    architecture='spce_only_shared_encoder_dselect_prototype'
    kind='sparse_moe_prototype'
    def __init__(self,action_dim,obs_dim,horizon,hidden=64,k=1,dense=False,first_action=None):
        super().__init__(action_dim,obs_dim,horizon,hidden,'moe_sboed',1.,first_action)
        self.kind='sparse_moe_prototype'
        self.dense=dense;self.k=k
        if not dense:self.router=DSelectGate(3*hidden+1,k).double()
        self.last_route=None
    def gate(self,context,hard=False):
        if self.dense:
            w=self.router(context.mean(1)).softmax(-1)
            return w,None,None
        return self.router(context.mean(1),hard=hard)
    def dispatch(self,context,weights):
        if not torch.isfinite(weights).all():raise FloatingPointError('Nonfinite sparse gate')
        output=context.new_zeros(len(context),self.action_dim)
        for index,expert in enumerate(self.experts):
            rows=(weights[:,index]>0).nonzero().flatten()
            if len(rows):
                value=expert(context[rows,index])*weights[rows,index,None]
                output=output.index_add(0,rows,value)
        return output
    def forward(self,features,stage):
        if hasattr(self,'first_design') and stage==0:
            self.last_route=None
            return self.first_design[None].expand(len(features),-1)
        context=self.context(features)
        weights,selectors,bits=self.gate(context,hard=not self.training and not self.dense)
        self.last_route=dict(weights=weights,selectors=selectors,bits=bits)
        return self.dispatch(context,weights)
    def regularizers(self,collapse_threshold=.75):
        """Advisory mild anti-collapse and selector-hardening terms, separate from EIG.

        No equal-use constraint. Coefficients/schedules require validation.
        """
        if self.last_route is None:return self.empty.sum()*0,self.empty.sum()*0
        w=self.last_route['weights'];load=(w.mean(0)-collapse_threshold).clamp_min(0).square().sum()
        q=self.last_route['selectors']
        entropy=w.sum()*0 if q is None else -(q*q.clamp_min(1e-15).log()).sum(-1).mean()
        return load,entropy

class MatchedDensePolicy(SparseMoESBOED):
    """DAD-style single policy head with the same shared history encoder.

    This is an architecture control; original DAD remains an external baseline.
    Count all trainable policy parameters, including the reference router.
    """
    def __init__(self,action_dim,obs_dim,horizon,hidden=64,first_action=None):
        super().__init__(action_dim,obs_dim,horizon,hidden,dense=True,first_action=first_action)
        target=sum(p.numel() for p in self.experts.parameters())+sum(p.numel() for p in self.router.parameters())
        inputs=4*(3*hidden+1)
        def count(w):return (inputs+1)*w+(w+1)*w+(w+1)*action_dim
        width=min(range(1,16*hidden+1),key=lambda w:abs(count(w)-target))
        del self.experts;del self.router
        self.head=nn.Sequential(nn.Linear(inputs,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,action_dim)).double()
        self.architecture='shared_history_parameter_matched_dense_policy_control'
    def forward(self,features,stage):
        if hasattr(self,'first_design') and stage==0:return self.first_design[None].expand(len(features),-1)
        return self.head(self.context(features).flatten(1))
