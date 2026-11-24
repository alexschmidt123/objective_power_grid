"""MoE-sBOED-v2 candidate: independently decoded routed design contributions.
Unregistered research candidate until explicit validation/promotion. No planner.
"""
import copy
import torch
from torch import nn
from src.research.deepseek_sboed import DeepSeekPolicy,FeatureFFN

class DecisionMoE(DeepSeekPolicy):
    kind='moe_sboed_v2_candidate'
    def __init__(self,horizon=30,action_scale=4.,obs_scale=3.,balance_factor=.001):
        super().__init__(horizon,action_scale,obs_scale,'full',balance_factor)
        self.decoders=nn.ModuleList([nn.Linear(16,2,bias=False) for _ in range(7)])
        # Same small readout distribution as the native shared design head.
        for d in self.decoders:nn.init.normal_(d.weight,std=.01)
        self.double();self.routing_mode='learned';self.substitution=None;self.last_decision_parts=None
    def __getstate__(self):
        s=super().__getstate__();s['last_decision_parts']=None;return s
    def forward(self,features,stage):
        if stage==0 or (self.training and self._last_stage is not None and stage<=self._last_stage):self._gate_cache=[]
        self._last_stage=stage;z=self.encode_context(features,stage)
        shared=self.emitter(z+self.shared(z));routed=torch.zeros_like(shared)
        scores=self.router(z).softmax(-1);weights,ids=scores.topk(3,-1)
        if self.routing_mode=='uniform':
            # Dense uniform-routing intervention, not sparse production inference.
            for e in range(7):routed=routed+self.decoders[e](self.routed[e](z))*weights.sum(-1,keepdim=True)/7
        elif not self.ablate_routed:
            if self.substitution is not None:
                target_stage,e=self.substitution
                if stage==target_stage:
                    ids=ids.clone();eligible=~(ids[:,:2]==e).any(-1);ids[eligible,2]=e
            for e in ids.unique().tolist():
                rows,slots=(ids==e).nonzero(as_tuple=True)
                value=self.decoders[e](self.routed[e](z[rows]))
                routed.index_add_(0,rows,value*weights[rows,slots,None])
        if self.training:self._gate_cache.append((scores,ids.detach()))
        self.last_dispatch=dict(indices=ids.detach(),scores=scores.detach(),selected_weights=weights.detach())
        self.last_features=z.detach();self.last_decision_parts=dict(shared=shared.detach(),routed=routed.detach())
        return shared+routed
    def counts(self):
        total=sum(p.numel() for p in self.parameters());one=sum(p.numel() for p in self.routed[0].parameters())+sum(p.numel() for p in self.decoders[0].parameters())
        return dict(total_parameters=total,activated_parameters_per_context=total-4*one,shared_experts=1,routed_experts=7,top_k=3,expert_readout='independent16to2_biasfree',full_softmax_selected_mass=True)

def matched_models(seed):
    torch.manual_seed(seed);native=DeepSeekPolicy();candidate=DecisionMoE()
    candidate.load_state_dict(native.state_dict(),strict=False)
    # Match shared encoder/head initialization across all four policies.
    shared=DeepSeekPolicy(variant='shared_only');shared.load_state_dict({k:v for k,v in native.state_dict().items() if not k.startswith(('routed.','router.'))},strict=True)
    dense=DeepSeekPolicy(variant='dense');dense.shared=FeatureFFN(71).double()
    dense.load_state_dict({k:v for k,v in native.state_dict().items() if not k.startswith(('routed.','router.','shared.'))},strict=False)
    assert sum(p.numel() for p in dense.parameters())==candidate.counts()['total_parameters']==8658
    return dict(v1=native,v2=candidate,dense=dense,shared=shared)
