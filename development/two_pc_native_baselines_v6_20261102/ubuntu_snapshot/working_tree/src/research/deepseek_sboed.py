"""DeepSeekMoE-inspired hidden-feature policy, trained with original DAD sPCE.

Paper: Dai etal., ACL2024, Sections3.1-3.3,4.4-4.5. Compact sBOED adaptation.
No predictive densities, teachers, competence refresh, posterior inference or
simulator/design optimization is imported or executed by this policy.
"""
import math
import torch
from torch import nn
from src.research.dad_location_policy import LocationDAD

class FeatureFFN(nn.Module):
    """Bias-free compact SwiGLU feature expert. No action is emitted here."""
    def __init__(self,width=8,dimension=16):
        super().__init__();self.up=nn.Linear(dimension,width,bias=False);self.gate=nn.Linear(dimension,width,bias=False);self.down=nn.Linear(width,dimension,bias=False)
        nn.init.normal_(self.down.weight,std=.02)
    def forward(self,z):return self.down(torch.nn.functional.silu(self.gate(z))*self.up(z))

class DeepSeekPolicy(LocationDAD):
    kind='deepseek_moe_sboed'
    def __init__(self,horizon=30,action_scale=4.,obs_scale=3.,variant='full',balance_factor=.001):
        super().__init__(horizon,action_scale,obs_scale,first_design_revision=1)
        if variant not in ['full','shared_only','routed_only','dense']:raise ValueError('Unknown predeclared control')
        self.kind='deepseek_moe_sboed';self.variant=variant;self.balance_factor=balance_factor;self.input_norm=nn.LayerNorm(16);self.time_vector=nn.Parameter(torch.zeros(16))
        self.emitter=nn.Sequential(nn.LayerNorm(16),nn.Linear(16,2));nn.init.normal_(self.emitter[-1].weight,std=.01);nn.init.zeros_(self.emitter[-1].bias)
        self.shared=FeatureFFN(66 if variant=='dense' else 8) if variant!='routed_only' else None
        n=8 if variant=='routed_only' else 7 if variant=='full' else 0
        self.routed=nn.ModuleList([FeatureFFN(8) for _ in range(n)]);self.top_k=4 if variant=='routed_only' else 3 if n else 0
        self.router=nn.Linear(16,n,bias=False) if n else None
        if self.router is not None:nn.init.normal_(self.router.weight,std=.01)
        self._gate_cache=[];self._last_stage=None;self.last_dispatch=None;self.last_features=None;self.ablate_shared=False;self.ablate_routed=False;self.double()
    def __getstate__(self):
        state=super().__getstate__();state['_gate_cache']=[];state['last_dispatch']=None;state['last_features']=None;state['_last_stage']=None;return state
    def encode_context(self,features,stage):
        # Exactly the original raw3-256ReLU-16sum history encoder; normalization
        # and explicit known time are declared new-policy stability adaptations.
        raw=super().encode(features,stage)
        return self.input_norm(raw)+(stage/self.horizon)*self.time_vector
    def forward(self,features,stage):
        if stage==0 or (self.training and self._last_stage is not None and stage<=self._last_stage):self._gate_cache=[]
        self._last_stage=stage
        z=self.encode_context(features,stage);u=z
        if self.shared is not None and not self.ablate_shared:u=u+self.shared(z)
        dispatch=None
        if self.router is not None and not self.ablate_routed:
            scores=self.router(z).softmax(-1);weights,indices=scores.topk(self.top_k,dim=-1)
            routed=z.new_zeros(z.shape)
            # Genuine sparse execution in BOTH training and deployment. A row
            # executes exactly top_k small FFNs; unselected FFNs are not called.
            for e in indices.unique().tolist():
                rows,slots=(indices==e).nonzero(as_tuple=True)
                values=self.routed[e](z[rows]);routed.index_add_(0,rows,values*weights[rows,slots,None])
            u=u+routed;dispatch=dict(indices=indices.detach(),scores=scores.detach(),selected_weights=weights.detach())
            if self.training:self._gate_cache.append((scores,indices.detach()))
        self.last_dispatch=dispatch;self.last_features=z.detach()
        return self.emitter(u)
    def balance_loss(self):
        if not self._gate_cache:return next(self.parameters()).sum()*0
        scores=torch.cat([x[0] for x in self._gate_cache]);ids=torch.cat([x[1] for x in self._gate_cache]);N=scores.shape[-1];K=ids.shape[-1]
        frequency=torch.nn.functional.one_hot(ids,N).sum(1).double().mean(0)*(N/K)
        probability=scores.mean(0)
        # PaperEq12-14: detached assignment frequency times full softmax mean.
        return self.balance_factor*(frequency.detach()*probability).sum()
    def routing_statistics(self):
        if not self._gate_cache:return {}
        scores=torch.cat([x[0].detach() for x in self._gate_cache]);ids=torch.cat([x[1] for x in self._gate_cache]);N=scores.shape[-1]
        frequency=torch.nn.functional.one_hot(ids,N).sum(1).double().mean(0)
        return dict(fraction_contexts_selected=frequency.cpu().tolist(),mean_softmax=scores.mean(0).cpu().tolist(),effective_gate_experts=float((-(scores*scores.clamp_min(1e-300).log()).sum(-1)).exp().mean()),distinct_selected_experts=int((frequency>0).sum()),mean_selected_gate_mass=float(scores.gather(1,ids).sum(-1).mean()),top_k=self.top_k)
    def counts(self):
        total=sum(p.numel() for p in self.parameters());one=sum(p.numel() for p in self.routed[0].parameters()) if len(self.routed) else 0
        activated=total-max(0,len(self.routed)-self.top_k)*one
        return dict(total_parameters=total,activated_parameters_per_context=activated,shared_experts=int(self.shared is not None),routed_experts=len(self.routed),top_k=self.top_k,expert_feature_dimension=16,routed_FFN_width=8,shared_FFN_width=66 if self.variant=='dense' else 8 if self.shared is not None else 0,FFN_linear_FLOPs_per_context=2*3*16*(66 if self.variant=='dense' else (int(self.shared is not None)+self.top_k)*8),history_encoder_linear_FLOPs_per_observation=2*(3*256+256*16),router_linear_FLOPs_per_context=2*16*len(self.routed),action_head_linear_FLOPs_per_context=2*16*2,FLOPs_definition='2perlinear multiplyaccumulate; excludesnonlinearities,normalization,topk,dispatch,simulator,likelihood; not measuredGPUFLOPs')

def load_deepseek(path,device='cpu'):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('method')!='DeepSeekMoE-sBOED':raise ValueError('Wrongcheckpoint')
    m=DeepSeekPolicy(saved['horizon'],saved['action_scale'],saved['obs_scale'],saved['variant'],saved['balance_factor']);m.load_state_dict(saved['state_dict'],strict=True);return m.to(device).eval()
