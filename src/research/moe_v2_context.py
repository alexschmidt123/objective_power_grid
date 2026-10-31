"""Single v2 revision: direct residual-context routed design readouts."""
import torch
from src.research.moe_v2_decision import DecisionMoE,matched_models

class ContextDecisionMoE(DecisionMoE):
    kind='moe_sboed_v2_context_readout_candidate'
    def forward(self,features,stage):
        if stage==0 or (self.training and self._last_stage is not None and stage<=self._last_stage):self._gate_cache=[]
        self._last_stage=stage;z=self.encode_context(features,stage)
        shared=self.emitter(z+self.shared(z));routed=torch.zeros_like(shared)
        scores=self.router(z).softmax(-1);weights,ids=scores.topk(3,-1)
        if self.routing_mode=='uniform':
            for e in range(7):routed=routed+self.decoders[e](z+self.routed[e](z))*weights.sum(-1,keepdim=True)/7
        elif not self.ablate_routed:
            if self.substitution is not None:
                target_stage,e=self.substitution
                if stage==target_stage:
                    ids=ids.clone();eligible=~(ids[:,:2]==e).any(-1);ids[eligible,2]=e
            for e in ids.unique().tolist():
                rows,slots=(ids==e).nonzero(as_tuple=True)
                value=self.decoders[e](z[rows]+self.routed[e](z[rows]))
                routed.index_add_(0,rows,value*weights[rows,slots,None])
        if self.training:self._gate_cache.append((scores,ids.detach()))
        self.last_dispatch=dict(indices=ids.detach(),scores=scores.detach(),selected_weights=weights.detach())
        self.last_features=z.detach();self.last_decision_parts=dict(shared=shared.detach(),routed=routed.detach())
        return shared+routed

def context_matched_models(seed):
    models=matched_models(seed);candidate=ContextDecisionMoE()
    candidate.load_state_dict(models['v2'].state_dict(),strict=True);models['v2']=candidate
    assert candidate.counts()['total_parameters']==sum(p.numel() for p in models['dense'].parameters())==8658
    return models
