"""V2: mathematically equivalent selected-expert kernels for offline training.

Architecture, parameters, gates, balance objective and inference are unchanged.
Only selected FFN activations are evaluated. Stacking small parameter tensors
is not evaluation of unselected expert functions.
"""
import torch
from src.research.deepseek_sboed import DeepSeekPolicy

class DeepSeekV2Policy(DeepSeekPolicy):
    method='DeepSeekMoE-sBOED-v2'
    def routed_feature_sum(self,z,weights,indices):
        if not self.training:return super().routed_feature_sum(z,weights,indices)
        up=torch.stack([e.up.weight for e in self.routed])[indices]
        gate=torch.stack([e.gate.weight for e in self.routed])[indices]
        down=torch.stack([e.down.weight for e in self.routed])[indices]
        u=torch.einsum('bd,bkhd->bkh',z,up)
        g=torch.einsum('bd,bkhd->bkh',z,gate)
        activated=torch.nn.functional.silu(g)*u
        values=torch.einsum('bkh,bkdh->bkd',activated,down)
        return (values*weights[:,:,None]).sum(1)
    def mask_unused_gradients(self):
        # Gather autograd creates zero tensors for unselected bank weights.
        # Original Adam skips grad=None experts. Preserve that exact behavior.
        if not self._gate_cache or not len(self.routed):return
        used=set(torch.cat([ids.reshape(-1) for _,ids in self._gate_cache]).unique().tolist())
        for index,expert in enumerate(self.routed):
            if index not in used:
                for parameter in expert.parameters():parameter.grad=None

def load_v2(path,device='cpu'):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('method')!='DeepSeekMoE-sBOED-v2':raise ValueError('Not a v2checkpoint')
    p=DeepSeekV2Policy(saved['horizon'],saved['action_scale'],saved['obs_scale'],saved['variant'],saved['balance_factor']);p.load_state_dict(saved['state_dict'],strict=True);return p.to(device).eval()
