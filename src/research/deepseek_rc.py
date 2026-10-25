
"""One-scalar routed contribution adaptation of the validated v1 policy."""
import math
import torch
from torch import nn
from src.research.deepseek_sboed import DeepSeekPolicy

class RoutedCalibrationPolicy(DeepSeekPolicy):
    method='DeepSeekMoE-sBOED-RC'
    def __init__(self,variant='full',initial_gain=1.,balance_factor=.001):
        if variant!='full':raise ValueError('Only one predeclared full candidate')
        super().__init__(variant=variant,balance_factor=balance_factor)
        if not .25<initial_gain<4.:raise ValueError('Gain initialization outside bounds')
        self.raw_gain=nn.Parameter(torch.tensor(math.log((initial_gain-.25)/(4.-initial_gain)),dtype=torch.float64))
        self.intervention='full'
        self.intervention_seed=7126101
        self._intervention_generator=None
    @property
    def gain(self):return .25+3.75*self.raw_gain.sigmoid()
    def routed_feature_sum(self,z,weights,indices):
        if self.intervention=='mask_routed':return z*0
        if self.intervention=='uniform_subset':
            if self._intervention_generator is None:self._intervention_generator=torch.Generator(device=z.device).manual_seed(self.intervention_seed)
            indices=torch.rand(len(z),7,device=z.device,generator=self._intervention_generator).topk(3,-1).indices
        if self.intervention in ('uniform_weights','uniform_subset'):
            weights=weights.sum(-1,keepdim=True).expand(-1,3)/3
        return self.gain*super().routed_feature_sum(z,weights,indices)
    def groups(self):
        return dict(encoder=list(self.encoder.parameters())+list(self.input_norm.parameters())+[self.time_vector],
                    shared=list(self.shared.parameters()),router=list(self.router.parameters()),
                    routed=[p for e in self.routed for p in e.parameters()],head=list(self.emitter.parameters()),gain=[self.raw_gain])
