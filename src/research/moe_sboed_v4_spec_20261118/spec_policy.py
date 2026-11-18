"""V4 values unchanged;warmup routes expert-parameter gradients by regime."""
import torch
from torch.func import functional_call
from policy import V4Actor
from spec_regimes import Regimes

class SpecActor(V4Actor):
    def configure(self,phase):
        super().configure(phase);self.spec_phase=phase
        if phase=='warmup':self.router.requires_grad_(False)
        self._record_exposure=False
    def forward(self,x,collect=False):
        previous=getattr(self,'_record_exposure',False)
        self._record_exposure=self.spec_phase=='warmup' and torch.is_grad_enabled()
        try:return super().forward(x,collect)
        finally:self._record_exposure=previous
    def packets(self,x):
        if self.spec_phase!='warmup':return super().packets(x)
        c=self.query(x)
        if torch.is_grad_enabled():
            masks,values=self.regimes(c,self.experts[3].semantic.opportunity)
            if getattr(self,'audit_masks',None) is not None:
                masks=self.audit_masks.to(x.device);assert masks.shape==(len(x),4)
            detached={name:value.detach() if torch.is_tensor(value) else value for name,value in c.items()}
        packets=[]
        for k,expert in enumerate(self.experts):
            if torch.is_grad_enabled():
                # Context derivatives are native;expert-parameter derivatives
                # originate only in the detached-context branch,scaled by m.
                parameters={name:p.detach() for name,p in expert.named_parameters()}
                z_shared,raw=functional_call(expert,parameters,(c,))
                z_parameter,_=expert(detached)
                z=z_shared+masks[:,k:k+1].to(z_parameter)*(z_parameter-z_parameter.detach())
            else:z,raw=expert(c)
            packets.append((z,raw))
        if torch.is_grad_enabled() and self._record_exposure:
            if not hasattr(self,'exposure'):
                self.exposure=dict(total=x.new_zeros(()),selected=x.new_zeros(4),stage_total=x.new_zeros(30),stage_selected=x.new_zeros(30,4))
            with torch.no_grad():
                stage=c['stage'];self.exposure['total'].add_(len(x));self.exposure['selected'].add_(masks.sum(0))
                self.exposure['stage_total'].index_add_(0,stage,torch.ones_like(stage,dtype=x.dtype))
                self.exposure['stage_selected'].index_add_(0,stage,masks.to(x))
        z=torch.stack([q[0] for q in packets],1);raw=torch.stack([q[1] for q in packets],1)
        return z,raw,x.new_full((len(x),4),.25)

def attach(actor,thresholds):
    actor.__class__=SpecActor;actor.regimes=Regimes(thresholds);actor.audit_masks=None
    actor.configure('warmup');return actor
