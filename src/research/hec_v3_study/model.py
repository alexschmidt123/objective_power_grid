"""Four-pathway collaborative V3; all interventions are explicitly configured."""
import torch
from torch.nn import functional as F
from frozen_feature_policy import FeatureMoE

class StudyMoE(FeatureMoE):
 def __init__(self,parent,seed=202611145001,arm='learned'):
  self.arm=arm;self.correction_enabled=arm in ['learned','router_only'];self.encoders_enabled=arm in ['learned','uniform','encoders_only']
  self.gate_table=None;self.swap_gates=False;self.remove_feature=None;self.observation_swap=False
  super().__init__(parent,seed)
 def configure(self,phase):
  super().configure(phase)
  if self.arm=='uniform' or not self.correction_enabled:self.router.requires_grad_(False)
  if self.arm=='uniform':self.prior_coordinator.requires_grad_(False);self.prior_response_gate.requires_grad_(False)
  if not self.encoders_enabled:
   for e in self.experts:e.innovation.requires_grad_(False)
 def compute(self,x,mode='learned',expert=None,collect=False):
  c=self.context(x)
  if mode=='expert':
   packet=self.experts[expert](c)
   if not self.encoders_enabled:packet['innovation']=torch.zeros_like(packet['innovation'])
   action=self.decode_one(c,expert,packet)
   return (action,dict(private=packet['private'])) if collect else action
  packets=[e(c) for e in self.experts]
  first=torch.stack([p['first'] for p in packets],1);response=torch.stack([p['response'] for p in packets],1)
  legacy=torch.stack([p['legacy'] for p in packets],1);innovation=torch.stack([p['innovation'] for p in packets],1)
  if not self.encoders_enabled:innovation=torch.zeros_like(innovation)
  projected=torch.stack([F.linear(p['first'],self.decoder.weight[:,128+64*k:128+64*(k+1)])+F.linear(p['response'],self.decoder.response_weight[:,64*k:64*(k+1)])+innovation[:,k] for k,p in enumerate(packets)],1)
  proposals=torch.stack([self.decode_one(c,k,dict(p,innovation=innovation[:,k])) for k,p in enumerate(packets)],1)
  remaining=(30-c['stage']).to(x.dtype)[:,None]/30;norms=torch.log1p(projected.square().mean(-1));offsets=(proposals-c['a0'][:,None])/4
  distances=torch.stack([(proposals[:,i]-proposals[:,j]).norm(dim=-1)/8 for i in range(4) for j in range(i+1,4)],-1)
  inp=torch.cat([c['s'],remaining,norms,offsets.flatten(1),distances],-1).detach()
  correction=self.router(inp) if self.correction_enabled else inp.new_zeros(len(inp),4)
  prior=self.prior_coordinator(torch.cat([c['s'],legacy.flatten(1)],-1))+self.prior_response_gate(response.flatten(1))
  gates=(10*((prior+correction)/10).tanh()).softmax(-1)
  if mode=='uniform' or self.arm=='uniform':gates=torch.full_like(gates,.25)
  if self.gate_table is not None:gates=self.gate_table[c['stage']]
  if self.swap_gates:gates=gates.roll(1,0)
  if self.observation_swap:
   # Holding designs and stage fixed, swap only observed observations; no future slots.
   alt=x.clone();alt[:,88:]=x[:,88:].roll(1,0)
   self.observation_swap=False
   try:_,other=self.compute(alt,collect=True)
   finally:self.observation_swap=True
   gates=other['gates']
  if self.remove_feature is not None:
   mask=gates.new_ones(4);mask[self.remove_feature]=0;gates=gates*mask;gates=gates/gates.sum(-1,keepdim=True)
  action=self.decoder(c,gates,first,response,innovation)
  if not collect:return action
  return action,dict(gates=gates,prior_logits=prior,correction=correction,router_input=inp,history=c['s'],commands=c['commands'],
    first=first,responses=response,innovation=innovation,projected=projected,proposals=proposals,scale=c['scale'])
