"""DAD's sum encoder generalized to normalized SIR time/count histories."""
import torch
from torch import nn
class SIRSumDAD(nn.Module):
 kind='dad';architecture='DAD_sum_encoder_256_16_SIR_normalized_history'
 def __init__(self,horizon):
  super().__init__();self.horizon=horizon
  self.first_design=nn.Parameter(torch.zeros(1,dtype=torch.float64))
  self.encoder=nn.Sequential(nn.Linear(2,256),nn.ReLU(),nn.Linear(256,16))
  self.emitter=nn.Linear(16,1);self.double()
 def forward(self,features,stage):
  if stage==0:return self.first_design[None].expand(len(features),-1)
  slots=features[:,1:].reshape(len(features),self.horizon,3)
  return self.emitter(self.encoder(slots[:,:stage,:2]).sum(1))

class SIRSumMLP(nn.Module):
 """SIR-specific strong single-policy base; normalized set history and stage."""
 checkpoint_kind='sir_sum_single_v1';kind='sparse_moe_prototype';control_method='sir_sum_single';architecture='SIR_sum16_stage193_single_64_64_tanh_v1'
 def __init__(self,horizon):
  super().__init__();self.horizon=horizon;self.dense=True;self.routing_override=None;self.last_route=None
  self.first_design=nn.Parameter(torch.zeros(1,dtype=torch.float64))
  self.sum_encoder=nn.Sequential(nn.Linear(2,256),nn.ReLU(),nn.Linear(256,16))
  self.head=nn.Sequential(nn.Linear(193,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1));self.double()
 def residual_context(self,features):
  slots=features[:,1:].reshape(len(features),self.horizon,3)
  z=(self.sum_encoder(slots[...,:2])*slots[...,2:3]).sum(1)
  return torch.cat((torch.nn.functional.pad(z,(0,176)),features[:,:1]),-1)
 def forward(self,features,stage):
  self.last_route=None
  if stage==0:return self.first_design[None].expand(len(features),-1)
  return self.head(self.residual_context(features))
