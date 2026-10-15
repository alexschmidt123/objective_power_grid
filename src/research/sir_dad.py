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
