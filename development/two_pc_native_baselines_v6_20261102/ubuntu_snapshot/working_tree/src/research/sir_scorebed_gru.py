"""Torch inference translation of pinned Flax IqbalGRUNet, not a new learner."""
import numpy as np
import torch
from torch import nn
class SIRScoreGRUPolicy(nn.Module):
 kind='scorebed';outputs_design=True;architecture='official_ScoreBED_IqbalGRU_SIR_v1'
 def __init__(self,arrays,horizon,b):
  super().__init__();self.horizon=horizon;self.end=b.end;self.population=b.population;self.gap=b.gap
  self.weights=nn.ParameterDict({key.replace('/','__'):nn.Parameter(torch.from_numpy(np.array(v,dtype=np.float32)),requires_grad=False) for key,v in arrays.items() if key=='log_std' or '/' in key})
 def dense(self,x,name):
  w=self.weights[name.replace('/','__')+'__kernel'];y=x@w
  bias=self.weights.get(name.replace('/','__')+'__bias',None)
  return y if bias is None else y+bias
 def encode(self,x):
  for layer in ['layers_0','layers_2']:x=torch.relu(self.dense(x,'encoder_net/'+layer))
  return self.dense(x,'encoder_net/layers_4')
 def cell(self,h,x,index):
  prefix=f'gru_cells_{index}/';r=(self.dense(x,prefix+'ir')+self.dense(h,prefix+'hr')).sigmoid();z=(self.dense(x,prefix+'iz')+self.dense(h,prefix+'hz')).sigmoid();n=(self.dense(x,prefix+'in')+r*self.dense(h,prefix+'hn')).tanh()
  return (1-z)*n+z*h
 def propose(self,f,stage,noise):
  slots=f[:,1:].reshape(len(f),self.horizon,3);state=[f.new_zeros(len(f),64,dtype=torch.float32) for _ in range(2)]
  for t in range(stage):
   token=torch.cat((slots[:,t,1:2],slots[:,t,:1]),-1).float();x=self.encode(token)
   for k in range(2):state[k]=self.cell(state[k],x,k);x=state[k]
  x=state[-1]
  for layer in ['layers_0','layers_2']:x=torch.relu(self.dense(x,'emitter_net/'+layer))
  raw=self.dense(x,'emitter_net/layers_4')+(self.weights['log_std'].exp()+1e-4)*noise.float()
  previous=slots[:,stage-1,0]*self.end if stage else f.new_zeros(len(f));lo=previous+self.gap;hi=self.end-(self.horizon-stage-1)*self.gap
  return (lo+raw[:,0].double().sigmoid()*(hi-lo))[:,None]
 def sample_design(self,f,stage,rng):
  z=torch.randn((len(f),1),generator=rng,dtype=torch.float64,device=f.device)
  return self.propose(f,stage,z)
 def forward(self,f,stage):return self.propose(f,stage,torch.zeros((len(f),1),device=f.device))
