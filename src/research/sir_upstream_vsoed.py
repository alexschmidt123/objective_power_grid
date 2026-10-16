"""Explicit SIR ODE model interface for pinned native vsOED BA actor-critic.

Only experiment-time percentages/observations reach its actor. Hidden ODE state
is private to the simulator, as the upstream SDE data index was private there.
"""
import math
import torch
from torch import nn

def physical_time(stage,percent,previous,horizon,end,gap):
 lo=previous.double()+gap;hi=end-(horizon-stage-1)*gap
 width=hi-lo
 if torch.any(width < -1e-9):raise ValueError('Infeasible SIR percentage history')
 return lo+percent.double().clamp(0,100)/100*width.clamp_min(0)

class SIRNativeModel:
 multimodel=False;include_nuisance=False;include_goal=False
 def __init__(self,benchmark,horizon):self.b=benchmark;self.horizon=horizon;self._state=None;self._expected_stage=0
 def rvs(self,n_sample):return torch.randn(n_sample,2)
 def log_poi_prior(self,params):return (-.5*params.square()-.5*math.log(2*math.pi)).sum(-1)
 def log_model_prior(self,*args,**kwargs):return 0
 def log_goal_prior(self,*args,**kwargs):return 0
 def reward_fun(self,*args,**kwargs):return 0
 def deterministic(self,stage,params,percent,xps=None):
  if stage==0:
   self._state=self.b.initial((len(params),));self._expected_stage=0
  if stage!=self._expected_stage or self._state is None or len(self._state)!=len(params):raise ValueError('Native SIR observations must be simulated chronologically')
  theta=self.b.inverse(params.double());time=physical_time(stage,percent,self._state[:,3:4],self.horizon,self.b.end,self.b.gap)
  mean,self._state=self.b.forward(theta,self._state,time);self._expected_stage+=1
  return mean
 def model(self,stage,params,percent,xps=None):
  with torch.no_grad():
   mean=self.deterministic(stage,params,percent,xps)
   return (mean+self.b.sigma*torch.randn_like(mean)).float()
 def xp_f(self,stage,xps,percent,ys):
  # Time is planned/observable; latent S,I,R values never enter actor or critic inputs.
  return self._state[:,3:4].float()
 def loglikeli(self,*args,**kwargs):
  raise NotImplementedError('This native BA adapter uses no model likelihood during training; common held-out scoring uses the explicit SIR benchmark, not this stateful sampler')

class UpstreamSIRActor(nn.Module):
 kind='vsoed';outputs_design=True;architecture='pinned_vsOED_GMM_SIR_ODE_percentage_history_adapter'
 def __init__(self,actor,horizon,benchmark):
  super().__init__();self.actor=actor;self.horizon=horizon;self.end=benchmark.end;self.gap=benchmark.gap;self.population=benchmark.population
 def forward(self,features,stage):
  slots=features[:,1:].reshape(len(features),self.horizon,3)
  x=torch.zeros(len(features),self.horizon+2*(self.horizon-1),dtype=torch.float32,device=features.device);x[:,stage]=1
  previous=features.new_zeros(len(features),1)
  for t in range(stage):
   time=slots[:,t,:1]*self.end;lo=previous+self.gap;hi=self.end-(self.horizon-t-1)*self.gap
   denominator=(hi-lo).clamp_min(torch.finfo(features.dtype).eps)
   percent=torch.where(hi>lo,100*(time-lo)/denominator,torch.zeros_like(time))
   x[:,self.horizon+t]=percent[:,0].float();previous=time
  begin=self.horizon+self.horizon-1
  x[:,begin:begin+stage]=(slots[:,:stage,1]*self.population).float()
  percent=self.actor(x)
  return physical_time(stage,percent,previous,self.horizon,self.end,self.gap)
