"""Reuse previous V5 networks;only replace conditioner evidence with role-specific history–belief views."""
import torch
from history_policy import V5Actor as PreviousActor,HistoryExpert as PreviousExpert
from role_evidence import evidence

class Expert(PreviousExpert):
 def forward(self,c):
  cmd=c['commands'].split([4,6,6,4],-1)[self.index]
  features=c['role_belief'][:,self.index]
  output=self.condition(torch.cat([c['history_context'],features,c['remaining'],cmd],-1))
  delta,gain=output.split([self.command_width,256],-1)
  adjusted=cmd+.25*delta.tanh()*(1-cmd.abs())
  first,response,innovation=self.semantic(c,adjusted)
  specialized=torch.cat([first,response,innovation],-1)*(1+.5*gain.tanh())
  raw=self.projection(specialized)
  return self.normalization(raw),raw

class Actor(PreviousActor):
 def __init__(self,parent,arm):
  super().__init__(parent,arm)
  for e in self.experts:e.__class__=Expert
 def packets(self,x):
  c=self.query(x);c['history_context'],_=self.history_context(x,self.query.history(x));c['role_belief']=evidence(c,self.query.inference.belief)
  packets=[e(c) for e in self.experts];z=torch.stack([v[0] for v in packets],1);raw=torch.stack([v[1] for v in packets],1)
  gates=x.new_full((len(x),4),.25) if self.arm=='uniform' else self.router(torch.cat([x,z.flatten(1),c['remaining'],c['summary']],-1)).softmax(-1)
  return z,raw,gates
