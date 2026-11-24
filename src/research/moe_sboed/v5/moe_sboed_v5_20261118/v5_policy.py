"""V5 preserves weighted expert slots;all other V4 policy modules are inherited unchanged."""
import torch
from torch import nn
from policy import V4Actor

class V5Actor(V4Actor):
    def __init__(self,parent,arm):
        super().__init__(parent,arm)
        # Same V4 module initialization first;only this decision head is replaced.
        self.action_head=nn.Sequential(nn.Linear(512,128),nn.SiLU(),nn.Linear(128,32),nn.SiLU(),nn.Linear(32,2))
        self.to(next(parent.context.history.parameters()))
        self.configure('joint')

    def decode(self,x,z,gates):
        assert z.shape==(len(x),4,128) and gates.shape==(len(x),4)
        weighted=gates.unsqueeze(-1)*z
        routed=weighted.flatten(1)
        assert routed.shape==(len(x),512)
        a_moe=4*self.action_head(routed).tanh()
        a_shared=self.shared_small(x).tanh()
        correction=.2*a_shared
        unclipped=a_moe+correction
        return unclipped.clamp(-4,4),dict(routed=routed,weighted=weighted,a_moe=a_moe,a_shared=a_shared,correction=correction,unclipped=unclipped)
