"""History-conditioned collaborative MoE;no new unrestricted history/action bypass."""
import math,torch
from torch import nn
from policy import V4Actor

class OrderedHistory(nn.Module):
    """Masked design/observation interactions and four ordered moments,not lossless sequence memory."""
    def __init__(self):
        super().__init__()
        self.token=nn.Sequential(nn.Linear(8,16),nn.SiLU())
        self.combine=nn.Sequential(nn.Linear(192,64),nn.SiLU())

    def forward(self,x,query_history):
        B=len(x);stage=x[:,:30].argmax(-1)
        designs=x[:,30:88].reshape(B,29,2)/4
        observations=(x[:,88:117]/5).tanh()
        prev_d=torch.cat([torch.zeros_like(designs[:,:1]),designs[:,:-1]],1)
        prev_y=torch.cat([torch.zeros_like(observations[:,:1]),observations[:,:-1]],1)
        tokens=torch.cat([designs,observations[...,None],designs*observations[...,None],
            (observations-prev_y)[...,None],designs-prev_d],-1)
        assert tokens.shape==(B,29,8)
        idx=torch.arange(29,device=x.device,dtype=x.dtype);valid=idx[None,:]<stage[:,None]
        age=(stage[:,None].to(x.dtype)-idx[None,:]).clamp_min(0)
        position=(idx+.5)/29
        basis=torch.stack([torch.ones_like(age),position.expand_as(age),torch.cos(math.pi*position).expand_as(age),torch.exp(-age/5)],-1)
        basis=basis*valid[...,None]
        encoded=self.token(tokens)
        moments=torch.einsum('btk,btc->bkc',basis,encoded)/stage.clamp_min(1).to(x.dtype)[:,None,None]
        h=self.combine(torch.cat([query_history,moments.flatten(1)],-1))
        return h,moments.flatten(1)

class HistoryExpert(nn.Module):
    def __init__(self,original):
        super().__init__();self.index,self.role=original.index,original.role
        self.semantic=original.semantic;self.projection=original.projection;self.normalization=original.normalization
        self.command_width=[4,6,6,4][self.index]
        self.condition=nn.Sequential(nn.Linear(64+14+1+self.command_width,32),nn.SiLU(),nn.Linear(32,self.command_width+256))
        nn.init.zeros_(self.condition[-1].weight);nn.init.zeros_(self.condition[-1].bias)

    def forward(self,c):
        command=c['commands'].split([4,6,6,4],-1)[self.index]
        output=self.condition(torch.cat([c['history_context'],c['summary'],c['remaining'],command],-1))
        delta,gain=output.split([self.command_width,256],-1)
        # Bounded command adjustment;neutral initialization exactly preserves the original command.
        cmd=command+.25*delta.tanh()*(1-command.abs())
        first,response,innovation=self.semantic(c,cmd)
        specialized=torch.cat([first,response,innovation],-1)
        assert specialized.shape[-1]==256
        # Multiplicative per-channel conditioning only;no additive history feature shortcut.
        specialized=specialized*(1+.5*gain.tanh())
        raw=self.projection(specialized)
        return self.normalization(raw),raw

class V5Actor(V4Actor):
    def __init__(self,parent,arm):
        super().__init__(parent,arm)
        self.history_context=OrderedHistory()
        self.experts=nn.ModuleList([HistoryExpert(e) for e in self.experts])
        self.to(next(parent.context.history.parameters()));self.configure('joint')

    def configure(self,phase):
        super().configure(phase)
        if hasattr(self,'history_context'):
            self.history_context.requires_grad_(phase!='frozen')
            for e in self.experts:e.condition.requires_grad_(phase!='frozen')

    def packets(self,x):
        c=self.query(x)
        h,moments=self.history_context(x,self.query.history(x))
        c['history_context']=h
        packets=[e(c) for e in self.experts]
        z=torch.stack([p[0] for p in packets],1);raw=torch.stack([p[1] for p in packets],1)
        if self.arm=='uniform':
            gates=x.new_full((len(x),4),.25)
            assert not gates.requires_grad and torch.equal(gates,torch.full_like(gates,.25))
        else:
            inp=torch.cat([x,z.flatten(1),c['remaining'],c['summary']],-1)
            assert inp.shape[-1]==644
            gates=self.router(inp).softmax(-1)
        return z,raw,gates
