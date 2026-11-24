"""Expert-dominant soft feature MoE;bounded isolated weak history correction."""
import copy
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from frozen_feature_policy import observed_prefix
from specialist_queries import SharedInference, SpecialistFeatures

ROLES = ['Posterior geometry','Hypothesis discrimination','Parameter sensitivity','Sequential opportunity']


class QueryContext(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.inference = SharedInference(parent)
        self.history = copy.deepcopy(parent.context.history)
        self.commander = copy.deepcopy(parent.context.commander)
        self.offset = nn.Parameter(parent.context.offset.detach().clone())

    def forward(self, x):
        c = self.inference(x)
        h = self.history(x)
        # a0 here is ONLY a deterministic posterior-centroid query reference.
        # The inherited SG/DAD-like policy is neither copied nor evaluated.
        command_input = torch.cat([h,c['a0']/4,c['mean'],torch.log1p(c['scale'].square()),c['summary'][:,8:]],-1)
        assert command_input.shape[-1] == 144
        c['commands'] = (self.commander(command_input)+self.offset[c['stage']]).tanh()
        return c


class V4Expert(nn.Module):
    def __init__(self,parent,index):
        super().__init__()
        self.index,self.role = index,ROLES[index]
        self.semantic = SpecialistFeatures(parent.experts[index],index)
        self.projection = nn.Sequential(nn.Linear(256,128),nn.SiLU())
        self.normalization = nn.LayerNorm(128,elementwise_affine=False,eps=1e-5)

    def forward(self,c):
        cmd = c['commands'].split([4,6,6,4],-1)[self.index]
        first,response,innovation = self.semantic(c,cmd)
        specialized = torch.cat([first,response,innovation],-1)
        assert specialized.shape[-1] == 256
        raw = self.projection(specialized)
        z = self.normalization(raw)
        return z,raw


class V4Actor(nn.Module):
    def __init__(self,parent,arm):
        super().__init__()
        assert arm in ['learned','uniform']
        self.arm,self.chunk = arm,4096
        self.query = QueryContext(parent)
        self.experts = nn.ModuleList([V4Expert(parent,k) for k in range(4)])
        # No proposed actions and no weak-shared output appear in these inputs.
        self.router = nn.Sequential(nn.Linear(644,64),nn.SiLU(),nn.Linear(64,4))
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self.action_head = nn.Sequential(nn.Linear(128,32),nn.SiLU(),nn.Linear(32,2))
        self.shared_small = nn.Sequential(nn.Linear(117,16),nn.SiLU(),nn.Linear(16,2))
        self.to(next(parent.context.history.parameters()))
        self.configure('joint')

    def configure(self,phase):
        assert phase in ['warmup','joint','frozen']
        self.requires_grad_(False)
        if phase != 'frozen':
            self.query.commander.requires_grad_(True)
            self.query.offset.requires_grad_(True)
            self.query.history.requires_grad_(phase=='joint')
            for e in self.experts:
                e.semantic.configure()
                e.projection.requires_grad_(True)
            self.action_head.requires_grad_(True)
            self.shared_small.requires_grad_(True)
            self.router.requires_grad_(self.arm=='learned')
        self.query.inference.eval()
        for e in self.experts:
            for name in ['anchor1','anchor3','opportunity']:
                if hasattr(e.semantic,name):getattr(e.semantic,name).eval()
        for p in self.parameters():p.grad=None

    def packets(self,x):
        c = self.query(x)
        packets = [e(c) for e in self.experts]
        z = torch.stack([p[0] for p in packets],1)
        raw = torch.stack([p[1] for p in packets],1)
        if self.arm == 'uniform':
            gates = x.new_full((len(x),4),.25)
            assert not gates.requires_grad and torch.equal(gates,torch.full_like(gates,.25))
        else:
            inp = torch.cat([x,z.flatten(1),c['remaining'],c['summary']],-1)
            assert inp.shape[-1] == 644
            gates = self.router(inp).softmax(-1)
        return z,raw,gates

    def decode(self,x,z,gates):
        routed = (gates[...,None]*z).sum(1)
        a_moe = 4*self.action_head(routed).tanh()
        a_shared = self.shared_small(x).tanh()
        correction = .2*a_shared
        unclipped = a_moe+correction
        action = unclipped.clamp(-4,4)
        return action,dict(routed=routed,a_moe=a_moe,a_shared=a_shared,
                           correction=correction,unclipped=unclipped)

    def compute(self,x,collect=False):
        z,raw,gates = self.packets(x)
        action,info = self.decode(x,z,gates)
        if collect:return action,dict(info,z=z,raw=raw,gates=gates)
        return action

    def forward(self,x,collect=False):
        shape = x.shape[:-1]
        flat = observed_prefix(x).reshape(-1,117)
        if collect:
            assert len(flat)<=self.chunk
            action,info = self.compute(flat,True)
            return action.reshape(*shape,2),info
        values=[]
        for xx in flat.split(self.chunk):
            def fn(v):return self.compute(v)
            values.append(checkpoint(fn,xx,use_reentrant=False) if torch.is_grad_enabled() else fn(xx))
        return torch.cat(values).reshape(*shape,2)
