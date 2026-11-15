"""V2-preserving collaborative feature MoE with four independently executable queries."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from v3_native import n
from hec_policy import HECActor,mean_signal,jacobian,squash,upper
from query_response import paired_discrimination,prospective_opportunity

def observed_prefix(x):
    if x.dtype!=torch.float32 or x.shape[-1]!=117:raise ValueError('Expected float32 history[...,117]')
    flat=x.reshape(-1,117);clock=flat[:,:30]
    if not bool((((clock==0)|(clock==1)).all()) and (clock.sum(-1)==1).all()):raise ValueError('Expected stage one-hot')
    past=torch.arange(29,device=x.device)[None]<clock.argmax(-1)[:,None]
    ds=torch.where(past[...,None],flat[:,30:88].reshape(-1,29,2),0);ys=torch.where(past,flat[:,88:],0)
    return torch.cat([clock,ds.flatten(1),ys],-1).reshape_as(x)

class SharedContext(nn.Module):
    def __init__(self,parent):
        super().__init__()
        for name in ['history','commander','belief','base']:setattr(self,name,copy.deepcopy(getattr(parent,name)))
        self.register_buffer('offset',(parent.stage_logits-parent.reference_logits).detach().clone());self.requires_grad_(False)
    @torch.no_grad()
    def forward(self,x):
        w,mu,sd=self.belief.params(x);a0=self.base(x);s,ctx=HECActor.context(self,x,w,mu,sd,a0)
        stage=x[:,:30].argmax(-1);commands=(self.commander(ctx)+self.offset[stage]).tanh()
        nodes,pw=self.belief.nodes(w,mu,sd);mean=(w[...,None]*mu).sum(1)
        scale=((w[...,None]*(mu.square()+sd.square())).sum(1)-mean.square()).clamp_min(1e-5).sqrt()
        return dict(x=x,w=w,mu=mu,sd=sd,a0=a0,s=s,stage=stage,commands=commands,nodes=nodes,pw=pw,mean=mean,scale=scale)

class FeatureExpert(nn.Module):
    roles=['region','hypothesis_discrimination','parameter_sensitivity','remaining_horizon']
    widths=[13,28,20,26]
    def __init__(self,parent,index):
        super().__init__();self.index=index;self.role=self.roles[index]
        names=[['region','anchor1'],['discriminate','adapter2'],['sensitivity','anchor3'],['sequential','adapter4','opportunity']][index]
        for name in names:setattr(self,name,copy.deepcopy(getattr(parent,name)))
        self.requires_grad_(False);device=parent.fusion[0].weight.device
        self.innovation=nn.Sequential(nn.Linear(self.widths[index],64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,128)).to(device)
        nn.init.zeros_(self.innovation[-1].weight);nn.init.zeros_(self.innovation[-1].bias)
        self.register_buffer('first_projection',parent.fusion[0].weight[:,128+64*index:128+64*(index+1)].detach().clone())
        self.register_buffer('response_projection',parent.response_fusion.weight[:,64*index:64*(index+1)].detach().clone())
    def forward(self,c):
        x=c['x'];w,mu,sd=c['w'],c['mu'],c['sd'];nodes,pw=c['nodes'],c['pw'];mean,scale,a0=c['mean'],c['scale'],c['a0']
        cmd=c['commands'].split([4,6,6,4],-1)[self.index]
        if self.index==0:
            center=(mu-mean[:,None])/scale[:,None]
            att=(w.clamp_min(1e-38).log()+(center*cmd[:,None]).sum(-1)/2).softmax(-1)
            tokens=torch.cat([mu/3,sd.log(),w[...,None]],-1)
            legacy=(att[...,None]*self.region(tokens)).sum(1);first=(att[...,None]*self.anchor1(tokens)).sum(1);response=legacy-first
            private=torch.cat([tokens,cmd[:,None].expand(-1,mu.shape[1],-1)],-1)
            innovation=(att[...,None]*self.innovation(private)).sum(1)
        elif self.index==1:
            direction=cmd[:,:2]+x.new_tensor([1.,0.]);direction=direction/(direction.square().sum(-1,keepdim=True)+1e-6).sqrt()
            probes=(a0[:,None]+x.new_tensor([0.,1.,-1.])[None,:,None]*direction[:,None]).clamp(-4,4)
            pred=mean_signal(nodes,probes);tilt=(nodes-mean[:,None])/scale[:,None]
            att=(pw.clamp_min(1e-38).log()+(tilt*cmd[:,None,2:]).sum(-1)/2).softmax(-1)
            legacy=(att[...,None]*self.discriminate(torch.cat([pred.transpose(1,2)/5,att[...,None]],-1))).sum(1)
            semantic,_,_,hp,hm=paired_discrimination(nodes,pw,pred,mean,scale,cmd)
            first=legacy;response=torch.cat([semantic,self.adapter2(torch.cat([semantic,hp/3,hm/3,a0/4,cmd,mean/3,scale.log()],-1))],-1)
            private=torch.cat([semantic,cmd,a0/4,mean/3,scale.log()],-1);innovation=self.innovation(private)
        elif self.index==2:
            design=(a0+cmd[:,4:]).clamp(-4,4);J=jacobian(nodes,design);fisher=4*torch.einsum('bn,bni,bnj->bij',pw,J,J)
            direction=cmd[:,:4]+x.new_tensor([1.,-1.,1.,-1.]);direction=direction/(direction.square().sum(-1,keepdim=True)+1e-6).sqrt()
            inp=torch.cat([squash(upper(fisher)),direction],-1);legacy=self.sensitivity(inp);first=self.anchor3(inp);response=legacy-first
            private=torch.cat([inp,cmd],-1);innovation=self.innovation(private)
        else:
            direction=cmd[:,:2]+x.new_tensor([0.,1.]);direction=direction/(direction.square().sum(-1,keepdim=True)+1e-6).sqrt()
            radius=1+.5*cmd[:,2:3];probes=(a0[:,None]+x.new_tensor([0.,1.,-1.])[None,:,None]*radius[:,None]*direction[:,None]).clamp(-4,4)
            q=self.opportunity(torch.cat([x[:,None].expand(-1,3,-1),probes],-1)).squeeze(-1);qdiff=(q-q[:,:1])/10
            pred=mean_signal(nodes,probes);pm=(pw[:,None]*pred).sum(-1);pv=(pw[:,None]*(pred-pm[:,:,None]).square()).sum(-1)
            remaining=(30-c['stage']).to(x.dtype)[:,None]/30
            legacy=self.sequential(torch.cat([qdiff,pm/5,torch.log1p(pv),remaining,cmd],-1))
            semantic=prospective_opportunity(nodes,pw,probes,cmd,c['stage']);first=legacy
            response=torch.cat([semantic,self.adapter4(torch.cat([semantic,qdiff,remaining,cmd],-1))],-1)
            private=torch.cat([semantic,probes.flatten(1)/4,qdiff,remaining,cmd],-1);innovation=self.innovation(private)
        projected=F.linear(first,self.first_projection)+F.linear(response,self.response_projection)+innovation
        return dict(first=first,response=response,legacy=legacy,innovation=innovation,projected=projected,private=private)

class SharedDecoder(nn.Module):
    def __init__(self,parent):
        super().__init__();self.register_buffer('weight',parent.fusion[0].weight.detach().clone());self.register_buffer('bias',parent.fusion[0].bias.detach().clone())
        self.register_buffer('response_weight',parent.response_fusion.weight.detach().clone())
        self.action_layer=copy.deepcopy(parent.fusion[2]);self.requires_grad_(False)
    def forward(self,c,g,first,response,innovation):
        # Retain original GEMM order for exact V2 initialization, rather than reassociate sums.
        pre=F.linear(torch.cat([c['s'],(g[...,None]*first).flatten(1)],-1),self.weight,self.bias)+F.linear((g[...,None]*response).flatten(1),self.response_weight)
        pre=pre+(g[...,None]*innovation).sum(1)
        delta=self.action_layer(F.relu(pre)).tanh();a0=c['a0']
        return a0+4*delta-a0*delta.abs()

class FeatureMoE(nn.Module):
    def __init__(self,parent,seed=202611145001):
        super().__init__();self.context=SharedContext(parent);self.decoder=SharedDecoder(parent);self.chunk=4096
        device=parent.fusion[0].weight.device
        self.prior_coordinator=copy.deepcopy(parent.coordinator).requires_grad_(False)
        self.prior_response_gate=copy.deepcopy(parent.response_gate).requires_grad_(False)
        with torch.random.fork_rng(devices=[device.index if device.index is not None else 0] if device.type=='cuda' else []):
            torch.manual_seed(seed);self.experts=nn.ModuleList([FeatureExpert(parent,k) for k in range(4)])
            self.router=nn.Sequential(nn.Linear(147,64),nn.SiLU(),nn.Linear(64,4)).to(device)
            nn.init.zeros_(self.router[-1].weight);nn.init.zeros_(self.router[-1].bias)
        self.configure('joint')
    def configure(self,phase):
        if phase not in ['experts','router','joint','frozen']:raise ValueError(phase)
        self.requires_grad_(False)
        for e in self.experts:e.innovation.requires_grad_(phase in ['experts','joint'])
        self.router.requires_grad_(phase in ['router','joint'])
        for p in self.parameters():p.grad=None
    def standalone(self,k):
        if k not in range(4):raise ValueError(k)
        return SingleFeaturePolicy(self.context,self.decoder,self.experts[k],k)
    def decode_one(self,c,k,packet):
        b=len(c['x']);first=c['x'].new_zeros(b,4,64);response=torch.zeros_like(first);innovation=c['x'].new_zeros(b,4,128)
        first[:,k]=packet['first'];response[:,k]=packet['response'];innovation[:,k]=packet['innovation']
        gates=c['x'].new_zeros(b,4);gates[:,k]=1
        return self.decoder(c,gates,first,response,innovation)
    def compute(self,x,mode='learned',expert=None,collect=False):
        c=self.context(x)
        if mode=='expert':
            if expert not in range(4):raise ValueError('Invalid expert')
            packet=self.experts[expert](c);action=self.decode_one(c,expert,packet)
            return (action,dict(private=packet['private'],projected=packet['projected'])) if collect else action
        if mode not in ['learned','uniform']:raise ValueError(mode)
        packets=[e(c) for e in self.experts]
        first=torch.stack([p['first'] for p in packets],1);response=torch.stack([p['response'] for p in packets],1)
        legacy=torch.stack([p['legacy'] for p in packets],1);innovation=torch.stack([p['innovation'] for p in packets],1)
        projected=torch.stack([p['projected'] for p in packets],1)
        proposals=torch.stack([self.decode_one(c,k,p) for k,p in enumerate(packets)],1)
        norms=torch.log1p(projected.square().mean(-1));remaining=(30-c['stage']).to(x.dtype)[:,None]/30
        offsets=(proposals-c['a0'][:,None])/4
        distances=torch.stack([(proposals[:,i]-proposals[:,j]).norm(dim=-1)/8 for i in range(4) for j in range(i+1,4)],-1)
        router_input=torch.cat([c['s'],remaining,norms,offsets.flatten(1),distances],-1).detach()
        correction=self.router(router_input)
        raw=self.prior_coordinator(torch.cat([c['s'],legacy.flatten(1)],-1))+self.prior_response_gate(response.flatten(1))+correction
        g=(10*(raw/10).tanh()).softmax(-1)
        if mode=='uniform':g=torch.full_like(g,.25)
        action=self.decoder(c,g,first,response,innovation)
        if not collect:return action
        return action,dict(history=c['s'],commands=c['commands'],gates=g,router_input=router_input,correction=correction,
            first=first,responses=response,legacy=legacy,innovation=innovation,projected=projected,proposals=proposals,
            private=[p['private'] for p in packets])
    def forward(self,history,mode='learned',expert=None,collect=False):
        shape=history.shape[:-1];flat=observed_prefix(history).reshape(-1,117)
        if collect:
            if len(flat)>self.chunk:raise ValueError('Use small explicit shape traces for collect')
            action,info=self.compute(flat,mode,expert,True);return action.reshape(*shape,2),info
        outputs=[]
        for chunk in flat.split(self.chunk):
            def fn(xx):return self.compute(xx,mode,expert)
            outputs.append(checkpoint(fn,chunk,use_reentrant=False) if torch.is_grad_enabled() else fn(chunk))
        return torch.cat(outputs).reshape(*shape,2)

class SingleFeaturePolicy(nn.Module):
    """Export contains common providers/decoder and exactly one current expert."""
    def __init__(self,context,decoder,expert,index):
        super().__init__();self.context=copy.deepcopy(context);self.decoder=copy.deepcopy(decoder);self.expert=copy.deepcopy(expert);self.index=index
    def forward(self,history):
        shape=history.shape[:-1];flat=observed_prefix(history).reshape(-1,117);outputs=[]
        for chunk in flat.split(4096):
            def fn(xx):
                c=self.context(xx);packet=self.expert(c)
                return FeatureMoE.decode_one(self,c,self.index,packet)
            outputs.append(checkpoint(fn,chunk,use_reentrant=False) if torch.is_grad_enabled() else fn(chunk))
        return torch.cat(outputs).reshape(*shape,2)
