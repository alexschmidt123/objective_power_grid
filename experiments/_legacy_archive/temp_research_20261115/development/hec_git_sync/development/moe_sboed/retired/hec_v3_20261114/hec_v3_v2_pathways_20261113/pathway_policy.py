"""V2's FOUR actual query pathways, each with a standalone design head."""
import copy
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from common import n
from hec_policy import HECActor,mean_signal,jacobian,squash,upper
from query_response import paired_discrimination,prospective_opportunity

def prefix(x):
    if x.dtype!=torch.float32 or x.shape[-1]!=117:raise ValueError('float32 history[...,117] required')
    f=x.reshape(-1,117);clock=f[:,:30]
    if not bool(((clock==0)|(clock==1)).all()) or not bool((clock.sum(-1)==1).all()):raise ValueError('Invalid stage onehot')
    past=torch.arange(29,device=x.device)[None]<clock.argmax(-1)[:,None]
    ds=torch.where(past[...,None],f[:,30:88].reshape(-1,29,2),0)
    ys=torch.where(past,f[:,88:],0)
    return torch.cat([clock,ds.flatten(1),ys],-1).reshape_as(x)

class Context(nn.Module):
    """Original V2 history,commander,belief,SG providers; frozen in this pilot."""
    def __init__(self,parent):
        super().__init__()
        for key in ['history','commander','belief','base']:setattr(self,key,copy.deepcopy(getattr(parent,key)))
        self.register_buffer('command_offset',(parent.stage_logits-parent.reference_logits).detach().clone())
        self.requires_grad_(False)
    @torch.no_grad()
    def forward(self,x):
        w,mu,sd=self.belief.params(x);a0=self.base(x)
        s,ctx=HECActor.context(self,x,w,mu,sd,a0)
        stage=x[:,:30].argmax(-1);commands=(self.commander(ctx)+self.command_offset[stage]).tanh()
        nodes,pw=self.belief.nodes(w,mu,sd);mean=(w[...,None]*mu).sum(1)
        scale=((w[...,None]*(mu.square()+sd.square())).sum(1)-mean.square()).clamp_min(1e-5).sqrt()
        return dict(x=x,w=w,mu=mu,sd=sd,a0=a0,s=s,stage=stage,commands=commands,nodes=nodes,pw=pw,mean=mean,scale=scale)

class Pathway(nn.Module):
    def __init__(self,parent,index):
        super().__init__();self.index=index
        names=[['region','anchor1'],['discriminate','adapter2'],['sensitivity','anchor3'],['sequential','adapter4','opportunity']][index]
        for key in names:setattr(self,key,copy.deepcopy(getattr(parent,key)))
        device=parent.fusion[0].weight.device
        self.head=nn.Sequential(nn.Linear(256,128),nn.ReLU(),copy.deepcopy(parent.fusion[2])).to(device)
        with torch.no_grad():
            self.head[0].weight.copy_(torch.cat([parent.fusion[0].weight[:,:128],parent.fusion[0].weight[:,128+64*index:128+64*(index+1)],parent.response_fusion.weight[:,64*index:64*(index+1)]],-1))
            self.head[0].bias.copy_(parent.fusion[0].bias)

    def packet(self,c):
        x=c['x'];w,mu,sd=c['w'],c['mu'],c['sd'];nodes,pw=c['nodes'],c['pw']
        mean,scale,a0=c['mean'],c['scale'],c['a0'];cmd=c['commands'].split([4,6,6,4],-1)[self.index]
        if self.index==0:
            center=(mu-mean[:,None])/scale[:,None]
            attention=(w.clamp_min(1e-38).log()+(center*cmd[:,None]).sum(-1)/2).softmax(-1)
            tokens=torch.cat([mu/3,sd.log(),w[...,None]],-1)
            legacy=(attention[...,None]*self.region(tokens)).sum(1)
            first=(attention[...,None]*self.anchor1(tokens)).sum(1);response=legacy-first
        elif self.index==1:
            direction=cmd[:,:2]+x.new_tensor([1.,0.]);direction=direction/(direction.square().sum(-1,keepdim=True)+1e-6).sqrt()
            probes=(a0[:,None]+x.new_tensor([0.,1.,-1.])[None,:,None]*direction[:,None]).clamp(-4,4)
            pred=mean_signal(nodes,probes);tilt=(nodes-mean[:,None])/scale[:,None]
            attention=(pw.clamp_min(1e-38).log()+(tilt*cmd[:,None,2:]).sum(-1)/2).softmax(-1)
            tokens=torch.cat([pred.transpose(1,2)/5,attention[...,None]],-1)
            legacy=(attention[...,None]*self.discriminate(tokens)).sum(1)
            semantic,plus,minus,hp,hm=paired_discrimination(nodes,pw,pred,mean,scale,cmd)
            inp=torch.cat([semantic,hp/3,hm/3,a0/4,cmd,mean/3,scale.log()],-1)
            first=legacy;response=torch.cat([semantic,self.adapter2(inp)],-1)
        elif self.index==2:
            design=(a0+cmd[:,4:]).clamp(-4,4);J=jacobian(nodes,design)
            F=4*torch.einsum('bn,bni,bnj->bij',pw,J,J)
            v=cmd[:,:4]+x.new_tensor([1.,-1.,1.,-1.]);v=v/(v.square().sum(-1,keepdim=True)+1e-6).sqrt()
            inp=torch.cat([squash(upper(F)),v],-1);legacy=self.sensitivity(inp);first=self.anchor3(inp);response=legacy-first
        else:
            direction=cmd[:,:2]+x.new_tensor([0.,1.]);direction=direction/(direction.square().sum(-1,keepdim=True)+1e-6).sqrt()
            radius=1+.5*cmd[:,2:3]
            probes=(a0[:,None]+x.new_tensor([0.,1.,-1.])[None,:,None]*radius[:,None]*direction[:,None]).clamp(-4,4)
            q=self.opportunity(torch.cat([x[:,None].expand(-1,3,-1),probes],-1)).squeeze(-1)
            qdiff=(q-q[:,:1])/10;pred=mean_signal(nodes,probes)
            pm=(pw[:,None]*pred).sum(-1);pv=(pw[:,None]*(pred-pm[:,:,None]).square()).sum(-1)
            remaining=(30-c['stage']).to(x.dtype)[:,None]/30
            legacy=self.sequential(torch.cat([qdiff,pm/5,torch.log1p(pv),remaining,cmd],-1))
            semantic=prospective_opportunity(nodes,pw,probes,cmd,c['stage'])
            first=legacy;response=torch.cat([semantic,self.adapter4(torch.cat([semantic,qdiff,remaining,cmd],-1))],-1)
        return first,response,legacy

    def design(self,c,first,response):
        delta=self.head(torch.cat([c['s'],first,response],-1)).tanh();a0=c['a0']
        return a0+4*delta-a0*delta.abs()

class Standalone(nn.Module):
    def __init__(self,context,pathway,clone=True):
        super().__init__();self.context=copy.deepcopy(context) if clone else context;self.expert=copy.deepcopy(pathway) if clone else pathway;self.chunk=4096
    def forward(self,history):
        shape=history.shape[:-1];flat=prefix(history).reshape(-1,117);outputs=[]
        for x in flat.split(self.chunk):
            def compute(xx):
                c=self.context(xx);first,response,_=self.expert.packet(c)
                return self.expert.design(c,first,response)
            if torch.is_grad_enabled():value=checkpoint(compute,x,use_reentrant=False)
            else:value=compute(x)
            outputs.append(value)
        return torch.cat(outputs).reshape(*shape,2)

class PathwayMoE(nn.Module):
    def __init__(self,parent):
        super().__init__();self.context=Context(parent)
        self.experts=nn.ModuleList([Pathway(parent,k) for k in range(4)])
        self.coordinator=copy.deepcopy(parent.coordinator);self.response_gate=copy.deepcopy(parent.response_gate)
        self.proposal_gate=nn.Linear(8,4,bias=False).to(parent.fusion[0].weight.device)
        nn.init.zeros_(self.proposal_gate.weight);self.chunk=4096;self.freeze_experts()
    def freeze_experts(self):
        self.context.requires_grad_(False);self.experts.requires_grad_(False)
        for p in self.experts.parameters():p.grad=None
        self.coordinator.requires_grad_(True);self.response_gate.requires_grad_(True);self.proposal_gate.requires_grad_(True)
    def standalone(self,k):return Standalone(self.context,self.experts[k])
    def components(self,history):
        shape=history.shape[:-1];flat=prefix(history).reshape(-1,117);ps=[];gs=[];ls=[];rs=[]
        for x in flat.split(self.chunk):
            c=self.context(x);packets=[expert.packet(c) for expert in self.experts]
            proposals=torch.stack([expert.design(c,*packet[:2]) for expert,packet in zip(self.experts,packets)],1)
            legacy=torch.stack([v[2] for v in packets],1);responses=torch.stack([v[1] for v in packets],1)
            raw=self.coordinator(torch.cat([c['s'],legacy.flatten(1)],-1))+self.response_gate(responses.flatten(1))+self.proposal_gate(proposals.detach().flatten(1))
            gates=(10*torch.tanh(raw/10)).softmax(-1)
            ps.append(proposals);gs.append(gates);ls.append(legacy);rs.append(responses)
        return torch.cat(ps).reshape(*shape,4,2),torch.cat(gs).reshape(*shape,4),torch.cat(ls).reshape(*shape,4,64),torch.cat(rs).reshape(*shape,4,64)
    def forward(self,history,mode='learned',expert=None,uniforms=None,collect=False):
        if mode=='expert':
            if expert not in range(4):raise ValueError('Invalid expert')
            # Only the requested expert is executed; no other expert or router.
            return Standalone(self.context,self.experts[expert],clone=False)(history)
        if mode not in ['learned','uniform','argmax']:raise ValueError(mode)
        proposals,g,_,_=self.components(history)
        if mode=='uniform':g=torch.full_like(g,.25)
        if mode=='argmax':chosen=g.argmax(-1)
        else:
            if uniforms is None:uniforms=torch.rand(history.shape[:-1],device=history.device)
            if uniforms.shape!=history.shape[:-1] or not bool(((uniforms>=0)&(uniforms<1)).all()):raise ValueError('Invalid selection uniforms')
            chosen=(uniforms[...,None]>=g.cumsum(-1)).sum(-1).clamp_max(3)
        action=proposals.gather(-2,chosen[...,None,None].expand(*chosen.shape,1,2)).squeeze(-2)
        if collect:return action,dict(proposals=proposals,weights=g,selected=chosen)
        return action
    def native_actor_loss(self,critic,history):
        clean=prefix(history);proposals,g,_,_=self.components(clean)
        values=torch.stack([critic(torch.cat([clean,proposals[...,k,:]],-1)).squeeze(-1) for k in range(4)],-1)
        return -(g*values).sum(-1).mean()
