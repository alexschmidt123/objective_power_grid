"""Offline, truth-excluded predictive uncertainty cohorts; unchanged V1 inference."""
import torch
from src.research.engine import replay_likelihood
from src.research.context_expert_gradients import protect_gradient,apply_context_gradients
THRESHOLD=.75
@torch.no_grad()
def belief(b,particles,actions,observations):
 # Prior-normalized posterior predictive dispersion at a frozen 3x3 design set.
 ll,_,_=replay_likelihood(b,particles,actions,observations);w=ll.softmax(-1);ess=w.square().sum(-1).reciprocal();pmax=w.max(-1).values
 B,L,_=particles.shape;vals=[];splits=[slice(None),slice(0,L//2),slice(L//2,None)]
 for sl in splits:
  pp=particles[:,sl];ww=ll[:,sl].softmax(-1);post=ww.new_zeros(B);prior=post.clone()
  for x in [-2.,0.,2.]:
   for y in [-2.,0.,2.]:
    a=pp.new_tensor([x,y]).expand(B,pp.shape[1],2);mu,_=b.forward(pp,b.initial((B,pp.shape[1])),a);mu=mu.squeeze(-1)
    post+=(ww*mu.square()).sum(-1)-(ww*mu).sum(-1).square();prior+=mu.var(-1,unbiased=False)
  vals.append(post/prior.clamp_min(1e-12))
 value=vals[0];agreement=((vals[1]>THRESHOLD)==(vals[2]>THRESHOLD));reliable=(ess>=64)&(pmax<=.1)&agreement&((vals[1]-vals[2]).abs()<=.2)&((value-THRESHOLD).abs()>=.1)
 return dict(value=value,ess=ess,pmax=pmax,reliable=reliable,group=torch.where(reliable,(value>THRESHOLD).long(),torch.full_like(ess,2,dtype=torch.long)))
def context_from(feature,snapshot):
 group=feature['group'];counts=torch.bincount(group,minlength=3);p=snapshot['scores'];ids=snapshot['indices'];exposure=torch.nn.functional.one_hot(ids,7).sum(1)*p
 affinity=torch.stack([exposure[group==c].mean(0) if counts[c] else p.new_zeros(7) for c in range(2)])
 return dict(group=group,counts=counts,affinity=affinity,valid=bool(counts[:2].min()>=32),stage=snapshot['stage'])
def collect(model,info,balance,ctx):
 ps=[p for e in model.routed for p in e.parameters()];out=[]
 for c in range(3):
  mask=(ctx['group']==c).to(info.dtype);loss=(-info*mask).mean()+mask.mean()*balance
  gg=torch.autograd.grad(loss,ps,retain_graph=True,allow_unused=True);out.append([torch.zeros_like(p) if g is None else g.detach() for p,g in zip(ps,gg)])
 return out
def apply(model,historical,oldctx,newgrads,newctx):
 # Always compute exact historical adaptation first; use it as guarded fallback.
 originals=[p.grad.clone() if p.grad is not None else torch.zeros_like(p) for e in model.routed for p in e.parameters()]
 base_record=apply_context_gradients(model,historical,oldctx);records=[];offset=0
 if newgrads is None:return dict(fallback=True,historical=base_record,adjusted=0)
 for e,expert in enumerate(model.routed):
  ps=list(expert.parameters());n=len(ps);gs=[torch.cat([g.reshape(-1) for g in row[offset:offset+n]]) for row in newgrads];G=torch.cat([g.reshape(-1) for g in originals[offset:offset+n]]);base=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in ps]);assert torch.allclose(sum(gs),G,rtol=1e-7,atol=1e-9)
  H,rec=protect_gradient(gs[0],gs[1],newctx['affinity'][:,e]);H=H+gs[2]
  good=rec['adjusted'] and bool(torch.isfinite(H).all()) and bool(H.norm()>0) and bool(torch.dot(H,G)>0) and bool(torch.dot(H,base)>0)
  if good:
   H=H*(base.norm()/H.norm());pos=0
   for p in ps:
    v=H[pos:pos+p.numel()].view_as(p);pos+=p.numel()
    if p.grad is not None:p.grad.copy_(v)
    elif bool((v!=0).any()):raise AssertionError('Inactive gradient activated')
   assert torch.dot(H,G)>0 and torch.allclose(H.norm(),base.norm(),rtol=1e-10,atol=1e-12)
  records.append(dict(expert=e,adjusted=good,conflict=rec['conflict'],reconstruction_error=float((sum(gs)-G).abs().max())));offset+=n
 return dict(fallback=False,adjusted=sum(x['adjusted'] for x in records),experts=records,cohort_counts=newctx['counts'].tolist(),historical=base_record)
