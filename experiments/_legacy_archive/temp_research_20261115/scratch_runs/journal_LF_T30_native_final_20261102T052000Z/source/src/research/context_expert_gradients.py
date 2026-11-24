
"""Context-protected routed-expert gradients for the UNCHANGED DeepSeekMoE-v1."""
from contextlib import contextmanager
import math,time
import torch

@contextmanager
def capture_context(model,stage):
    snapshot={}
    def observe(module,args,output):
        if int(args[1])==stage:
            snapshot.update(z=module.last_features.detach(),scores=module.last_dispatch['scores'].detach(),indices=module.last_dispatch['indices'].detach(),stage=stage)
    hook=model.register_forward_hook(observe)
    try:yield snapshot
    finally:hook.remove()

def partition_context(snapshot):
    z=snapshot['z'].detach()
    variance=z.var(0,unbiased=False);coordinate=int(variance.argmax())
    cut=z[:,coordinate].median()
    group=(z[:,coordinate]>cut).long()
    counts=torch.bincount(group,minlength=2)
    valid=bool(variance[coordinate]>1e-16 and counts.min()>0)
    p=snapshot['scores'];ids=snapshot['indices']
    exposure=torch.nn.functional.one_hot(ids,p.shape[-1]).sum(1)*p
    affinity=torch.stack([exposure[group==c].mean(0) if counts[c]>0 else p.new_zeros(p.shape[-1]) for c in range(2)])
    return dict(valid=valid,group=group,counts=counts,coordinate=coordinate,threshold=float(cut),affinity=affinity.detach(),stage=snapshot['stage'])

def collect_context_gradients(model,terminal_information,balance,context):
    """Two cohorts sum EXACTLY to the original loss gradient, including balance."""
    parameters=[p for e in model.routed for p in e.parameters()]
    result=[]
    for c in range(2):
        mask=(context['group']==c).to(terminal_information.dtype).detach()
        loss=(-terminal_information*mask).mean()+mask.mean()*balance
        gradients=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
        result.append([torch.zeros_like(p) if g is None else g.detach() for p,g in zip(parameters,gradients)])
    return result

def protect_gradient(g0,g1,affinity):
    """Directional optimizer adaptation, not a claimed unbiased gradient of a new loss."""
    G=g0+g1;norm=G.norm()
    record=dict(conflict=False,adjusted=False,fallback=False,norm=float(norm),cosine=None,anchor=None)
    a,b=g0.norm(),g1.norm()
    if a==0 or b==0 or norm==0:return G,record
    dot=torch.dot(g0,g1);record['cosine']=float(dot/(a*b))
    if dot>=0 or abs(float(affinity[0]-affinity[1]))<=1e-12:return G,record
    anchor=int(affinity.argmax());u,v=(g0,g1) if anchor==0 else (g1,g0)
    projected=v-(torch.dot(v,u)/u.square().sum())*u
    H=u+projected
    record.update(conflict=True,anchor=anchor,protected_other_dot=float(torch.dot(projected,u)))
    # Protect descent alignment of the ORIGINAL summed cohort gradient.
    if not torch.isfinite(H).all() or H.norm()==0 or torch.dot(H,G)<=0:
        record['fallback']=True;return G,record
    H=H*(norm/H.norm())
    record.update(adjusted=True,original_alignment=float(torch.dot(H,G)),norm_after=float(H.norm()),direction_change_L2=float((H-G).norm()))
    return H,record

def apply_context_gradients(model,gradients,context):
    offset=0;records=[]
    for e,expert in enumerate(model.routed):
        parameters=list(expert.parameters());count=len(parameters)
        first=torch.cat([v.reshape(-1) for v in gradients[0][offset:offset+count]])
        second=torch.cat([v.reshape(-1) for v in gradients[1][offset:offset+count]])
        primary=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1) for p in parameters])
        G=first+second
        error=float((G-primary).abs().max())
        if error>1e-8*max(1.,float(primary.norm())):raise AssertionError('Cohort gradients do not reconstruct full-horizon gradient')
        H,record=protect_gradient(first,second,context['affinity'][:,e])
        # Retain the exact original .grad when no adaptation is admitted.
        if record['adjusted'] and (primary.norm()==0 or torch.dot(H,primary)<=0):
            record.update(adjusted=False,fallback=True)
        if record['adjusted']:
            H=H*(primary.norm()/H.norm())
            position=0
            for p in parameters:
                value=H[position:position+p.numel()].view_as(p);position+=p.numel()
                if p.grad is not None:p.grad.copy_(value)
                elif bool((value!=0).any()):p.grad=value.clone()
        record.update(expert=e,cohort_sum_max_error=error,affinity=context['affinity'][:,e].cpu().tolist())
        records.append(record);offset+=count
    return dict(stage=context['stage'],coordinate=context['coordinate'],threshold=context['threshold'],cohort_counts=context['counts'].cpu().tolist(),experts=records,conflicts=sum(r['conflict'] for r in records),adjusted=sum(r['adjusted'] for r in records),fallbacks=sum(r['fallback'] for r in records))
