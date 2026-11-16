"""V5-B refines existing V4 credit with replicated continuation uncertainty.

It is not a new route-intervention mechanism. It adds no deployment parameters.
"""
import copy, torch
from src.research.engine import rollout,generator
from src.research.moe_v5_diagnostics import Work

def swap_one(ids,slot,rank):
    selected=torch.nn.functional.one_hot(ids,7).sum(1).bool()
    candidates=torch.arange(7,device=ids.device).expand(len(ids),7)[~selected].reshape(-1,4)
    add=candidates.gather(1,rank[:,None]).squeeze(1);drop=ids.gather(1,slot[:,None]).squeeze(1)
    changed=ids.clone();changed.scatter_(1,slot[:,None],add[:,None]);return changed,drop,add

def route_output(model,z,ids):
    scores=model.router(z).softmax(-1);u=z+model.shared(z);r=z.new_zeros(z.shape)
    for e in ids.unique().tolist():
        rows,col=(ids==e).nonzero(as_tuple=True);r.index_add_(0,rows,model.routed[e](z[rows])*scores[rows,e,None])
    return model.emitter(u+r)

@torch.no_grad()
def replicated_credit(benchmark,model,seed,batch=1,negatives=16,repeats=4,work=None):
    if repeats<4:raise ValueError('At least four continuation replicates required')
    work=work or Work()
    if 2*batch*repeats>16 or negatives>32:raise ValueError('Tiny credit-smoke allowance exceeded')
    if seed in {1001,8831001}:raise ValueError('Reserved/evaluation randomness forbidden')
    p=copy.deepcopy(model).eval();g=generator(seed);h=model.horizon
    truth=benchmark.sample((batch,),g);bank=benchmark.sample((batch,negatives),g)
    prefix_noise=torch.randn(batch,h,benchmark.obs_dim,dtype=torch.float64,generator=g)
    stages=torch.randint(0,h,(batch,),generator=g);slots=torch.randint(0,3,(batch,),generator=g);ranks=torch.randint(0,4,(batch,),generator=g)
    saved={};advantages=[];prefix_reference=None
    for rep in range(repeats):
        future=torch.randn(batch,h,benchmark.obs_dim,dtype=torch.float64,generator=g)
        noise=torch.where((torch.arange(h)[None,:,None]>=stages[:,None,None]),future,prefix_noise)
        z=truth.new_zeros(batch,16);drop=torch.zeros(batch,dtype=torch.long);add=drop.clone();records={}
        def record(module,args,out):
            stage=int(args[1]);rows=(stages==stage).nonzero().flatten()
            if len(rows):
                ids=module.last_dispatch['indices'][rows];new,d,a=swap_one(ids,slots[rows],ranks[rows]);records[stage]=(rows,new)
                z[rows]=module.last_features[rows];drop[rows]=d;add[rows]=a
        hook=p.register_forward_hook(record)
        try:
            work.rollout(batch,h,negatives);base=rollout(benchmark,p,h,batch,seed,negatives,truth=truth,noise=noise,prior_particles=bank)
        finally:hook.remove()
        def intervene(module,args,out):
            entry=records.get(int(args[1]))
            if entry is None:return out
            rows,ids=entry;changed=out.clone();changed[rows]=route_output(module,module.last_features[rows],ids);return changed
        hook=p.register_forward_hook(intervene)
        try:
            work.rollout(batch,h,negatives);alt=rollout(benchmark,p,h,batch,seed,negatives,truth=truth,noise=noise,prior_particles=bank)
        finally:hook.remove()
        for stage in range(h):
            rows=stages>=stage
            assert torch.equal(base['features'][stage][rows],alt['features'][stage][rows])
        prefixes=torch.stack([base['features'][int(stages[i])][i] for i in range(batch)])
        if prefix_reference is None:prefix_reference=prefixes
        else:assert torch.equal(prefix_reference,prefixes)
        advantages.append(alt['info'][:,-1]-base['info'][:,-1]);saved=dict(z=z.detach(),drop=drop,add=add,stages=stages)
    saved.update(advantages=torch.stack(advantages).detach(),work=work,same_prefix=True,
        interpretation='Noise-replicated joint-history utility proxy at fixed generating theta, not exact posterior-conditional competence')
    return saved

def confidence_loss(router,z,drop,add,advantages):
    if advantages.ndim!=2 or advantages.shape[0]<4 or not torch.isfinite(advantages).all():raise ValueError('Finite replicated utilities required')
    a=advantages.detach();mean=a.mean(0);se=a.std(0,unbiased=True)/a.shape[0]**.5
    # Predeclared conservative heuristic, not a guaranteed small-sample CI.
    weight=((mean.abs()-2*se).clamp_min(0)/(mean.abs()+1e-12)).clamp_max(1).detach()
    logits=router(z.detach());difference=logits.gather(1,add[:,None]).squeeze(1)-logits.gather(1,drop[:,None]).squeeze(1)
    loss=(weight*torch.nn.functional.softplus(-mean.sign().detach()*difference)).mean()
    return loss,dict(mean=mean.tolist(),mcse=se.tolist(),weights=weight.tolist(),
        interpretation='Heuristic replicated-noise confidence filtering; not a significance claim')

def refine_router_credit(model,record,legacy_utility,refinement=0.,fraction=.05):
    """eta0 executes exact V4; positive eta blends losses instead of double credit."""
    if refinement==0:return legacy_utility.add_credit(model,record,fraction)
    if not 0<refinement<=.1:raise ValueError('Only the predeclared small refinement is allowed')
    first=dict(z=record['z'],drop=record['drop'],add=record['add'],advantage=record['advantages'][0])
    old,_=legacy_utility.ranking_loss(model.router,first['z'],first['drop'],first['add'],first['advantage'])
    new,meta=confidence_loss(model.router,record['z'],record['drop'],record['add'],record['advantages'])
    grad=torch.autograd.grad((1-refinement)*old+refinement*new,model.router.weight)[0].detach()
    base=model.router.weight.grad
    if base is None or base.norm()==0 or grad.norm()==0 or not torch.isfinite(grad).all():return dict(applied=False,**meta)
    delta=grad*min(1.,float(fraction*base.norm()/grad.norm()));model.router.weight.grad.add_(delta)
    return dict(applied=True,delta_norm=float(delta.norm()),fraction_cap=fraction,refinement=refinement,**meta)
