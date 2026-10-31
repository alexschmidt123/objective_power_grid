"""Training-only paired route interventions. Historical V1 class stays unchanged."""
import copy,torch
from src.research.engine import rollout,generator

def swapped_ids(ids,slot,rank):
 selected=torch.nn.functional.one_hot(ids,7).sum(1).bool();allids=torch.arange(7,device=ids.device).expand(len(ids),7);other=allids[~selected].reshape(len(ids),4);add=other.gather(1,rank[:,None]).squeeze(1);drop=ids.gather(1,slot[:,None]).squeeze(1);out=ids.clone();out.scatter_(1,slot[:,None],add[:,None]);return out,drop,add

def route_output(model,z,ids):
 scores=model.router(z).softmax(-1);u=z+model.shared(z);r=z.new_zeros(z.shape)
 for e in ids.unique().tolist():
  rows,col=(ids==e).nonzero(as_tuple=True);r.index_add_(0,rows,model.routed[e](z[rows])*scores[rows,e,None])
 return model.emitter(u+r)

@torch.no_grad()
def paired_credit(b,model,seed,batch=64,contrasts=1024):
 p=copy.deepcopy(model).eval();g=generator(seed);truth=b.sample((batch,),g);bank=b.sample((batch,contrasts),g);noise=torch.randn(batch,30,b.obs_dim,generator=g,dtype=torch.float64);stages=torch.randint(0,30,(batch,),generator=g);slots=torch.randint(0,3,(batch,),generator=g);ranks=torch.randint(0,4,(batch,),generator=g);zsave=truth.new_zeros(batch,16);drop=torch.zeros(batch,dtype=torch.long,device=truth.device);add=drop.clone();ids_by_stage={};prefix_ok=[]
 def record(module,args,output):
  stage=int(args[1]);rows=(stages==stage).nonzero().flatten()
  if len(rows):
   ids=module.last_dispatch['indices'][rows];new,d,a=swapped_ids(ids,slots[rows],ranks[rows]);ids_by_stage[stage]=(rows,new);zsave[rows]=module.last_features[rows];drop[rows]=d;add[rows]=a
 hook=p.register_forward_hook(record)
 try:r0=rollout(b,p,30,batch,seed,contrasts,truth=truth,noise=noise,prior_particles=bank)
 finally:hook.remove()
 def intervene(module,args,output):
  stage=int(args[1]);entry=ids_by_stage.get(stage)
  if entry is None:return output
  rows,ids=entry;out=output.clone();out[rows]=route_output(module,module.last_features[rows],ids);return out
 hook=p.register_forward_hook(intervene)
 try:r1=rollout(b,p,30,batch,seed,contrasts,truth=truth,noise=noise,prior_particles=bank)
 finally:hook.remove()
 for t in range(30):
  rows=stages>=t
  assert torch.equal(r0['features'][t][rows],r1['features'][t][rows]),'Intervention prefixes differ'
  before=stages>t
  assert torch.equal(r0['actions'][t][before],r1['actions'][t][before])
 advantage=(r1['info'][:,-1]-r0['info'][:,-1]).detach()
 return dict(z=zsave.detach(),drop=drop,add=add,advantage=advantage,stages=stages,baseline=r0['info'][:,-1].detach(),alternative=r1['info'][:,-1].detach(),same_prefix=True,truth=truth.detach(),noise=noise.detach(),particles=bank.detach())

def ranking_loss(router,z,drop,add,advantage):
 a=advantage.detach();rms=a.square().mean().sqrt();safe=bool(torch.isfinite(a).all()) and bool(rms>=1e-8) and bool(rms<=100)
 if not safe:return router.weight.sum()*0,dict(enabled=False,rms=float(rms))
 bounded=(a/rms).clamp(-1,1).detach();logits=router(z.detach());difference=logits.gather(1,add[:,None]).squeeze(1)-logits.gather(1,drop[:,None]).squeeze(1);loss=(bounded.abs()*torch.nn.functional.softplus(-bounded.sign()*difference)).mean()
 return loss,dict(enabled=True,rms=float(rms),mean_advantage=float(a.mean()),positive_fraction=float((a>0).double().mean()),maximum_normalized_advantage=float(bounded.abs().max()))

def add_credit(model,record,fraction=.05):
 loss,meta=ranking_loss(model.router,record['z'],record['drop'],record['add'],record['advantage']);g=torch.autograd.grad(loss,model.router.weight)[0].detach();base=model.router.weight.grad
 if base is None or not torch.isfinite(g).all() or g.norm()==0 or base.norm()==0:meta.update(applied=False);return meta
 scale=min(1.,float(fraction*base.norm()/g.norm()));delta=g*scale;assert delta.norm()<=fraction*base.norm()*(1+1e-10);model.router.weight.grad.add_(delta);meta.update(applied=True,scale=scale,auxiliary_gradient_norm=float(delta.norm()),primary_router_gradient_norm=float((base-delta).norm()),fraction_cap=fraction);return meta
