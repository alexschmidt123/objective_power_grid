"""Domain adapters and frozen prior-predictive hypothesis partitions; OFFLINE ONLY."""
import copy,math
import torch
from torch import nn
from src.research.benchmarks import make_benchmark

class PredictiveAdapter:
 def __init__(self,raw,config_path=None):
  self.raw=copy.deepcopy(raw)
  if config_path is not None:self.raw['_config_path']=str(config_path)
  self.b=make_benchmark(self.raw)
  if self.b.name=='location_finding':
   # Fixed5x5 covering the canonical feasible box, not selected by EIG.
   lo,hi=self.b.design_bounds;axis=torch.linspace(lo,hi,5,dtype=torch.float64);self.designs=torch.cartesian_prod(axis,axis)
  elif self.b.name=='sir_ode':self.designs=torch.linspace(.1,self.b.end,16,dtype=torch.float64)[:,None]
  else:
   h=int(raw['experiment']['step_number']);actions=[]
   for t in range(h):actions.append(self.b.feasible(torch.zeros(1,self.b.action_dim,dtype=torch.float64),actions,h))
   self.designs=torch.cat(actions)
 def policy_options(self,horizon):
  return dict(horizon=horizon,action_dim=self.b.action_dim,obs_dim=self.b.obs_dim,action_scale=self.b.action_scale if self.b.state_dim==0 else 1.,obs_scale=self.b.obs_scale if self.b.state_dim==0 else 1.,ordered_history=self.b.state_dim>0)
 @torch.no_grad()
 def signatures(self,theta,chunk=8192):
  shape=theta.shape[:-1];flat=theta.reshape(-1,theta.shape[-1]);result=[]
  for start in range(0,len(flat),chunk):
   x=flat[start:start+chunk];state=self.b.initial((len(x),));values=[]
   for d in self.designs:
    y,state=self.b.forward(x,state,d.to(x).expand(len(x),-1));values.append(y)
   result.append(torch.cat(values,-1))
  return torch.cat(result).reshape(*shape,-1)

class FrozenPartition(nn.Module):
 """Nearest-centroid f(theta) in a frozen standardized predictive signature."""
 def __init__(self,mean,scale,centers):
  super().__init__();self.register_buffer('mean',mean);self.register_buffer('scale',scale);self.register_buffer('centers',centers)
 @torch.no_grad()
 def classify(self,adapter,theta):
  shape=theta.shape[:-1];flat=theta.reshape(-1,theta.shape[-1]);labels=[]
  for start in range(0,len(flat),8192):
   s=adapter.signatures(flat[start:start+8192]);z=(s-self.mean.to(s))/self.scale.to(s);dist=(z[:,None]-self.centers.to(z)[None]).square().sum(-1);labels.append(dist.argmin(-1))
  return torch.cat(labels).reshape(shape)

@torch.no_grad()
def fit_partition(adapter,n=8192,validation=4096,seed=86404011):
 torch.set_num_threads(4);g=torch.Generator().manual_seed(seed);theta=adapter.b.sample((n,),g);sig=adapter.signatures(theta);mean=sig.mean(0);scale=(sig.var(0,unbiased=False)+adapter.b.sigma**2).sqrt().clamp_min(1e-8);z=(sig-mean)/scale
 if float(z.var(0,unbiased=False).sum())<1e-8:raise ValueError('Degenerate predictive signatures')
 centers=[z[torch.randint(n,(1,),generator=g)[0]]]
 for _ in range(3):
  dist=torch.stack([(z-c).square().sum(-1) for c in centers]).min(0).values
  centers.append(z[torch.multinomial(dist/dist.sum(),1,generator=g)[0]])
 centers=torch.stack(centers)
 for it in range(100):
  label=torch.cdist(z,centers).argmin(-1);counts=torch.bincount(label,minlength=4)
  if counts.min()==0:raise ValueError('Empty predictive cluster; no policy training permitted')
  updated=torch.stack([z[label==c].mean(0) for c in range(4)])
  change=float((updated-centers).abs().max());centers=updated
  if change<1e-10:break
 partition=FrozenPartition(mean,scale,centers);vg=torch.Generator().manual_seed(seed+1);vtheta=adapter.b.sample((validation,),vg);v=adapter.signatures(vtheta);vz=(v-mean)/scale;vl=partition.classify(adapter,vtheta);counts=torch.bincount(vl,minlength=4);frequency=counts.double()/validation;overall=vz.mean(0);total=(vz-overall).square().sum(-1).mean();within=torch.stack([(vz[vl==c]-vz[vl==c].mean(0)).square().sum(-1).mean() for c in range(4)]);between=sum(frequency[c]*(vz[vl==c].mean(0)-overall).square().sum() for c in range(4));within_ratio=within/total
 valid=bool(frequency.min()>=.05 and between/total>=.05 and within_ratio.min()>=.01)
 report=dict(valid=valid,groups=4,fit_samples=n,validation_samples=validation,seed=seed,validation_seed=seed+1,frequency=frequency.tolist(),within_variance_relative_to_total=within_ratio.tolist(),between_fraction=float(between/total),iterations=it+1,reference_designs=adapter.designs.tolist(),method='standardized predictive mean signatures; frozen kmeans++/Lloyd; noise-floor scaling',labels_are_not_physical_strategies=True)
 if adapter.b.name=='location_finding':
  swapped=vtheta[:,[2,3,0,1]];invariant=bool(torch.equal(vl,partition.classify(adapter,swapped)));report['source_exchangeability']=invariant;valid=valid and invariant;report['valid']=valid
 if not valid:raise ValueError('Partition failed predeclared nondegeneracy/symmetry checks: '+str(report))
 return partition,report
