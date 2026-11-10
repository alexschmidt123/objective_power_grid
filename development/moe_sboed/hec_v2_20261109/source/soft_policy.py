"""Soft feature slots and distinct supervised capabilities;unchanged native AC."""
import sys,copy,math,json,inspect,time
sys.dont_write_bytecode=True
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from native_common import O,digest,put
from src.research.vendor.vsoed.pg_vsoed import PGvsOED

class CollaborativeActor(nn.Module):
 def __init__(self,cfg):
  super().__init__();self.encoder=nn.Sequential(nn.Linear(117,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU());self.experts=nn.ModuleList([])
  for seed in cfg['expert_seeds']:
   with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []):
    torch.manual_seed(seed);self.experts.append(nn.Sequential(nn.Linear(128,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU()))
  self.router=nn.Linear(128,4);self.decision=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Linear(128,2));self.task_heads=nn.ModuleList([nn.Linear(64,n) for n in [16,8,5,1]])
 def parts(self,x,detach_encoder=False):
  s=self.encoder(x);s=s.detach() if detach_encoder else s;z=torch.stack([expert(s) for expert in self.experts],-2);g=(10*torch.tanh(self.router(s)/10)).softmax(-1);return s,z,g
 def forward(self,x,disable=None,stage_gates=None,uniform=False):
  s,z,g=self.parts(x)
  if stage_gates is not None:g=stage_gates.expand_as(g)
  if uniform:g=torch.full_like(g,.25)
  if disable is not None:
   mask=torch.ones(4,device=g.device,dtype=g.dtype);mask[disable]=0;g=g*mask;g=g/g.sum(-1,keepdim=True)
  slots=(g.unsqueeze(-1)*z).flatten(-2);return -4+8*self.decision(slots).sigmoid()

def prefix_inputs(ds,ys,stages):
 n=len(ds);t=torch.arange(29,device=ds.device)[None];mask=t<stages[:,None];x=torch.zeros(n,117,device=ds.device,dtype=ds.dtype);x.scatter_(1,stages[:,None],1);x[:,30:88]=(ds[:,:29]*mask[:,:,None]).reshape(n,58);x[:,88:117]=ys[:,:29,0]*mask;return x

def discounted_returns(rewards,gamma=.9):
 out=torch.empty(len(rewards),30,device=rewards.device,dtype=rewards.dtype);out[:,29]=rewards[:,29]+rewards[:,30]
 for t in range(28,-1,-1):out[:,t]=rewards[:,t]+gamma*out[:,t+1]
 return out

class Tasks:
 def __init__(self,a,enabled,cfg):
  self.a=a;self.enabled=enabled;self.cfg=cfg;self.rng=np.random.default_rng(cfg['auxiliary_sampling_seed']);self.hypotheses=np.random.default_rng(cfg['hypothesis_label_seed']).standard_normal((16,2,2));self.calls=0;self.seconds=0.;self.steps=0;self.last={}
 @torch.no_grad()
 def labels(self,ds,ys,theta,stages,rewards):
  n=len(ds);src=theta.reshape(n,2,2);edges=torch.tensor([-1.,0.,1.],device=theta.device);cell=torch.bucketize(src.contiguous(),edges);bins=cell[:,:,0]*4+cell[:,:,1];coarse=F.one_hot(bins,16).to(theta.dtype).mean(1);center=src.mean(1);delta=src[:,0]-src[:,1];fine=torch.stack([center[:,0]*math.sqrt(2),center[:,1]*math.sqrt(2),(delta[:,0].square()-2)/math.sqrt(8),delta[:,0]*delta[:,1]/2,(delta[:,1].square()-2)/math.sqrt(8)],-1)
  hyp=torch.tensor(self.hypotheses,device=theta.device,dtype=theta.dtype);dist=(hyp[None,None]-ds[:,:29,None,None]).square().sum(-1);mean=(.1+(dist+1e-4).reciprocal().sum(-1)).log();ll=-2*(ys[:,:29,0,None]-mean).square();mask=torch.arange(29,device=ds.device)[None]<stages[:,None];prefix=(ll*mask[:,:,None]).sum(1);dis=((prefix[:,0::2]-prefix[:,1::2])/stages.clamp_min(1)[:,None]/10).clamp(-5,5);planning=discounted_returns(rewards)[torch.arange(n,device=ds.device),stages,None]/10;self.calls+=n*29*16
  return [coarse,dis,fine,planning]
 def losses(self,x,targets):
  _,z,_=self.a.actor_net.parts(x,detach_encoder=True);pred=[head(z[:,k]) for k,head in enumerate(self.a.actor_net.task_heads)];losses=[-(targets[0]*pred[0].log_softmax(-1)).sum(-1).mean()]+[F.mse_loss(pred[k],targets[k]) for k in [1,2,3]];return losses,pred
 def add_gradient(self):
  beg=time.monotonic();a=self.a;device=next(a.actor_net.parameters()).device;n=self.cfg['auxiliary_prefix_batch'];b=len(a.params);idx=torch.tensor(self.rng.integers(b,size=n),device=device);stage=torch.tensor(self.rng.integers(30,size=n),device=device);ds=a.ds_hist[idx].detach();ys=a.ys_hist[idx].detach();x=prefix_inputs(ds,ys,stage);targets=self.labels(ds,ys,a.params[idx].detach(),stage,a.rewards_hist[idx].detach());native_norm=math.sqrt(sum(float(p.grad.square().sum()) for p in a.actor_net.parameters() if p.grad is not None));parameters=list(a.actor_net.experts.parameters())+list(a.actor_net.task_heads.parameters())
  with torch.enable_grad():losses,pred=self.losses(x,targets);loss=sum(w*loss for w,loss in zip(self.cfg['task_weights'],losses));grads=torch.autograd.grad(loss,parameters)
  step=int(a.update)+1;warm=step<=self.cfg['guided_iterations'];weight=1. if warm else self.cfg['joint_auxiliary_multiplier'];cap=self.cfg['guided_gradient_cap'] if warm else self.cfg['joint_gradient_cap'];gnorm=math.sqrt(sum(float(g.square().sum()) for g in grads));scale=min(weight,cap*native_norm/(gnorm+1e-12)) if self.enabled else 0.;assert all(torch.isfinite(g).all() for g in grads)
  if scale:
   for p,g in zip(parameters,grads):
    if p.grad is None:p.grad=scale*g
    else:p.grad.add_(g,alpha=scale)
  self.steps+=1;self.seconds+=time.monotonic()-beg;self.last=dict(iteration=step,task_losses=[float(l.detach()) for l in losses],auxiliary_norm=gnorm,native_gradient_norm=native_norm,scale=scale,auxiliary_native_ratio=scale*gnorm/max(native_norm,1e-30),planning_target='native discounted realized remaining variational reward /10',prefixes=n,enabled=self.enabled)

def install(a,enabled=True):
 cfg=json.loads((O/'CONFIG_LOCKED.json').read_text());device=next(a.actor_net.parameters()).device
 with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []):
  torch.manual_seed(cfg['actor_seed']);actor=CollaborativeActor(cfg).to(device)
 assert all(not opt.state for opt in [a.actor_optimizer,a.critic_optimizer]);a.actor_net=actor;a.actor_target_net=copy.deepcopy(actor);a.actor_optimizer=torch.optim.Adam(actor.parameters(),lr=.001);a.actor_lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(a.actor_optimizer,gamma=.9999);tasks=Tasks(a,enabled,cfg['tasks']);a._soft_tasks=tasks
 def hook(opt,args,kwargs):tasks.add_gradient()
 a.actor_optimizer.register_step_pre_hook(hook);return inspect.getsource(PGvsOED.train)
