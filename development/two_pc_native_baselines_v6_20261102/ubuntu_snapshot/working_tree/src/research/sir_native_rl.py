"""SIR ODE adaptation of unchanged pinned RL-BOED REDQ networks/update kernels.

Only observed normalized (time, infected-count) pairs enter the native encoder.
Replay stores the native tanh action, not physical state-dependent time.
"""
import json,time
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from src.research.native_rl_reference import TYPES
from src.research.baseline_fidelity import rl_optimizer_epoch_completed

from src.research.resource_ledger import with_training_ledger
def spec(h):return SimpleNamespace(observation_space=SimpleNamespace(flat_dim=2),action_space=SimpleNamespace(flat_dim=1,shape=(1,1,1,1),is_discrete=False),max_episode_length=h)
def history(f,h):
 slots=f[:,1:].reshape(len(f),h,3)
 return slots[...,:2].float(),slots[...,-1:].float()
def network(h,critic=False):
 kw=dict(encoder_sizes=[128,128],encoding_dim=64,emitter_sizes=[128,128],encoder_nonlinearity=nn.ReLU,emitter_nonlinearity=nn.ReLU)
 if not critic:kw.update(init_std=np.sqrt(1/3),min_std=np.exp(-20),max_std=1.)
 return TYPES['AdaptiveMLPQFunction' if critic else 'AdaptiveTanhGaussianPolicy'](spec(h),**kw)

class NativeSIRRLPolicy(nn.Module):
 kind='rl_sboed';outputs_design=True;bound=1.;architecture='published_RL_BOED_REDQ_SIR_ODE_native_v1'
 def __init__(self,horizon,benchmark):
  super().__init__();self.horizon=horizon;self.end=benchmark.end;self.gap=benchmark.gap;self.native=network(horizon)
 def sample(self,f,deterministic=False,rng=None):
  x,mask=history(f,self.horizon);dist,_=self.native(x,mask);normal=dist._normal.base_dist
  z=normal.loc if deterministic else normal.loc+normal.scale*torch.randn(normal.loc.shape,dtype=normal.loc.dtype,device=normal.loc.device,generator=rng)
  a=z.tanh();return a,dist.log_prob(a,pre_tanh_value=z)
 def design_from_unit_action(self,unit,actions,horizon):
  previous=actions[-1][:,0] if actions else unit.new_zeros(len(unit))
  lo=previous+self.gap;hi=self.end-(horizon-len(actions)-1)*self.gap
  return (lo+(hi-lo)*(unit[:,0].double()+1)/2)[:,None]
 def forward(self,f,stage):
  unit=self.sample(f)[0];slots=f[:,1:].reshape(len(f),self.horizon,3)
  previous=slots[:,stage-1,0]*self.end if stage else f.new_zeros(len(f))
  lo=previous+self.gap;hi=self.end-(self.horizon-stage-1)*self.gap
  return (lo+(hi-lo)*(unit[:,0].double()+1)/2)[:,None]

class Replay:
 def __init__(self,capacity,inputs):self.capacity=int(capacity);self.inputs=inputs;self.position=self.count=0;self.data=torch.empty(capacity,2*inputs+3,dtype=torch.float32)
 def add(self,data):
  ids=(torch.arange(len(data),device=data.device)+self.position)%self.capacity;self.data[ids]=data;self.position=(self.position+len(data))%self.capacity;self.count=min(self.count+len(data),self.capacity)
 def sample(self,batch,h):
  ids=np.random.randint(self.count,size=batch);v=self.data[torch.as_tensor(ids,device=self.data.device)];d=self.inputs;x,mask=history(v[:,:d],h);nx,nmask=history(v[:,-d:],h)
  return dict(observation=x,mask=mask,next_observation=nx,next_mask=nmask,action=v[:,d:d+1],reward=v[:,d+1:d+2],terminal=v[:,d+2:d+3])

def algorithm(p,h):
 a=TYPES['SAC'](spec(h),p.native,[network(h,True) for _ in range(2)],None,None,gradient_steps_per_itr=64,min_buffer_size=100000,buffer_batch_size=4096,target_update_tau=.001,policy_lr=.0001,qf_lr=.0003,discount=.9,M=2,ent_anneal_rate=1/1.4e4,eval_env=None)
 a.to(next(p.parameters()).device);return a

def train_sir_native_rl(b,h,s,seed,directory,trajectory_budget,observer=None):
 from src.research.engine import rollout
 if b.name!='sir_ode':raise ValueError('SIR-only native adapter')
 torch.manual_seed(seed);np.random.seed(seed);start=time.perf_counter();p=NativeSIRRLPolicy(h,b);a=algorithm(p,h);replay=Replay(int(s.get('rl_native_replay_capacity',1000000)),1+3*h)
 batch=100;rb=4096;gs=64;warm=100000;L=int(s.get('rl_native_contrasts',100000))
 if s.get('updates',3)<=2:batch=2;rb=4;gs=1;warm=1
 episodes=updates=gradients=0;records=[]
 while episodes<trajectory_budget:
  if observer is not None and not observer.allow_next_unit(p,episodes,updates,a._policy_optimizer):break
  n=min(batch,trajectory_budget-episodes);updates+=1
  with torch.no_grad():r=rollout(b,p,h,n,seed+updates*104729,L)
  info=r['info'];reward=info-torch.cat((torch.zeros_like(info[:,:1]),info[:,:-1]),1)
  inputs=torch.stack(r['features'],1).reshape(n*h,-1).float();nxt=torch.stack(r['features'][1:]+[r['final_features']],1).reshape_as(inputs).float()
  actions=torch.stack(r['actions'],1)[...,0];previous=torch.cat((actions.new_zeros(n,1),actions[:,:-1]),1);lo=previous+b.gap;hi=b.end-(h-torch.arange(h,device=actions.device)-1)*b.gap
  unit=(2*(actions-lo)/(hi-lo)-1).reshape(-1,1).float();done=torch.zeros(n,h);done[:,-1]=1
  replay.add(torch.cat((inputs,unit,reward.reshape(-1,1).float(),done.reshape(-1,1),nxt),-1));episodes+=n
  if replay.count>=warm:
   for _ in range(gs):
    aloss,qloss,entropy=a.optimize_policy(replay.sample(rb,h));a._update_targets();gradients+=1
    if not all(torch.isfinite(x).all() for x in [aloss,*qloss,entropy,a._log_alpha]):raise FloatingPointError('Nonfinite native SIR REDQ update')
  if rl_optimizer_epoch_completed(replay.count,warm):a._target_entropy-=a._ent_anneal_rate
  if updates%100==0 or episodes==trajectory_budget:
   row=dict(primary_episodes=episodes,collection_iterations=updates,gradient_updates=gradients,training_spce=float(info[:,-1].mean()),alpha=float(a._log_alpha.exp().detach()),target_entropy=a._target_entropy);records.append(row);print('native SIR RL-BOED',row,flush=True)
  if observer is not None and observer.check(p,episodes,updates,a._policy_optimizer,force=episodes==trajectory_budget):break
 journal=observer.finish(p) if observer else {};p.eval();torch.save(dict(kind='rl_sboed',architecture=p.architecture,horizon=h,state_dict=p.state_dict()),directory/'rl_sboed.pt')
 return p,with_training_ledger(dict(training='pinned_original_RL_BOED_REDQ_SIR_ODE_adaptation',generated_training_trajectories=episodes,training_forward_trajectories=episodes,contrast_training_trajectories=episodes*L,forward_particle_stages=episodes*h*(L+1),training_seconds=time.perf_counter()-start,history=records,gradient_updates=gradients,critic_updates=2*gradients,actor_updates=gradients,temperature_updates=gradients,gamma=.9,target_rate=.001,gradient_steps_per_collection=gs,replay_batch=rb,replay_capacity=replay.capacity,contrasts=L,adaptations='Observed normalized time/count history; native tanh action maps affinely to remaining feasible chronological interval; replay retains unit action; native REDQ update kernels unchanged; not a published SIR reproduction',resource_departures=['source-location native hyperparameters transferred to deterministic SIR ODE; total1M episodes'],**journal),h)
