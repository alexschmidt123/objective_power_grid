"""Pinned RL-BOED source-location networks and update kernels, not generic SAC.

The loader executes unchanged upstream class bodies while replacing only legacy
Garage imports/trainer metadata. Vendored originals and their hashes are kept.
The experiment loop uses native SAC.optimize_policy and _update_targets.
"""
import ast,abc,copy,math,json,hashlib,time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.research.baseline_fidelity import rl_optimizer_epoch_completed
VENDOR=Path(__file__).parent/'vendor/rl_boed'

def load_classes(path,names,env):
 tree=ast.parse(Path(path).read_text());nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in names]
 if {n.name for n in nodes}!=set(names):raise ValueError('Reference class missing')
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
 return [env[n] for n in names]

class PolicyMetadata(nn.Module):
 def __init__(self,env_spec,name):super().__init__();self._env_spec=env_spec;self._name=name

def reference_types():
 env=dict(__name__=__name__,torch=torch,nn=nn,F=F,np=np,copy=copy,abc=abc,deque=deque,RLAlgorithm=object,StochasticPolicy=PolicyMetadata)
 load_classes(VENDOR/'garage/_functions.py',['NonLinearity'],env)
 load_classes(VENDOR/'garage/modules/multi_headed_mlp_module.py',['MultiHeadedMLPModule'],env)
 load_classes(VENDOR/'garage/modules/mlp_module.py',['MLPModule'],env)
 from torch.distributions import Normal,Independent
 env.update(Normal=Normal,Independent=Independent)
 load_classes(VENDOR/'garage/distributions/tanh_normal.py',['TanhNormal'],env)
 load_classes(VENDOR/'rl/pyro/modules/gaussian_mlp_module.py',['GaussianMLPBaseModule','GaussianMLPTwoHeadedModule'],env)
 load_classes(VENDOR/'rl/pyro/policies/adaptive_tanh_gaussian_policy.py',['AdaptiveTanhGaussianPolicy'],env)
 load_classes(VENDOR/'rl/pyro/q_functions/adaptive_mlp_q_function.py',['AdaptiveMLPQFunction'],env)
 load_classes(VENDOR/'rl/pyro/algos/sac.py',['SAC'],env)
 return env
TYPES=reference_types()
for _name in ['NonLinearity','MultiHeadedMLPModule','MLPModule','TanhNormal','GaussianMLPBaseModule','GaussianMLPTwoHeadedModule','AdaptiveTanhGaussianPolicy','AdaptiveMLPQFunction','SAC']:globals()[_name]=TYPES[_name]

def specification(horizon):
 return SimpleNamespace(observation_space=SimpleNamespace(flat_dim=3),action_space=SimpleNamespace(flat_dim=2,shape=(1,1,1,2),is_discrete=False),max_episode_length=horizon)

def native_history(features,horizon):
 slots=features[:,1:].reshape(len(features),horizon,4)
 # Exact NormalizedEnv._apply_normalize_obs for [-4,-4,-3]..[4,4,10].
 raw=torch.cat(((slots[...,:2]+1.)/2.,(slots[...,2:3]*3.+3.)/13.),-1).float()
 return raw,slots[...,-1:].float()

def make_actor(h):
 return TYPES['AdaptiveTanhGaussianPolicy'](specification(h),encoder_sizes=[128,128],encoding_dim=64,emitter_sizes=[128,128],encoder_nonlinearity=nn.ReLU,emitter_nonlinearity=nn.ReLU,init_std=np.sqrt(1/3),min_std=np.exp(-20),max_std=1.)
def make_critic(h):
 return TYPES['AdaptiveMLPQFunction'](specification(h),encoder_sizes=[128,128],encoding_dim=64,emitter_sizes=[128,128],encoder_nonlinearity=nn.ReLU,emitter_nonlinearity=nn.ReLU)

class NativeRLPolicy(nn.Module):
 kind='rl_sboed';outputs_design=True;bound=4.;architecture='published_RL_BOED_source_REDQ_native_v1'
 def __init__(self,horizon):super().__init__();self.horizon=horizon;self.native=make_actor(horizon)
 def sample(self,features,deterministic=False,rng=None):
  x,mask=native_history(features,self.horizon);dist,_=self.native(x,mask)
  normal=dist._normal.base_dist
  z=normal.loc if deterministic else normal.loc+normal.scale*torch.randn(normal.loc.shape,dtype=normal.loc.dtype,device=normal.loc.device,generator=rng)
  action=z.tanh();return action,dist.log_prob(action,pre_tanh_value=z)
 def forward(self,f,stage):return self.sample(f)[0].double()*4.

class Replay:
 def __init__(self,capacity,inputs):
  self.capacity=int(capacity);self.position=self.count=0
  self.data=torch.empty(self.capacity,2*inputs+4,dtype=torch.float32)
 def add(self,data):
  ids=(torch.arange(len(data),device=data.device)+self.position)%self.capacity;self.data[ids]=data
  self.position=(self.position+len(data))%self.capacity;self.count=min(self.count+len(data),self.capacity)
 def sample(self,batch,h):
  # Same uniform-with-replacement transition distribution as PathBuffer.
  ids=np.random.randint(self.count,size=batch);v=self.data[torch.as_tensor(ids,device=self.data.device)];dim=(v.shape[-1]-4)//2
  x,mask=native_history(v[:,:dim],h);nx,nmask=native_history(v[:,-dim:],h)
  return dict(observation=x,mask=mask,next_observation=nx,next_mask=nmask,action=v[:,dim:dim+2],reward=v[:,dim+2:dim+3],terminal=v[:,dim+3:dim+4])

def make_algorithm(policy,horizon):
 q=[make_critic(horizon) for _ in range(2)]
 algorithm=TYPES['SAC'](specification(horizon),policy.native,q,None,None,gradient_steps_per_itr=64,min_buffer_size=100000,buffer_batch_size=4096,target_update_tau=.001,policy_lr=.0001,qf_lr=.0003,discount=.9,M=2,ent_anneal_rate=1/1.4e4,eval_env=None)
 algorithm.to(next(policy.parameters()).device)
 return algorithm

def train_native_rl(b,h,s,seed,directory,trajectory_budget,observer=None):
 from src.research.engine import rollout
 if b.name!='location_finding' or tuple(b.design_bounds)!=(-4.,4.):raise ValueError('Pinned native RL adapter is currently verified for bounded source location only')
 torch.manual_seed(seed);np.random.seed(seed);started=time.perf_counter();p=NativeRLPolicy(h);algo=make_algorithm(p,h)
 capacity=int(s.get('rl_native_replay_capacity',1000000));replay=Replay(capacity,1+4*h)
 batch=100;replay_batch=4096;grad_steps=64;min_buffer=100000;L=int(s.get('rl_native_contrasts',100000))
 if s.get('updates',3)<=2:batch=2;replay_batch=4;grad_steps=1;min_buffer=1
 episodes=updates=gradient_updates=0;history=[]
 while episodes<trajectory_budget:
  n=min(batch,trajectory_budget-episodes);updates+=1
  with torch.no_grad():r=rollout(b,p,h,n,seed+updates*104729,L)
  info=r['info'];reward=info-torch.cat((torch.zeros_like(info[:,:1]),info[:,:-1]),1)
  inputs=torch.stack(r['features'],1).reshape(n*h,-1).float();next_inputs=torch.stack(r['features'][1:]+[r['final_features']],1).reshape_as(inputs).float()
  actions=torch.stack(r['actions'],1).reshape(n*h,2).float()/4.;done=torch.zeros(n,h);done[:,-1]=1
  replay.add(torch.cat((inputs,actions,reward.reshape(-1,1).float(),done.reshape(-1,1),next_inputs),-1))
  episodes+=n
  if replay.count>=min_buffer:
   for _ in range(grad_steps):
    aloss,qloss,entropy=algo.optimize_policy(replay.sample(replay_batch,h));algo._update_targets();gradient_updates+=1
    if not all(torch.isfinite(x).all() for x in [aloss,*qloss,entropy,algo._log_alpha]):raise FloatingPointError('Nonfinite native RL update; no clipping/fallback introduced')
  # Native trainer collects the entire warmup before its first optimizer epoch.
  if rl_optimizer_epoch_completed(replay.count,min_buffer):algo._target_entropy-=algo._ent_anneal_rate
  if updates%100==0 or episodes==trajectory_budget:
   row=dict(primary_episodes=episodes,collection_iterations=updates,gradient_updates=gradient_updates,training_spce=float(info[:,-1].mean()),alpha=float(algo._log_alpha.exp().detach()),target_entropy=algo._target_entropy)
   history.append(row);print('native RL-BOED',row,flush=True)
  if observer is not None and observer.check(p,episodes,updates,algo._policy_optimizer,force=episodes==trajectory_budget):break
 journal=observer.finish(p) if observer is not None else {}
 p.eval();torch.save(dict(kind='rl_sboed',architecture=p.architecture,horizon=h,state_dict=p.state_dict()),directory/'rl_sboed.pt')
 return p,dict(training='pinned_original_RL_BOED_source_SAC_REDQ',publication_algorithm_equivalence='reference_kernels_and_native_networks; see executable tests',generated_training_trajectories=episodes,training_forward_trajectories=episodes,contrast_training_trajectories=episodes*L,forward_particle_stages=episodes*h*(L+1),training_seconds=time.perf_counter()-started,history=history,gradient_updates=gradient_updates,critic_updates=2*gradient_updates,actor_updates=gradient_updates,temperature_updates=gradient_updates,gamma=.9,target_rate=.001,policy_lr=.0001,critic_lr=.0003,gradient_steps_per_collection=grad_steps,replay_batch=replay_batch,replay_capacity=capacity,contrasts=L,resource_departures=['total1M versus source README about2M primary episodes; contrasts and replay capacity preserved'],stochastic_deployment='native source.py policy evaluation samples tanh Gaussian; independent decision RNG',**journal)
