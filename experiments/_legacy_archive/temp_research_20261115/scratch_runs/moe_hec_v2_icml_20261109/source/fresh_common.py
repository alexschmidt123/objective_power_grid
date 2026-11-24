"""Exact original fresh seed101 native framework;no trained state inherited."""
import sys,copy,json,pathlib,random
sys.dont_write_bytecode=True
import torch
from native_common import *
def build_fresh(device='cpu'):
 random.seed(101);profile=json.loads((P/'frozen_protocol.json').read_text())['profile'];a,logger,initialization=original.build(profile,device,candidate=True,seed=101)
 a.use_PCE=False;a.discount=profile['discount']
 assert a.update==0 and not hasattr(a,'_runtime_buffer');assert a.actor_optimizer.state=={} and a.critic_optimizer.state=={}
 assert all(len(o.state)==0 for o in allopts(a));assert a.actor_optimizer.param_groups[0]['lr']==.001
 return dict(algorithm=a,rng=original.rng(),initialization=initialization,profile=profile)
def fallback_state():
 path=O/'inputs/original.pt';assert sha(path)=='886a82e6e84f35e4bfb65d5ccb3f55370c565c23d906740901d5bc5ea95bd2c7';return torch.load(path,map_location='cpu',weights_only=False)['actor_state']
