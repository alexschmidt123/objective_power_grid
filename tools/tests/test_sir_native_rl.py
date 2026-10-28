import json
import pytest
import torch
from src.research.benchmarks import SIR
from src.research.sir_native_rl import NativeSIRRLPolicy,algorithm,train_sir_native_rl
from src.research.engine import rollout

def benchmark():return SIR({'sir_ode':{'integration_steps_per_stage':512},'observation':{'noise_sigma':1.}})

def test_native_sir_observed_history_and_replay_one_update(tmp_path):
 torch.set_num_threads(2);b=benchmark();p,t=train_sir_native_rl(b,5,dict(updates=2,rl_native_contrasts=8,rl_native_replay_capacity=64),101,tmp_path,4)
 assert t['generated_training_trajectories']==4 and t['gradient_updates']==2 and t['gamma']==.9
 r=rollout(b,p,5,3,1001,8);a=torch.stack(r['actions'],1)[...,0]
 assert (a[:,0]>=b.gap).all() and (a[:,-1]<=b.end).all() and ((a[:,1:]-a[:,:-1])>=b.gap-1e-10).all()
 assert torch.isfinite(r['info']).all()
 saved=torch.load(tmp_path/'rl_sboed.pt',map_location='cpu',weights_only=False);q=NativeSIRRLPolicy(5,b);q.load_state_dict(saved['state_dict'])
 z=rollout(b,q,5,3,1001,8)
 torch.testing.assert_close(torch.stack(r['actions']),torch.stack(z['actions']),atol=0,rtol=0)

def test_native_unit_action_mapping_inverts_without_hidden_state():
 b=benchmark();p=NativeSIRRLPolicy(5,b);u=torch.tensor([[-.6],[.4]],dtype=torch.float32)
 actions=[torch.tensor([[10.],[25.]],dtype=torch.float64)]
 t=p.design_from_unit_action(u,actions,5);lo=actions[-1]+b.gap;hi=b.end-3*b.gap
 torch.testing.assert_close(2*(t-lo)/(hi-lo)-1,u.double(),atol=1e-12,rtol=1e-12)
