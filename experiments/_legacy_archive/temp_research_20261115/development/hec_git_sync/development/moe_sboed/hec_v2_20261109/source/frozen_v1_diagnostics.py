import time,numpy as np,torch
from native_common import *
from restore_v1 import restore,V1
original.setup();leases=original.acquire();tick=time.monotonic();state=restore();actor=state['algorithm'].actor_net.eval().requires_grad_(False);hist=np.load(V1/'evaluation/diagnostic_candidate_histories.npz');raw=np.load(V1/'evaluation/competency_raw.npz');stages=[3,12,24];X=torch.tensor(hist['X'][256:288][:,stages],device='cuda').reshape(-1,117);stagec=torch.tensor(hist['commands'][:256].mean(0)[stages],device='cuda')[None].expand(32,3,20).reshape(-1,20)
with torch.no_grad():
 a,z=actor(X,collect=True);zero,_=actor(X,command_override=torch.zeros_like(z['commands']),collect=True);stage,_=actor(X,command_override=stagec,collect=True);effects={}
 for k,sl in enumerate([slice(0,4),slice(4,10),slice(10,16),slice(16,20)]):
  c=z['commands'].clone();c[:,sl]=(c[:,sl]+.25).clamp(-.95,.95);aa,zz=actor(X,command_override=c,collect=True);effects[str(k+1)]=dict(feature_RMS_change=float((zz['features'][:,k]-z['features'][:,k]).square().mean().sqrt()),action_RMS_change=float((aa-a).square().mean().sqrt()))
 cka={}
 for k in range(4):
  for j in range(k+1,4):
   vals=[]
   for t in range(3):
    x=z['features'].reshape(32,3,4,64)[:,t,k];y=z['features'].reshape(32,3,4,64)[:,t,j];x=x-x.mean(0);y=y-y.mean(0);vals.append(float((x.T@y).square().sum()/((x.T@x).norm()*(y.T@y).norm()).clamp_min(1e-30)))
   cka[f'{k+1}-{j+1}']=vals
assert all(v['feature_RMS_change']>0 for v in effects.values())
put(O/'evidence/FROZEN_V1_DIAGNOSTICS.json',dict(status='COMPLETE',histories=32,stages=stages,expert_command_effects=effects,zero_action_RMS=float((zero-a).square().mean().sqrt()),stage_action_RMS=float((stage-a).square().mean().sqrt()),cross_task_normalized_MSE={m:raw[f'{m}_probe_losses'][:,:,256:].mean((2,3)).tolist() for m in ['candidate','prepared']},within_stage_CKA=cka,training_updates=0,new_rollouts=0,functional_not_EIG=True,seconds=time.monotonic()-tick));print('FROZEN_V1_DIAGNOSTICS_COMPLETE')
