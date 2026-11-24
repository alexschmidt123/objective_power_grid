"""Run only after upstream exits and watcher verifies idleGPU;no optimizer steps."""
import sys,time
sys.dont_write_bytecode=True
import torch
from native_common import *
from fresh_common import build_fresh
from soft_policy import install
original.setup();leases=original.acquire();a=build_fresh('cuda')['algorithm'];install(a,enabled=False);torch.save(dict(actor_state={k:v.detach().cpu() for k,v in a.actor_net.state_dict().items()},actor_digest=digest(a.actor_net.state_dict()),critic_digest=digest(a.critic_net.state_dict())),O/'inputs/fresh_soft_initial.pt');assert digest(a.actor_net.state_dict())==torch.load(O/'inputs/fresh_soft_initial_CPU.pt',map_location='cpu',weights_only=False)['actor_digest'];torch.cuda.reset_peak_memory_stats();x=torch.randn(10000,30,117,device='cuda');loss=-a.critic_net(torch.cat([x,a.actor_net(x)],-1)).mean();loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for name,p in a.actor_net.named_parameters() if not name.startswith('task_heads.'));torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated();assert peak<22*1024**3;assert all(not op.state for op in allopts(a));
from evaluate_soft import actor_load,simulate
from native_reward import stateless_rewards
with torch.no_grad():
 outside,_=actor_load('original');reference,_=actor_load('reference');g=torch.Generator(device='cuda').manual_seed(202611088301);truth=torch.randn(4,4,device='cuda',generator=g);noise=torch.randn(4,30,device='cuda',generator=g);u=torch.rand(4,30,device='cuda',generator=g);row=simulate(a.actor_net,truth,noise,u,'candidate');ref=simulate(reference,truth,noise,u,'reference',outside=outside);a.params=truth;a.ds_hist=torch.stack(row['actions'],1);a.ys_hist=torch.stack(row['observations'],1);a.rewards_hist=stateless_rewards(a,a.ds_hist,a.ys_hist,truth)
a._soft_tasks.add_gradient();assert a._soft_tasks.last['scale']==0
put(O/'evidence/GPU_PREFLIGHT.json',dict(status='PASS',training_updates=0,reference_and_soft_full_horizon_fixture_observations=240,GPU_auxiliary_checked=True,batch=10000,peak_bytes=peak,actor_parameters=sum(p.numel() for p in a.actor_net.parameters()),critic_parameters=sum(p.numel() for p in a.critic_net.parameters()),GPU=torch.cuda.get_device_name()));print('SOFT_NATIVE_BATCH_GPU_PREFLIGHT_PASS')
