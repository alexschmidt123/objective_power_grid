"""CPU essential correctness,no training or scientific bank."""
import os,sys,copy,inspect,time
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
import torch,numpy as np
from native_common import *
from fresh_common import build_fresh
from soft_policy import install,prefix_inputs,discounted_returns
beg=time.monotonic();torch.set_num_threads(2);a=build_fresh()['algorithm'];rng=torch.get_rng_state().clone();src=install(a,enabled=False);assert torch.equal(rng,torch.get_rng_state());(O/'source/native_train_snapshot.py.txt').write_text(src);torch.save(dict(actor_state=a.actor_net.state_dict(),actor_digest=digest(a.actor_net.state_dict())),O/'inputs/fresh_soft_initial_CPU.pt');assert len({digest(e.state_dict()) for e in a.actor_net.experts})==4
b=8;ds=torch.randn(b,30,2).clamp(-4,4);ys=torch.randn(b,30,1);theta=torch.randn(b,4);rewards=torch.randn(b,31);stages=torch.tensor([0,3,5,9,12,18,24,29]);x=prefix_inputs(ds,ys,stages)
for i,t in enumerate(stages.tolist()):torch.testing.assert_close(x[i:i+1],a.form_actor_input(t,ds[i:i+1,:t],ys[i:i+1,:t]),atol=0,rtol=0)
d2=ds.clone();y2=ys.clone()
for i,t in enumerate(stages):d2[i,t:]+=3;y2[i,t:]-=4
torch.testing.assert_close(x,prefix_inputs(d2,y2,stages),atol=0,rtol=0);task=a._soft_tasks;targets=task.labels(ds,ys,theta,stages,rewards);other=task.labels(d2,y2,theta,stages,rewards);torch.testing.assert_close(targets[1],other[1],rtol=0,atol=0)
# Source-label symmetry and the native terminal convention.
swapped=theta.reshape(b,2,2).flip(1).reshape(b,4);swap=task.labels(ds,ys,swapped,stages,rewards)
for k in [0,2]:torch.testing.assert_close(targets[k],swap[k],rtol=0,atol=0)
v=discounted_returns(rewards);assert torch.equal(v[:,29],rewards[:,29]+rewards[:,30]);torch.testing.assert_close(v[:,24],sum((.9**(t-24))*rewards[:,t] for t in range(24,30))+.9**5*rewards[:,30])
for k in range(4):
 a.actor_net.zero_grad(set_to_none=True);losses,_=task.losses(x,targets);losses[k].backward();norms=[norm([p.grad for p in e.parameters()]) for e in a.actor_net.experts];assert norms[k]>0 and all(vv==0 for j,vv in enumerate(norms) if j!=k);assert all(p.grad is None for module in [a.actor_net.encoder,a.actor_net.router,a.actor_net.decision] for p in module.parameters())
a.actor_net.zero_grad(set_to_none=True);out=a.actor_net(x);assert ((out>=-4)&(out<=4)).all();loss=-a.critic_net(torch.cat([x,out],-1)).mean();loss.backward();assert all(norm([p.grad for p in module.parameters()])>0 for module in [a.actor_net.encoder,a.actor_net.router,a.actor_net.decision,*a.actor_net.experts]);g=a.actor_net.parts(x)[2];assert (g>0).all();torch.testing.assert_close(g.sum(-1),torch.ones(b));a.params=theta;a.ds_hist=ds;a.ys_hist=ys;a.rewards_hist=rewards
before=[None if p.grad is None else p.grad.clone() for p in a.actor_net.parameters()];task.add_gradient()
for p,z in zip(a.actor_net.parameters(),before):assert (p.grad is None) if z is None else torch.equal(p.grad,z)
assert all(not op.state for op in allopts(a));put(O/'evidence/SOFT_TEST_RESULTS.json',dict(status='PASS',all_four_experts_native_gradient=True,router_and_decision_gradient=True,task_gradient_isolation=True,encoder_detached_from_task_losses=True,zero_auxiliary_exact=True,no_future_input_leak=True,hypothesis_labels_prefix_only=True,source_label_permutation_invariant=True,native_discounted_terminal_convention=True,action_bounds=True,outer_RNG_preserved=True,training_updates=0,seconds=time.monotonic()-beg));print('SOFT_ESSENTIAL_TESTS_PASS')
