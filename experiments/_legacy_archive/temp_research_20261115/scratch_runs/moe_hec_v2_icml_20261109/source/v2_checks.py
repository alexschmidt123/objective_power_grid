import copy,time,warnings,torch,numpy as np
from native_common import *
from restore_v1 import restore,V1
from prepare_v2 import construct,calibrate
from migrate_v2 import migrate,phase
from soft_policy import prefix_inputs
from hec_policy import jacobian,mean_signal
from v2_tasks import labels,native_logpost
original.setup();leases=original.acquire();tick=time.monotonic();state=restore();a=state['algorithm'];parent=copy.deepcopy(a.actor_net);gen=torch.Generator(device='cuda').manual_seed(202611099211);ds=a._runtime_buffer['ds_hist'][:60];ys=a._runtime_buffer['ys_hist'][:60];stage=torch.arange(30,device='cuda').repeat(2);x=prefix_inputs(ds,ys,stage);model=construct(a);calibrate(model,x)
with torch.no_grad():
 pa=parent(x);out,info=model(x,collect=True);torch.testing.assert_close(out,pa,rtol=1e-6,atol=1e-6);initial_error=float((out-pa).abs().max());assert bool(((out>=-4)&(out<=4)).all());model.stage_only=True;_,si=model(x,collect=True);assert torch.equal(si['commands'][:30],si['commands'][30:]) and torch.equal(si['gates'][:30],si['gates'][30:]);model.stage_only=False
 tt,receipt=labels(a,parent,model,x,info,gen,4);assert all(torch.isfinite(t).all() for t in tt);assert receipt['predictive_observations']<=len(x)*3*4*2
 # Known exact analytic parameter Jacobian;new E2 uses cross-group Gaussian KL.
 th=torch.randn(8,4,device='cuda',dtype=torch.float64,generator=gen);ac=torch.randn(8,2,device='cuda',dtype=torch.float64,generator=gen)
with torch.enable_grad():
 th.requires_grad_(True);m=mean_signal(th[:,None],ac[:,None])[:,0,0];auto=torch.autograd.grad(m.sum(),th)[0];torch.testing.assert_close(auto,jacobian(th[:,None],ac)[:,0],rtol=1e-10,atol=1e-10)
with torch.no_grad():
 p=info['pair_plus'];q=info['pair_minus'];m=info['pred'];literal=(p[:,None,:,None]*q[:,None,None,:]*(m[:,:,:,None]-m[:,:,None,:]).square()/.5).sum((-1,-2));torch.testing.assert_close(torch.log1p(literal),info['semantic2'][:,:3],rtol=2e-4,atol=1e-5)
 # Perturb only disposable validation tensors to expose all innovation gradients.
 model.response_fusion.weight[:,64:128].add_(.001);model.response_fusion.weight[:,192:256].add_(.001)
phase(model,'C');model.zero_grad();out=model(x);loss=-(out*torch.tensor([1.,-.7],device='cuda')).mean();loss.backward();norms={}
for name in ['history','commander','coordinator','region','discriminate','sensitivity','sequential','adapter2','adapter4','response_gate','response_fusion']:
 module=getattr(model,name);g=[p.grad for p in module.parameters()];assert all(v is not None and torch.isfinite(v).all() for v in g);norms[name]=norm(g);assert norms[name]>0,name
# Full native batch/target path;all optimizers intercepted,matching prior no-update fixture.
moment=a.actor_optimizer.state[a.actor_net.history[0].weight]['exp_avg'].clone();migrate(a,'candidate',False);assert torch.equal(a.actor_optimizer.state[a.actor_net.history[0].weight]['exp_avg'],moment);phase(a.actor_net,'C');a.n_newtraj=4;a.n_critic_update=1;a.n_post_approx_update=1
class Log:
 def __init__(self):self.logger=self;self.tabular=self
 def clear(self):pass
 def record(self,*a):pass
 def log(self,*a):pass
 def push_prefix(self,*a):pass
 def pop_prefix(self,*a):pass
 def dump_all(self):pass
l=Log();a.dowel=l;a.post_approx.dowel=l;a.save_path=None;before=digest(a.actor_net.state_dict());counts=[0];episodes=[0];asses=a.asses
for op in allopts(a):
 def no_step(*args,**kwargs):counts[0]+=1
 op.step=no_step
def simulate(*args,**kwargs):episodes[0]+=int(kwargs.get('n_traj',args[0] if args else 0));return asses(*args,**kwargs)
a.asses=simulate
with torch.device('cuda'),warnings.catch_warnings():
 warnings.simplefilter('ignore');a.train(n_update=14002,n_newtraj=4,n_batch=10000,n_buffer_init=20000,n_buffer_max=1000000,buffer_device='cuda',discount=1.,n_critic_update=1,n_post_approx_update=1,actor_optimizer=a.actor_optimizer,critic_optimizer=a.critic_optimizer,actor_lr_scheduler=a.actor_lr_scheduler,critic_lr_scheduler=a.critic_lr_scheduler,lr_target=.1,design_noise_scale=a.design_noise_scale,design_noise_decay=.9999,on_policy=False,use_PCE=False,transition=0,frozen=-1,dowel=l,save_path=None)
assert counts[0]==32 and episodes[0]==4 and a.update==14002;assert digest(a.actor_net.state_dict())==before;assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.actor_net.parameters());assert len(a._runtime_buffer['params'])==1000000
put(O/'evidence/V2_CHECKS.json',dict(status='PASS',scientific_optimizer_updates=0,initial_candidate_parent_max_error=initial_error,initial_match_checked_before_disposable_head_perturbation=True,stage_commands_and_coordination_bitwise_history_invariant=True,all_expert_and_command_gradients=norms,exact_Jacobian_and_cross_hypothesis_KL=True,valid_two_step_counterfactual_prefixes=True,compatible_Adam_preserved=True,full_native_batch=10000,native_actor_rows=300000,intercepted_steps=counts[0],fixture_episodes=episodes[0],seconds=time.monotonic()-tick,peak_GPU_bytes=torch.cuda.max_memory_allocated()));print('V2_CORE_NATIVE_CHECKS_PASS',flush=True)
