"""No-update mathematical, leakage, gradient, shape and native-loop fixtures."""
import copy,time,math,warnings
from native_common import *
from restore_sg import restore
from prepare_specialists import create_model
from hec_policy import mean_signal,jacobian,advance
from specialist_tasks import targets
from soft_policy import prefix_inputs
from normalized_reward import normalize_native_post
class Log:
 def __init__(self):self.logger=self;self.tabular=self
 def clear(self):pass
 def record(self,*args):pass
 def log(self,*args):pass
 def push_prefix(self,*args):pass
 def pop_prefix(self,*args):pass
 def dump_all(self):pass
original.setup();leases=original.acquire();state=restore();a=state['algorithm'];tick=time.monotonic();model=create_model(a);gen=torch.Generator(device='cuda').manual_seed(202611099111);t=torch.arange(30,device='cuda').repeat(2);ds=torch.randn(60,30,2,device='cuda',generator=gen);ys=torch.randn(60,30,1,device='cuda',generator=gen);x=prefix_inputs(ds,ys,t)
with torch.no_grad():
 out,info=model(x,collect=True);base=model.base(x);assert torch.equal(out,base);assert ((out>=-4)&(out<=4)).all();w,mu,sd=model.belief.params(x);nodes,weights=model.belief.nodes(w,mu,sd);mean=(weights[...,None]*nodes).sum(1);want=(w[...,None]*mu).sum(1);torch.testing.assert_close(mean,want,rtol=1e-5,atol=1e-5);cov=torch.einsum('bn,bni,bnj->bij',weights,nodes-mean[:,None],nodes-mean[:,None]);exact=torch.einsum('bm,bmi,bmj->bij',w,mu-want[:,None],mu-want[:,None])+torch.diag_embed((w[...,None]*sd.square()).sum(1));torch.testing.assert_close(cov,exact,rtol=2e-4,atol=2e-5)
 # Prefix-only posterior packing matches original native inference parameters.
 for tt in [1,12,29]:
  sl=t==tt;net=a.post_approx.poi_post_nets[tt-1][0];inp=a.post_approx.form_input(tt-1,ds[sl,:tt],ys[sl,:tt]);f=net.feature_net(inp);torch.testing.assert_close(w[sl],net.weight_net(f).exp(),rtol=0,atol=0)
 noisy_ds=ds.clone();noisy_ys=ys.clone();mask=torch.arange(30,device='cuda')[None]>=t[:,None];noisy_ds[mask]=100;noisy_ys[mask]=-100;assert torch.equal(x,prefix_inputs(noisy_ds,noisy_ys,t))
 # Distinct commands really change model queries before fusion, even at zero output head.
 commanded,ci=model(x,command_override=torch.full((60,20),.4,device='cuda'),collect=True);assert (ci['F']-info['F']).abs().max()>0;assert (ci['probes']-info['probes']).abs().max()>0;assert (ci['attention']-info['attention']).abs().max()>0
 labels,receipt=targets(model,x,info,gen);assert all(torch.isfinite(v).all() for v in labels)
 # Analytic known-normal KL and sigma=.5 constant are not learned.
 m=info['pred'];p=info['pw'];v=(p[:,None]*(m-(p[:,None]*m).sum(-1,keepdim=True)).square()).sum(-1);pair=(p[:,None,:,None]*p[:,None,None,:]*(m[:,:,:,None]-m[:,:,None,:]).square()/(2*.25)).sum((-1,-2));torch.testing.assert_close(pair,v/.25,rtol=1e-4,atol=1e-4)
 # Exact gradients under the actual two-source log-signal likelihood.
 theta=torch.randn(8,4,device='cuda',generator=gen,dtype=torch.float64);act=torch.randn(8,2,device='cuda',generator=gen,dtype=torch.float64)
with torch.enable_grad():
 theta.requires_grad_(True);signal=mean_signal(theta[:,None],act[:,None])[:,0,0];auto=torch.autograd.grad(signal.sum(),theta)[0];ana=jacobian(theta[:,None],act)[:,0];torch.testing.assert_close(auto,ana,rtol=1e-10,atol=1e-10)
 # Open a disposable nonzero residual head to test all command/expert gradients.
 with torch.no_grad():model.fusion[-1].weight.fill_(.0001)
 model.set_phase('C');out=model(x);loss=-(out*torch.tensor([1.,-.7],device='cuda')).mean();loss.backward();norms={}
 for name,module in [('commander',model.commander),('coordinator',model.coordinator),('history',model.history)]+[(f'expert{k+1}',m) for k,m in enumerate(model.experts())]:
  grads=[p.grad for p in module.parameters()];assert all(v is not None and torch.isfinite(v).all() for v in grads);norms[name]=norm(grads);assert norms[name]>0,name
 # Chunked recomputation preserves the full policy gradient.
 model.zero_grad();model.chunk=17;out=model(x);out.square().mean().backward();chunkgrad=[p.grad.clone() if p.grad is not None else None for p in model.parameters()];model.zero_grad();model.chunk=4096;out=model(x);out.square().mean().backward()
 for p,g in zip(model.parameters(),chunkgrad):
  if g is not None:torch.testing.assert_close(p.grad,g,rtol=2e-3,atol=2e-5)
# Full native 10,000x30 batch and target path, intercept all updates.
from migration import migrate
old_history_weight=a.actor_net.encoder[0].weight;history_moment=a.actor_optimizer.state[old_history_weight]['exp_avg'].clone();migrate(a,True,prepared=False);assert a.actor_net.history[0].weight is old_history_weight;assert torch.equal(a.actor_optimizer.state[old_history_weight]['exp_avg'],history_moment);normalize_native_post(a);a.actor_net.set_phase('C');a.n_newtraj=4;a.n_critic_update=1;a.n_post_approx_update=1;a.dowel=l=Log();a.post_approx.dowel=l;a.save_path=None;before=digest(a.actor_net.state_dict());counts=[0];episodes=[0];asses=a.asses

def count(*args,**kwargs):counts[0]+=1
for op in allopts(a):op.step=count

def sim(*args,**kwargs):episodes[0]+=int(kwargs.get('n_traj',args[0] if args else 0));return asses(*args,**kwargs)
a.asses=sim
with torch.device('cuda'),warnings.catch_warnings():
 warnings.simplefilter('ignore');a.train(n_update=12002,n_newtraj=4,n_batch=10000,n_buffer_init=20000,n_buffer_max=1000000,buffer_device='cuda',discount=1.,n_critic_update=1,n_post_approx_update=1,actor_optimizer=a.actor_optimizer,critic_optimizer=a.critic_optimizer,actor_lr_scheduler=a.actor_lr_scheduler,critic_lr_scheduler=a.critic_lr_scheduler,lr_target=.1,design_noise_scale=a.design_noise_scale,design_noise_decay=.9999,on_policy=False,use_PCE=False,transition=0,frozen=-1,dowel=l,save_path=None)
print('NATIVE_FIXTURE_COUNTS',counts[0],episodes[0],a.update,flush=True)
assert counts[0]==32 and episodes[0]==4 and a.update==12002;assert digest(a.actor_net.state_dict())==before;assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.actor_net.parameters());assert len(a._runtime_buffer['params'])==1000000
put(O/'evidence/CHECKS.json',dict(status='PASS',scientific_optimizer_updates=0,gradient_norms=norms,posterior_prefix_packing_exact=True,prior_quadrature_moments=True,analytic_jacobian_autograd_agreement=True,exact_pair_normal_KL_identity=True,future_history_leakage_test=True,commands_change_model_queries=True,initial_policy_bitwise_SG=True,compatible_history_Adam_bitwise_preserved=True,checkpoint_chunk_gradient_consistency=True,full_native_batch=10000,native_actor_rows=300000,intercepted_native_steps=counts[0],no_extra_replay_initialization=True,seconds=time.monotonic()-tick,peak_bytes=torch.cuda.max_memory_allocated()));print('HEC_ALL_CHECKS_PASS',flush=True)
