import copy,time,io,warnings,torch
from common import *
from v3_policy import V3Actor,proposal_features
from native_v3 import install,instrument_simulator
from prepare import dataset
from v2_tasks import native_logpost

def main():
 original.setup();lease=original.acquire();tick=time.monotonic();s=restore();a=s['algorithm'];model=V3Actor(copy.deepcopy(a.actor_net)).cuda();x,w,cal,score,threshold,idx=dataset(a,model);xx=x[::31][:60]
 with torch.no_grad():
  g,mu,info=model.parts(xx);torch.testing.assert_close(g,torch.full_like(g,.25),rtol=0,atol=0);assert torch.isfinite(mu).all();gen=torch.Generator(device='cuda').manual_seed(CFG['seed']);u=torch.rand(len(xx),device='cuda',generator=gen);eps=torch.randn(len(xx),2,device='cuda',generator=gen);act,di=model.draw(xx,u,eps);lp,r=model.log_prob(act,g,mu);assert torch.isfinite(lp).all();torch.testing.assert_close(r.sum(-1),torch.ones(len(xx),device='cuda'),rtol=1e-5,atol=1e-5);assert bool((act.abs()<4).all())
  # Every expert is feasible at every stage; future saved entries are masked.
  allx=x[::160];assert len(allx)==30
  for k in range(4):assert bool((model.draw(allx,torch.full((30,),.5,device='cuda'),torch.zeros(30,2,device='cuda'),fixed=k)[0].abs()<4).all())
  dd=a._runtime_buffer['ds_hist'][:30].clone();yy=a._runtime_buffer['ys_hist'][:30].clone();tt=torch.arange(30,device='cuda');p=prefix_inputs(dd,yy,tt)
  for j in range(30):dd[j,j:]=999;yy[j,j:]=-999
  torch.testing.assert_close(p,prefix_inputs(dd,yy,tt),rtol=0,atol=0)
  # Independent exact Gaussian pairwise KL and analytic Fisher/Jacobian checks.
  base,summary,nodes,pw,var,bf=model.provider(xx);pr=info['proposals'];pred=mean_signal(nodes,pr);lit=(pw[:,None,:,None]*pw[:,None,None,:]*(pred[:,:,:,None]-pred[:,:,None,:]).square()/.5).sum((-1,-2));torch.testing.assert_close(torch.expm1(info['model_features'][...,1]),lit,rtol=1e-3,atol=1e-5)
  # Activate disposable router/heads for nonzero derivative checks, not training.
  model.router[-1].weight.normal_(0,.01)
  for e in model.experts:e[-1].weight.normal_(0,.01)
  stage=copy.deepcopy(model);stage.stage_only=True;paired=x[160:162];sg=stage.parts(paired)[0];torch.testing.assert_close(sg[0],sg[1],rtol=0,atol=0)
  dynamic=model.parts(paired)[0];assert float((dynamic[0]-dynamic[1]).abs().max())>0
 with torch.enable_grad():
  th=torch.randn(6,4,device='cuda',dtype=torch.float64,requires_grad=True);ac=torch.randn(6,2,device='cuda',dtype=torch.float64);v=mean_signal(th[:,None],ac[:,None])[:,0,0];torch.testing.assert_close(torch.autograd.grad(v.sum(),th)[0],jacobian(th[:,None],ac)[:,0],rtol=1e-10,atol=1e-10)
  g,mu,_=model.parts(xx);ep=torch.randn(len(xx),4,2,device='cuda');acs=4*(mu+model.sigma*ep).tanh();q=(acs*acs.new_tensor([.6,-.4])).sum(-1)-.1*acs.square().sum(-1);ps=list(model.parameters());direct=-(g*q).sum(-1).mean();rb=-(g.detach()*q).sum(-1).mean()-(g*q.detach()).sum(-1).mean();dg=torch.autograd.grad(direct,ps,retain_graph=True);rg=torch.autograd.grad(rb,ps)
  for d,r in zip(dg,rg):torch.testing.assert_close(d,r,rtol=2e-4,atol=2e-6)
  norms={};offset=0
  for name,mod in [('encoder',model.encoder),('experts',model.experts),('router',model.router)]:
   params=list(mod.parameters());gr=dg[offset:offset+len(params)];offset+=len(params);norms[name]=norm(gr);assert norms[name]>0 and all(torch.isfinite(v).all() for v in gr)
  # State/RNG round trip for the actual stochastic actor.
 with torch.no_grad():
  buf=io.BytesIO();torch.save(model.state_dict(),buf);buf.seek(0);clone=copy.deepcopy(model);clone.load_state_dict(torch.load(buf,map_location='cuda',weights_only=False));gen=torch.Generator(device='cuda').manual_seed(873);state=gen.get_state();uu=torch.rand(len(xx),device='cuda',generator=gen);ee=torch.randn(len(xx),2,device='cuda',generator=gen);one=model.draw(xx,uu,ee)[0];gen.set_state(state);two=clone.draw(xx,torch.rand(len(xx),device='cuda',generator=gen),torch.randn(len(xx),2,device='cuda',generator=gen))[0];assert torch.equal(one,two)
 prevalence={str(t):dict(validation_group_fractions=((score[(x[:,:30].argmax(-1)==t)&~cal]>threshold[t]).float().mean(0)).tolist(),threshold=threshold[t].tolist()) for t in range(30)}
 put(O/'evidence/CONDITION_PREFLIGHT.json',dict(status='PASS',train_prefixes=int(cal.sum()),validation_prefixes=int((~cal).sum()),every_stage_represented=True,prior_stage_ties_not_arbitrarily_split=True,prevalence=prevalence,not_proof_of_design_competence=True))
 # Full native batch with actual V3 actor/target path; no optimizer may step.
 del model,stage,clone;torch.cuda.empty_cache();moment=a.actor_optimizer.state[a.actor_net.history[0].weight]['exp_avg'].clone();migration=install(a,False);assert torch.equal(moment,a.actor_optimizer.state[a.actor_net.encoder[0].weight]['exp_avg']);a.n_newtraj=4;a.n_critic_update=1;a.n_post_approx_update=1
 class Log:
  def __init__(self):self.logger=self;self.tabular=self
  def clear(self):pass
  def record(self,*a):pass
  def log(self,*a):pass
  def push_prefix(self,*a):pass
  def pop_prefix(self,*a):pass
  def dump_all(self):pass
 sim_hooks,sim_counts=instrument_simulator(a);l=Log();a.dowel=l;a.post_approx.dowel=l;a.save_path=None;before=digest(a.actor_net.state_dict());steps=[0];episodes=[0];asses=a.asses
 for op in allopts(a):
  def nostep(*args,**kwargs):steps[0]+=1
  op.step=nostep
 def sim(*args,**kwargs):episodes[0]+=int(kwargs.get('n_traj',args[0] if args else 0));return asses(*args,**kwargs)
 a.asses=sim
 with torch.device('cuda'),warnings.catch_warnings():
  warnings.simplefilter('ignore');a.train(n_update=16002,n_newtraj=4,n_batch=10000,n_buffer_init=20000,n_buffer_max=1000000,buffer_device='cuda',discount=1.,n_critic_update=1,n_post_approx_update=1,actor_optimizer=a.actor_optimizer,critic_optimizer=a.critic_optimizer,actor_lr_scheduler=a.actor_lr_scheduler,critic_lr_scheduler=a.critic_lr_scheduler,lr_target=.1,design_noise_scale=0.,design_noise_decay=.9999,on_policy=False,use_PCE=False,transition=0,frozen=-1,dowel=l,save_path=None)
 assert sim_counts==dict(episodes=4,observations=120,batch_calls=30);assert steps[0]==32 and episodes[0]==4 and a.update==16002;assert digest(a.actor_net.state_dict())==before;assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.actor_net.parameters());assert len(a._runtime_buffer['params'])==1000000
 with torch.no_grad():
  terminal=native_logpost(a,a.ds_hist,a.ys_hist,a.params,torch.full((len(a.params),),30,device='cuda'));prior=native_logpost(a,a.ds_hist,a.ys_hist,a.params,torch.zeros(len(a.params),device='cuda',dtype=torch.long));torch.testing.assert_close(a.rewards_hist.sum(-1),terminal-prior,rtol=1e-4,atol=2e-4)
 put(O/'evidence/IMPLEMENTATION_CHECKS.json',dict(status='PASS',seconds=time.monotonic()-tick,scientific_optimizer_steps=0,fixture_episodes=episodes[0],fixture_actual_simulator_counts=sim_counts,intercepted_optimizer_calls=steps[0],full_native_batch=10000,actor_history_rows=300000,all_30_stages_feasible_all_4_experts=True,stage_gate_bitwise_history_invariant=True,model_informed_router_full_chain_gradient_identity=True,model_feature_likelihood_and_Jacobian_checks=True,no_future_prefix_leak=True,checkpoint_RNG_repeat_bitwise=True,full_horizon_native_telescope=True,compatible_encoder_Adam_preserved=True,gradient_norms=norms,peak_GPU_bytes=torch.cuda.max_memory_allocated(),transformed_native_train_sha256=migration['source_sha256']));(O/'evidence/NATIVE_TRAIN_TRANSFORMED.py.txt').write_text(migration['source']);print('V3_IMPLEMENTATION_CHECKS_PASS',flush=True)
if __name__=='__main__':main()
