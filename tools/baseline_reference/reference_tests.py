"""Executable official-source comparisons. CPU only; no campaign GPU contention."""
import os,sys,ast,json,math,copy,importlib.util,traceback,tempfile
from pathlib import Path
from types import SimpleNamespace
ROOT=Path('/home/qiqi/Documents/objective_power_grid_baseline_fidelity_20261017');C=Path('/home/qiqi/Documents/objective_power_grid/experiments/10172026/10172026_original_baseline_fidelity');ROOT=Path(os.environ.get('BASELINE_CHECKOUT',str(ROOT)));C=Path(os.environ.get('BASELINE_AUDIT',str(C)));REF=C/'upstream_references'
sys.path[:0]=[str(ROOT),str(C/'reference_dependencies')]
import numpy as np,torch
from torch import nn
from src.research.benchmarks import LocationFinding
from src.research.engine import rollout,generator
from src.research.policies import history_features
from src.research.dad_location_policy import LocationDAD
from src.research.native_rl_reference import TYPES,make_actor,make_critic,NativeRLPolicy,make_algorithm,native_history,load_classes
import pyro,pyro.distributions as dist
from pyro import poutine
from pyro.contrib.util import lexpand,iter_plates_to_shape
from contextlib import ExitStack
from abc import ABC
import torch.nn.functional as F
B=LocationFinding({'location_finding':{'design_bounds':[-4,4]},'observation':{'noise_sigma':.5}})
torch.set_num_threads(2);rows=[]
def test(name,fn):
 try:value=fn();rows.append(dict(test=name,status='PASS',evidence=value))
 except Exception:rows.append(dict(test=name,status='FAIL',traceback=traceback.format_exc()))
 print(rows[-1],flush=True)
def env():return dict(torch=torch,nn=nn,math=math,np=np,pyro=pyro,dist=dist,lexpand=lexpand,poutine=poutine,ExitStack=ExitStack,iter_plates_to_shape=iter_plates_to_shape,ABC=ABC,EPS=2**-22,is_bad=lambda x:not torch.isfinite(x).all())
def dad_reference():
 e=env();load_classes(REF/'dad/location_finding.py',['EncoderNetwork','EmitterNetwork','HiddenObjects'],e);load_classes(REF/'dad/neural/modules.py',['SetEquivariantDesignNetwork'],e)
 enc=e['EncoderNetwork']((1,2),1,256,16).double();em=e['EmitterNetwork'](16,(1,2)).double();net=e['SetEquivariantDesignNetwork'](enc,em,torch.ones(1,2,dtype=torch.float64)*.01)
 local=LocationDAD(5,4,3,first_design_revision=1)
 with torch.no_grad():
  for a,b in [(local.encoder[0],enc.linear1),(local.encoder[2],enc.output_layer),(local.emitter,em.linear)]:a.weight.copy_(b.weight);a.bias.copy_(b.bias)
 return e,net,local

def dad_test():
 torch.manual_seed(11);e,ref,p=dad_reference();rng=generator(101);truth=B.sample((3,),rng);particles=B.sample((3,7),rng);noise=torch.randn(3,5,1,generator=rng,dtype=torch.float64)
 a=rollout(B,p,5,3,1001,7,truth=truth,noise=noise,prior_particles=particles)
 theta=torch.cat((truth[:,None],particles),1).reshape(3,8,2,2)
 pairs=[];cum=torch.zeros(3,8,dtype=torch.float64);actions=[];logprobs=[]
 model=object.__new__(e['HiddenObjects']);model.base_signal=.1;model.max_signal=1e-4
 for t in range(5):
  raw=ref(*pairs).reshape(-1,2)
  if t==0:raw=raw.expand(3,-1)
  action=B.feasible(raw,[],5);mean=e['HiddenObjects'].forward_map(model,action[:,None,None,:],theta)
  y=mean[:,0]+.5*noise[:,t];lp=dist.Normal(mean,.5).log_prob(y[:,None]).sum(-1);logprobs.append(lp);cum=cum+lp
  actions.append(action);pairs.append((action[:,None,:],y))
 torch.testing.assert_close(torch.stack(actions),torch.stack(a['actions']),rtol=1e-11,atol=1e-11)
 torch.testing.assert_close(cum,a['log_likelihood'],rtol=1e-10,atol=1e-10)
 # Execute the original loss body against deterministic trace log-probabilities.
 le=env();le['warn_if_nan']=lambda *x:None;load_classes(REF/'dad/contrastive/mi.py',['MutualInformation','PriorContrastiveEstimation'],le)
 def trace(primary):return SimpleNamespace(nodes={str(t):dict(subtype='observation_sample',log_prob=lp[:,0] if primary else lp[:,1:].T) for t,lp in enumerate(logprobs)})
 lossfn=le['PriorContrastiveEstimation'](3,7);lossfn.get_primary_rollout=lambda *args,**kw:trace(True);lossfn.get_contrastive_rollout=lambda *args,**kw:trace(False)
 native_loss=lossfn.differentiable_loss(None)-math.log(8);loss=-a['info'][:,-1].mean();torch.testing.assert_close(native_loss,loss,rtol=1e-10,atol=1e-10)
 native_loss.backward();loss.backward()
 pairs=[(ref.encoder.linear1,p.encoder[0]),(ref.encoder.output_layer,p.encoder[2]),(ref.emitter.linear,p.emitter)]
 for x,y in pairs:
  for n in ['weight','bias']:torch.testing.assert_close(getattr(x,n).grad,getattr(y,n).grad,rtol=1e-8,atol=1e-9)
 oa=torch.optim.Adam(ref.parameters(),lr=5e-5,betas=(.8,.998));ob=torch.optim.Adam(p.parameters(),lr=5e-5,betas=(.8,.998));oa.step();ob.step()
 for x,y in pairs:
  for n in ['weight','bias']:torch.testing.assert_close(getattr(x,n),getattr(y,n),rtol=1e-9,atol=1e-10)
 return dict(actions_likelihood_loss_gradients_one_Adam_step=True,rtol=1e-8,atol=1e-9,scope='Exact upstream class/loss bodies; deterministic trace harness replaces stochastic trace construction only; common bounded mapping applied to both')

def rl_physics():
 e=env();load_classes(REF/'rl_boed/pyro/models/adaptive_experiment_model.py',['ExperimentModel','SourceModel'],e)
 m=e['SourceModel'](n_parallel=3);th=torch.randn(3,1,2,2);x=torch.randn(3,1,1,2);y=torch.randn(3,1,1)
 fn=pyro.condition(m.make_model(),data={'theta':th,'y':y});tr=poutine.trace(fn).get_trace(x);tr.compute_log_prob();lp=tr.nodes['y']['log_prob'].reshape(3)
 mean,_=B.propagate(th.reshape(3,4).double(),B.initial((3,)),x.reshape(3,2).double());expected=B.log_likelihood(y.reshape(3,1).double(),mean)
 torch.testing.assert_close(lp.double(),expected,rtol=3e-5,atol=3e-5)
 return dict(native_Pyro_source_likelihood=True,max_absolute_error=float((lp-expected).abs().max()),prior='four independent standard-normal coordinates')

def rl_network_rewards_update():
 torch.manual_seed(12);p=NativeRLPolicy(5);algo=make_algorithm(p,5)
 f=torch.randn(8,21,dtype=torch.float64);slots=f[:,1:].reshape(8,5,4);slots[...,3]=1;slots[:,3:,3]=0
 x,mask=native_history(f,5);expected=torch.cat(((slots[...,:2]*4+4)/8,(slots[...,2:3]*3+3)/13),-1).float();torch.testing.assert_close(x,expected,rtol=0,atol=0)
 native_dist,_=p.native(x,mask);g=torch.Generator().manual_seed(99);a,lp=p.sample(f,rng=g)
 z=native_dist._normal.base_dist.loc+native_dist._normal.base_dist.scale*torch.randn(8,2,generator=torch.Generator().manual_seed(99))
 torch.testing.assert_close(a,z.tanh(),rtol=0,atol=0);torch.testing.assert_close(lp,native_dist.log_prob(a,pre_tanh_value=z),rtol=0,atol=0)
 # Independent direct source load of update class; identical state and draws.
 scope=dict(torch=torch,nn=nn,F=F,np=np,copy=copy,deque=__import__('collections').deque,RLAlgorithm=object)
 cls=load_classes(REF/'rl_boed/pyro/algos/sac.py',['SAC'],scope)[0]
 reference=copy.deepcopy(algo);reference.__class__=cls
 data=dict(observation=x,mask=mask,next_observation=x,next_mask=mask,action=a.detach(),reward=torch.randn(8,1),terminal=torch.zeros(8,1))
 torch.manual_seed(77);np.random.seed(77);first=algo.optimize_policy(data);algo._update_targets()
 torch.manual_seed(77);np.random.seed(77);second=reference.optimize_policy(data);reference._update_targets()
 for n,v in algo.policy.state_dict().items():torch.testing.assert_close(v,reference.policy.state_dict()[n],rtol=0,atol=0)
 for left,right in zip(algo._qfs,reference._qfs):
  for n,v in left.state_dict().items():torch.testing.assert_close(v,right.state_dict()[n],rtol=0,atol=0)
 torch.testing.assert_close(algo._log_alpha,reference._log_alpha,rtol=0,atol=0)
 # Original factorized reward body against common cumulative PCE.
 scope=dict(torch=torch,Env=object,LOWER=0,UPPER=1,TERMINAL=2);cls=load_classes(REF/'rl_boed/pyro/envs/adaptive_design_env.py',['AdaptiveDesignEnv'],scope)[0]
 likelihood=torch.randn(5,8,6,dtype=torch.float64);state=object.__new__(cls);state.bound_type=0;state.log_products=torch.zeros(8,6,dtype=torch.float64);state.last_logsumprod=torch.logsumexp(state.log_products,0);state.thetas=None
 rewards=[]
 for lp0 in likelihood:
  state.model=SimpleNamespace(get_likelihoods=lambda *a,lp0=lp0:lp0[...,None]);rewards.append(state.get_reward(None,None))
 scores=likelihood.sum(0);terminal=scores[0]-torch.logsumexp(scores,0)+math.log(8)
 torch.testing.assert_close(torch.stack(rewards).sum(0),terminal,rtol=1e-12,atol=1e-12)
 return dict(native_network_outputs_distribution_logprob_and_one_full_update=True,native_factorized_reward=True,updated_parameters_bitwise_equal=True,discount=.9,critics=2,subset=2)

def vsoed_test():
 sys.path.insert(0,str(REF/'vsoed'));import vsOED as original
 from vsOED.models import SOURCE as RefSource
 from src.research.vendor.vsoed.models import SOURCE
 from src.research.upstream_vsoed import train_upstream_vsoed,Logger
 from src.research.vendor.vsoed import PGvsOED,POST_APPROX
 th=torch.randn(8,4);x=torch.randn(8,2);aa,sa=SOURCE().deterministic(th,x);zz,sz=RefSource().deterministic(th,x)
 torch.testing.assert_close(aa,zz,rtol=0,atol=0);torch.testing.assert_close(aa.double(),B.propagate(th.double(),B.initial((8,)),x.double())[0],rtol=1e-6,atol=1e-6)
 profile=dict(updates=1,new_trajectories=3,replay_batch=4,initial_replay=8,replay_capacity=16,posterior_updates=1,critic_updates=1,actor_lr=.001,critic_lr=.001,posterior_lr=.001,lr_decay=.9999,discount=.9,exploration=.5,exploration_decay=.9999,target_rate=.1,transition=0,log_every=1,n_incre=3)
 with tempfile.TemporaryDirectory() as d:
  p,t=train_upstream_vsoed(B,3,{},101,Path(d),profile)
  saved=torch.load(Path(d)/'vsoed.pt',weights_only=False)
  torch.manual_seed(101);np.random.seed(101);model=RefSource();logger=Logger(1)
  post=original.POST_APPROX(n_stage=3,n_design=2,n_obs=1,n_pois=[4],model_weight=0,poi_weight=1,goal_weight=0,n_mixture=8,mu_bounds={'poi':[[[-6,6]]*4]},max_sigmas={'poi':[[1.]*4]},truncnorm_info={'poi':[None]},prior=model,n_incre=3,share_interm_net=False,use_NFs=False,activate=nn.ReLU,poi_post_lrs=[.001]*3,poi_post_lr_scheduler_gammas=[.9999]*3,log_every=1,dowel=logger)
  alg=original.PGvsOED(3,4,2,1,model,model,[(-4.,4.)]*2,model.reward_fun,post.reward_fun,post_approx=post,actor_dimns=[256]*3,critic_dimns=[256]*3,activate=nn.ReLU)
  ao=torch.optim.Adam(alg.actor_net.parameters(),lr=.001);co=torch.optim.Adam(alg.critic_net.parameters(),lr=.001)
  alg.train(n_update=1,n_newtraj=3,n_batch=4,n_buffer_init=8,n_buffer_max=16,buffer_device=torch.device('cpu'),discount=.9,n_critic_update=1,n_post_approx_update=1,actor_optimizer=ao,critic_optimizer=co,actor_lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(ao,.9999),critic_lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(co,.9999),lr_target=.1,design_noise_scale=.5,design_noise_decay=.9999,transition=0,on_policy=False,use_PCE=False,frozen=-1,log_every=1,dowel=logger,save_path=None)
  for n,v in saved['actor_state'].items():torch.testing.assert_close(v,alg.actor_net.state_dict()[n],rtol=0,atol=0)
  for n,v in saved['critic_state'].items():torch.testing.assert_close(v,alg.critic_net.state_dict()[n],rtol=0,atol=0)
  for k,nets in post.poi_post_nets.items():
   for i,net in enumerate(nets):
    for n,v in saved['posterior_states'][str(k)][i].items():torch.testing.assert_close(v,net.state_dict()[n],rtol=0,atol=0)
 return dict(native_model_actor_critic_GMM_posterior_one_update=True,all_parameters_bitwise_equal=True,model_discrepancy='Executable upstream SOURCE applies natural log to signal; shared benchmark follows executable source')
for name,fn in [('DAD_actions_physics_loss_gradients_update',dad_test),('RL_native_Pyro_physics',rl_physics),('RL_networks_rewards_native_update',rl_network_rewards_update),('vsOED_native_full_update',vsoed_test)]:test(name,fn)
(C/'baseline_reference_tests.json').write_text(json.dumps(dict(status='PASS' if all(x['status']=='PASS' for x in rows) else 'FAIL',tests=rows,ScoreBED='Separate JAX tests pending'),indent=2))
raise SystemExit(any(x['status']=='FAIL' for x in rows))
