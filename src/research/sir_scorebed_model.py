"""SIR ODE adapter for pinned official ScoreBED, matching Torch RK4 numerics.

Dynamic RK4 subdivision is nondifferentiable; derivatives hold its integer
count fixed, exactly as the common Torch simulator. A custom VJP caches RK4
sensitivities and does not differentiate backward through dynamic loops.
"""
import math
import jax,jax.numpy as jnp
from policy_learning.bed_models import LocationFinding
from policy_learning.bed_models.double_pendulum import DoublePendulum
from policy_learning.policies import DADNet,IqbalGRUNet
jax.config.update('jax_enable_x64',True)

def make_evolve(population,minimum_steps):
 def rhs(s,theta):
  rate=theta[0]*s[0]*s[1]/population;recovery=theta[1]*s[1]
  return jnp.array([-rate,rate-recovery,recovery])
 def rk(s,theta,dt):
  a=rhs(s,theta);b=rhs(s+dt*a/2,theta);c=rhs(s+dt*b/2,theta);d=rhs(s+dt*c,theta)
  return s+dt*(a+2*b+2*c+d)/6
 def count(theta,duration):return jax.lax.stop_gradient(jnp.maximum(minimum_steps,jnp.ceil(duration*jnp.sum(theta)/.2).astype(jnp.int32)))
 def primal(theta,initial,duration):
  n=count(theta,duration);dt=duration/n
  def body(c):return c[0]+1,rk(c[1],theta,dt)
  _,s=jax.lax.while_loop(lambda c:c[0]<n,body,(0,initial))
  return jnp.where((duration>=0)&(n<=20000),s,jnp.nan)
 def sensitivities(theta,initial,duration):
  n=count(theta,duration);dt=duration/n
  J=jnp.concatenate((jnp.eye(3),jnp.zeros((3,3))),1)
  def body(c):
   i,s,J=c
   A,B,C=jax.jacfwd(rk,argnums=(0,1,2))(s,theta,dt)
   J=A@J;J=J.at[:,3:5].add(B);J=J.at[:,5].add(C/n)
   return i+1,rk(s,theta,dt),J
  _,s,J=jax.lax.while_loop(lambda c:c[0]<n,body,(0,initial,J))
  valid=(duration>=0)&(n<=20000)
  return jnp.where(valid,s,jnp.nan),jnp.where(valid,J,jnp.nan)
 @jax.custom_vjp
 def evolve(theta,initial,duration):return primal(theta,initial,duration)
 def fwd(theta,initial,duration):
  s,J=sensitivities(theta,initial,duration);return s,J
 def bwd(J,cotangent):
  v=J.T@cotangent
  return v[3:5],v[:3],v[5]
 evolve.defvjp(fwd,bwd)
 return evolve

class SIRScoreModel(LocationFinding):
 def __init__(self,T=5,population=500.,i0=2.,t_end=100.,min_spacing=.1,integration_steps_per_stage=512,beta_mean=.5,gamma_mean=.1,log_std=.5,sigma=1.,init_seed=101,**kwargs):
  super().__init__(T=T,d=1,k=2,sigma=sigma,init_seed=init_seed,standardise=False,dtype=jnp.float64)
  self.theta_shape=(2,);self.theta_dim=2
  self.population=population;self.i0=i0;self.end=t_end;self.gap=min_spacing
  self.log_mean=jnp.log(jnp.array([beta_mean,gamma_mean]));self.log_std=log_std
  self.evolve=make_evolve(population,integration_steps_per_stage)
  # Sampler-only reference object: never construct or simulate a pendulum.
  # Native correlated samplers require2channels; retain both and use their
  # first-coordinate marginal (bound4) for the scalar SIR time action.
  self.reference_sampler=object.__new__(DoublePendulum)
  for key,value in dict(T=T,d=2,dtype=jnp.float64,design_bounds=jnp.array([4.,2.]),EPS=1e-6,_policy_parameterisation='x').items():setattr(self.reference_sampler,key,value)
  self.reference_sampler.design_bijector=lambda x:jnp.clip(x,-self.reference_sampler.design_bounds,self.reference_sampler.design_bounds)
  self.reference_sampler.design_bijector_inv=lambda x:x
 def initial(self,batch_shape=()):return jnp.broadcast_to(jnp.array([self.population-self.i0,self.i0,0.]),(*batch_shape,3))
 def feasible(self,raw,previous,stage):
  lower=previous+self.gap;upper=self.end-(self.T-stage-1)*self.gap
  return lower+jax.nn.sigmoid(raw)*(upper-lower)
 def get_policy_output_activation(self,activation_type):return lambda x:x
 def sample_prior(self,key,batch_shape):
  key,sub=jax.random.split(key)
  theta=jnp.exp(self.log_mean+self.log_std*jax.random.normal(sub,(*batch_shape,1,2),dtype=jnp.float64))
  return jnp.broadcast_to(theta,(*batch_shape,self.T,2)),key
 def sample_designs(self,key,batch_shape,theta=None,sampler_type=None,sampler_kwargs=None):
  # Draw the pinned Gaussian sampler in unconstrained coordinates, then apply
  # the same causal feasible-time map as the policy. No latent parameters used.
  sampler_type=sampler_type or 'uniform'
  sampler_kwargs=dict(sampler_kwargs or {})
  raw,aux,key=self.reference_sampler.sample_designs(key,batch_shape,theta=None,sampler_type=sampler_type,sampler_kwargs=sampler_kwargs)
  last=jnp.zeros(batch_shape);times=[]
  for t in range(self.T):
   last=self.feasible(raw[...,t,0],last,t);times.append(last)
  return jnp.stack(times,-1)[...,None],{},key
 def _get_cached_fast_design_sampler(self,sampler_type,sampler_kwargs):
  from functools import partial
  return partial(self.sample_designs,sampler_type=sampler_type,sampler_kwargs=sampler_kwargs)
 def means(self,xi,theta):
  shape=xi.shape[:-2];flatx=xi.reshape(-1,self.T);flatth=theta.reshape(-1,self.T,2)
  def one(times,th):
   def scan(c,x):
    s,last=c;s=self.evolve(th[0],s,x-last);return (s,x),s[1]
   _,values=jax.lax.scan(scan,(self.initial(),0.),times)
   return values
  return jax.vmap(one)(flatx,flatth).reshape(*shape,self.T,1)
 def data_reparam(self,epsilon,xi,theta):return self.means(xi,theta)+self.sigma*epsilon,{}
 def sample_data(self,key,xi,theta,noise_mult=None):
  key,sub=jax.random.split(key);mean=self.means(xi,theta)
  scale=self.sigma if noise_mult is None else self.sigma*jnp.asarray(noise_mult)[...,None,None]
  return mean+scale*jax.random.normal(sub,mean.shape,dtype=jnp.float64),{},key
 def sample(self,key,batch_shape,theta=None,xi=None,design_aux_data=None,sampler_type=None,sampler_kwargs=None):
  if theta is None:theta,key=self.sample_prior(key,batch_shape)
  if xi is None:xi,_,key=self.sample_designs(key,batch_shape,sampler_type=sampler_type,sampler_kwargs=sampler_kwargs)
  y,aux,key=self.sample_data(key,xi,theta);return y,xi,theta,aux,key
 def fast_sample(self,key,batch_shape,sampler_type=None,sampler_kwargs=None):return self.sample(key,batch_shape,sampler_type=sampler_type,sampler_kwargs=sampler_kwargs)
 def reparam_sample(self,params,policy_net,theta,epsilon,policy_epsilon):
  if hasattr(policy_net,'gru_step'):
   return self.reparam_sample_gru(params,policy_net,theta,epsilon,policy_epsilon)
  shape=epsilon.shape[:-2];enc=jnp.ones((*shape,policy_net.encoding_dim),dtype=jnp.float32)*policy_net.empty_value
  state=self.initial(shape);last=jnp.zeros(shape);ys=[];xs=[]
  evolve=jax.vmap(self.evolve) if shape else self.evolve
  for t in range(self.T):
   raw=policy_net.apply({'params':params},enc=enc,method='emitter')[...,0]
   x=self.feasible(raw,last,t)
   state=evolve(theta[...,0,:],state,x-last)
   y=state[...,1]+self.sigma*epsilon[...,t,0]
   enc=enc+policy_net.apply({'params':params},y=y[...,None],x=x[...,None],method='encoder')
   ys.append(y);xs.append(x);last=x
  return jnp.stack(ys,-1)[...,None],jnp.stack(xs,-1)[...,None],{}
 def reparam_sample_gru(self,params,net,theta,epsilon,policy_epsilon):
  shape=epsilon.shape[:-2];variables={'params':params};carry=net.apply(variables,shape,method='init_gru_state');state=self.initial(shape);last=jnp.zeros(shape);ys=[];xs=[]
  evolve=jax.vmap(self.evolve) if shape else self.evolve
  for t in range(self.T):
   raw=net.apply(variables,carry[-1],z=policy_epsilon[...,t,:],method='emitter_from_state')[...,0]
   x=self.feasible(raw,last,t);state=evolve(theta[...,0,:],state,x-last);y=state[...,1]+self.sigma*epsilon[...,t,0]
   token=net.apply(variables,y=y[...,None],x=x[...,None],method='encoder');carry,_=net.apply(variables,carry,token,method='gru_step')
   ys.append(y);xs.append(x);last=x
  return jnp.stack(ys,-1)[...,None],jnp.stack(xs,-1)[...,None],{}
 def prior_log_lik(self,theta):
  th=theta[...,0,:];z=(jnp.log(th)-self.log_mean)/self.log_std
  return jnp.sum(-.5*z*z-jnp.log(th)-math.log(self.log_std*math.sqrt(2*math.pi)),-1)
 def per_step_data_log_lik(self,yT,xT,theta,aux_data=None,noise_mult=None):
  mean=self.means(xT,theta)[...,0];scale=self.sigma if noise_mult is None else self.sigma*jnp.asarray(noise_mult)[...,None]
  return -.5*((yT[...,0]-mean)/scale)**2-jnp.log(scale*math.sqrt(2*math.pi))
 def data_log_lik(self,yT,xT,theta,aux_data=None,noise_mult=None):return self.per_step_data_log_lik(yT,xT,theta,aux_data,noise_mult).sum(-1)
 def log_lik(self,yT,xT,theta,aux_data=None,noise_mult=None):return self.data_log_lik(yT,xT,theta,aux_data,noise_mult)
 def raw_designs_and_log_jacobian(self,xi):
  times=xi[...,0];previous=jnp.concatenate((jnp.zeros_like(times[...,:1]),times[...,:-1]),-1)
  width=self.end-jnp.arange(self.T-1,-1,-1)*self.gap-previous-self.gap
  u=(times-previous-self.gap)/width
  raw=jnp.log(u)-jnp.log1p(-u)
  logdet=jnp.sum(jnp.log(width)+jnp.log(u)+jnp.log1p(-u),-1)
  return raw,logdet
 def marginal_design_log_lik(self,xi,sampler_type='uniform'):
  # Exact continuous density for the native uniform component pushed forward.
  # Native random_sampler includes clipped/atomic components: no single smooth
  # Lebesgue log-density/score exists for that mixed measure. Native ScoreBED
  # trains grad log p(y|xi,theta), so q's score is never used by its objective.
  if sampler_type!='uniform':raise NotImplementedError('Native mixed design measure has no global smooth marginal score; not required by native score matching')
  raw,logdet=self.raw_designs_and_log_jacobian(xi)
  return jnp.where(jnp.all(jnp.abs(raw)<4,axis=-1),-self.T*jnp.log(8.)-logdet,-jnp.inf)
 def marginal_design_score(self,xi,sampler_type='uniform'):
  return jax.grad(lambda x:self.marginal_design_log_lik(x,sampler_type))(xi)

class SIRScorePolicyNet(DADNet):
 time_scale:float=100.
 count_scale:float=500.
 def encoder(self,y,x,**kwargs):
  return self.encoder_net(jnp.concatenate((y/self.count_scale,x/self.time_scale),-1).astype(jnp.float32))

class SIRScoreGRUNet(IqbalGRUNet):
 time_scale:float=100.
 count_scale:float=500.
 def encoder(self,y,x,**kwargs):
  return self.encoder_net(jnp.concatenate((y/self.count_scale,x/self.time_scale),-1).astype(jnp.float32))
