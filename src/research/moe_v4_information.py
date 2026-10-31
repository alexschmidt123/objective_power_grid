"""Stable finite-bank coarse/fine densities and actual continuation rollouts."""
import math,torch
from src.research.engine import generator,logmeanexp
from src.research.policies import history_features

def densities(loglik,labels):
 n=loglik.shape[1];same=labels==labels[:,:1];logZ=torch.logsumexp(loglik,1);total=loglik[:,0]-logZ+math.log(n);group=torch.logsumexp(loglik.masked_fill(~same,-torch.inf),1)-logZ;prior=same.sum(1).to(loglik.dtype)/n;coarse=group-prior.log();return total,coarse,total-coarse

def targets(total,coarse):
 # Both inputs include the empty-history prefix at column0.
 fine=total-coarse;ic=coarse[:,1:]-coarse[:,:-1];iF=fine[:,1:]-fine[:,:-1];fc=coarse[:,-1:]-coarse[:,1:];fF=fine[:,-1:]-fine[:,1:]
 return torch.stack((ic,iF,fc,fF),-1).detach()

def paired_training_draws(b,h,B,L,seed):
 g=generator(seed);truth=b.sample((B,),g);bank=b.sample((L,),g);noise=torch.randn(B,h,b.obs_dim,generator=g,dtype=torch.float64);return truth,bank,noise

def role_rollout(b,p,h,truth,bank,noise,partition,adapter):
 B=len(truth);L=len(bank);theta=torch.cat((truth[:,None],bank[None].expand(B,-1,-1)),1);state=b.initial((B,L+1));cum=torch.zeros(B,L+1,dtype=torch.float64)
 # Classification is OFFLINE, independent of policy inputs. Shared iid bank
 # retains the same expected per-trajectory finite-L objective; batch covariance
 # differs from independently resampled outer-case banks, for ALL matched arms.
 with torch.no_grad():
  truth_labels=partition.classify(adapter,truth);bank_labels=partition.classify(adapter,bank);labels=torch.cat((truth_labels[:,None],bank_labels[None].expand(B,-1)),1)
 actions=[];obs=[];features=[];info=[];coarse=[cum[:,0].detach().clone()];totals=[cum[:,0].detach().clone()];routes=[];ess=[]
 for stage in range(h):
  f=history_features(b,actions,obs,B,h);latent=p(f,stage);action=b.feasible(latent,actions,h);mean,state=b.forward(theta,state,action[:,None].expand(-1,L+1,-1));y=mean[:,0]+b.sigma*noise[:,stage];cum=cum+b.log_likelihood(y[:,None],mean);value=cum[:,0]-logmeanexp(cum);info.append(value);actions.append(action);obs.append(y);features.append(f)
  with torch.no_grad():
   jt,jc,jf=densities(cum.detach(),labels);totals.append(jt);coarse.append(jc);ess.append(torch.exp(2*torch.logsumexp(cum.detach(),1)-torch.logsumexp(2*cum.detach(),1)))
   if getattr(p,'last_dispatch',None) is not None:routes.append(p.last_dispatch)
 total=torch.stack(totals,1);coarse=torch.stack(coarse,1);role=targets(total,coarse)
 return dict(truth=truth,particles=bank[None].expand(B,-1,-1),actions=actions,observations=obs,features=features,info=torch.stack(info,1),role_targets=role,coarse_density=coarse,total_density=total,routing=routes,posterior_ESS=torch.stack(ess,1),group_prior_counts=torch.bincount(bank_labels,minlength=4),log_likelihood=cum)
