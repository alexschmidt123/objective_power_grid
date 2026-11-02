"""PIMoE research prototype: predictive densities, never expert action policies.

SMC is a continuous rejuvenated approximation, not an exact inference certificate.
Readiness must independently check its predictive stability before benchmark use.
"""
import copy,math,time
import numpy as np
import torch
from torch import nn

DTYPE=torch.float64

def rng(seed,device='cpu'):
    return torch.Generator(device=device).manual_seed(int(seed))

def conditional_entropy(benchmark):
    if benchmark.name!='location_finding' or benchmark.obs_dim!=1:
        raise ValueError('Entropy identity validated only for scalar additive Gaussian location model')
    return .5*math.log(2*math.pi*math.e*benchmark.sigma**2)

def responses(b,theta,actions):
    """Symmetric simulator responses, no latent coordinates enter neural nets."""
    n,k=len(theta),len(actions)
    means,_=b.forward(theta[:,None].expand(n,k,-1),b.initial((n,k)),actions[None].expand(n,k,-1))
    return means[...,0]

def count_work(b,key,value):
    ledger=getattr(b,'pimoe_work',None)
    if ledger is not None:ledger[key]=ledger.get(key,0)+int(value)

class AdaptiveBelief:
    def __init__(self,benchmark,n=2048,seed=101,moves=8):
        self.b=benchmark;self.g=rng(seed);self.theta=benchmark.sample((n,),self.g)
        self.logw=torch.full((n,),-math.log(n),dtype=DTYPE);self.actions=[];self.ys=[];self.moves=moves;self.records=[]
    @property
    def weights(self):return self.logw.softmax(0)
    def loglik(self,theta,actions=None,ys=None):
        actions=self.actions if actions is None else actions;ys=self.ys if ys is None else ys
        if not actions:return theta.new_zeros(len(theta))
        mean=responses(self.b,theta,torch.stack(actions));measure=torch.stack(ys).reshape(1,-1)
        count_work(self.b,'scalar_conditional_likelihood_evaluations',mean.numel())
        return (-.5*((measure-mean)/self.b.sigma).square()-math.log(self.b.sigma*math.sqrt(2*math.pi))).sum(-1)
    def _resample(self,n=None):
        n=n or len(self.theta);u=(torch.arange(n,dtype=DTYPE)+torch.rand((),generator=self.g))/n
        ids=torch.searchsorted(self.weights.cumsum(0),u).clamp_max(len(self.theta)-1)
        self.theta=self.theta[ids].clone();self.logw=self.theta.new_full((n,),-math.log(n))
    def _move(self,action=None,y=None,beta=1.,steps=None):
        n=len(self.theta);steps=self.moves if steps is None else steps
        # Covariance fixed during each move kernel; label-symmetrized, full support.
        cov=torch.cov(self.theta.T)+torch.eye(4,dtype=DTYPE)*1e-4
        perm=torch.tensor([2,3,0,1]);cov=.5*(cov+cov[perm][:,perm]);chol=torch.linalg.cholesky(cov)
        def target(theta):
            z=self.b.log_prior_z(theta)+self.loglik(theta)
            if action is not None:
                mu=responses(self.b,theta,action[None])[:,0]
                count_work(self.b,'scalar_conditional_likelihood_evaluations',mu.numel())
                z=z+beta*(-.5*((y-mu)/self.b.sigma).square()-math.log(self.b.sigma*math.sqrt(2*math.pi)))
            return z
        current=target(self.theta);accepted=0
        for k in range(steps):
            global_move=(k%8==7)
            if global_move:proposal=self.b.sample((n,),self.g)
            else:
                scale=.25 if k%3==0 else 1.19
                proposal=self.theta+scale*torch.randn(n,4,generator=self.g,dtype=DTYPE)@chol.T
            val=target(proposal);ratio=val-current
            if global_move:ratio=ratio+self.b.log_prior_z(self.theta)-self.b.log_prior_z(proposal)
            keep=torch.rand(n,generator=self.g,dtype=DTYPE).log()<ratio
            self.theta=torch.where(keep[:,None],proposal,self.theta);current=torch.where(keep,val,current);accepted+=int(keep.sum())
        # Label permutation is an exact target symmetry, not a separate mode.
        swap=(torch.rand(n,generator=self.g)<.5) if getattr(self.b,'source_label_symmetry',True) else torch.zeros(n,dtype=torch.bool)
        self.theta=torch.where(swap[:,None],self.theta[:,perm],self.theta)
        return accepted/max(1,n*steps)
    def update(self,action,y):
        action=action.detach().double().clone();y=torch.as_tensor(y,dtype=DTYPE).reshape(())
        if not torch.isfinite(y) or not torch.isfinite(action).all():raise ValueError('Nonfinite history')
        beta=0.;bridges=0;accept=[];minimum_ess=len(self.theta)
        while beta<1-1e-10:
            if bridges>64:raise RuntimeError('Tempering failed; inference not certified')
            count_work(self.b,'scalar_conditional_likelihood_evaluations',len(self.theta))
            ll=-.5*((y-responses(self.b,self.theta,action[None])[:,0])/self.b.sigma).square()-math.log(self.b.sigma*math.sqrt(2*math.pi))
            def ess(delta):return float((self.logw+delta*ll).softmax(0).square().sum().reciprocal())
            target=.7*float(self.weights.square().sum().reciprocal());left,right=0.,1-beta
            if ess(right)<target:
                for _ in range(32):
                    mid=(left+right)/2
                    if ess(mid)<target:right=mid
                    else:left=mid
                delta=max(left,1e-10)
            else:delta=right
            beta=min(1.,beta+delta);self.logw=self.logw+delta*ll;self.logw-=torch.logsumexp(self.logw,0)
            minimum_ess=min(minimum_ess,float(self.weights.square().sum().reciprocal()))
            self._resample();accept.append(self._move(action,y,beta));bridges+=1
        self.actions.append(action);self.ys.append(y)
        self.records.append(dict(stage=len(self.actions),particles=len(self.theta),tempering_bridges=bridges,minimum_ESS=minimum_ess,move_acceptance=float(np.mean(accept)),continuous_rejuvenation=True))
        return self
    def refine(self,n,steps=32):
        if n<len(self.theta):raise ValueError('Refinement cannot silently reduce particles')
        self._resample(n);accept=self._move(steps=steps)
        self.records.append(dict(refined_particles=n,move_steps=steps,move_acceptance=accept));return self
    def fork(self,seed):
        q=copy.copy(self);q.theta=self.theta.clone();q.logw=self.logw.clone();q.actions=list(self.actions);q.ys=list(self.ys);q.records=list(self.records);q.g=rng(seed);return q

class DensityHead(nn.Module):
    def __init__(self,width=48,components=3):
        super().__init__();self.components=components
        self.net=nn.Sequential(nn.Linear(10,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,3*components))
    def forward(self,x,center,spread,sigma):
        z=self.net(x).reshape(*x.shape[:-1],3,self.components)
        weight=z[...,0,:].softmax(-1)
        loc=center[...,None]+spread[...,None]*z[...,1,:]
        scale=(sigma**2+(spread[...,None]*torch.nn.functional.softplus(z[...,2,:]))**2).sqrt()
        return weight,loc,scale

class PredictiveMoE(nn.Module):
    def __init__(self,experts=4,width=48,components=3):
        super().__init__();self.expert_count=experts;self.components=components
        self.router=nn.Sequential(nn.Linear(11,32),nn.Tanh(),nn.Linear(32,experts)) if experts>1 else None
        self.experts=nn.ModuleList([DensityHead(width,components) for _ in range(experts)])
        self.double()
    def forward(self,means,anchor_response,actions,weights,stage,sigma,bounds):
        # means[A,P], anchors[P,9]; neural inputs contain responses, not theta.
        A,P=means.shape;device=next(self.parameters()).device
        means=means.to(device);anchor_response=anchor_response.to(device);actions=actions.to(device);weights=weights.to(device)
        center=(means*weights).sum(-1);variance=((means-center[:,None]).square()*weights).sum(-1);spread=(variance+sigma**2).sqrt()
        anchored=(anchor_response-anchor_response.mean(0))/(anchor_response.std(0,unbiased=False)+sigma)
        normal=(means-center[:,None])/spread[:,None]
        phi=torch.cat([normal[...,None],anchored[None].expand(A,-1,-1),torch.full((A,P,1),stage/30,device=device,dtype=DTYPE)],-1)
        membership=self.router(phi).softmax(-1) if self.router is not None else means.new_ones(A,P,1)
        mass=(membership*weights[None,:,None]).sum(1);conditional=membership*weights[None,:,None]/mass[:,None,:].clamp_min(1e-12)
        region=(conditional*means[:,:,None]).sum(1);var=(conditional*(means[:,:,None]-region[:,None,:]).square()).sum(1)
        region_sd=(var+sigma**2).sqrt();skew=(conditional*((means[:,:,None]-region[:,None,:])/region_sd[:,None,:]).pow(3)).sum(1)
        global_skew=(weights*((means-center[:,None])/spread[:,None]).pow(3)).sum(-1)
        outw=[];outm=[];outs=[]
        for e,head in enumerate(self.experts):
            features=torch.stack([region[:,e]/3,region_sd[:,e].log(),skew[:,e],mass[:,e],center/3,spread.log(),global_skew,actions[:,0]/bounds,actions[:,1]/bounds,torch.full_like(center,stage/30)],-1)
            w,m,s=head(features,region[:,e],region_sd[:,e],sigma);outw.append(mass[:,e,None]*w);outm.append(m);outs.append(s)
        result=dict(weights=torch.cat(outw,-1),means=torch.cat(outm,-1),scales=torch.cat(outs,-1),membership=membership,mass=mass,region=region,regional_variance=var)
        return result
    @staticmethod
    def log_prob(result,y):
        # y[A,S]; normalized mixture component axis last.
        z=(y[...,None]-result['means'][:,None])/result['scales'][:,None]
        lp=-.5*z.square()-result['scales'][:,None].log()-.5*math.log(2*math.pi)
        return torch.logsumexp(lp+result['weights'][:,None].clamp_min(1e-300).log(),-1)
    @staticmethod
    def cdf(result,y):
        z=(y[...,None]-result['means'][:,None])/result['scales'][:,None]
        return (result['weights'][:,None]*.5*(1+torch.erf(z/math.sqrt(2)))).sum(-1)
    @classmethod
    def entropy(cls,result,nodes=24):
        x,w=np.polynomial.hermite.hermgauss(nodes);x=torch.as_tensor(x,dtype=DTYPE,device=result['means'].device);w=torch.as_tensor(w/math.sqrt(math.pi),dtype=DTYPE,device=x.device)
        A,K=result['means'].shape;y=(result['means'][...,None]+math.sqrt(2)*result['scales'][...,None]*x).reshape(A,-1)
        logp=cls.log_prob(result,y).reshape(A,K,nodes)
        return -(logp*w*result['weights'][...,None]).sum((-1,-2))

def state_features(b,belief,actions):
    lo,hi=b.design_bounds;axis=torch.linspace(lo,hi,3,dtype=DTYPE);anchors=torch.cartesian_prod(axis,axis)
    return dict(means=responses(b,belief.theta,actions).T.detach(),anchor_response=responses(b,belief.theta,anchors).detach(),actions=actions.detach(),weights=belief.weights.detach(),stage=len(belief.actions),sigma=b.sigma,bounds=b.action_scale)

def direct_reference(means,weights,sigma,seed=1004,outer=512):
    """Independent outer y draws; exact mixture denominator for current SMC belief.

    Uncertainty is conditional on approximate belief. Posterior error is separate.
    No sPCE objective or truth-included contrast denominator is used.
    """
    g=rng(seed);A,P=means.shape;u=(torch.arange(outer,dtype=DTYPE)+torch.rand(outer,generator=g,dtype=DTYPE))/outer
    ids=torch.searchsorted(weights.cumsum(0),u).clamp_max(P-1)
    y=means[:,ids]+sigma*torch.randn(A,outer,generator=g,dtype=DTYPE)
    logp=-.5*((y[:,:,None]-means[:,None,:])/sigma).square()-math.log(sigma*math.sqrt(2*math.pi))
    logp=torch.logsumexp(logp+weights.clamp_min(1e-300).log()[None,None],-1)
    scores=-logp-.5*math.log(2*math.pi*math.e*sigma**2)
    return dict(EIG=scores.mean(-1),SE=scores.std(-1)/math.sqrt(outer),y=y,scores=scores)

def predictive_loss(model,features,targets,reference,reference_se):
    result=model(**features);y=targets.to(result['means'].device);nll=-model.log_prob(result,y).mean()
    # Region-specific predictive fit, independent posterior draws per candidate.
    # Aggregate mixture likelihood plus physical response dispersion specialization.
    info=model.entropy(result)-.5*math.log(2*math.pi*math.e*features['sigma']**2)
    ref=reference.to(info);se=reference_se.to(info);d=ref[:,None]-ref[None,:]
    uncertainty=2*(se[:,None].square()+se[None,:].square()).sqrt();mask=torch.triu(d.abs()>uncertainty,diagonal=1)
    rank=torch.nn.functional.softplus(-d.sign()*(info[:,None]-info[None,:]))[mask].mean() if mask.any() else info.sum()*0
    variance=(result['mass']*result['regional_variance']).sum(-1)
    mu=features['means'].to(info);ww=features['weights'].to(info);center=(mu*ww).sum(-1)
    total=((mu-center[:,None]).square()*ww).sum(-1)+features['sigma']**2
    specialization=(variance/total).mean()
    loss=nll+.1*rank+.02*specialization
    return loss,dict(NLL=float(nll.detach()),ranking_loss=float(rank.detach()),identifiability_loss=float(specialization.detach()),confident_pairs=int(mask.sum())),result

class PIMoEPlanner:
    """Bounded two-step belief lookahead; no hidden truth and no action experts."""
    def __init__(self,model,benchmark,seed=1004,particles=2048,candidates=24,shortlist=2,fantasies=2,two_step=True):
        self.model=model.eval();self.b=benchmark;self.g=rng(seed);self.belief=AdaptiveBelief(benchmark,particles,seed+1);self.candidates=candidates;self.shortlist=shortlist;self.fantasies=fantasies;self.two_step=two_step;self.seed=seed;self.timings=[]
    def candidates_for(self):
        lo,hi=self.b.design_bounds
        return lo+(hi-lo)*torch.rand(self.candidates,2,generator=self.g,dtype=DTYPE)
    def information(self,belief,actions):
        with torch.no_grad():r=self.model(**state_features(self.b,belief,actions));return (self.model.entropy(r)-conditional_entropy(self.b)).cpu()
    def select(self):
        begin=time.monotonic();actions=self.candidates_for();scores=self.information(self.belief,actions)
        # Local candidate refinement keeps the exact feasible physical box.
        top=actions[scores.topk(min(3,len(scores))).indices];lo,hi=self.b.design_bounds
        refine=(top+(.125*(hi-lo))*torch.randn(top.shape,generator=self.g,dtype=DTYPE)).clamp(lo,hi)
        actions=torch.cat([actions,refine]);scores=self.information(self.belief,actions)
        ids=scores.topk(min(self.shortlist,len(scores))).indices;objective=scores[ids].clone()
        if self.two_step and len(self.belief.actions)<29:
            for j,index in enumerate(ids):
                futures=[]
                for k in range(self.fantasies):
                    choice=int(torch.multinomial(self.belief.weights,1,generator=self.g));mu=responses(self.b,self.belief.theta[choice:choice+1],actions[index:index+1])[0,0]
                    y=mu+self.b.sigma*torch.randn((),generator=self.g,dtype=DTYPE)
                    child=self.belief.fork(self.seed+len(self.belief.actions)*100003+j*101+k)
                    child.update(actions[index],y);futures.append(self.information(child,actions).max())
                objective[j]+=torch.stack(futures).mean()
        answer=actions[ids[objective.argmax()]];self.timings.append(time.monotonic()-begin);return answer
    def observe(self,action,y):self.belief.update(action,y)
