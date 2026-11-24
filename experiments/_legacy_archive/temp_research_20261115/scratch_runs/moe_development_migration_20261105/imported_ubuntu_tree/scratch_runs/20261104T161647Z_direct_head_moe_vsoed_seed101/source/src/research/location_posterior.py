"""Truth-independent, tempered resample-move inference for explicit locations.

Every update targets p(theta) times the actual observed likelihoods. Adaptive
tempering controls weight ESS, systematic resampling prevents weight collapse,
and symmetric Metropolis moves rejuvenate static parameters. Finite populations
can still miss modes; diagnostics and particle sensitivity remain necessary.
"""
import math
import torch


def ess(logw):
    return logw.softmax(-1).square().sum(-1).reciprocal()


class LocationSMC:
    def __init__(self, benchmark, batch, count, rng, moves=8, threshold=.8):
        if count<4 or moves<1: raise ValueError('SMC requires >=4 particles and >=1 moves')
        self.b,self.rng,self.moves,self.threshold=benchmark,rng,moves,threshold
        # Each episode has its own random stream and tempering schedule. A
        # difficult observation in another episode must not affect this one.
        self._children=None
        if batch>1:
            self._children=[LocationSMC(benchmark,1,count,
                torch.Generator(device=rng.device).manual_seed(
                    (rng.initial_seed()+1000003*i)%(2**63-1)),moves,threshold)
                for i in range(batch)]
            self.theta=torch.cat([child.theta for child in self._children],0)
        else:
            self.theta=benchmark.sample((batch,count),rng)
        self.logw=torch.zeros(batch,count,dtype=torch.float64)
        self.actions,self.observations,self.diagnostics=[],[],[]
        self.scale=torch.full((batch,1,1),.15,dtype=torch.float64)

    def loglike(self, theta, actions=None, observations=None):
        actions=self.actions if actions is None else actions
        observations=self.observations if observations is None else observations
        lp=torch.zeros(theta.shape[:-1],dtype=torch.float64)
        # Chunking histories bounds memory; all measured observations contribute.
        for start in range(0,len(actions),16):
            a=torch.stack(actions[start:start+16],1)
            y=torch.stack(observations[start:start+16],1)
            mean,_=self.b.forward(theta[:,:,None].expand(-1,-1,a.shape[1],-1),None,
                                  a[:,None].expand(-1,theta.shape[1],-1,-1))
            lp+=self.b.log_likelihood(y[:,None],mean).sum(-1)
        return lp

    def resample(self):
        b,n=self.logw.shape
        u=(torch.arange(n,dtype=torch.float64)[None]+torch.rand(b,1,generator=self.rng,dtype=torch.float64))/n
        cdf=self.logw.softmax(-1).cumsum(-1); cdf[:,-1]=1
        index=torch.searchsorted(cdf.contiguous(),u.contiguous()).clamp_max(n-1)
        self.theta=self.theta.gather(1,index[...,None].expand(-1,-1,4))
        self.logw.zero_()
        return index

    def update(self, action, observation):
        action,observation=action.detach(),observation.detach()
        if self._children is not None:
            diagnostics=[child.update(action[i:i+1],observation[i:i+1])
                         for i,child in enumerate(self._children)]
            self.theta=torch.cat([child.theta for child in self._children],0)
            self.scale=torch.cat([child.scale for child in self._children],0)
            self.logw.zero_()
            self.actions.append(action);self.observations.append(observation)
            d=dict(tempering_steps=max(x['tempering_steps'] for x in diagnostics),
                   tempering_steps_by_episode=[x['tempering_steps'] for x in diagnostics],
                   minimum_ess=min(x['minimum_ess'] for x in diagnostics),
                   final_ess=self.theta.shape[1],
                   mh_acceptance=[x['mh_acceptance'][0] for x in diagnostics],
                   proposal_scale=[x['proposal_scale'][0] for x in diagnostics],
                   independent_episode_updates=True)
            self.diagnostics.append(d)
            return d
        old=self.loglike(self.theta)
        new=self.loglike(self.theta,[action],[observation])
        beta=0.;temperatures=[];acceptances=[];minimum=math.inf
        while beta<1-1e-12:
            if len(temperatures)>=256:raise RuntimeError('SMC tempering failed to reach the posterior')
            remaining=1-beta
            if float(ess(self.logw+remaining*new).min())>=self.threshold*self.theta.shape[1]:delta=remaining
            else:
                lo,hi=0.,remaining
                for _ in range(24):
                    mid=(lo+hi)/2
                    if float(ess(self.logw+mid*new).min())>=self.threshold*self.theta.shape[1]:lo=mid
                    else:hi=mid
                delta=max(lo,min(remaining,1e-8))
            beta=min(1.,beta+delta)
            self.logw+=delta*new
            minimum=min(minimum,float(ess(self.logw).min()))
            index=self.resample()
            old,new=old.gather(1,index),new.gather(1,index)
            target=self.b.log_prior_z(self.theta)+old+beta*new
            # Proposal scales are held fixed during each group of MH transitions.
            # Mixing large and small symmetric random walks helps narrow modes.
            rates=[]
            for move in range(self.moves):
                factor=1. if move%2==0 else .2
                proposal=self.theta+factor*self.scale*torch.randn(self.theta.shape,generator=self.rng,dtype=torch.float64)
                proposed_old=self.loglike(proposal)
                proposed_new=self.loglike(proposal,[action],[observation])
                proposed_target=self.b.log_prior_z(proposal)+proposed_old+beta*proposed_new
                keep=torch.rand(target.shape,generator=self.rng,dtype=torch.float64).log()<proposed_target-target
                self.theta=torch.where(keep[...,None],proposal,self.theta)
                old,new=torch.where(keep,proposed_old,old),torch.where(keep,proposed_new,new)
                target=torch.where(keep,proposed_target,target)
                rates.append(keep.double().mean(-1))
            rate=torch.stack(rates).mean(0)
            # Exact label-swap transitions preserve the symmetric source posterior.
            swap=torch.rand(target.shape,generator=self.rng,dtype=torch.float64)<.5
            self.theta=torch.where(swap[...,None],self.theta.reshape(*target.shape,2,2).flip(-2).flatten(-2),self.theta)
            self.scale=(self.scale*torch.exp(.5*(rate[:,None,None]-.3))).clamp(.0001,1.)
            temperatures.append(beta);acceptances.append(rate)
        self.actions.append(action);self.observations.append(observation)
        d=dict(tempering_steps=len(temperatures),minimum_ess=minimum,
               final_ess=self.theta.shape[1],mh_acceptance=torch.stack(acceptances).mean(0).cpu().tolist(),
               proposal_scale=self.scale[:,0,0].cpu().tolist())
        self.diagnostics.append(d)
        return d


# Design selection is supplied by location_variational.py.

@torch.no_grad()
def posterior_metrics(benchmark, result, settings, seed, return_artifacts=False):
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    with torch.device(device):
        rng=torch.Generator(device=device).manual_seed(seed)
        count=max(4,settings['posterior_particles']//2)
        smc=LocationSMC(benchmark,len(result['truth']),count,rng,settings.get('smc_moves',8))
        rng2=torch.Generator(device=device).manual_seed(seed+15485863)
        other=LocationSMC(benchmark,len(result['truth']),count,rng2,settings.get('smc_moves',8))
        for a,y in zip(result['actions'],result['observations']):
            smc.update(a.to(device),y.to(device));other.update(a.to(device),y.to(device))
        samples=torch.cat((smc.theta,other.theta),1)
        lp=smc.loglike(samples)
        reference=samples[torch.arange(len(samples)),lp.argmax(-1)].reshape(-1,2,2)
        sources=samples.reshape(len(samples),-1,2,2)
        direct=(sources-reference[:,None]).square().sum((-1,-2))
        swapped=(sources.flip(-2)-reference[:,None]).square().sum((-1,-2))
        aligned=torch.where((swapped<direct)[...,None,None],sources.flip(-2),sources).flatten(-2)
        estimate=aligned.mean(1);truth=result['truth'].to(device)
        e=(estimate-truth).square();se=(estimate.reshape(-1,2,2).flip(1).flatten(1)-truth).square()
        e=torch.where((se.sum(-1)<e.sum(-1))[:,None],se,e)
        variance=aligned.var(1,unbiased=False)
        # The two populations use independent random streams and never resample
        # from each other. Agreement is useful but cannot prove mode coverage.
        half=aligned.shape[1]//2
        discrepancy=(aligned[:,:half].mean(1)-aligned[:,half:].mean(1)).square().mean(-1).sqrt()
        rates=torch.tensor([d['mh_acceptance'] for d in smc.diagnostics+other.diagnostics]).mean(0)
        rows=[]
        for i in range(len(samples)):
            rows.append(dict(estimate=estimate[i].cpu().tolist(),mse=float(e[i].mean()),normalized_mse=float(e[i].mean()),
                squared_error_by_parameter=dict(zip(benchmark.param_names,e[i].cpu().tolist())),
                posterior_variance=variance[i].cpu().tolist(),importance_ess=float(samples.shape[1]),
                mh_acceptance=float(rates[i]),posterior_particles=samples.shape[1],posterior_steps=settings.get('smc_moves',8),
                truth_excluded=True,inference_requires_sensitivity_check=True,
                inference_degenerate=bool(rates[i]<.01 or discrepancy[i]>.25*variance[i].mean().sqrt()+.01),
                inference='two_independent_tempered_resample_move_SMC_populations',
                independent_replica_mean_disagreement=float(discrepancy[i]),
                final_population_ess=float(samples.shape[1]),ess_is_after_resampling=True,
                tempering_steps=sum(d['tempering_steps'] for d in smc.diagnostics)))
        weights=torch.full(samples.shape[:2],1/samples.shape[1],dtype=torch.float64)
        artifacts=dict(posterior_samples=aligned.cpu().numpy(),posterior_sample_weights=weights.cpu().numpy())
    return (rows,artifacts) if return_artifacts else rows
