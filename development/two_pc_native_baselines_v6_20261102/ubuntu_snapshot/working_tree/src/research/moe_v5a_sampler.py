"""V5-only conditional SMC adapter; never used by baseline trainers.

Finite, correlated resample-move particles and stopped inference gradients make
this an explicitly approximate auxiliary, not a certified conditional bound.
"""
import time,math
import torch
from src.research.location_posterior import LocationSMC
from src.research.moe_v5_conditional import PosteriorQuality,require_quality

class AuditedSMC(LocationSMC):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.resampling_unique=[]
    def resample(self):
        index=super().resample();self.resampling_unique.append(float(index.unique().numel()/index.numel()));return index

@torch.no_grad()
def sample_prefix(benchmark,actions,observations,negatives,seed,particles=512,moves=8):
    if benchmark.name!='location_finding':raise ValueError('Only the verified location likelihood is adapted')
    if len(actions)!=len(observations) or negatives>particles:raise ValueError('Invalid prefix/particle dimensions')
    if any(len(a)!=1 or len(y)!=1 for a,y in zip(actions,observations)):raise ValueError('One independently sampled history per pool')
    started=time.monotonic();calls=benchmark.calls
    rng=torch.Generator(device=torch.empty(0).device).manual_seed(int(seed))
    smc=AuditedSMC(benchmark,1,particles,rng,moves=moves)
    for action,y in zip(actions,observations):smc.update(action.detach(),y.detach())
    theta=smc.theta[0];unique=float(torch.unique(theta,dim=0).shape[0]/particles)
    acceptance=min((float(x['mh_acceptance'][0]) for x in smc.diagnostics),default=1.)
    pre_ess=min((x['minimum_ess']/particles for x in smc.diagnostics),default=1.)
    finite=bool(torch.isfinite(theta).all());quality_ok=finite and unique>=.5 and acceptance>=.005 and pre_ess>=.79
    selected=torch.randperm(particles,generator=rng,device=theta.device)[:negatives]
    draws=theta[selected].detach()
    q=PosteriorQuality(len(actions),renewed=True,exact=False,validated=quality_ok,ess_fraction=pre_ess,unique_fraction=unique,
        mixing_checked=acceptance>=.005,symmetry_checked=True,proposal_weights_checked=True,
        sampling_gradient='SMC/resampling and history-conditioning distribution detached; heuristic policy gradient')
    require_quality(q,len(actions))
    return draws,q,dict(status='PASS_EMPIRICAL_POOL_CHECKS',conditioning_observations=len(actions),
        particle_count=particles,negative_count=negatives,pre_resampling_minimum_ess_fraction=pre_ess,
        final_ess_is_uniform_after_resampling=True,final_unique_fraction=unique,
        minimum_mh_acceptance=acceptance,resampling_unique_minimum=min(smc.resampling_unique,default=1.),
        symmetry_label_fraction=float((theta[:,0]<theta[:,2]).double().mean()),
        likelihood_particle_stages=benchmark.calls-calls,posterior_proposals=particles*moves*sum(x['tempering_steps'] for x in smc.diagnostics),
        seconds=time.monotonic()-started,exact=False,independent_draws_certified=False,
        estimate_type='APPROXIMATE_POSTERIOR_HEURISTIC_NOT_A_CERTIFIED_LOWER_BOUND'),theta

def conditional_auxiliary(benchmark,policy,truth,noise,stage,seed,negatives=128,particles=512,moves=8):
    """One uniform time/sample yields T*C, an unbiased stage-sum *selection*.

    Inference and selection are separate from the primary RNG. Only the
    prefix[:stage] is supplied to SMC. No y_stage or truth enters inference.
    """
    from src.research.engine import rollout
    batch=len(truth)
    if batch!=1:raise ValueError('The frozen first candidate uses one auxiliary case/update')
    start=time.monotonic();calls=benchmark.calls
    result=rollout(benchmark,policy,policy.horizon,batch,seed,0,truth=truth,noise=noise,prior_particles=truth.new_empty(batch,0,4))
    a=[x.detach() for x in result['actions'][:stage]];y=[x.detach() for x in result['observations'][:stage]]
    # The validated CPU sampler is used verbatim; all inference wall time is
    # charged while the GPU lease remains held. Only small draws return to GPU.
    with torch.device('cpu'):
        draws,q,stats,_=sample_prefix(benchmark,[x.cpu() for x in a],[x.cpu() for x in y],negatives,seed+15485863,particles,moves)
    draws=draws.to(truth.device)
    action=result['actions'][stage];observation=result['observations'][stage]
    theta=torch.cat((truth[:,None],draws[None]),1)
    mean,_=benchmark.forward(theta,benchmark.initial(theta.shape[:2]),action[:,None].expand(-1,negatives+1,-1))
    ll=benchmark.log_likelihood(observation[:,None],mean)
    c=ll[:,0]-torch.logsumexp(ll,-1)+math.log(negatives+1)
    if not torch.isfinite(c).all():raise FloatingPointError('Nonfinite V5-A conditional score')
    estimate=policy.horizon*c.mean()
    stats.update(stage=stage,conditional_score=float(c.detach().mean()),stage_sum_estimate=float(estimate.detach()),
        stage_sampling_probability=1/policy.horizon,stage_sampling_weight=policy.horizon,
        total_auxiliary_likelihood_particle_stages=benchmark.calls-calls,
        seconds=time.monotonic()-start,extra_primary_replay_trajectories=1)
    return estimate,stats
