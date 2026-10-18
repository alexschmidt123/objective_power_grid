"""Shared offline-policy rollouts, trajectory likelihoods and metrics."""
import math
import time
import torch
from src.research.policies import history_features


def generator(seed):
    # Respect the training device context; evaluation uses the CPU default.
    return torch.Generator(device=torch.empty(0).device).manual_seed(int(seed))


def logmeanexp(value, dim=-1):
    return torch.logsumexp(value,dim)-math.log(value.shape[dim])


def replay_likelihood(benchmark, particles, actions, observations):
    """Particles have shape (batch, candidates, parameters); never include truth
    unless explicitly constructing the sPCE denominator elsewhere.
    """
    batch,count,_ = particles.shape
    states = benchmark.initial((batch,count))
    cumulative = torch.zeros(batch,count,dtype=torch.float64)
    scores = []
    for action,y in zip(actions,observations):
        mean,states = benchmark.forward(particles,states,action[:,None].expand(-1,count,-1))
        cumulative = cumulative + benchmark.log_likelihood(y[:,None],mean)
        scores.append(cumulative)
    return cumulative,states,scores


def rollout(benchmark, policy, horizon, batch, seed, contrasts, *, truth=None,
            noise=None, method=None, settings=None, exploration=0., prior_particles=None, track_likelihood=True):
    if method=='myopic':raise ValueError('Myopic is removed: online design optimization is outside the study')
    if not track_likelihood and contrasts:raise ValueError('Sample-only rollout cannot compute sPCE contrasts')
    rng = generator(seed)
    decision_rng=generator(seed+44000001)
    theta0 = benchmark.sample((batch,),rng) if truth is None else truth
    particles = (benchmark.sample((batch,contrasts),rng) if prior_particles is None else prior_particles)
    theta = torch.cat((theta0[:,None],particles),1)
    count = theta.shape[1]
    state = benchmark.initial((batch,count))
    cumulative = torch.zeros(batch,count,dtype=torch.float64)
    actions,observations,features,latents,infos,timings,routers = [],[],[],[],[],[],[]
    for stage in range(horizon):
        f = history_features(benchmark,actions,observations,batch,horizon)
        started = time.perf_counter()
        if method == 'random':
            latent = benchmark.random_latent(batch,decision_rng)
        else:
            if policy.kind=='rl_sboed' and hasattr(policy,'design_from_unit_action'):
                latent=policy.design_from_unit_action(policy.sample(f,rng=decision_rng)[0],actions,horizon)
            else:
                latent = (policy.sample(f,rng=decision_rng)[0]*policy.bound).double() if policy.kind=='rl_sboed' else policy(f,stage)
            if exploration:
                latent = latent+exploration*torch.randn(latent.shape,generator=decision_rng,dtype=torch.float64)
            if policy.kind in {'moe_sboed','uniform_moe','matched_dense'}:
                weights,proposals = policy.components(f)
                record={'weights':weights.detach().tolist(),'proposals':proposals.detach().tolist()}
                if hasattr(policy,'role_weights'):
                    record.update(critic_weights=policy.role_weights(f,'critic').detach().tolist(),posterior_weights=policy.role_weights(f,'posterior').detach().tolist())
                routers.append(record)
        action = latent if getattr(policy,'outputs_design',False) else benchmark.feasible(latent,actions,horizon)
        timings.append(time.perf_counter()-started)
        sample_fn=getattr(benchmark,'sample_observation',None)
        if not track_likelihood and callable(sample_fn):
            y,state=sample_fn(theta,state,action[:,None].expand(-1,count,-1),rng)
            if y.ndim==3:y=y[:,0]
        else:
            means,state = benchmark.forward(theta,state,action[:,None].expand(-1,count,-1))
            eps = (torch.randn(batch,benchmark.obs_dim,generator=rng,dtype=torch.float64)
                   if noise is None else noise[:,stage])
            y = means[:,0]+benchmark.sigma*eps
            if track_likelihood:cumulative = cumulative+benchmark.log_likelihood(y[:,None],means)
        if contrasts:
            infos.append(cumulative[:,0]-logmeanexp(cumulative))
        features.append(f); latents.append(latent); actions.append(action); observations.append(y)
    return dict(truth=theta0,particles=particles,actions=actions,observations=observations,
                features=features,latents=latents,log_likelihood=cumulative,
                info=torch.stack(infos,1) if infos else None,
                final_features=history_features(benchmark,actions,observations,batch,horizon),
                decision_seconds=timings,routing=routers,
                planning_diagnostics=[])


def posterior_metrics(benchmark, result, settings, seed, *, return_artifacts=False):
    """Truth-excluded prior importance sampling + independent-chain MH refinement.

    Location samples are aligned to a truth-independent posterior reference,
    then the point estimate is scored with the better source permutation.
    Diagnostics do not certify convergence; sensitivity runs remain necessary.
    """
    if getattr(benchmark,'corrected_location',False):
        from src.research.location_posterior import posterior_metrics as location_metrics
        return location_metrics(benchmark,result,settings,seed,return_artifacts)
    rng = generator(seed)
    batch = len(result['truth'])
    particles = benchmark.sample((batch,settings['posterior_particles']),rng)
    lp,particle_states,_ = replay_likelihood(benchmark,particles,result['actions'],result['observations'])
    weights = lp.softmax(-1)
    importance_ess = weights.square().sum(-1).reciprocal()
    chains = settings['posterior_chains']
    idx = torch.multinomial(weights,chains,replacement=True,generator=rng)
    current = particles.gather(1,idx[...,None].expand(-1,-1,benchmark.param_dim))
    z = benchmark.transform(current)
    current_lp = lp.gather(1,idx)+benchmark.log_prior_z(z)
    accepted = torch.zeros(batch,chains,dtype=torch.float64)
    samples = []
    iterations = settings['posterior_steps']
    for step in range(iterations):
        proposal = z+settings['posterior_proposal_scale']*torch.randn(z.shape,generator=rng,dtype=torch.float64)
        proposed_theta = benchmark.inverse(proposal)
        likelihood,_,_ = replay_likelihood(benchmark,proposed_theta,result['actions'],result['observations'])
        proposed_lp = likelihood+benchmark.log_prior_z(proposal)
        keep = torch.rand(batch,chains,generator=rng,dtype=torch.float64).log() < proposed_lp-current_lp
        z = torch.where(keep[...,None],proposal,z)
        current_lp = torch.where(keep,proposed_lp,current_lp)
        accepted += keep
        if step >= iterations//2:
            samples.append(benchmark.inverse(z).clone())
    if samples:
        chain_samples = torch.stack(samples,1)  # batch,steps,chains,parameters
        samples = chain_samples.flatten(1,2)
        sample_weights = torch.full(samples.shape[:2],1/samples.shape[1],dtype=torch.float64)
    else:
        samples,sample_weights = particles,weights
    if benchmark.name == 'location_finding':
        reference = particles[torch.arange(batch),lp.argmax(-1)].reshape(batch,2,2)
        sources = samples.reshape(batch,-1,2,2)
        direct = (sources-reference[:,None]).square().sum((-1,-2))
        swapped = (sources.flip(-2)-reference[:,None]).square().sum((-1,-2))
        samples = torch.where((swapped<direct)[...,None,None],sources.flip(-2),sources).flatten(-2)
    estimate = (samples*sample_weights[...,None]).sum(1)
    target = result['truth']
    errors = (estimate-target).square()
    if benchmark.name == 'location_finding':
        swap_errors = (estimate.reshape(batch,2,2).flip(1).flatten(1)-target).square()
        errors = torch.where((swap_errors.sum(-1)<errors.sum(-1))[:,None],swap_errors,errors)
    variance = ((samples-estimate[:,None]).square()*sample_weights[...,None]).sum(1)
    rows = []
    for i in range(batch):
        acceptance = float(accepted[i].mean()/iterations) if iterations else None
        row = dict(estimate=estimate[i].tolist(),mse=float(errors[i].mean()),
                   normalized_mse=float((errors[i]/benchmark.prior_variance).mean()),
                   squared_error_by_parameter=dict(zip(benchmark.param_names,errors[i].tolist())),
                   posterior_variance=variance[i].tolist(),importance_ess=float(importance_ess[i]),
                   mh_acceptance=acceptance,
                   posterior_particles=settings['posterior_particles'],posterior_steps=iterations,
                   truth_excluded=True,
                   inference_requires_sensitivity_check=True,
                   inference_degenerate=bool(importance_ess[i]<2 or (acceptance is not None and acceptance==0)))
        if benchmark.name.startswith('ieee'):
            n = benchmark.param_dim//2
            row.update(mse_M=float(errors[i,:n].mean()),mse_K=float(errors[i,n:].mean()))
        rows.append(row)
    if return_artifacts:
        return rows,dict(prior_particles=particles.numpy(),prior_log_likelihood=lp.numpy(),
                         prior_posterior_weights=weights.numpy(),prior_terminal_states=particle_states.numpy(),
                         posterior_samples=samples.numpy(),posterior_sample_weights=sample_weights.numpy())
    return rows
