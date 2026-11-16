"""IEEE9 adapter and posterior-predictive training for belief-value MoE.

The planner support is separate from the sPCE evaluation support and truth.
Both experts receive full-remaining-horizon score-function gradients. A shared
Q(b,a) learns matched posterior-predictive branch returns. Routing selects one
expert's proposal, never an average. No PPO rollback or baseline initialization.
"""
import copy
import json
import time
import numpy as np
import torch
from scipy.special import logsumexp
from src.policies.belief_moe import BeliefMoEPolicy, ARCHITECTURE


class ParticleContext:
    def __init__(self, engine, theta, states=None, logw=None):
        self.engine = engine
        self.theta = np.asarray(theta).copy()
        b, p, d = self.theta.shape
        self.states = (engine.observer.initial_state(b*p).reshape(b, p, d)
                       if states is None else states.copy())
        self.logw = np.full((b, p), -np.log(p)) if logw is None else logw.copy()

    def clone(self):
        return ParticleContext(self.engine, self.theta, self.states, self.logw)

    def update(self, duration, observation):
        b, p, d = self.theta.shape
        prediction = self.engine.observer.propagate(
            self.theta.reshape(-1, d), self.states.reshape(-1, d), np.repeat(duration, p))
        means = prediction.observations.reshape(b, p, self.engine.n_obs)
        self.states = prediction.terminal_state.reshape(b, p, d)
        self.logw += -.5*np.sum(((observation[:, None]-means)/self.engine.sigma)**2, -1)
        self.logw -= logsumexp(self.logw, axis=1, keepdims=True)

    def inputs(self, actions, observations, stage):
        e = self.engine
        # Physical-state scales are fixed model coordinates, not test statistics.
        n = self.theta.shape[-1]//2
        scales = np.r_[np.full(n, .1), np.full(n, .01)]
        normalized = 2*(self.theta-e.lower)/(e.upper-e.lower)-1
        tokens = np.concatenate([normalized, np.tanh(self.states/scales)], -1)
        weights = np.exp(self.logw)
        history = e.features(actions, observations, len(self.theta), stage).numpy()
        lo, hi = e.observer.bounds
        lower = np.asarray(actions[-1])+e.min_separation if actions else np.full(len(self.theta), lo)
        upper = hi-(e.horizon-stage-1)*e.min_separation
        entropy = -(weights*self.logw).sum(1)/np.log(weights.shape[1])
        ess = 1/(weights**2).sum(1)/weights.shape[1]
        context = np.column_stack([history, (lower-lo)/(hi-lo),
            np.full(len(self.theta), (upper-lo)/(hi-lo)), entropy, ess])
        return tuple(torch.as_tensor(x, dtype=torch.float32) for x in (tokens, weights, context))


class ContinuousBeliefMoE(BeliefMoEPolicy):
    def __init__(self, engine, planner_particles=128):
        super().__init__(2*len(engine.lower), 1+engine.horizon*(engine.n_obs+2)+4)
        self.planner_particles = planner_particles
        self.horizon, self.n_obs = engine.horizon, engine.n_obs

    def rollout(self, engine, rng, batch, *, stochastic, initial=None,
                history=None, stage_start=0, selector=None, noise=None):
        if selector is not None or history is not None or stage_start:
            raise ValueError('Use the explicit posterior continuation interface')
        return rollout(engine, self, rng, batch, stochastic=stochastic, initial=initial, noise=noise)


def new_belief(engine, rng, batch, count):
    return ParticleContext(engine, rng.uniform(engine.lower, engine.upper,
        size=(batch, count, len(engine.lower))))


def rollout(engine, policy, rng, batch, *, stochastic, initial=None, noise=None,
            belief=None, history=None, stage_start=0, first_latent=None, first_logp=None):
    from src.objectives.eig.continuous_eig import feasible_duration
    theta, states = engine.sample(rng, batch) if initial is None else initial
    theta, states = theta.copy(), states.copy()
    particles = theta.shape[1]
    actions, observations = ([], []) if history is None else copy.deepcopy(history)
    # Separate RNG stream; changing planner size does not change scoring truth,
    # contrast draws, or observation noise.
    if noise is None:
        noise = rng.normal(size=(batch, engine.horizon-stage_start, engine.n_obs))
    if belief is None:
        planner_rng = np.random.default_rng(int(rng.integers(2**63-1)))
        belief = new_belief(engine, planner_rng, batch, policy.planner_particles)
    else:
        belief = belief.clone()
    likelihood = np.zeros((batch, particles))
    infos, logps, snapshots, routing, means_record, units = [], [], [], [], [], []
    for stage in range(stage_start, engine.horizon):
        inputs = belief.inputs(actions, observations, stage)
        snapshots.append((belief.clone(), copy.deepcopy((actions, observations)), inputs))
        if stage == stage_start and first_latent is not None:
            latent, logp = first_latent, first_logp
            selected = torch.full((batch,), -1)
            q = means = torch.zeros(batch, policy.n_experts)
        else:
            latent, logp, selected, q, means = policy.choose(inputs, stochastic=stochastic)
        unit = torch.sigmoid(latent.double()).detach().numpy()
        duration = feasible_duration(unit, actions, engine.observer.bounds, engine.min_separation, engine.horizon)
        prediction = engine.observer.propagate(theta.reshape(-1, theta.shape[-1]),
            states.reshape(-1, states.shape[-1]), np.repeat(duration, particles))
        predicted = prediction.observations.reshape(batch, particles, engine.n_obs)
        states = prediction.terminal_state.reshape(states.shape)
        y = predicted[:, 0]+engine.sigma*noise[:, stage-stage_start]
        likelihood += -.5*np.sum(((y[:, None]-predicted)/engine.sigma)**2, -1)
        infos.append(likelihood[:, 0]-logsumexp(likelihood, axis=1)+np.log(particles))
        belief.update(duration, y)
        actions.append(duration.copy()); observations.append(y.copy())
        logps.append(logp); units.append(unit); means_record.append(predicted[:, 0].copy())
        routing.append({'selected_experts': selected.tolist(), 'predicted_remaining_eig': q.detach().tolist(),
                        'expert_latent_means': means.detach().tolist()})
    return {'info': np.stack(infos, 1), 'actions': actions, 'observations': observations,
        'logprobs': logps, 'snapshots': snapshots, 'routing': routing,
        'states': states, 'theta': theta, 'log_likelihood': likelihood,
        'true_means': means_record, 'unit_actions': units, 'decisions': [], 'controller_seconds': []}


def fantasy_initial(belief, rng, contrasts):
    """Truth and alternatives are independent draws from the current posterior.

    The returned fantasy truth is never identified in the planner inputs.
    """
    indices = np.stack([rng.choice(belief.theta.shape[1], size=contrasts+1,
                                  p=np.exp(w)) for w in belief.logw])
    rows = np.arange(len(indices))[:, None]
    return belief.theta[rows, indices], belief.states[rows, indices]


def improve(engine, policy, optimizer, rng, batch):
    # Prefixes come from actual prior episodes, not evaluator records.
    with torch.no_grad():
        behavior = rollout(engine, policy, rng, batch, stochastic=True)
    stage = int(rng.integers(engine.horizon))
    belief, history, inputs = behavior['snapshots'][stage]
    initial = fantasy_initial(belief, rng, engine.contrasts)
    noise = rng.normal(size=(batch, engine.horizon-stage, engine.n_obs))
    distributions = policy.distributions(inputs)
    latent = distributions.sample()
    logp = distributions.log_prob(latent)
    # Responsibilities depend on pre-action value estimates, not noisy sampled
    # outcomes. Both experts retain an exploration floor while specialization
    # can develop; no equal-use or action-separation penalty is imposed.
    with torch.no_grad():
        predicted = policy.q_values(inputs, distributions.mean)
        responsibility = .2/policy.n_experts + .8*torch.softmax(predicted/.25, -1)
    branches, rewards = [], []
    for k in range(policy.n_experts):
        branch = rollout(engine, policy, rng, batch, stochastic=False,
            initial=initial, belief=belief, history=history, stage_start=stage,
            noise=noise, first_latent=latent[:, k], first_logp=logp[:, k])
        branches.append(branch)
        rewards.append(torch.as_tensor(branch['info'][:, -1], dtype=torch.float32))
    target = torch.stack(rewards, 1)
    q = policy.q_values(inputs, latent.detach())
    actor_losses = []
    for k, branch in enumerate(branches):
        # Continuations use deterministic deployment routing. The score gradient
        # updates the initial proposal only, with the continuation held fixed.
        # A baseline predicted for the mean action is independent of the sampled
        # action. Never subtract Q at the sampled action in REINFORCE.
        advantage = target[:, k]-predicted[:, k]
        actor_losses.append(-(responsibility[:, k]*advantage*
                              logp[:, k]).mean())
    actor = torch.stack(actor_losses).sum()
    critic = .5*(q-target).square().mean()
    loss = actor+critic
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite posterior MoE training loss')
    optimizer.zero_grad(); loss.backward()
    expert_gradients = [float(torch.sqrt(sum(p.grad.square().sum() for p in expert.parameters()
        if p.grad is not None))) for expert in policy.experts]
    norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.))
    optimizer.step()
    return {'stage': stage, 'behavior_eig': float(behavior['info'][:, -1].mean()),
        'branch_remaining_eig': target.mean(0).tolist(), 'actor_loss': float(actor.detach()),
        'value_loss': float(critic.detach()), 'gradient_norm': norm,
        'expert_gradient_norms': expert_gradients,
        'responsibility': responsibility.mean(0).tolist(),
        'behavior_trajectories': batch, 'counterfactual_trajectories': batch*policy.n_experts,
        'simulated_scoring_stages': batch*(engine.horizon+policy.n_experts*(engine.horizon-stage))}


def train(engine, args, directory):
    torch.manual_seed(args.seed)
    policy = ContinuousBeliefMoE(engine, args.planner_particles)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)
    records, updates = [], []
    best, best_state, best_update = -float('inf'), None, None
    start = time.monotonic()
    for update in range(args.updates+1):
        if update:
            updates.append({'update': update, **improve(engine, policy, optimizer, rng, args.batch_size)})
        if update % args.validate_every == 0 or update == args.updates:
            with torch.no_grad():
                result = rollout(engine, policy, np.random.default_rng(args.seed+900000),
                                 args.validation_systems, stochastic=False)
            score = float(result['info'][:, -1].mean())
            if not np.isfinite(score):
                raise FloatingPointError('Non-finite MoE validation')
            routing = []
            for t, record in enumerate(result['routing']):
                means = np.asarray(record['expert_latent_means'])
                routing.append({'stage': t, 'selection_counts': np.bincount(record['selected_experts'],
                    minlength=policy.n_experts).tolist(),
                    'mean_absolute_latent_difference': float(np.abs(means[:, 0]-means[:, 1]).mean())})
            records.append({'update': update, 'validation_utility': score,
                            'routing': routing, 'elapsed_seconds': time.monotonic()-start})
            if score > best:
                best, best_state, best_update = score, copy.deepcopy(policy.state_dict()), update
                torch.save({'state_dict': best_state, 'architecture': ARCHITECTURE,
                    'settings': vars(args), 'best_update': best_update, 'method': 'moe_sboed',
                    'particle_width': policy.particle_width, 'context_width': policy.context_width,
                    'n_experts': 2}, directory/'moe_sboed.pth')
            (directory/'moe_sboed_training.json').write_text(json.dumps(records, indent=2)+'\n')
            (directory/'moe_sboed_optimizer_diagnostics.json').write_text(json.dumps(updates, indent=2)+'\n')
            print(f'[belief-value-moe] update={update}/{args.updates} validation_spce={score:.6f} '
                  f'elapsed={time.monotonic()-start:.1f}s', flush=True)
    policy.load_state_dict(best_state)
    (directory/'moe_sboed_training_diagnostics.json').write_text(json.dumps({
        'architecture': ARCHITECTURE, 'best_update': best_update, 'best_validation_utility': best,
        'completed_updates': args.updates, 'planner_particles': args.planner_particles,
        'n_experts': 2, 'teacher': None, 'truth_in_actor_inputs': False,
        'training': 'posterior-predictive paired full-horizon score-function gradients and action-value regression',
        'evaluation': 'hard selection of expert mean by predicted remaining EIG; no averaging',
        'behavior_trajectories': args.updates*args.batch_size,
        'counterfactual_trajectories': args.updates*args.batch_size*2,
        'simulated_scoring_stages': sum(x['simulated_scoring_stages'] for x in updates),
        'posterior_update': 'weighted independent prior support, propagated under actual history; no resampling',
        'specialization_established': False, 'convergence_established': False}, indent=2)+'\n')
    return policy, time.monotonic()-start
