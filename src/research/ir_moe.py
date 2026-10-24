"""Integrated information-aware sparse routing; deployment is neural inference only."""
import math
import torch
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop
from src.research.deepseek_sboed_v3 import DeepSeekV3Policy, CountedBenchmark
from src.research.engine import generator, rollout, replay_likelihood


def ordered_log_probability(logits, indices):
    """Exact Plackett-Luce ordered probability, without replacement."""
    if indices.shape[-1] > logits.shape[-1]:
        raise ValueError('Selection exceeds population')
    mask = torch.zeros_like(logits, dtype=torch.bool)
    terms = []
    for j in range(indices.shape[-1]):
        index = indices[:, j:j+1]
        if mask.gather(1, index).any():
            raise ValueError('Repeated expert')
        remaining = logits.masked_fill(mask, -torch.inf)
        terms.append(remaining.gather(1, index).squeeze(1) - remaining.logsumexp(-1))
        mask = mask.scatter(1, index, True)
    return torch.stack(terms, -1).sum(-1)


def sample_ordered(logits, k, rng=None):
    # Exponential race is equivalent to Gumbel-top-k; strictly positive exponentials.
    exponential = torch.empty_like(logits).exponential_(generator=rng)
    return (logits.detach() - exponential.log()).topk(k, -1).indices


class IRDeepSeekPolicy(DeepSeekV3Policy):
    method = 'IR-DeepSeekMoE-sBOED'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.variant != 'full':
            raise ValueError('IR has one shared and seven routed experts, top3')
        self._score_cache = []
        self._history_score_cache = []
        self._routing_diagnostic_cache = []
        self.routing_generator = None
        self.intervention = 'learned'
        self.intervention_seed = 410101
        self._intervention_generator = None
    def __getstate__(self):
        state = super().__getstate__()
        state['_score_cache'] = []
        state['_history_score_cache'] = []
        state['_routing_diagnostic_cache'] = []
        state['routing_generator'] = None
        state['_intervention_generator'] = None
        return state
    def forward(self, features, stage):
        if stage == 0:
            self._gate_cache = []
            self._score_cache = []
            self._history_score_cache = []
            self._routing_diagnostic_cache = []
        z = self.encode_context(features, stage)
        logits = self.router(z)
        scores = logits.softmax(-1)
        if self.training:
            indices = sample_ordered(logits, self.top_k, self.routing_generator)
            # Holding the physical history fixed prevents a spurious derivative
            # through previous actions, but keeps current encoder parameter credit.
            conditional_logits = self.router(self.encode_context(features.detach(), stage))
            conditional = ordered_log_probability(conditional_logits, indices)
            full = ordered_log_probability(logits, indices)
            self._score_cache.append(conditional)
            # Explicit stochastic-computation-graph correction: earlier continuous
            # actions affect later routing probabilities through physical history.
            # full-conditional has zero value and ONLY this history derivative.
            self._history_score_cache.append(full-conditional)
            self._gate_cache.append((scores, indices.detach()))
            self._routing_diagnostic_cache.append((z.detach(), logits.detach(), indices.detach()))
        elif self.intervention == 'uniform':
            if self._intervention_generator is None:
                self._intervention_generator = torch.Generator(device=z.device).manual_seed(self.intervention_seed)
            indices = sample_ordered(torch.zeros_like(logits), self.top_k, self._intervention_generator)
        else:
            indices = logits.topk(self.top_k, -1).indices
        weights = scores.gather(1, indices)
        if self.intervention == 'uniform':
            # Match v1 diagnostic: preserve learned top3 total routed mass.
            weights = scores.topk(self.top_k, -1).values.sum(-1, keepdim=True).expand_as(weights)/self.top_k
        shared = self.shared(z)
        routed = self.routed_feature_sum(z, weights, indices)
        if self.intervention == 'mask_shared': shared = shared*0
        if self.intervention == 'mask_routed': routed = routed*0
        self.last_dispatch = dict(indices=indices.detach(), scores=scores.detach(), selected_weights=weights.detach())
        self.last_features = z.detach()
        self.last_contribution = dict(shared_norm=float(shared.detach().norm(dim=-1).mean()), routed_norm=float(routed.detach().norm(dim=-1).mean()), context_norm=float(z.detach().norm(dim=-1).mean()))
        return self.emitter(z + shared + routed)
    def score_variance(self, advantages):
        # Conditional direct-router score gradient per trajectory, holding history
        # and encoder fixed. Exact PL score, no extra backward or simulator calls.
        contributions = None
        for stage, (z, logits, ids) in enumerate(self._routing_diagnostic_cache):
            mask = torch.zeros_like(logits, dtype=torch.bool)
            score = torch.zeros_like(logits)
            for slot in range(self.top_k):
                index = ids[:, slot:slot+1]
                remaining = logits.masked_fill(mask, -torch.inf)
                score += torch.nn.functional.one_hot(index[:, 0], logits.shape[-1]) - remaining.softmax(-1)
                mask = mask.scatter(1, index, True)
            gradient = advantages[:, stage, None, None].detach()*score[:, :, None]*z[:, None, :]
            contributions = gradient if contributions is None else contributions + gradient
        vectors = contributions.flatten(1)
        return dict(direct_router_per_trajectory_variance_trace=float(vectors.var(0).sum()),direct_router_minibatch_estimator_variance_trace=float(vectors.var(0).sum()/len(vectors)),score_variance_scope='Direct conditional router weight gradient; history-path correction excluded; current minibatch fixed policy')

    def routing_loss(self, prefix_information, baseline):
        logp = torch.stack(self._score_cache, 1) + torch.stack(self._history_score_cache, 1)
        advantages = reward_to_go(prefix_information).detach() - baseline.detach()[None]
        if logp.shape != advantages.shape:
            raise ValueError('Routing and prefix stage mismatch')
        return -(advantages*logp).sum(1).mean(), advantages


def reward_to_go(prefix_information):
    # prefix column j is U_(j+1). Route index j is decision t=j+1.
    previous = torch.cat([torch.zeros_like(prefix_information[:, :1]), prefix_information[:, :-1]], 1)
    return prefix_information[:, -1:] - previous


def integrated_spce(benchmark, policy, horizon, batch, seed, contrasts, chunk=1024, recompute=True):
    """Exact terminal pathwise sPCE plus detached matching prefix estimates.

    Prefixes use the identical truth and prior contrast samples as terminal sPCE.
    Detached prefixes are control variates/score rewards, never pathwise losses.
    """
    if benchmark.name != 'location_finding' or contrasts < 1:
        raise ValueError('Validated positive-contrast location experiment only')
    rng = generator(seed)
    truth = benchmark.sample((batch,), rng)
    particles = benchmark.sample((batch, contrasts), rng)
    noise = torch.stack([torch.randn(batch, benchmark.obs_dim, generator=rng, dtype=torch.float64) for _ in range(horizon)], 1)
    counts = dict(executed_parameter_stage_propagations=0, executed_scalar_likelihood_evaluations=0)
    counted = CountedBenchmark(benchmark, counts)
    if policy.training: policy.routing_generator = generator(seed+710000003)
    result = rollout(counted, policy, horizon, batch, seed, 0, truth=truth, noise=noise, prior_particles=particles[:, :0])
    actions, observations = result['actions'], result['observations']
    true_terminal = result['log_likelihood'][:, 0]
    with torch.no_grad():
        _, _, prefix_true = replay_likelihood(counted, truth[:, None], [a.detach() for a in actions], [y.detach() for y in observations])
        prefix_true = torch.stack([x[:, 0] for x in prefix_true], 1)
        prefix_denominator = prefix_true.clone()
    denominator = true_terminal
    def chunk_scores(parameters, *history):
        # Autograd checkpoint replay runs without the caller's default device mode.
        with torch.device(parameters.device):
            likelihood, _, prefixes = replay_likelihood(counted, parameters, history[:horizon], history[horizon:])
        with torch.no_grad():
            prefix_sums = torch.stack([x.detach().logsumexp(-1) for x in prefixes], 1)
        return likelihood.logsumexp(-1), prefix_sums
    for first in range(0, contrasts, chunk):
        parameters = particles[:, first:first+chunk]
        if recompute and torch.is_grad_enabled():
            with set_checkpoint_early_stop(False):
                terminal, prefixes = checkpoint(chunk_scores, parameters, *actions, *observations, use_reentrant=False, preserve_rng_state=False)
        else:
            terminal, prefixes = chunk_scores(parameters, *actions, *observations)
        denominator = torch.logaddexp(denominator, terminal)
        prefix_denominator = torch.logaddexp(prefix_denominator, prefixes.detach())
    result['info'] = (true_terminal-denominator+math.log(contrasts+1))[:, None]
    result['prefix_information'] = prefix_true-prefix_denominator+math.log(contrasts+1)
    result['work_counts'] = counts
    return result


def load_ir(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['method'] != IRDeepSeekPolicy.method: raise ValueError('Not IR checkpoint')
    model = IRDeepSeekPolicy(saved['horizon'], saved['action_scale'], saved['obs_scale'], saved['variant'], saved['balance_factor'])
    model.load_state_dict(saved['state_dict'], strict=True)
    return model.to(device).eval()
