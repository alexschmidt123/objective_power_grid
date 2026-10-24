"""Higher-contrast native terminal sPCE with checkpointed likelihood replay."""
import math
import torch
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop
from src.research.deepseek_sboed_v2 import DeepSeekV2Policy
from src.research.engine import generator, rollout, replay_likelihood

class DeepSeekV3Policy(DeepSeekV2Policy):
    method = 'DeepSeekMoE-sBOED-v3'
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.expert_optimizer_updates=[0]*len(self.routed)
    def mask_unused_gradients(self):
        super().mask_unused_gradients()
        for i,expert in enumerate(self.routed):
            # Adam executes when gradient exists, including a legitimate zero.
            if any(p.grad is not None for p in expert.parameters()):
                self.expert_optimizer_updates[i]+=1
    def counts(self):
        result=super().counts()
        result['expert_updates_with_gradient_tensor']=self.expert_optimizer_updates.copy()
        return result

class CountedBenchmark:
    def __init__(self, benchmark, counts):
        self.benchmark, self.counts = benchmark, counts
    def __getattr__(self, key): return getattr(self.benchmark, key)
    def forward(self, theta, state, action):
        self.counts['executed_parameter_stage_propagations'] += theta.numel() // self.param_dim
        return self.benchmark.forward(theta, state, action)
    def log_likelihood(self, y, mean):
        result = self.benchmark.log_likelihood(y, mean)
        self.counts['executed_scalar_likelihood_evaluations'] += result.numel()
        return result

def streamed_spce(benchmark, policy, horizon, batch, seed, contrasts, chunk=1024, recompute=True):
    if benchmark.name != 'location_finding': raise ValueError('Validated stateless location adapter only')
    if contrasts < 1 or chunk < 1: raise ValueError('Positive contrasts/chunk required')
    rng = generator(seed)
    truth = benchmark.sample((batch,), rng)
    particles = benchmark.sample((batch, contrasts), rng)
    noise = torch.stack([torch.randn(batch, benchmark.obs_dim, generator=rng, dtype=torch.float64) for _ in range(horizon)], 1)
    counts = dict(executed_parameter_stage_propagations=0, executed_scalar_likelihood_evaluations=0)
    counted = CountedBenchmark(benchmark, counts)
    result = rollout(counted, policy, horizon, batch, seed, 0, truth=truth, noise=noise, prior_particles=particles[:, :0])
    actions, observations = result['actions'], result['observations']
    true_loglik = result['log_likelihood'][:, 0]
    log_denominator = true_loglik
    def chunk_logsum(parameters, *history):
        likelihood, _, _ = replay_likelihood(counted, parameters, history[:horizon], history[horizon:])
        return torch.logsumexp(likelihood, -1)
    for first in range(0, contrasts, chunk):
        parameters = particles[:, first:first+chunk]
        if recompute and torch.is_grad_enabled():
            with set_checkpoint_early_stop(False):
                part = checkpoint(chunk_logsum, parameters, *actions, *observations, use_reentrant=False, preserve_rng_state=False)
        else:
            part = chunk_logsum(parameters, *actions, *observations)
        log_denominator = torch.logaddexp(log_denominator, part)
    result['info'] = (true_loglik - log_denominator + math.log(contrasts+1))[:, None]
    result['work_counts'] = counts
    result['particles'] = particles
    result['training_information_scope'] = 'terminal full-horizon sPCE; intermediate scores not returned'
    return result

def load_v3(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('method') != DeepSeekV3Policy.method: raise ValueError('Not v3')
    p = DeepSeekV3Policy(state['horizon'], state['action_scale'], state['obs_scale'], state['variant'], state['balance_factor'])
    p.load_state_dict(state['state_dict'], strict=True)
    return p.to(device).eval()
