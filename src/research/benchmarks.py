"""Continuous benchmark contracts, with explicit likelihoods and prior transforms.

Policies receive observed history only. Grid particles each carry their own
physical state. SIR integrates from fixed initial conditions along actual times.
"""
import math
import numpy as np
import torch

DTYPE = torch.float64


def normal_log_prob(z):
    return (-.5 * z.square() - .5 * math.log(2 * math.pi)).sum(-1)


class Benchmark:
    obs_dim = 1
    action_dim = 1
    state_dim = 0
    def initial(self, shape):
        return torch.empty(*shape, 0, dtype=DTYPE)

    def log_likelihood(self, observation, mean):
        return (-.5 * ((observation - mean) / self.sigma).square()
                - math.log(self.sigma * math.sqrt(2 * math.pi))).sum(-1)

    def encode_action(self, action):
        return action / self.action_scale

    def transform(self, theta):
        return theta

    def inverse(self, z):
        return z

    def log_prior_z(self, z):
        return normal_log_prob(z)

    def forward(self, theta, state, action):
        self.calls += theta.numel() // self.param_dim
        return self.propagate(theta, state, action)


class LocationFinding(Benchmark):
    """Two-source log-signal model; explicit bounds select vsOED Case 1a."""
    name = 'location_finding'
    action_dim = 2
    param_dim = 4
    param_names = ['source1_x', 'source1_y', 'source2_x', 'source2_y']
    action_scale = 1.
    obs_scale = 3.
    def __init__(self, raw):
        self.sigma = float(raw.get('observation', {}).get('noise_sigma', .5))
        self.calls = 0
        self.prior_variance = torch.ones(4, dtype=DTYPE)
        bounds = raw.get('location_finding', {}).get('design_bounds')
        self.design_bounds = [float(x) for x in bounds] if bounds is not None else None
        self.random_design_distribution=raw.get('location_finding',{}).get('random_design_distribution','uniform_box' if self.design_bounds is not None else 'standard_normal')
        self.paper_dad = bool(raw.get('research',{}).get('paper_protocol'))
        self.corrected_location = bool(raw.get('research',{}).get('paper_protocol',{}).get('corrected_methods'))
        if self.design_bounds is not None:
            if len(self.design_bounds) != 2 or self.design_bounds[0] >= self.design_bounds[1]:
                raise ValueError('Invalid location design bounds')
            self.action_scale = max(abs(x) for x in self.design_bounds)

    def sample(self, shape, generator):
        return torch.randn(*shape, 4, generator=generator, dtype=DTYPE)

    def feasible(self, latent, actions, horizon):
        if self.design_bounds is not None:
            lo,hi = self.design_bounds
            return lo+(hi-lo)*latent.sigmoid()
        return latent

    def random_latent(self, batch, generator):
        if self.design_bounds is not None:
            u = torch.rand(batch, 2, generator=generator, dtype=DTYPE).clamp(1e-12, 1-1e-12)
            if self.random_design_distribution=='truncated_standard_normal':
                lo,hi=self.design_bounds
                cdf=lambda z:.5*(1+math.erf(z/math.sqrt(2)))
                physical=math.sqrt(2)*torch.erfinv(2*(cdf(lo)+u*(cdf(hi)-cdf(lo)))-1)
                return torch.logit(((physical-lo)/(hi-lo)).clamp(1e-12,1-1e-12))
            if self.random_design_distribution!='uniform_box':raise ValueError('Bounded Random requires uniform_box or truncated_standard_normal')
            return torch.logit(u)
        return torch.randn(batch, 2, generator=generator, dtype=DTYPE)

    def propagate(self, theta, state, action):
        sources = theta.reshape(*theta.shape[:-1], 2, 2)
        distance = (sources - action.unsqueeze(-2)).square().sum(-1)
        mean = (.1 + (1e-4 + distance).reciprocal().sum(-1)).log().unsqueeze(-1)
        return mean, state


class SIR(Benchmark):
    name = 'sir_ode'
    param_dim = 2
    state_dim = 4  # S,I,R,time
    param_names = ['beta', 'gamma']
    def __init__(self, raw):
        cfg = raw.get('sir_ode', {})
        self.population = float(cfg.get('population', 500.))
        self.i0 = float(cfg.get('i0', 2.))
        self.end = float(cfg.get('t_end', 100.))
        self.gap = float(cfg.get('min_spacing', .1))
        self.steps = int(cfg.get('integration_steps_per_stage', 512))
        self.log_mean = torch.tensor([math.log(cfg.get('beta_mean', .5)),
                                      math.log(cfg.get('gamma_mean', .1))], dtype=DTYPE)
        self.log_std = float(cfg.get('log_std', .5))
        self.sigma = float(raw.get('observation', {}).get('noise_sigma', 1.))
        self.action_scale, self.obs_scale = self.end, self.population
        self.prior_variance = ((math.exp(self.log_std**2)-1)
                               * (2*self.log_mean+self.log_std**2).exp())
        self.calls = 0
        if self.steps < 1 or not 0 < self.i0 < self.population or self.gap <= 0:
            raise ValueError('Invalid SIR integration/initial/spacing settings')

    def sample(self, shape, generator):
        z = torch.randn(*shape, 2, generator=generator, dtype=DTYPE)
        return self.inverse(z)

    def transform(self, theta):
        return (theta.log() - self.log_mean) / self.log_std

    def inverse(self, z):
        return (z*self.log_std + self.log_mean).exp()

    def initial(self, shape):
        out = torch.zeros(*shape, 4, dtype=DTYPE)
        out[..., 0], out[..., 1] = self.population-self.i0, self.i0
        return out

    def feasible(self, latent, actions, horizon):
        lower = actions[-1][..., 0] + self.gap if actions else torch.full_like(latent[..., 0], self.gap)
        upper = self.end - (horizon-len(actions)-1)*self.gap
        if torch.any(lower >= upper):
            raise ValueError('SIR horizon/spacing leaves no feasible time interval')
        return (lower + latent[..., 0].sigmoid()*(upper-lower)).unsqueeze(-1)

    def random_latent(self, batch, generator):
        unit = torch.rand(batch, 1, generator=generator, dtype=DTYPE).clamp(1e-9, 1-1e-9)
        return torch.logit(unit)

    def propagate(self, theta, state, action):
        # Batched adaptive-size integration: fixed RK4 count over each interval.
        # Rate-dependent subdivision bounds dt*(beta+gamma) <= .2 per particle;
        # additional zero-length steps keep the batch vectorized.
        duration = action[..., 0] - state[..., 3]
        counts = torch.maximum(torch.full_like(duration, self.steps, dtype=torch.long),
                               torch.ceil(duration.detach()*theta.detach().sum(-1)/.2).long())
        if torch.any(duration < 0) or int(counts.max()) > 20000:
            raise ValueError('SIR invalid time or excessively stiff prior draw')
        dt = duration / counts
        current = state[..., :3]
        beta, gamma = theta[..., 0], theta[..., 1]
        def rhs(y):
            incidence = beta*y[..., 0]*y[..., 1]/self.population
            recovery = gamma*y[..., 1]
            return torch.stack((-incidence, incidence-recovery, recovery), -1)
        # Checkpoint blocks bound training memory for large contrast budgets;
        # recomputation uses the identical integrator and preserves gradients.
        for start in range(0,int(counts.max()),32):
            stop = min(start+32,int(counts.max()))
            def block(y,interval,start=start,stop=stop):
                for k in range(start,stop):
                    step = (interval*(k < counts)).unsqueeze(-1)
                    a = rhs(y)
                    b = rhs(y + step*a/2)
                    c = rhs(y + step*b/2)
                    d = rhs(y + step*c)
                    y = y + step*(a+2*b+2*c+d)/6
                return y
            if torch.is_grad_enabled() and (current.requires_grad or dt.requires_grad):
                from torch.utils.checkpoint import checkpoint
                current = checkpoint(block,current,dt,use_reentrant=False)
            else:
                current = block(current,dt)
        if not torch.isfinite(current).all() or torch.any(current < -1e-7):
            raise FloatingPointError('SIR integration failed; particles are not dropped')
        return current[..., 1:2], torch.cat((current, action), -1)


class Grid(Benchmark):
    def __init__(self, raw):
        from src.config import load_config
        from src.domains.swing.continuous_rocof import EndpointRocofObserver, MaxRocofObserver
        self.name = raw['system']['name']
        cfg = load_config(raw['_config_path'])
        sw = cfg.swing
        if sw.get('reset_after_probe', False):
            raise ValueError('Research grid requires full state carry, no reset')
        rc = raw['research']
        self.kind = rc.get('observation_kind', 'endpoint_rocof')
        observer_class = {'endpoint_rocof': EndpointRocofObserver,
                          'max_absolute_rocof': MaxRocofObserver}[self.kind]
        self.observer = observer_class(cfg, duration_bounds=tuple(sw['probe_duration_bounds']),
                                      injection_bus=1, amplitude=.05,
                                      window=float(rc.get('window', 3.5)))
        self.physical_calls = 0
        original_propagate = self.observer._propagate
        def counted_propagate(theta,*args,**kwargs):
            self.physical_calls += len(theta)
            return original_propagate(theta,*args,**kwargs)
        self.observer._propagate = counted_propagate
        self.lower = torch.tensor(sw['M_lower_nodes']+sw['K_lower_nodes'], dtype=DTYPE)
        self.upper = torch.tensor(sw['M_upper_nodes']+sw['K_upper_nodes'], dtype=DTYPE)
        self.param_dim = len(self.lower)
        self.state_dim = self.param_dim
        buses = sw['dynamic_machine_buses']
        self.param_names = [f'{p}_bus{b}' for p in ['M','K'] for b in buses]
        self.prior_variance = (self.upper-self.lower).square()/12
        self.sigma = float(raw['observation']['noise_sigma'])
        self.gap = float(raw['experiment'].get('min_duration_separation', .01))
        self.action_scale, self.obs_scale = self.observer.bounds[1], .05
        self.calls = 0

    def sample(self, shape, generator):
        unit = torch.rand(*shape, self.param_dim, generator=generator, dtype=DTYPE)
        return self.lower + unit*(self.upper-self.lower)

    def transform(self, theta):
        return torch.logit(((theta-self.lower)/(self.upper-self.lower)).clamp(1e-12,1-1e-12))

    def inverse(self, z):
        return self.lower + z.sigmoid()*(self.upper-self.lower)

    def log_prior_z(self, z):
        return (-torch.nn.functional.softplus(z)-torch.nn.functional.softplus(-z)).sum(-1)

    def initial(self, shape):
        return torch.from_numpy(self.observer.initial_state(math.prod(shape))).reshape(*shape,self.state_dim)

    def feasible(self, latent, actions, horizon):
        from src.objectives.eig.continuous_pathwise import feasible_duration_torch
        return feasible_duration_torch(latent[...,0].sigmoid(), [x[...,0] for x in actions],
                                       self.observer.bounds, self.gap, horizon).unsqueeze(-1)

    def random_latent(self, batch, generator):
        return torch.logit(torch.rand(batch,1,generator=generator,dtype=DTYPE).clamp(1e-9,1-1e-9))

    def propagate(self, theta, state, action):
        from src.objectives.eig.continuous_pathwise import SwingStage
        shape = theta.shape[:-1]
        mean, final = SwingStage.apply(self.observer, theta.reshape(-1,self.param_dim),
                                      state.reshape(-1,self.state_dim), action[...,0].reshape(-1))
        return mean.reshape(*shape,1), final.reshape(*shape,self.state_dim)


def make_benchmark(raw):
    name = raw['system']['name']
    cls = {'location_finding': LocationFinding, 'sir_ode': SIR,
           'ieee9': Grid, 'ieee14': Grid}.get(name)
    if cls is None:
        raise ValueError(f'Unsupported research benchmark: {name}')
    benchmark = cls(raw)
    if benchmark.sigma <= 0 or not math.isfinite(benchmark.sigma):
        raise ValueError('Positive finite observation noise required')
    return benchmark
