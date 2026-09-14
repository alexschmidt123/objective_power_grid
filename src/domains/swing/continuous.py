"""Online, history-conditioned swing observations for continuous probe duration.

Each stage lasts the fixed recording window, even if injection stops earlier.
The terminal physical state must be passed into the next stage. There is no
equilibrium-bank approximation and no interpolation on a duration catalog.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import logsumexp

from src.domains.swing.design import build_simulator
from src.observations.likelihood import evenly_spaced_indices


@dataclass(frozen=True)
class StageResponse:
    observations: np.ndarray
    terminal_state: np.ndarray


class ContinuousSwingObserver:
    """Batched deterministic simulator; every M/K row has its own state.

    Public arrays use (systems, ...) shape. States contain all retained rotor
    angles followed by all speed deviations. Noise is applied after simulation,
    never to the latent state. Physical bus numbering is one-based at this API.
    """

    def __init__(self, cfg, *, duration_bounds, injection_bus, amplitude,
                 n_obs=5, window=3.0, rtol=1e-8, atol=1e-10):
        if str(cfg.raw.get('system', {}).get('name', '')).lower() != 'ieee9':
            raise ValueError('The initial continuous protocol is restricted to IEEE9')
        self.sim = build_simulator(cfg)
        self.N = self.sim.N
        self.bounds = tuple(float(v) for v in duration_bounds)
        self.window = float(window)
        if len(self.bounds) != 2 or not 0 < self.bounds[0] < self.bounds[1] <= self.window:
            raise ValueError('Require 0 < duration lower < upper <= recording window')
        self.bus = int(injection_bus) - 1
        if not 0 <= self.bus < self.sim.physical_input_map.shape[0]:
            raise ValueError('Invalid physical injection bus')
        self.amplitude = float(amplitude)
        if not np.isfinite(self.amplitude):
            raise ValueError('Nonfinite amplitude')
        self.n_obs = int(n_obs)
        if self.n_obs < 1:
            raise ValueError('Continuous frequency observation requires N_obs >= 1')
        # Preserve existing five-point protocol, including the first post-step
        # point, to isolate the change in design/state carry from sensor changes.
        steps = int(round(self.window / self.sim.ode_dt))
        if not np.isclose(steps * self.sim.ode_dt, self.window, atol=1e-12):
            raise ValueError('Recording window must be an integer number of ODE steps')
        self.times = (evenly_spaced_indices(steps, self.n_obs) + 1) * self.sim.ode_dt
        self.rtol, self.atol = float(rtol), float(atol)

    def initial_state(self, count):
        return np.broadcast_to(np.r_[self.sim.theta0, self.sim.omega0],
                               (int(count), 2 * self.N)).copy()

    def propagate(self, theta, state, duration):
        theta = np.asarray(theta, dtype=np.float64)
        state = np.asarray(state, dtype=np.float64)
        if theta.ndim != 2 or theta.shape[1] != 2 * self.N or state.shape != theta.shape:
            raise ValueError('Expected matching (systems, 2*N) theta and state arrays')
        count = len(theta)
        durations = np.broadcast_to(np.asarray(duration, dtype=np.float64), (count,))
        if (not np.all(np.isfinite(theta)) or not np.all(np.isfinite(state)) or
                not np.all(np.isfinite(durations)) or np.any(theta[:, :self.N] <= 0)):
            raise ValueError('Nonfinite input or nonpositive inertia')
        if np.any(durations < self.bounds[0]) or np.any(durations > self.bounds[1]):
            raise ValueError('Duration outside declared continuous bounds')
        M, K = theta[:, :self.N], theta[:, self.N:]
        decay = K / (2 * np.pi) + self.sim.D_nodes
        injection = self.amplitude * self.sim.physical_input_map[self.bus]

        def rhs(t, flat):
            y = flat.reshape(count, 2 * self.N)
            angles, omega = y[:, :self.N], y[:, self.N:]
            coupling = (self.sim.B * np.sin(angles[:, :, None] - angles[:, None, :])).sum(-1)
            pulse = np.where(t <= durations, .5 * (1 - np.cos(2*np.pi*t/durations)), 0.)
            acceleration = (self.sim.P_m - coupling - decay * omega + pulse[:, None] * injection) / M
            return np.concatenate((omega, acceleration), axis=1).ravel()

        sample_times = np.unique(np.r_[self.times, self.window])
        sol = solve_ivp(rhs, (0., self.window), state.ravel(), t_eval=sample_times,
                        method='DOP853', rtol=self.rtol, atol=self.atol,
                        max_step=min(self.bounds[0] / 8., .025))
        if not sol.success or not np.all(np.isfinite(sol.y)):
            raise RuntimeError('Continuous swing integration failed: ' + sol.message)
        values = sol.y.T.reshape(len(sample_times), count, 2*self.N)
        obs = values[np.searchsorted(sample_times, self.times), :, self.N + self.sim.observation_bus].T / (2*np.pi)
        return StageResponse(obs.copy(), values[-1].copy())


class ContinuousParticleBelief:
    """Stateful likelihood for fixed M/K particles under the applied history.

    No evaluator truth is accepted by this class. Candidate states advance
    under the same actual duration, even when their posterior weights are low.
    """

    def __init__(self, observer, particles, sigma):
        self.observer = observer
        self.particles = np.asarray(particles, dtype=np.float64).copy()
        self.states = observer.initial_state(len(self.particles))
        self.log_weights = np.full(len(self.particles), -np.log(len(self.particles)))
        self.sigma = float(sigma)
        if not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError('Positive finite observation sigma required')
        self.stage = 0

    def predict(self, duration):
        return self.observer.propagate(self.particles, self.states, duration)

    def update(self, duration, observation):
        observation = np.asarray(observation, dtype=np.float64)
        if observation.shape != (self.observer.n_obs,) or not np.all(np.isfinite(observation)):
            raise ValueError('Invalid observation vector')
        prediction = self.predict(duration)
        loglike = -.5 * np.sum(((observation - prediction.observations) / self.sigma)**2, axis=1)
        normalizer = logsumexp(self.log_weights + loglike)
        self.log_weights += loglike - normalizer
        self.states = prediction.terminal_state
        self.stage += 1
        return float(normalizer - self.observer.n_obs * np.log(self.sigma * np.sqrt(2*np.pi)))

    def expected_information(self, duration, *, noise_samples, rng):
        """Conditional MI Monte Carlo estimate for a proposed next duration.

        This is a one-step design score, not a terminal EIG evaluation statistic.
        It leaves the actual history unchanged. Fantasies use the current
        posterior and candidate-specific carried states.
        """
        means = self.predict(duration).observations
        indices = rng.choice(len(means), size=int(noise_samples), p=np.exp(self.log_weights))
        noise = rng.normal(size=(len(indices), self.observer.n_obs))
        y = means[indices] + self.sigma * noise
        likelihood = -.5 * np.sum(((y[:, None, :] - means[None, :, :]) / self.sigma)**2, axis=-1)
        return float(np.mean(-.5*np.sum(noise**2, axis=-1) - logsumexp(likelihood+self.log_weights, axis=1)))
