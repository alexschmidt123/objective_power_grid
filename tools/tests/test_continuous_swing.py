import unittest
from pathlib import Path
import numpy as np
import torch
from scipy.integrate import solve_ivp

from src.config import load_config
from src.domains.swing.continuous import ContinuousSwingObserver, ContinuousParticleBelief
from src.domains.swing.design import Design


class ContinuousSwingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = load_config(Path(__file__).resolve().parents[2] / 'configs/ieee9_eig.yaml')
        cls.observer = ContinuousSwingObserver(cfg, duration_bounds=(.2,3.),
                                               injection_bus=1, amplitude=.05)
        sw = cfg.swing
        cls.theta = np.r_[
            (np.asarray(sw['M_lower_nodes'])+sw['M_upper_nodes'])/2,
            (np.asarray(sw['K_lower_nodes'])+sw['K_upper_nodes'])/2][None, :]

    def test_matches_independent_full_episode_integration(self):
        obs, theta = self.observer, self.theta
        durations = [1.75317, 2.40123, .77231]
        state = obs.initial_state(1)
        recorded = []
        for duration in durations:
            result = obs.propagate(theta, state, duration)
            recorded.append(result.observations[0])
            state = result.terminal_state
        def rhs(t, y):
            k = min(int(t // obs.window), 2)
            return obs.sim._rhs(t-obs.window*k, y, theta[0,:3], theta[0,3:],
                                Design(.05, 0, durations[k]))
        times = np.concatenate([obs.times+obs.window*k for k in range(3)])
        reference = solve_ivp(rhs, (0.,3*obs.window), obs.initial_state(1)[0],
                              t_eval=times, rtol=1e-10, atol=1e-12, max_step=.002)
        self.assertTrue(reference.success)
        np.testing.assert_allclose(np.ravel(recorded), reference.y[3]/(2*np.pi), atol=2e-8)
        np.testing.assert_allclose(state[0], reference.y[:,-1], atol=2e-7)
        reset = obs.propagate(theta, obs.initial_state(1), durations[-1])
        self.assertGreater(np.max(np.abs(reset.observations-recorded[-1])), 1e-5)

    def test_particle_states_are_independent_and_history_conditioned(self):
        particles = np.concatenate([self.theta, self.theta*1.1])
        belief = ContinuousParticleBelief(self.observer, particles, .005)
        before = belief.states.copy()
        proposal = belief.predict(1.75317)
        np.testing.assert_array_equal(before, belief.states)
        belief.update(1.75317, proposal.observations[0])
        np.testing.assert_allclose(belief.states, proposal.terminal_state)
        self.assertGreater(np.max(np.abs(belief.states[0]-belief.states[1])), 1e-6)
        self.assertAlmostEqual(np.exp(belief.log_weights).sum(), 1.)
        second = belief.predict(2.40123)
        reset = self.observer.propagate(particles, before, 2.40123)
        self.assertGreater(np.max(np.abs(second.observations-reset.observations)), 1e-5)
        saved = belief.states.copy()
        score = belief.expected_information(.77231, noise_samples=8, rng=np.random.default_rng(1))
        self.assertTrue(np.isfinite(score))
        np.testing.assert_array_equal(saved, belief.states)

    def test_recording_window_and_continuous_duration(self):
        np.testing.assert_allclose(self.observer.times,
                                   [.0015625,.7515625,1.5015625,2.25,3.])
        for d in [.1, 3.1, np.nan]:
            with self.assertRaises(ValueError):
                self.observer.propagate(self.theta, self.observer.initial_state(1), d)
        a = self.observer.propagate(self.theta, self.observer.initial_state(1), 1.75317)
        b = self.observer.propagate(self.theta, self.observer.initial_state(1), 1.75318)
        self.assertGreater(np.max(np.abs(a.observations-b.observations)), 1e-10)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_matches_cpu_with_particle_specific_carried_states(self):
        from src.domains.swing.continuous_cuda import CudaContinuousSwingObserver
        cfg = load_config(Path(__file__).resolve().parents[2] / 'configs/ieee9_eig.yaml')
        gpu = CudaContinuousSwingObserver(cfg, duration_bounds=(.2,3.), injection_bus=1, amplitude=.05)
        theta = np.concatenate([self.theta, self.theta*1.1])
        cpu_state = gpu.initial_state(2)
        gpu_state = cpu_state.copy()
        for duration in [[.20731,2.40123], [1.75317,.77231], [2.97,1.38]]:
            a = self.observer.propagate(theta, cpu_state, duration)
            b = gpu.propagate(theta, gpu_state, duration)
            np.testing.assert_allclose(a.observations,b.observations,atol=2e-6)
            np.testing.assert_allclose(a.terminal_state,b.terminal_state,atol=2e-5)
            cpu_state, gpu_state = a.terminal_state,b.terminal_state


if __name__ == '__main__':
    unittest.main()
