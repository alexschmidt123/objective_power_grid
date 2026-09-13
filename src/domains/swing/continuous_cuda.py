"""Batched online CUDA propagation using the existing swing RK4 equations."""
from __future__ import annotations
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from src.domains.swing.cuda import CUDA_DELTA_F_KERNEL
from src.domains.swing.continuous import ContinuousSwingObserver, StageResponse


def _carry_kernel_source():
    # Reuse the production equations verbatim. Assert every transformation so a
    # future kernel edit cannot silently reintroduce equilibrium resets.
    replacements = {
        'simulate_delta_f_trajectories(': 'simulate_continuous_stage(',
        'double *out_df\n': 'double *out_df,\n    const double *state_in,\n    double *state_out,\n    const int n_obs,\n    const int *obs_indices\n',
        'y[i] = theta0[i];': 'y[i] = state_in[idx * 2 * N + i];',
        'y[N + i] = omega0[i];': 'y[N + i] = state_in[idx * 2 * N + N + i];',
        'out_df[idx * n_steps + s] = y[N + pb] / (2.0 * pi);': '''
        for (int j = 0; j < n_obs; ++j) {
            if (s == obs_indices[j]) out_df[idx * n_obs + j] = y[N + pb] / (2.0 * pi);
        }
        if (s == n_steps - 1) {
            for (int j = 0; j < 2 * N; ++j) state_out[idx * 2 * N + j] = y[j];
        }''',
    }
    source = CUDA_DELTA_F_KERNEL
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError('Production kernel changed; review carry-state transformation: ' + old)
        source = source.replace(old, new)
    return source


class CudaContinuousSwingObserver(ContinuousSwingObserver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Compile and execute on the same primary context as Torch/control.
        from src.control.cuda_control import _ensure_ctrl_kernel, _PrimaryCtx
        _ensure_ctrl_kernel()
        with _PrimaryCtx():
            self.module = SourceModule(_carry_kernel_source())
            self.kernel = self.module.get_function("simulate_continuous_stage")

    def propagate(self, theta, state, duration):
        theta = np.asarray(theta, dtype=np.float64)
        state = np.ascontiguousarray(state, dtype=np.float64)
        if theta.ndim != 2 or theta.shape[1] != 2*self.N or state.shape != theta.shape:
            raise ValueError('Expected matching (systems, 2*N) theta and state arrays')
        count = len(theta)
        duration = np.broadcast_to(np.asarray(duration, dtype=np.float64), (count,)).copy()
        if (not np.all(np.isfinite(theta)) or not np.all(np.isfinite(state)) or
                not np.all(np.isfinite(duration)) or np.any(theta[:,:self.N] <= 0)):
            raise ValueError('Invalid theta/state/duration')
        if np.any(duration < self.bounds[0]) or np.any(duration > self.bounds[1]):
            raise ValueError('Duration outside declared bounds')
        output = np.empty((count, self.n_obs), dtype=np.float64)
        final = np.empty_like(state)
        n_steps = int(round(self.window/self.sim.ode_dt))
        indices = np.asarray(np.rint(self.times/self.sim.ode_dt)-1, dtype=np.int32)
        from src.control.cuda_control import _PrimaryCtx
        with _PrimaryCtx():
            self.kernel(
                np.int32(count), np.int32(n_steps), np.int32(self.N),
                cuda.In(np.arange(count, dtype=np.int32)),
                cuda.In(np.ascontiguousarray(theta[:,:self.N])),
                cuda.In(np.ascontiguousarray(theta[:,self.N:])),
                cuda.In(np.ascontiguousarray(self.sim.B)), cuda.In(self.sim.P_m), cuda.In(self.sim.D_nodes),
                cuda.In(self.sim.theta0), cuda.In(self.sim.omega0),
                cuda.In(np.full(count, self.amplitude)),
                cuda.In(np.tile(self.sim.physical_input_map[self.bus], (count,1))),
                np.int32(self.sim.observation_bus), cuda.In(duration), np.float64(self.sim.ode_dt),
                cuda.Out(output), cuda.In(state), cuda.Out(final), np.int32(self.n_obs), cuda.In(indices),
                block=(128,1,1), grid=((count+127)//128,1))
        if not np.all(np.isfinite(output)) or not np.all(np.isfinite(final)):
            raise RuntimeError('Nonfinite continuous CUDA trajectory')
        return StageResponse(output, final)
