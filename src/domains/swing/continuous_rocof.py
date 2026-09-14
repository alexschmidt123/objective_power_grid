"""Carried-state maximum absolute RoCoF and its active-sample linearization."""
import numpy as np
from src.domains.swing.continuous_cuda import CudaContinuousSwingObserver


class MaxRocofObserver(CudaContinuousSwingObserver):
    """Keep the physical maximum unchanged; differentiate its active sample.

    At a unique maximum, d(max |r_i|) = sign(r_k) d(r_k). Re-selecting k
    separately for coordinate finite differences can yield a spurious chained
    derivative near a peak switch. Ties remain nondifferentiable; the first
    active sample gives a branch derivative, not a smoothness guarantee.
    """
    def __init__(self, cfg, **kwargs):
        from src.domains.swing.continuous_cuda import _carry_kernel_source
        from src.control.cuda_control import _PrimaryCtx
        from pycuda.compiler import SourceModule
        super().__init__(cfg, n_obs=1, **kwargs)
        down=max(1,int(np.floor(1./(float(cfg.swing['fs_hz'])*self.sim.ode_dt))))
        self.rocof_sample_dt=down*self.sim.ode_dt
        source=_carry_kernel_source()
        old='    for (int s = 0; s < n_steps; ++s) {'
        assert source.count(old)==1
        source=source.replace(old,'    double previous = state_in[idx * 2 * N + N + pb];\n    out_df[idx] = 0.;\n'+old)
        old='        for (int j = 0; j < n_obs; ++j) {\n            if (s == obs_indices[j]) out_df[idx * n_obs + j] = y[N + pb] / (2.0 * pi);\n        }'
        assert old in source
        prefix=f'if ((s+1) % {down} == 0) {{ double slope=(y[N+pb]-previous)/(2.0*pi*{down}*dt); '
        suffix=' previous=y[N+pb]; }'
        forward=source.replace(old,prefix+'out_df[idx]=fmax(out_df[idx],fabs(slope));'+suffix)
        signature='const int *obs_indices\n'
        assert source.count(signature)==1
        capture=source.replace(signature,signature.rstrip()+', int *peak_index, double *peak_sign\n')
        capture=capture.replace('    out_df[idx] = 0.;',
            '    out_df[idx] = 0.; peak_index[idx]=0; peak_sign[idx]=1.;')
        capture=capture.replace(old,prefix+
            'if (peak_index[idx]==0 || fabs(slope)>out_df[idx]) { peak_index[idx]=s+1; peak_sign[idx]=(slope<0.?-1.:1.); } '
            'out_df[idx]=fmax(out_df[idx],fabs(slope));'+suffix)
        selected=source.replace(signature,signature.rstrip()+', const int *peak_index, const double *peak_sign\n')
        selected=selected.replace(old,prefix+
            'if (s+1==peak_index[idx]) out_df[idx]=peak_sign[idx]*slope;'+suffix)
        with _PrimaryCtx():
            self.module=SourceModule(forward)
            self.kernel=self.module.get_function('simulate_continuous_stage')
            self.peak_module=SourceModule(capture)
            self.peak_kernel=self.peak_module.get_function('simulate_continuous_stage')
            self.derivative_module=SourceModule(selected)
            self.derivative_kernel=self.derivative_module.get_function('simulate_continuous_stage')
        self.times=np.array([self.window])

    def prepare_derivative(self, theta, state, duration):
        import pycuda.driver as cuda
        index=np.empty(len(theta),dtype=np.int32)
        sign=np.empty(len(theta),dtype=np.float64)
        self._propagate(theta,state,duration,kernel=self.peak_kernel,
                        extra_args=(cuda.Out(index),cuda.Out(sign)))
        return index,sign

    def propagate_derivative(self, theta, state, duration, active_sample):
        import pycuda.driver as cuda
        index,sign=active_sample
        return self._propagate(theta,state,duration,kernel=self.derivative_kernel,
                              extra_args=(cuda.In(index),cuda.In(sign)))
