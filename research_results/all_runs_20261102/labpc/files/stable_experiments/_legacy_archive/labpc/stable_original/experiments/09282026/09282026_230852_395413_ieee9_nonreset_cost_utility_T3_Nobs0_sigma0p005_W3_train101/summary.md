# Cost utility

Utility = -u_ctrl / u_max. Higher is better.
u_max = 0.5; required posterior joint safety probability = 0.9.
Safety requires BOTH the frequency nadir and maximum absolute RoCoF limits.
Coverage is model-based; the safety rates below are independently evaluated.

| Method | Mean utility | Mean control | Joint safety | Frequency safety | RoCoF safety |
|---|---:|---:|---:|---:|---:|
| dad | -0.770798 | 0.385399 | 1.000000 | 1.000000 | 1.000000 |
| rl_sboed | -0.770737 | 0.385368 | 1.000000 | 1.000000 | 1.000000 |
| step_dad | -0.770798 | 0.385399 | 1.000000 | 1.000000 | 1.000000 |
| myopic | -0.770424 | 0.385212 | 1.000000 | 1.000000 | 1.000000 |
| fixed | -0.766342 | 0.383171 | 1.000000 | 1.000000 | 1.000000 |
| random | -0.770859 | 0.385429 | 1.000000 | 1.000000 | 1.000000 |

No infeasible case is silently capped, dropped, or resampled.
The terminal control window is checked; probe-period safety is not enforced by this implementation.
The oracle uses each method's actual terminal state. It is a diagnostic, not subtracted from utility.
