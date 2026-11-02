# Summary — ieee9 (msc_based)

Observation: 5 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee9`
- experiment_type: `msc_based`
- observation_mode: `sampled_delta_f`
- N_obs: 5
- N_sim: 2560
- T: 3
- config: `/home/grads/g/g.lin/Documents/objective_power_grid/experiments/09112026_101907_ieee9_msc_q0p9_full_T3-5_train101-303_eval1001-1005_Nobs5_sigma0p005/source_snapshot/configs/ieee9_mocu.yaml`

## Comparison

| Method | Posterior MSC |
| --- | ---: |
| DAD | 0.383672 |
| RL-sBOED | 0.383125 |
| Step-DAD | 0.383516 |
| Myopic | 0.383281 |
| Fixed | 0.383359 |
| Random | 0.383521 |


## Empirical safety rate

| Method | Safety rate | Safe outcomes / evaluated outcomes | Physical systems |
|---|---:|---:|---:|
| DAD | 95.31% | 122 / 128 | 128 |
| RL-sBOED | 96.88% | 124 / 128 | 128 |
| Step-DAD | 94.53% | 121 / 128 | 128 |
| Myopic | 98.44% | 126 / 128 | 128 |
| Fixed | 98.44% | 126 / 128 | 128 |
| Random | 96.31% | 3945 / 4096 | 128 |

## Selected control and oracle MSC

| Method | Selected control | Oracle MSC |
|---|---:|---:|
| DAD | 0.383672 | 0.370213 |
| RL-sBOED | 0.383125 | 0.370213 |
| Step-DAD | 0.383516 | 0.370213 |
| Myopic | 0.383281 | 0.370213 |
| Fixed | 0.383359 | 0.370213 |
| Random | 0.383521 | 0.370213 |

Posterior coverage: 0.9.

Safety evaluation: physical_simulation.
Posterior MSC is primary; physical safety is independently evaluated and is not certified by coverage.
Safety requires both physical frequency and RoCoF limits to hold in the declared scenario.
Rates average repeats within each physical system before averaging across systems.
These are single-run estimates; across-seed standard deviations require multiple runs.
Coverage is a decision preference, not a measured safety rate or engineering acceptance threshold.
