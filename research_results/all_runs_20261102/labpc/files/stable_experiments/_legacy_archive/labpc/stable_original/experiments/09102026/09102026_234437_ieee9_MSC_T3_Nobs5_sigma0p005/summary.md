# Summary — ieee9 (msc_based)

Observation: 5 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee9`
- experiment_type: `msc_based`
- observation_mode: `sampled_delta_f`
- N_obs: 5
- N_sim: 2560
- T: 3
- config: `/home/grads/g/g.lin/Documents/objective_power_grid/experiments/09102026_234434_ieee9_msc_smoke/ieee9_msc.yaml`

## Comparison

| Method | Posterior MSC |
| --- | ---: |
| DAD | 0.395000 |
| RL-sBOED | 0.390000 |
| Step-DAD | 0.395000 |
| Myopic | 0.392500 |
| Fixed | 0.400000 |
| Random | 0.392813 |


## Empirical safety rate

| Method | Safety rate | Safe outcomes / evaluated outcomes | Physical systems |
|---|---:|---:|---:|
| DAD | 100.00% | 4 / 4 | 4 |
| RL-sBOED | 100.00% | 4 / 4 | 4 |
| Step-DAD | 100.00% | 4 / 4 | 4 |
| Myopic | 100.00% | 4 / 4 | 4 |
| Fixed | 100.00% | 4 / 4 | 4 |
| Random | 100.00% | 32 / 32 | 4 |

## Selected control and oracle MSC

| Method | Selected control | Oracle MSC |
|---|---:|---:|
| DAD | 0.395000 | 0.378186 |
| RL-sBOED | 0.390000 | 0.378186 |
| Step-DAD | 0.395000 | 0.378186 |
| Myopic | 0.392500 | 0.378186 |
| Fixed | 0.400000 | 0.378186 |
| Random | 0.392813 | 0.378186 |

Posterior coverage: 0.95.

Safety evaluation: oracle_threshold_proxy.
Posterior MSC is primary; physical safety is independently evaluated and is not certified by coverage.
Safety requires both physical frequency and RoCoF limits to hold in the declared scenario.
Rates average repeats within each physical system before averaging across systems.
These are single-run estimates; across-seed standard deviations require multiple runs.
Coverage is a decision preference, not a measured safety rate or engineering acceptance threshold.
