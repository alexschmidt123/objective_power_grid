# Summary — ieee9 (objective_based)

Observation: 5 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee9`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 5
- N_sim: 2560
- T: 3
- config: `/home/grads/g/g.lin/Documents/objective_power_grid/experiments/09112026_003158_ieee9_mocu_q080_q085_labpc_queue/q085/source_snapshot/configs/ieee9_mocu.yaml`

## Comparison

| Method | Terminal posterior MOCU |
| --- | ---: |
| DAD | 0.009723 |
| RL-sBOED | 0.009380 |
| Step-DAD | 0.009730 |
| Myopic | 0.009636 |
| Fixed | 0.009543 |
| Random | 0.010238 |


## Empirical safety rate

| Method | Safety rate | Safe outcomes / evaluated outcomes | Physical systems |
|---|---:|---:|---:|
| DAD | 94.53% | 121 / 128 | 128 |
| RL-sBOED | 92.97% | 119 / 128 | 128 |
| Step-DAD | 94.53% | 121 / 128 | 128 |
| Myopic | 92.19% | 118 / 128 | 128 |
| Fixed | 94.53% | 121 / 128 | 128 |
| Random | 94.04% | 3852 / 4096 | 128 |

Posterior coverage: 0.85.

Posterior MOCU is primary; safety rate and control magnitude are physical diagnostics.
Safety requires both physical frequency and RoCoF limits to hold in the declared scenario.
Rates average repeats within each physical system before averaging across systems.
These are single-run estimates; across-seed standard deviations require multiple runs.
Coverage is a decision preference, not a measured safety rate or engineering acceptance threshold.
