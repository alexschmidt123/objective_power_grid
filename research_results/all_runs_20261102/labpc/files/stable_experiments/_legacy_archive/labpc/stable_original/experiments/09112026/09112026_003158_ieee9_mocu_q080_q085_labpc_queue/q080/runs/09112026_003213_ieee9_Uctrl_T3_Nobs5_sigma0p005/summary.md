# Summary — ieee9 (objective_based)

Observation: 5 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee9`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 5
- N_sim: 2560
- T: 3
- config: `/home/grads/g/g.lin/Documents/objective_power_grid/experiments/09112026_003158_ieee9_mocu_q080_q085_labpc_queue/q080/source_snapshot/configs/ieee9_mocu.yaml`

## Comparison

| Method | Terminal posterior MOCU |
| --- | ---: |
| DAD | 0.008737 |
| RL-sBOED | 0.008495 |
| Step-DAD | 0.008859 |
| Myopic | 0.008798 |
| Fixed | 0.008621 |
| Random | 0.009264 |


## Empirical safety rate

| Method | Safety rate | Safe outcomes / evaluated outcomes | Physical systems |
|---|---:|---:|---:|
| DAD | 92.97% | 119 / 128 | 128 |
| RL-sBOED | 91.41% | 117 / 128 | 128 |
| Step-DAD | 92.19% | 118 / 128 | 128 |
| Myopic | 91.41% | 117 / 128 | 128 |
| Fixed | 92.97% | 119 / 128 | 128 |
| Random | 91.99% | 3768 / 4096 | 128 |

Posterior coverage: 0.8.

Posterior MOCU is primary; safety rate and control magnitude are physical diagnostics.
Safety requires both physical frequency and RoCoF limits to hold in the declared scenario.
Rates average repeats within each physical system before averaging across systems.
These are single-run estimates; across-seed standard deviations require multiple runs.
Coverage is a decision preference, not a measured safety rate or engineering acceptance threshold.
