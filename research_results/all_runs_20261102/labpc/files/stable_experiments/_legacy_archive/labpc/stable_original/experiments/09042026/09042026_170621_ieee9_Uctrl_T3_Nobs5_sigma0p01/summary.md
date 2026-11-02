# Summary — ieee9 (objective_based)

Observation: 5 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee9`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 5
- N_sim: 2560
- T: 3
- config: `/home/grads/g/g.lin/Documents/objective_power_grid/configs/ieee9_mocu.yaml`

## Comparison

| Method | mean_MOCU | mean_u_ctrl | safety_rate | valid | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.000000 | 0.370213 | 1.000 | VALID | — |
| DAD | 0.035774 | 0.410000 | 1.000 | VALID | 29 |
| RL-sBOED | 0.036083 | 0.410000 | 1.000 | VALID | 13 |
| Step-DAD | 0.035636 | 0.410000 | 1.000 | VALID | 51 |
| Myopic | 0.035498 | 0.410000 | 1.000 | VALID | 32 |
| Fixed | 0.035739 | 0.410000 | 1.000 | VALID | 1 |
| Random | 0.035873 | 0.410000 | 1.000 | VALID | 4039 |

Notes:

- `mean_MOCU` = mean terminal posterior Yoon MOCU on common held-out systems: `u_ctrl − posterior_mean(u_optimal)`.
- Raw `u_ctrl − u_ctrl_opt` is a separate held-out perfect-information diagnostic and is not the posterior MOCU.
- Methods with safety rate below 0.95 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_MOCU.
- `mean_u_ctrl` remains a secondary physical-control metric.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.95; mean_MOCU ↓)

Myopic, Step-DAD, Fixed, DAD, Random, RL-sBOED
