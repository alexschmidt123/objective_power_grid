# Summary — ieee9 (objective_based)

Observation: 5 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee9`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 5
- N_sim: 2560
- T: 4
- config: `/home/grads/g/g.lin/Documents/objective_power_grid/configs/ieee9_mocu_duration_v2_best.yaml`

## Comparison

| Method | mean_MOCU | mean_u_ctrl | safety_rate | valid | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.000000 | 0.370213 | 1.000 | VALID | — |
| DAD | 0.024215 | 0.393516 | 0.984 | VALID | 10 |
| RL-sBOED | 0.025093 | 0.393750 | 0.984 | VALID | 6 |
| MoE-sBOED | 0.023992 | 0.393594 | 0.984 | VALID | 4 |
| Myopic | 0.026635 | 0.393594 | 0.961 | VALID | 51 |
| Fixed | 0.024929 | 0.393359 | 0.977 | VALID | 1 |
| Random | 0.026931 | 0.396289 | 0.985 | VALID | 4094 |

Notes:

- `mean_MOCU` = mean safety-aware OCU on common held-out systems: u_ctrl + λ(u_opt−u_ctrl)_+ + ρ·1[unsafe] − u_opt.
- Raw `u_ctrl − u_ctrl_opt` remains a diagnostic; unsafe under-control is penalized and cannot improve `mean_MOCU`.
- Methods with safety rate below 0.95 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_MOCU.
- `mean_u_ctrl` remains a secondary physical-control metric.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.95; mean_MOCU ↓)

MoE-sBOED, DAD, Fixed, RL-sBOED, Myopic, Random
