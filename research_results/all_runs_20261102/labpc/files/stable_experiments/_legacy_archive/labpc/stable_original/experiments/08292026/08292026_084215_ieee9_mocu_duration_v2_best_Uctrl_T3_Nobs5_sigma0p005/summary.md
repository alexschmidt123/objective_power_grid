# Summary — ieee9 (objective_based)

Observation: 5 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee9`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 5
- N_sim: 2560
- T: 3
- config: `/home/grads/g/g.lin/Documents/objective_power_grid/configs/ieee9_mocu_duration_v2_best.yaml`

## Comparison

| Method | mean_MOCU | mean_u_ctrl | safety_rate | valid | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.000000 | 0.370213 | 1.000 | VALID | — |
| DAD | 0.024365 | 0.394453 | 0.992 | VALID | 4 |
| RL-sBOED | 0.024365 | 0.394453 | 0.992 | VALID | 4 |
| MoE-sBOED | 0.025347 | 0.394531 | 0.977 | VALID | 3 |
| Myopic | 0.026098 | 0.394609 | 0.961 | VALID | 23 |
| Fixed | 0.025508 | 0.395156 | 0.984 | VALID | 1 |
| Random | 0.027374 | 0.397068 | 0.988 | VALID | 4039 |

Notes:

- `mean_MOCU` = mean safety-aware OCU on common held-out systems: u_ctrl + λ(u_opt−u_ctrl)_+ + ρ·1[unsafe] − u_opt.
- Raw `u_ctrl − u_ctrl_opt` remains a diagnostic; unsafe under-control is penalized and cannot improve `mean_MOCU`.
- Methods with safety rate below 0.95 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_MOCU.
- `mean_u_ctrl` remains a secondary physical-control metric.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.95; mean_MOCU ↓)

DAD, RL-sBOED, MoE-sBOED, Fixed, Myopic, Random
