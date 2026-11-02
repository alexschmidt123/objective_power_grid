# Summary — ieee9 (objective_based)

Observation: 5 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee9`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 5
- N_sim: 2560
- T: 4
- config: `/home/grads/g/g.lin/Documents/objective_power_grid/configs/ieee9_mocu_duration_candidate.yaml`

## Comparison

| Method | mean_MOCU | mean_u_ctrl | safety_rate | valid | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.000000 | 0.370213 | 1.000 | VALID | — |
| DAD | 0.017678 | 0.387891 | 1.000 | VALID | 6 |
| RL-sBOED | 0.020264 | 0.388516 | 0.969 | VALID | 2 |
| MoE-sBOED | 0.019310 | 0.388047 | 0.977 | VALID | 3 |
| Myopic | 0.017763 | 0.386094 | 0.961 | VALID | 30 |
| Fixed | 0.018174 | 0.388125 | 0.984 | VALID | 1 |
| Random | 0.023648 | 0.392900 | 0.984 | VALID | 4094 |

Notes:

- `mean_MOCU` = mean safety-aware OCU on common held-out systems: u_ctrl + λ(u_opt−u_ctrl)_+ + ρ·1[unsafe] − u_opt.
- Raw `u_ctrl − u_ctrl_opt` remains a diagnostic; unsafe under-control is penalized and cannot improve `mean_MOCU`.
- Methods with safety rate below 0.95 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_MOCU.
- `mean_u_ctrl` remains a secondary physical-control metric.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.95; mean_MOCU ↓)

DAD, Myopic, Fixed, MoE-sBOED, RL-sBOED, Random
