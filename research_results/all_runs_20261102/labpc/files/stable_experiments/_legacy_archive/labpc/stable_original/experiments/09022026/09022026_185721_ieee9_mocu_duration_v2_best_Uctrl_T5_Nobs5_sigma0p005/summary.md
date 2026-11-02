# Summary — ieee9 (objective_based)

Observation: 5 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee9`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 5
- N_sim: 2560
- T: 5
- config: `/home/grads/g/g.lin/Documents/objective_power_grid/configs/ieee9_mocu.yaml`

## Comparison

| Method | mean_MOCU | mean_u_ctrl | safety_rate | valid | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.000000 | 0.370213 | 1.000 | VALID | — |
| DAD | 0.026081 | 0.393281 | 0.969 | VALID | 14 |
| RL-sBOED | 0.026311 | 0.392969 | 0.961 | VALID | 17 |
| Myopic | 0.021924 | 0.391875 | 0.984 | VALID | 74 |
| Fixed | 0.023993 | 0.391875 | 0.977 | VALID | 1 |
| Random | 0.026635 | 0.395767 | 0.983 | VALID | 4096 |
| Step-DAD | 0.025566 | 0.392891 | 0.977 | VALID | 43 |

Notes:

- `mean_MOCU` = mean safety-aware OCU on common held-out systems: u_ctrl + λ(u_opt−u_ctrl)_+ + ρ·1[unsafe] − u_opt.
- Raw `u_ctrl − u_ctrl_opt` remains a diagnostic; unsafe under-control is penalized and cannot improve `mean_MOCU`.
- Methods with safety rate below 0.95 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_MOCU.
- `mean_u_ctrl` remains a secondary physical-control metric.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.95; mean_MOCU ↓)

Myopic, Fixed, Step-DAD, DAD, RL-sBOED, Random
