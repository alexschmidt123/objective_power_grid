# Summary — ieee9 (objective_based)

Observation: 5 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee9`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 5
- N_sim: 2560
- T: 6
- config: `/home/grads/g/g.lin/Documents/objective_power_grid/configs/ieee9_mocu.yaml`

## Comparison

| Method | mean_MOCU | mean_u_ctrl | safety_rate | valid | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.000000 | 0.370213 | 1.000 | VALID | — |
| DAD | 0.022491 | 0.391563 | 0.977 | VALID | 7 |
| RL-sBOED | 0.022793 | 0.391875 | 0.984 | VALID | 20 |
| Myopic | 0.022075 | 0.391094 | 0.992 | VALID | 107 |
| Fixed | 0.021116 | 0.391328 | 1.000 | VALID | 1 |
| Random | 0.026025 | 0.395127 | 0.983 | VALID | 4096 |
| Step-DAD | 0.022754 | 0.392188 | 0.977 | VALID | 23 |

Notes:

- `mean_MOCU` = mean safety-aware OCU on common held-out systems: u_ctrl + λ(u_opt−u_ctrl)_+ + ρ·1[unsafe] − u_opt.
- Raw `u_ctrl − u_ctrl_opt` remains a diagnostic; unsafe under-control is penalized and cannot improve `mean_MOCU`.
- Methods with safety rate below 0.95 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_MOCU.
- `mean_u_ctrl` remains a secondary physical-control metric.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.95; mean_MOCU ↓)

Fixed, Myopic, DAD, Step-DAD, RL-sBOED, Random
