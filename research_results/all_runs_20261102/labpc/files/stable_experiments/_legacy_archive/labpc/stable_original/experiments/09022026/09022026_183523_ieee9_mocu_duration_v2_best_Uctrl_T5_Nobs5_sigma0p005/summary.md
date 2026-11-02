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
| DAD | 0.022568 | 0.392656 | 0.992 | VALID | 16 |
| RL-sBOED | 0.022929 | 0.392578 | 0.984 | VALID | 8 |
| Myopic | 0.023137 | 0.392031 | 0.984 | VALID | 87 |
| Fixed | 0.022287 | 0.392500 | 1.000 | VALID | 1 |
| Random | 0.026390 | 0.395828 | 0.984 | VALID | 4096 |
| Step-DAD | 0.024182 | 0.392188 | 0.969 | VALID | 52 |

Notes:

- `mean_MOCU` = mean safety-aware OCU on common held-out systems: u_ctrl + λ(u_opt−u_ctrl)_+ + ρ·1[unsafe] − u_opt.
- Raw `u_ctrl − u_ctrl_opt` remains a diagnostic; unsafe under-control is penalized and cannot improve `mean_MOCU`.
- Methods with safety rate below 0.95 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_MOCU.
- `mean_u_ctrl` remains a secondary physical-control metric.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.95; mean_MOCU ↓)

Fixed, DAD, RL-sBOED, Myopic, Step-DAD, Random
