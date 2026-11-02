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
| Oracle | 0.000000 | 0.378186 | 1.000 | VALID | — |
| DAD | 0.037101 | 0.410000 | 1.000 | VALID | 4 |
| RL-sBOED | 0.036723 | 0.410000 | 1.000 | VALID | 4 |
| Step-DAD | 0.037101 | 0.410000 | 1.000 | VALID | 4 |
| Myopic | 0.035563 | 0.410000 | 1.000 | VALID | 2 |
| Fixed | 0.033564 | 0.410000 | 1.000 | VALID | 1 |
| Random | 0.034891 | 0.410000 | 1.000 | VALID | 32 |

Notes:

- `mean_MOCU` = mean safety-aware OCU on common held-out systems: u_ctrl + λ(u_opt−u_ctrl)_+ + ρ·1[unsafe] − u_opt.
- Raw `u_ctrl − u_ctrl_opt` remains a diagnostic; unsafe under-control is penalized and cannot improve `mean_MOCU`.
- Methods with safety rate below 0.95 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_MOCU.
- `mean_u_ctrl` remains a secondary physical-control metric.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.95; mean_MOCU ↓)

Fixed, Random, Myopic, RL-sBOED, DAD, Step-DAD
