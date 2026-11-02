# Summary — ieee14 (objective_based)

Observation: 5 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee14`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 5
- N_sim: 2560
- T: 3
- config: `/home/grads/g/g.lin/Documents/objective_power_grid/configs/ieee14_mocu_duration_selected.yaml`

## Comparison

| Method | mean_MOCU | mean_u_ctrl | safety_rate | valid | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.000000 | 0.171414 | 1.000 | VALID | — |
| DAD | 0.071944 | 0.233203 | 0.961 | VALID | 1 |
| RL-sBOED | 0.066722 | 0.233672 | 0.969 | VALID | 2 |
| MoE-sBOED | 0.068409 | 0.233203 | 0.969 | VALID | 10 |
| Myopic | 0.064915 | 0.236328 | 1.000 | VALID | 22 |
| Fixed | 0.066790 | 0.238203 | 1.000 | VALID | 1 |
| Random | 0.074481 | 0.243240 | 0.984 | VALID | 4077 |

Notes:

- `mean_MOCU` = mean safety-aware OCU on common held-out systems: u_ctrl + λ(u_opt−u_ctrl)_+ + ρ·1[unsafe] − u_opt.
- Raw `u_ctrl − u_ctrl_opt` remains a diagnostic; unsafe under-control is penalized and cannot improve `mean_MOCU`.
- Methods with safety rate below 0.95 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_MOCU.
- `mean_u_ctrl` remains a secondary physical-control metric.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.95; mean_MOCU ↓)

Myopic, RL-sBOED, Fixed, MoE-sBOED, DAD, Random
