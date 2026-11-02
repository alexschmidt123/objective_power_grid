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
| DAD | 0.069693 | 0.236406 | 0.977 | VALID | 1 |
| RL-sBOED | 0.064836 | 0.236250 | 1.000 | VALID | 4 |
| MoE-sBOED | 0.063375 | 0.233828 | 0.984 | VALID | 5 |
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

MoE-sBOED, RL-sBOED, Myopic, Fixed, DAD, Random
