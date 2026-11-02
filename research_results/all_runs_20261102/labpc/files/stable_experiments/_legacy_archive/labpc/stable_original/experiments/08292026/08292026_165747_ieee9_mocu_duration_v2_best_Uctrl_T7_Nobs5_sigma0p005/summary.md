# Summary — ieee9 (objective_based)

Observation: 5 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee9`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 5
- N_sim: 2560
- T: 7
- config: `/home/grads/g/g.lin/Documents/objective_power_grid/configs/ieee9_mocu_duration_v2_best.yaml`

## Comparison

| Method | mean_MOCU | mean_u_ctrl | safety_rate | valid | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.000000 | 0.370213 | 1.000 | VALID | — |
| DAD | 0.022729 | 0.391563 | 0.969 | VALID | 35 |
| RL-sBOED | 0.022712 | 0.391719 | 0.984 | VALID | 82 |
| MoE-sBOED | 0.022208 | 0.391172 | 0.984 | VALID | 17 |
| Myopic | 0.025888 | 0.391797 | 0.961 | VALID | 114 |
| Fixed | 0.023299 | 0.391484 | 0.961 | VALID | 1 |
| Random | 0.025319 | 0.394456 | 0.983 | VALID | 4096 |

Notes:

- `mean_MOCU` = mean safety-aware OCU on common held-out systems: u_ctrl + λ(u_opt−u_ctrl)_+ + ρ·1[unsafe] − u_opt.
- Raw `u_ctrl − u_ctrl_opt` remains a diagnostic; unsafe under-control is penalized and cannot improve `mean_MOCU`.
- Methods with safety rate below 0.95 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_MOCU.
- `mean_u_ctrl` remains a secondary physical-control metric.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.95; mean_MOCU ↓)

MoE-sBOED, RL-sBOED, DAD, Fixed, Random, Myopic
