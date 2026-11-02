# Summary — ieee14 (objective_based)

Observation: 5 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee14`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 5
- N_sim: 2560
- T: 3
- config: `/tmp/ieee14_mocu_candidate_fast.yaml`

## Comparison

| Method | mean_MOCU | mean_u_ctrl | safety_rate | valid | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.000000 | 0.378501 | 1.000 | VALID | — |
| DAD | 0.019086 | 0.397344 | 0.992 | VALID | 1 |
| RL-sBOED | 0.020327 | 0.398828 | 1.000 | VALID | 3 |
| Myopic | 0.021857 | 0.397500 | 0.977 | VALID | 48 |
| Fixed | 0.020516 | 0.398516 | 0.992 | VALID | 1 |
| Random | 0.021687 | 0.399463 | 0.990 | VALID | 4077 |
| Step-DAD | 0.019744 | 0.397500 | 0.984 | VALID | 2 |

Notes:

- `mean_MOCU` = mean safety-aware OCU on common held-out systems: u_ctrl + λ(u_opt−u_ctrl)_+ + ρ·1[unsafe] − u_opt.
- Raw `u_ctrl − u_ctrl_opt` remains a diagnostic; unsafe under-control is penalized and cannot improve `mean_MOCU`.
- Methods with safety rate below 0.95 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_MOCU.
- `mean_u_ctrl` remains a secondary physical-control metric.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.95; mean_MOCU ↓)

DAD, Step-DAD, RL-sBOED, Fixed, Random, Myopic
