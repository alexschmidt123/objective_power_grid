# E1 ICML results: current verified status

Location confirmation:46policies, seed1002,2000paired cases,L1000000. Each trained method has1Mtotal primary episodes; residuals500kbase+500kphase. No checkpoint/method selected using confirmation scores. Earlier large-seed partial outputs are retained separately.

| Method | Location T30 mean ± seed SD | SIR T5 | IEEE9 T3 | IEEE14 T3 |
|---|---:|---|---|---|
| Random | 8.38644 (untrained) | Pending | Pending | Pending |
| Optimized Fixed | 8.80009 ± 0.02565 | Pending | Pending | Pending |
| DAD | 9.58136 ± 0.58249 | Pending | Pending | Pending |
| Native RL-sBOED | 9.47741 ± 0.10248 | Pending | Pending | Pending |
| Native vsOED-G-I | 8.49532 ± 0.29519 | Pending | Pending | Pending |
| Native ScoreBED | 9.73956 ± 0.36378 | Pending | Pending | Pending |
| MoE-sBOED | 10.79669 ± 0.12616 | Pending | Pending | Pending |
| ResMoE-sBOED | 10.61491 ± 0.12052 | Pending | Pending | Pending |
| SwitchMoE-sBOED | 10.60240 ± 0.28375 | Pending | Pending | Pending |
| SwitchResMoE-sBOED | 10.59451 ± 0.07195 | Pending | Pending | Pending |

Native algorithms include documented benchmark/resource adaptations. Fixed-budget completion does not prove convergence. Five training replicates limit precision. High EIG alone does not establish expert specialization or temporal persistence. SIR readiness/full training and grid readiness/full training remain separate incomplete stages. No real-grid deployment validation is claimed.
