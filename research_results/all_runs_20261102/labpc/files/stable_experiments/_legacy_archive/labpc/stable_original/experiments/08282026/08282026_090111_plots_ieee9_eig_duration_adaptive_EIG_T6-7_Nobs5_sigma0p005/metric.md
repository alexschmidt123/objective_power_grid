# Mean terminal EIG ± std (n = 3 seeds) (higher better)

| Method | T=6 | T=7 |
| --- | --- | --- |
| DAD | 2.0703 ± 0.0365 | 2.2017 ± 0.0280 |
| RL-SBOED | 2.0757 ± 0.0407 | 2.2108 ± 0.0281 |
| Myopic | 2.0491 ± 0.0403 | 2.1774 ± 0.0316 |
| Fixed | 2.0599 ± 0.0382 | 2.1916 ± 0.0306 |
| Random | 1.8833 ± 0.0049 | 2.0074 ± 0.0023 |
| MoE-sBOED | 2.0382 ± 0.0833 | 2.1380 ± 0.0778 |

Random is averaged over 32 randomized design sequences per held-out system; deterministic methods use one sequence per system.

## Paired EIG improvements

Intervals use a hierarchical seed-then-system bootstrap. Positive differences favor the method on the left.

| T | Comparison | Mean difference | Hierarchical 95% CI | Seeds |
| --- | --- | --- | --- | --- |
| 6 | dad_eig - fixed_open_loop | 0.0104 | [0.0061, 0.0153] | 3 |
| 6 | dad_eig - myopic_delta_h | 0.0212 | [0.0146, 0.0289] | 3 |
| 6 | dad_eig - random | 0.1870 | [0.1394, 0.2382] | 3 |
| 6 | rl_sboed_eig - fixed_open_loop | 0.0158 | [0.0124, 0.0191] | 3 |
| 6 | rl_sboed_eig - myopic_delta_h | 0.0266 | [0.0220, 0.0314] | 3 |
| 6 | rl_sboed_eig - random | 0.1924 | [0.1412, 0.2464] | 3 |
| 7 | dad_eig - fixed_open_loop | 0.0101 | [0.0059, 0.0152] | 3 |
| 7 | dad_eig - myopic_delta_h | 0.0243 | [0.0178, 0.0318] | 3 |
| 7 | dad_eig - random | 0.1943 | [0.1521, 0.2379] | 3 |
| 7 | rl_sboed_eig - fixed_open_loop | 0.0193 | [0.0149, 0.0239] | 3 |
| 7 | rl_sboed_eig - myopic_delta_h | 0.0335 | [0.0273, 0.0410] | 3 |
| 7 | rl_sboed_eig - random | 0.2034 | [0.1617, 0.2463] | 3 |
