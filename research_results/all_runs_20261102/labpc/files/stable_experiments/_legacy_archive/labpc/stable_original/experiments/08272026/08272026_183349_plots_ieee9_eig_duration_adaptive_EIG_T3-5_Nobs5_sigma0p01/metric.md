# Mean terminal EIG ± std (n = 3 seeds) (higher better)

| Method | T=3 | T=4 | T=5 |
| --- | --- | --- | --- |
| DAD | 0.7188 ± 0.0054 | 0.8542 ± 0.0209 | 0.9733 ± 0.0249 |
| RL-SBOED | 0.7187 ± 0.0055 | 0.8646 ± 0.0146 | 0.9765 ± 0.0272 |
| Myopic | 0.7000 ± 0.0186 | 0.8349 ± 0.0143 | 0.9569 ± 0.0216 |
| Fixed | 0.7250 ± 0.0163 | 0.8533 ± 0.0144 | 0.9806 ± 0.0224 |
| Random | 0.5830 ± 0.0022 | 0.7125 ± 0.0029 | 0.8239 ± 0.0023 |
| MoE-sBOED | 0.7055 ± 0.0146 | 0.8442 ± 0.0021 | 0.9672 ± 0.0360 |

Random is averaged over 32 randomized design sequences per held-out system; deterministic methods use one sequence per system.

## Paired EIG improvements

Intervals use a hierarchical seed-then-system bootstrap. Positive differences favor the method on the left.

| T | Comparison | Mean difference | Hierarchical 95% CI | Seeds |
| --- | --- | --- | --- | --- |
| 3 | dad_eig - fixed_open_loop | -0.0062 | [-0.0194, 0.0008] | 3 |
| 3 | dad_eig - myopic_delta_h | 0.0188 | [0.0032, 0.0303] | 3 |
| 3 | dad_eig - random | 0.1357 | [0.0989, 0.1758] | 3 |
| 3 | rl_sboed_eig - fixed_open_loop | -0.0062 | [-0.0191, 0.0007] | 3 |
| 3 | rl_sboed_eig - myopic_delta_h | 0.0188 | [0.0034, 0.0301] | 3 |
| 3 | rl_sboed_eig - random | 0.1357 | [0.0994, 0.1740] | 3 |
| 4 | dad_eig - fixed_open_loop | 0.0009 | [-0.0051, 0.0082] | 3 |
| 4 | dad_eig - myopic_delta_h | 0.0193 | [0.0117, 0.0281] | 3 |
| 4 | dad_eig - random | 0.1418 | [0.0971, 0.1878] | 3 |
| 4 | rl_sboed_eig - fixed_open_loop | 0.0113 | [0.0089, 0.0139] | 3 |
| 4 | rl_sboed_eig - myopic_delta_h | 0.0297 | [0.0254, 0.0343] | 3 |
| 4 | rl_sboed_eig - random | 0.1521 | [0.1100, 0.1973] | 3 |
| 5 | dad_eig - fixed_open_loop | -0.0073 | [-0.0119, -0.0025] | 3 |
| 5 | dad_eig - myopic_delta_h | 0.0164 | [0.0107, 0.0223] | 3 |
| 5 | dad_eig - random | 0.1494 | [0.1021, 0.1955] | 3 |
| 5 | rl_sboed_eig - fixed_open_loop | -0.0041 | [-0.0089, 0.0013] | 3 |
| 5 | rl_sboed_eig - myopic_delta_h | 0.0195 | [0.0127, 0.0270] | 3 |
| 5 | rl_sboed_eig - random | 0.1525 | [0.1035, 0.1985] | 3 |
