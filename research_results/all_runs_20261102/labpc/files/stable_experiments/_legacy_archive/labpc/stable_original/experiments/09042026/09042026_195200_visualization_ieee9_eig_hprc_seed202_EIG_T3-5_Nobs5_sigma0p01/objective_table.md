# Mean terminal EIG ± std (n = 1 seeds) (higher better)

| Method | T=3 | T=4 | T=5 |
| --- | --- | --- | --- |
| DAD | 0.6797 ± nan | 0.8374 ± nan | 0.9568 ± nan |
| RL-SBOED | 0.6797 ± nan | 0.8508 ± nan | 0.9617 ± nan |
| Step-DAD | 0.6867 ± nan | 0.8381 ± nan | 0.9585 ± nan |
| Myopic | 0.6751 ± nan | 0.8219 ± nan | 0.9360 ± nan |
| Fixed | 0.6999 ± nan | 0.8389 ± nan | 0.9588 ± nan |
| Random | 0.5882 ± nan | 0.7189 ± nan | 0.8263 ± nan |

Random is averaged over 32 randomized design sequences per held-out system; deterministic methods use one sequence per system.

## Paired EIG improvements

Intervals use a hierarchical seed-then-system bootstrap. Positive differences favor the method on the left.

| T | Comparison | Mean difference | Hierarchical 95% CI | Seeds |
| --- | --- | --- | --- | --- |
| 3 | dad_eig - fixed_open_loop | -0.0202 | [-0.0226, -0.0179] | 1 |
| 3 | dad_eig - myopic_delta_h | 0.0046 | [0.0014, 0.0079] | 1 |
| 3 | dad_eig - random | 0.0915 | [0.0613, 0.1239] | 1 |
| 3 | rl_sboed_eig - fixed_open_loop | -0.0202 | [-0.0227, -0.0178] | 1 |
| 3 | rl_sboed_eig - myopic_delta_h | 0.0046 | [0.0015, 0.0078] | 1 |
| 3 | rl_sboed_eig - random | 0.0915 | [0.0613, 0.1239] | 1 |
| 4 | dad_eig - fixed_open_loop | -0.0016 | [-0.0036, 0.0003] | 1 |
| 4 | dad_eig - myopic_delta_h | 0.0154 | [0.0116, 0.0196] | 1 |
| 4 | dad_eig - random | 0.1185 | [0.0849, 0.1529] | 1 |
| 4 | rl_sboed_eig - fixed_open_loop | 0.0119 | [0.0103, 0.0135] | 1 |
| 4 | rl_sboed_eig - myopic_delta_h | 0.0289 | [0.0245, 0.0336] | 1 |
| 4 | rl_sboed_eig - random | 0.1319 | [0.0970, 0.1666] | 1 |
| 5 | dad_eig - fixed_open_loop | -0.0020 | [-0.0040, -0.0000] | 1 |
| 5 | dad_eig - myopic_delta_h | 0.0208 | [0.0170, 0.0251] | 1 |
| 5 | dad_eig - random | 0.1305 | [0.0979, 0.1647] | 1 |
| 5 | rl_sboed_eig - fixed_open_loop | 0.0029 | [0.0017, 0.0041] | 1 |
| 5 | rl_sboed_eig - myopic_delta_h | 0.0257 | [0.0212, 0.0305] | 1 |
| 5 | rl_sboed_eig - random | 0.1353 | [0.1022, 0.1691] | 1 |
