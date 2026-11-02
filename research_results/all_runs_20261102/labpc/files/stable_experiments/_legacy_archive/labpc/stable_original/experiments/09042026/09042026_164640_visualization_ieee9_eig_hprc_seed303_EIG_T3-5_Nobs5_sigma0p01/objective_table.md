# Mean terminal EIG ± std (n = 1 seeds) (higher better)

| Method | T=3 | T=4 | T=5 |
| --- | --- | --- | --- |
| DAD | 0.7003 ± nan | 0.8506 ± nan | 0.9502 ± nan |
| RL-SBOED | 0.6994 ± nan | 0.8509 ± nan | 0.9622 ± nan |
| Step-DAD | 0.7003 ± nan | 0.8505 ± nan | 0.9504 ± nan |
| Myopic | 0.6751 ± nan | 0.8219 ± nan | 0.9360 ± nan |
| Fixed | 0.6999 ± nan | 0.8389 ± nan | 0.9588 ± nan |
| Random | 0.5882 ± nan | 0.7189 ± nan | 0.8263 ± nan |

Random is averaged over 32 randomized design sequences per held-out system; deterministic methods use one sequence per system.

## Paired EIG improvements

Intervals use a hierarchical seed-then-system bootstrap. Positive differences favor the method on the left.

| T | Comparison | Mean difference | Hierarchical 95% CI | Seeds |
| --- | --- | --- | --- | --- |
| 3 | dad_eig - fixed_open_loop | 0.0005 | [0.0002, 0.0008] | 1 |
| 3 | dad_eig - myopic_delta_h | 0.0252 | [0.0211, 0.0298] | 1 |
| 3 | dad_eig - random | 0.1122 | [0.0814, 0.1452] | 1 |
| 3 | rl_sboed_eig - fixed_open_loop | -0.0004 | [-0.0011, 0.0001] | 1 |
| 3 | rl_sboed_eig - myopic_delta_h | 0.0243 | [0.0203, 0.0287] | 1 |
| 3 | rl_sboed_eig - random | 0.1112 | [0.0805, 0.1440] | 1 |
| 4 | dad_eig - fixed_open_loop | 0.0117 | [0.0101, 0.0133] | 1 |
| 4 | dad_eig - myopic_delta_h | 0.0287 | [0.0245, 0.0333] | 1 |
| 4 | dad_eig - random | 0.1317 | [0.0983, 0.1665] | 1 |
| 4 | rl_sboed_eig - fixed_open_loop | 0.0120 | [0.0104, 0.0138] | 1 |
| 4 | rl_sboed_eig - myopic_delta_h | 0.0290 | [0.0246, 0.0339] | 1 |
| 4 | rl_sboed_eig - random | 0.1320 | [0.0974, 0.1667] | 1 |
| 5 | dad_eig - fixed_open_loop | -0.0086 | [-0.0108, -0.0064] | 1 |
| 5 | dad_eig - myopic_delta_h | 0.0142 | [0.0105, 0.0181] | 1 |
| 5 | dad_eig - random | 0.1238 | [0.0913, 0.1583] | 1 |
| 5 | rl_sboed_eig - fixed_open_loop | 0.0034 | [0.0018, 0.0050] | 1 |
| 5 | rl_sboed_eig - myopic_delta_h | 0.0262 | [0.0220, 0.0307] | 1 |
| 5 | rl_sboed_eig - random | 0.1359 | [0.1021, 0.1697] | 1 |
