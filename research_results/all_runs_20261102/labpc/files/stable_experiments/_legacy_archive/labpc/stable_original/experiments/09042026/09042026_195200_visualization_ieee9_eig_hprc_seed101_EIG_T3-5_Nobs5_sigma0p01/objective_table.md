# Mean terminal EIG ± std (n = 1 seeds) (higher better)

| Method | T=4 | T=5 |
| --- | --- | --- |
| DAD | 0.8319 ± nan | 0.9457 ± nan |
| RL-SBOED | 0.8479 ± nan | 0.9524 ± nan |
| Step-DAD | 0.8360 ± nan | 0.9487 ± nan |
| Myopic | 0.8219 ± nan | 0.9360 ± nan |
| Fixed | 0.8389 ± nan | 0.9588 ± nan |
| Random | 0.7189 ± nan | 0.8263 ± nan |

Random is averaged over 32 randomized design sequences per held-out system; deterministic methods use one sequence per system.

## Paired EIG improvements

Intervals use a hierarchical seed-then-system bootstrap. Positive differences favor the method on the left.

| T | Comparison | Mean difference | Hierarchical 95% CI | Seeds |
| --- | --- | --- | --- | --- |
| 4 | dad_eig - fixed_open_loop | -0.0070 | [-0.0089, -0.0052] | 1 |
| 4 | dad_eig - myopic_delta_h | 0.0100 | [0.0061, 0.0141] | 1 |
| 4 | dad_eig - random | 0.1130 | [0.0797, 0.1474] | 1 |
| 4 | rl_sboed_eig - fixed_open_loop | 0.0090 | [0.0076, 0.0106] | 1 |
| 4 | rl_sboed_eig - myopic_delta_h | 0.0260 | [0.0218, 0.0305] | 1 |
| 4 | rl_sboed_eig - random | 0.1290 | [0.0943, 0.1638] | 1 |
| 5 | dad_eig - fixed_open_loop | -0.0130 | [-0.0148, -0.0113] | 1 |
| 5 | dad_eig - myopic_delta_h | 0.0098 | [0.0061, 0.0137] | 1 |
| 5 | dad_eig - random | 0.1194 | [0.0871, 0.1536] | 1 |
| 5 | rl_sboed_eig - fixed_open_loop | -0.0064 | [-0.0081, -0.0047] | 1 |
| 5 | rl_sboed_eig - myopic_delta_h | 0.0164 | [0.0122, 0.0208] | 1 |
| 5 | rl_sboed_eig - random | 0.1260 | [0.0925, 0.1600] | 1 |
