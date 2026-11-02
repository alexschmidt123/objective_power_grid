# Mean terminal EIG ± std (n = 1 seeds) (higher better)

| Method | T=3 | T=4 | T=5 |
| --- | --- | --- | --- |
| DAD | 0.6931 ± nan | 0.8209 ± nan | 0.9345 ± nan |
| RL-SBOED | 0.6926 ± nan | 0.8200 ± nan | 0.9306 ± nan |
| Myopic | 0.6761 ± nan | 0.7927 ± nan | 0.9063 ± nan |
| Fixed | 0.6925 ± nan | 0.8205 ± nan | 0.9242 ± nan |
| Random | 0.3731 ± nan | 0.4707 ± nan | 0.5607 ± nan |
| MoE-sBOED | 0.6933 ± nan | 0.7882 ± nan | 0.9282 ± nan |

Random is averaged over 32 randomized design sequences per held-out system; deterministic methods use one sequence per system.

## Paired EIG improvements

Intervals use a hierarchical seed-then-system bootstrap. Positive differences favor the method on the left.

| T | Comparison | Mean difference | Hierarchical 95% CI | Seeds |
| --- | --- | --- | --- | --- |
| 3 | dad_eig - fixed_open_loop | 0.0007 | [0.0001, 0.0012] | 1 |
| 3 | dad_eig - myopic_delta_h | 0.0170 | [0.0129, 0.0215] | 1 |
| 3 | dad_eig - random | 0.3200 | [0.2554, 0.3911] | 1 |
| 3 | rl_sboed_eig - fixed_open_loop | 0.0001 | [-0.0004, 0.0006] | 1 |
| 3 | rl_sboed_eig - myopic_delta_h | 0.0165 | [0.0124, 0.0210] | 1 |
| 3 | rl_sboed_eig - random | 0.3195 | [0.2547, 0.3922] | 1 |
| 4 | dad_eig - fixed_open_loop | 0.0004 | [-0.0002, 0.0010] | 1 |
| 4 | dad_eig - myopic_delta_h | 0.0282 | [0.0185, 0.0402] | 1 |
| 4 | dad_eig - random | 0.3502 | [0.2767, 0.4323] | 1 |
| 4 | rl_sboed_eig - fixed_open_loop | -0.0005 | [-0.0009, -0.0001] | 1 |
| 4 | rl_sboed_eig - myopic_delta_h | 0.0273 | [0.0178, 0.0391] | 1 |
| 4 | rl_sboed_eig - random | 0.3493 | [0.2751, 0.4290] | 1 |
| 5 | dad_eig - fixed_open_loop | 0.0103 | [0.0071, 0.0134] | 1 |
| 5 | dad_eig - myopic_delta_h | 0.0282 | [0.0158, 0.0417] | 1 |
| 5 | dad_eig - random | 0.3737 | [0.2987, 0.4560] | 1 |
| 5 | rl_sboed_eig - fixed_open_loop | 0.0063 | [0.0041, 0.0087] | 1 |
| 5 | rl_sboed_eig - myopic_delta_h | 0.0243 | [0.0121, 0.0380] | 1 |
| 5 | rl_sboed_eig - random | 0.3698 | [0.2931, 0.4519] | 1 |
