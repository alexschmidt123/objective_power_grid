# Mean terminal EIG ± std (n = 3 seeds) (higher better)

| Method | T=3 | T=4 | T=5 | T=6 |
| --- | --- | --- | --- | --- |
| DAD | 1.5617 ± 0.0065 | 1.7689 ± 0.0229 | 1.9247 ± 0.0443 | 2.0703 ± 0.0365 |
| RL-SBOED | 1.5617 ± 0.0063 | 1.7553 ± 0.0356 | 1.9277 ± 0.0336 | 2.0757 ± 0.0407 |
| Myopic | 1.5280 ± 0.0046 | 1.7355 ± 0.0216 | 1.9067 ± 0.0348 | 2.0491 ± 0.0403 |
| Fixed | 1.5281 ± 0.0070 | 1.7470 ± 0.0184 | 1.9229 ± 0.0325 | 2.0599 ± 0.0382 |
| Random | 1.3562 ± 0.0055 | 1.5664 ± 0.0034 | 1.7383 ± 0.0021 | 1.8833 ± 0.0049 |
| MoE-sBOED | 1.3691 ± 0.0775 | 1.6609 ± 0.0822 | 1.7290 ± 0.0845 | 2.0382 ± 0.0833 |

Random is averaged over 32 randomized design sequences per held-out system; deterministic methods use one sequence per system.

## Paired EIG improvements

Intervals use a hierarchical seed-then-system bootstrap. Positive differences favor the method on the left.

| T | Comparison | Mean difference | Hierarchical 95% CI | Seeds |
| --- | --- | --- | --- | --- |
| 3 | dad_eig - fixed_open_loop | 0.0336 | [0.0300, 0.0373] | 3 |
| 3 | dad_eig - myopic_delta_h | 0.0337 | [0.0290, 0.0390] | 3 |
| 3 | dad_eig - random | 0.2055 | [0.1644, 0.2480] | 3 |
| 3 | rl_sboed_eig - fixed_open_loop | 0.0336 | [0.0299, 0.0373] | 3 |
| 3 | rl_sboed_eig - myopic_delta_h | 0.0337 | [0.0291, 0.0389] | 3 |
| 3 | rl_sboed_eig - random | 0.2055 | [0.1662, 0.2458] | 3 |
| 4 | dad_eig - fixed_open_loop | 0.0218 | [0.0139, 0.0283] | 3 |
| 4 | dad_eig - myopic_delta_h | 0.0334 | [0.0254, 0.0409] | 3 |
| 4 | dad_eig - random | 0.2025 | [0.1568, 0.2480] | 3 |
| 4 | rl_sboed_eig - fixed_open_loop | 0.0083 | [-0.0112, 0.0267] | 3 |
| 4 | rl_sboed_eig - myopic_delta_h | 0.0199 | [0.0019, 0.0350] | 3 |
| 4 | rl_sboed_eig - random | 0.1890 | [0.1394, 0.2429] | 3 |
| 5 | dad_eig - fixed_open_loop | 0.0018 | [-0.0101, 0.0125] | 3 |
| 5 | dad_eig - myopic_delta_h | 0.0180 | [0.0063, 0.0277] | 3 |
| 5 | dad_eig - random | 0.1864 | [0.1311, 0.2418] | 3 |
| 5 | rl_sboed_eig - fixed_open_loop | 0.0048 | [-0.0061, 0.0125] | 3 |
| 5 | rl_sboed_eig - myopic_delta_h | 0.0210 | [0.0118, 0.0296] | 3 |
| 5 | rl_sboed_eig - random | 0.1894 | [0.1430, 0.2374] | 3 |
| 6 | dad_eig - fixed_open_loop | 0.0104 | [0.0061, 0.0153] | 3 |
| 6 | dad_eig - myopic_delta_h | 0.0212 | [0.0146, 0.0289] | 3 |
| 6 | dad_eig - random | 0.1870 | [0.1394, 0.2382] | 3 |
| 6 | rl_sboed_eig - fixed_open_loop | 0.0158 | [0.0124, 0.0191] | 3 |
| 6 | rl_sboed_eig - myopic_delta_h | 0.0266 | [0.0220, 0.0314] | 3 |
| 6 | rl_sboed_eig - random | 0.1924 | [0.1412, 0.2464] | 3 |
