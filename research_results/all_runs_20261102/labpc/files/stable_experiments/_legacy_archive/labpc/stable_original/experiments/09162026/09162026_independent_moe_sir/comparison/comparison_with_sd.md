# SIR ODE comparison with evaluation-seed SD

Mean ± sample SD across five evaluation-seed means (1001–1005). Training seed: 101. Each evaluation uses the same 512 test systems. This SD does not measure variability across independently trained policies.

| Method | T=3 | T=4 | T=5 |
|---|---:|---:|---:|
| MoE-sBOED | 6.8525 ± 0.0030 | 6.8702 ± 0.0046 | 6.8817 ± 0.0041 |
| DAD | 6.8493 ± 0.0071 | 6.8634 ± 0.0060 | 6.8747 ± 0.0049 |
| RL-sBOED | 6.8523 ± 0.0034 | 6.8695 ± 0.0062 | 6.8831 ± 0.0062 |
| Step-DAD | 6.8513 ± 0.0057 | 6.8674 ± 0.0056 | 6.8880 ± 0.0029 |
| Myopic | 6.7108 ± 0.0125 | 6.7437 ± 0.0124 | 6.7596 ± 0.0135 |
| Fixed | 6.2816 ± 0.0101 | 6.4501 ± 0.0091 | 6.5360 ± 0.0044 |
| Random | 3.7783 ± 0.0167 | 3.8593 ± 0.0167 | 3.9241 ± 0.0167 |
