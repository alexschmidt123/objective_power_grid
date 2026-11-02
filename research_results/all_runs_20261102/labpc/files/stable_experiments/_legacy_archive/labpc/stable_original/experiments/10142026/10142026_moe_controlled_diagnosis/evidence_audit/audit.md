# MoE evidence audit

unresolved pending completed architecture controls and inference sensitivity

| Checkpoint | EIG | MSE | Training seconds |
|---|---:|---:|---:|
| original | 7.326120792222526 | 0.14491523848765056 | 1891.7864411261398 |
| component | 8.650043989674257 | 0.14193486639626657 | 1828.2797218298074 |
| slower | 8.819067662395133 | 0.11346794045546531 | 1882.3553197381552 |

| Architecture | EIG |
|---|---:|
| A | 10.66259855342537 |
| B | 8.31630832989122 |
| C | 8.242002540010274 |
| D | 8.650043989674257 |
| capacity | 5.543529865833229 |

Correction: pending completed controlled evidence. No sparse-MoE advantage established.

Historical score distributions were not logged; fresh checkpoint probes are explicitly distinguished.

The paired failure analysis supports a policy weakness. Larger-budget inference results are required to assess the secondary MSE contribution.

All comparisons use the same existing evaluator, unchanged L and episode budgets.
